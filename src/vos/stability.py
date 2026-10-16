"""Ground-truth-free temporal stability metrics.

In production you never have ground truth, yet a client still asks
"does the mask stay on the object, without flicker or dropped frames?".
These metrics answer that from the predictions alone, so they can be
computed for every request and exported to Prometheus (see
``src/monitoring/drift_detector.py``).

Per object we measure
---------------------
temporal IoU (tIoU)        IoU(mask_t, mask_t+1). Low → flicker / jitter, but also
                           drops for genuinely fast motion.
aligned IoU                IoU after shifting mask_t so its centroid matches mask_t+1.
                           Removes translation for free (no optical flow); what is left is
                           change of *shape* — flicker, leakage, deformation.
warped IoU                 IoU(warp(mask_t → t+1), mask_t+1) using dense optical flow.
                           Motion-compensated, so it isolates *segmentation* instability
                           from *object* motion (a.k.a. warping error / flow consistency).
area jump rate             share of transitions with |ΔA| / A > 30 %.
empty / dropout frames     frames with an empty mask; an *isolated dropout* is an empty
                           frame surrounded by non-empty ones (a "frame drop" in a roto job).
lost / recovered events    visible→empty and empty→visible transitions.
centroid jitter            median |second difference| of the centroid (px/frame²).
confidence                 mean foreground probability, share of uncertain pixels, and the
                           SAM 2 object-presence logit (when supplied by the segmenter).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

AREA_JUMP_THRESHOLD = 0.30


@dataclass
class StabilityReport:
    obj_id: int
    num_frames: int
    tiou_mean: float
    tiou_p05: float
    aligned_iou_mean: float
    aligned_iou_p05: float
    warped_iou_mean: float
    warped_iou_p05: float
    area_cv: float
    area_jump_rate: float
    empty_frames: int
    empty_rate: float
    isolated_dropouts: int
    longest_empty_run: int
    lost_events: int
    recovered_events: int
    centroid_jitter_px: float
    confidence_mean: float
    uncertain_fraction_mean: float
    object_score_min: float

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in asdict(self).items()}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    if union == 0:
        return float("nan")
    return np.count_nonzero(a & b) / union


def aligned_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU after translating ``a`` so that its centroid coincides with ``b``'s (integer pixels)."""
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    dy = int(round(yb.mean() - ya.mean()))
    dx = int(round(xb.mean() - xa.mean()))
    h, w = a.shape
    ys, xs = ya + dy, xa + dx
    keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    shifted = np.zeros_like(a, dtype=bool)
    shifted[ys[keep], xs[keep]] = True
    return _iou(shifted, b)


def backward_flow(frame_t: np.ndarray, frame_t1: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Dense Farnebäck flow from frame t+1 back to frame t, at full resolution.

    Backward flow lets us *sample* mask_t at the location each pixel of t+1 came from,
    which avoids holes that forward splatting would create.
    """
    g0 = cv2.cvtColor(frame_t, cv2.COLOR_RGB2GRAY) if frame_t.ndim == 3 else frame_t
    g1 = cv2.cvtColor(frame_t1, cv2.COLOR_RGB2GRAY) if frame_t1.ndim == 3 else frame_t1
    h, w = g0.shape
    if scale != 1.0:
        g0s = cv2.resize(g0, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        g1s = cv2.resize(g1, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        g0s, g1s = g0, g1
    flow = cv2.calcOpticalFlowFarneback(g1s, g0s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    if scale != 1.0:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    return flow


def warp_mask(mask_t: np.ndarray, flow_back: np.ndarray) -> np.ndarray:
    """Warp mask_t into frame t+1 using backward flow (t+1 → t)."""
    h, w = mask_t.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx + flow_back[..., 0].astype(np.float32)
    map_y = gy + flow_back[..., 1].astype(np.float32)
    warped = cv2.remap(mask_t.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    return warped > 0.5


def _runs(flags: list[bool]) -> list[int]:
    runs, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def object_stability(
    obj_id: int,
    masks: dict[int, np.ndarray],
    frames: dict[int, np.ndarray] | None = None,
    confidence: dict[int, float] | None = None,
    uncertain_fraction: dict[int, float] | None = None,
    object_score: dict[int, float | None] | None = None,
    flow_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> StabilityReport:
    """Compute a :class:`StabilityReport` from one object's per-frame masks.

    ``frames`` (RGB uint8, same keys as ``masks``) enables the flow-compensated metric.
    ``flow_cache`` lets several objects in the same video share the flow computation.
    """
    idx = sorted(masks)
    seq = [masks[i].astype(bool) for i in idx]
    n = len(seq)
    areas = np.array([np.count_nonzero(m) for m in seq], dtype=float)
    empty = [a == 0 for a in areas]

    tious, aious, wious, jumps = [], [], [], []
    for k in range(n - 1):
        a, b = seq[k], seq[k + 1]
        if areas[k] > 0 and areas[k + 1] > 0:
            tious.append(_iou(a, b))
            aious.append(aligned_iou(a, b))
            jumps.append(abs(areas[k + 1] - areas[k]) / max(areas[k], 1.0) > AREA_JUMP_THRESHOLD)
            if frames is not None and idx[k] in frames and idx[k + 1] in frames:
                key = (idx[k], idx[k + 1])
                flow = flow_cache.get(key) if flow_cache is not None else None
                if flow is None:
                    flow = backward_flow(frames[idx[k]], frames[idx[k + 1]])
                    if flow_cache is not None:
                        flow_cache[key] = flow
                wious.append(_iou(warp_mask(a, flow), b))

    vis_areas = areas[areas > 0]
    area_cv = float(vis_areas.std() / vis_areas.mean()) if vis_areas.size > 1 else float("nan")

    dropouts = sum(1 for k in range(1, n - 1) if empty[k] and not empty[k - 1] and not empty[k + 1])
    empty_runs = _runs(empty)
    lost = sum(1 for k in range(1, n) if empty[k] and not empty[k - 1])
    recovered = sum(1 for k in range(1, n) if not empty[k] and empty[k - 1])

    centroids = []
    for m, a in zip(seq, areas, strict=True):
        if a > 0:
            ys, xs = np.nonzero(m)
            centroids.append((xs.mean(), ys.mean()))
        else:
            centroids.append(None)
    accels = []
    for k in range(1, n - 1):
        c0, c1, c2 = centroids[k - 1], centroids[k], centroids[k + 1]
        if c0 is not None and c1 is not None and c2 is not None:
            accels.append(math.hypot(c2[0] - 2 * c1[0] + c0[0], c2[1] - 2 * c1[1] + c0[1]))

    def pct(v: list[float], q: float) -> float:
        v = [x for x in v if not math.isnan(x)]
        return float(np.percentile(v, q)) if v else float("nan")

    def mean(v: list[float]) -> float:
        v = [x for x in v if x is not None and not math.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    conf = [confidence[i] for k, i in enumerate(idx) if confidence and i in confidence and areas[k] > 0]
    unc = [uncertain_fraction[i] for i in idx if uncertain_fraction and i in uncertain_fraction]
    scores = [object_score[i] for i in idx if object_score and object_score.get(i) is not None]

    return StabilityReport(
        obj_id=obj_id,
        num_frames=n,
        tiou_mean=mean(tious),
        tiou_p05=pct(tious, 5),
        aligned_iou_mean=mean(aious),
        aligned_iou_p05=pct(aious, 5),
        warped_iou_mean=mean(wious),
        warped_iou_p05=pct(wious, 5),
        area_cv=area_cv,
        area_jump_rate=float(np.mean(jumps)) if jumps else float("nan"),
        empty_frames=int(sum(empty)),
        empty_rate=float(np.mean(empty)) if n else float("nan"),
        isolated_dropouts=int(dropouts),
        longest_empty_run=int(max(empty_runs) if empty_runs else 0),
        lost_events=int(lost),
        recovered_events=int(recovered),
        centroid_jitter_px=float(np.median(accels)) if accels else float("nan"),
        confidence_mean=mean(conf),
        uncertain_fraction_mean=mean(unc),
        object_score_min=float(min(scores)) if scores else float("nan"),
    )
