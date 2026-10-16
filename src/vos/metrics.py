"""DAVIS-standard video object segmentation metrics.

This is a faithful re-implementation of the official ``davis2017-evaluation``
toolkit (Pont-Tuset et al.), with two deliberate fixes that are documented in
the code review:

1. The official ``db_statistics`` casts bin indices to ``uint8``; for videos
   longer than 255 frames the indices overflow and the *decay* metric is
   silently wrong.  We use ``int64``.
2. Sparse ground truth (SA-V is annotated every 4th frame) is supported:
   only frames that have GT are scored.

Definitions
-----------
J  (region similarity)   = |P ∩ G| / |P ∪ G|            (1.0 when both are empty)
F  (contour accuracy)     = 2·Pc·Rc / (Pc + Rc) on mask boundaries, with a
                            tolerance of ``bound_th`` × image diagonal (0.008 → 8 px at 480p)
J&F                       = (J_mean + F_mean) / 2
Recall  (O)               = fraction of frames with score > 0.5
Decay   (D)               = mean(score in first quarter) − mean(score in last quarter)
                            → *positive decay means the mask degrades over time*,
                            which is exactly the "does segmentation persist?" question.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Region similarity J
# ---------------------------------------------------------------------------


def jaccard(gt: np.ndarray, pred: np.ndarray, void: np.ndarray | None = None) -> float:
    """Region similarity (IoU). Returns 1.0 if both masks are empty (DAVIS convention)."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if void is not None:
        keep = ~void.astype(bool)
        gt = gt & keep
        pred = pred & keep
    inter = np.count_nonzero(gt & pred)
    union = np.count_nonzero(gt | pred)
    if union == 0:
        return 1.0
    return inter / union


# ---------------------------------------------------------------------------
# Contour accuracy F
# ---------------------------------------------------------------------------


def seg2bmap(seg: np.ndarray) -> np.ndarray:
    """One-pixel-wide boundary map of a binary mask (port of DAVIS ``_seg2bmap``)."""
    seg = seg.astype(bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = (seg ^ e) | (seg ^ s) | (seg ^ se)
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def _disk(radius: int) -> np.ndarray:
    """Same structuring element as ``skimage.morphology.disk`` (no skimage dependency)."""
    r = int(radius)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def boundary_tolerance_px(shape: tuple[int, int], bound_th: float = 0.008) -> int:
    if bound_th >= 1:
        return int(bound_th)
    return int(math.ceil(bound_th * float(np.linalg.norm(shape))))


def f_measure(gt: np.ndarray, pred: np.ndarray, void: np.ndarray | None = None, bound_th: float = 0.008) -> float:
    """Boundary F-measure with a distance tolerance (port of DAVIS ``f_measure``)."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if void is not None:
        keep = ~void.astype(bool)
        gt = gt & keep
        pred = pred & keep

    bound_pix = boundary_tolerance_px(gt.shape, bound_th)
    fg_boundary = seg2bmap(pred)
    gt_boundary = seg2bmap(gt)
    kernel = _disk(bound_pix)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), kernel).astype(bool)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), kernel).astype(bool)

    gt_match = gt_boundary & fg_dil
    fg_match = fg_boundary & gt_dil
    n_fg = int(np.count_nonzero(fg_boundary))
    n_gt = int(np.count_nonzero(gt_boundary))

    if n_fg == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = np.count_nonzero(fg_match) / n_fg
        recall = np.count_nonzero(gt_match) / n_gt

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# ---------------------------------------------------------------------------
# Statistics over a sequence
# ---------------------------------------------------------------------------


@dataclass
class Stat:
    mean: float
    recall: float
    decay: float

    def as_dict(self) -> dict:
        return asdict(self)


def statistics(per_frame: np.ndarray) -> Stat:
    """Mean, recall (>0.5) and decay of a per-frame score vector (NaNs ignored)."""
    v = np.asarray(per_frame, dtype=float)
    if v.size == 0 or np.all(np.isnan(v)):
        return Stat(float("nan"), float("nan"), float("nan"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = float(np.nanmean(v))
        recall = float(np.nanmean(v[~np.isnan(v)] > 0.5))
        n_bins = 4
        # FIX vs. official toolkit: int64 instead of uint8 (overflow for > 255 frames)
        ids = (np.round(np.linspace(1, len(v), n_bins + 1) + 1e-10) - 1).astype(np.int64)
        bins = [v[ids[i] : ids[i + 1] + 1] for i in range(n_bins)]
        decay = float(np.nanmean(bins[0]) - np.nanmean(bins[3]))
    return Stat(mean, recall, decay)


# ---------------------------------------------------------------------------
# Sequence-level evaluation
# ---------------------------------------------------------------------------


@dataclass
class ObjectScore:
    obj_id: int
    frames: list[int]
    j: list[float]
    f: list[float]
    j_stat: Stat
    f_stat: Stat

    @property
    def jf(self) -> float:
        return (self.j_stat.mean + self.f_stat.mean) / 2


@dataclass
class SequenceScore:
    sequence: str
    objects: list[ObjectScore]

    def _mean(self, attr: str, field: str) -> float:
        vals = [getattr(getattr(o, attr), field) for o in self.objects]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def summary(self) -> dict:
        j_mean = self._mean("j_stat", "mean")
        f_mean = self._mean("f_stat", "mean")
        return {
            "sequence": self.sequence,
            "num_objects": len(self.objects),
            "J&F": (j_mean + f_mean) / 2,
            "J_mean": j_mean,
            "J_recall": self._mean("j_stat", "recall"),
            "J_decay": self._mean("j_stat", "decay"),
            "F_mean": f_mean,
            "F_recall": self._mean("f_stat", "recall"),
            "F_decay": self._mean("f_stat", "decay"),
        }


def evaluate_sequence(
    name: str,
    gt: dict[int, dict[int, np.ndarray]],
    pred: dict[int, dict[int, np.ndarray]],
    eval_frames: list[int] | None = None,
    skip_first_last: bool = True,
    bound_th: float = 0.008,
) -> SequenceScore:
    """Score predicted masks against (possibly sparse) GT.

    eval_frames      restrict scoring to these original-video frame indices
                     (used when the video was processed with a frame stride).
    skip_first_last  DAVIS / SA-V convention: the first annotated frame carries the
                     prompt and the last one is excluded by the official toolkits.
    Missing predictions count as an empty mask (i.e. the object was "lost").
    """
    objects: list[ObjectScore] = []
    for obj_id in sorted(gt):
        frames = sorted(gt[obj_id])
        if eval_frames is not None:
            allowed = set(eval_frames)
            frames = [f for f in frames if f in allowed]
        if skip_first_last and len(frames) > 2:
            frames = frames[1:-1]
        js, fs = [], []
        for f in frames:
            g = gt[obj_id][f]
            p = pred.get(obj_id, {}).get(f)
            if p is None:
                p = np.zeros_like(g, dtype=bool)
            js.append(jaccard(g, p))
            fs.append(f_measure(g, p, bound_th=bound_th))
        objects.append(
            ObjectScore(
                obj_id=obj_id,
                frames=frames,
                j=js,
                f=fs,
                j_stat=statistics(np.array(js)),
                f_stat=statistics(np.array(fs)),
            )
        )
    return SequenceScore(sequence=name, objects=objects)


def aggregate(scores: list[SequenceScore]) -> dict:
    """Dataset-level J&F: mean over *objects* of all sequences (DAVIS 2017 convention)."""
    objs = [o for s in scores for o in s.objects]
    if not objs:
        return {}

    def m(vals: list[float]) -> float:
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    j = m([o.j_stat.mean for o in objs])
    f = m([o.f_stat.mean for o in objs])
    return {
        "num_sequences": len(scores),
        "num_objects": len(objs),
        "J&F": (j + f) / 2,
        "J_mean": j,
        "J_recall": m([o.j_stat.recall for o in objs]),
        "J_decay": m([o.j_stat.decay for o in objs]),
        "F_mean": f,
        "F_recall": m([o.f_stat.recall for o in objs]),
        "F_decay": m([o.f_stat.decay for o in objs]),
    }
