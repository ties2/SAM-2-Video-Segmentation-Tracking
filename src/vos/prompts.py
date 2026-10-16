"""Derive SAM 2 prompts from a ground-truth mask.

Semi-supervised VOS benchmarks give the full first-frame mask, but a real
client clicks once or drags a box.  Comparing the three prompt types answers
"how much user effort does the tracker need?".

Click placement
---------------
A centroid can fall *outside* a non-convex object (a ring, a person with arms
out), so the first positive click is the interior pixel farthest from the
boundary (maximum of the distance transform).  Extra clicks are chosen by
farthest-point sampling among deep interior pixels, which spreads them over
the object instead of stacking them in one blob.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.vos.types import Prompt


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("cannot build a box from an empty mask")
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def interior_points(mask: np.ndarray, n: int = 1, min_depth_ratio: float = 0.35) -> np.ndarray:
    """``n`` well-separated positive click locations [x, y] inside ``mask``."""
    if not mask.any():
        raise ValueError("cannot place clicks on an empty mask")
    padded = np.pad(mask.astype(np.uint8), 1)  # treat the image border as background
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y0, x0 = np.unravel_index(np.argmax(dt), dt.shape)
    points = [(float(x0), float(y0))]
    if n == 1:
        return np.array(points, dtype=np.float32)

    deep = dt >= max(1.0, min_depth_ratio * dt.max())
    ys, xs = np.nonzero(deep)
    if xs.size == 0:
        ys, xs = np.nonzero(mask)
    cand = np.stack([xs, ys], axis=1).astype(np.float32)
    dist = np.linalg.norm(cand - np.array(points[0], dtype=np.float32), axis=1)
    while len(points) < n and cand.shape[0] > 0:
        k = int(np.argmax(dist))
        if dist[k] <= 0:
            break
        points.append((float(cand[k, 0]), float(cand[k, 1])))
        dist = np.minimum(dist, np.linalg.norm(cand - cand[k], axis=1))
    return np.array(points, dtype=np.float32)


def prompt_from_mask(mask: np.ndarray, kind: str, frame_idx: int = 0) -> Prompt:
    """Build a prompt of ``kind`` in {"mask", "box", "click", "click3"} from a GT mask."""
    mask = mask.astype(bool)
    if kind == "mask":
        return Prompt(kind="mask", frame_idx=frame_idx, mask=mask)
    if kind == "box":
        return Prompt(kind="box", frame_idx=frame_idx, box=mask_to_box(mask))
    if kind.startswith("click"):
        n = int(kind[len("click") :] or 1)
        pts = interior_points(mask, n=n)
        return Prompt(kind="points", frame_idx=frame_idx, points=pts, labels=np.ones(len(pts), dtype=np.int32))
    raise ValueError(f"unknown prompt kind: {kind}")
