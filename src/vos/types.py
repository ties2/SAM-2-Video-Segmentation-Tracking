"""Plain data classes shared by every VOS module.

Conventions used everywhere in the package
-----------------------------------------
* Masks are ``numpy.ndarray`` of dtype ``bool`` and shape ``(H, W)``.
* Frame indices are 0-based positions in the *original* video.
* Object ids are positive integers (0 is reserved for background, as in DAVIS).
* Ground truth may be *sparse*: SA-V is annotated every 4th frame, so GT is a
  ``dict[obj_id, dict[frame_idx, mask]]`` and only annotated frames are scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

PromptKind = Literal["mask", "box", "points"]


@dataclass
class Prompt:
    """A user interaction on one frame for one object.

    kind="mask"   -> ``mask`` (H, W) bool
    kind="box"    -> ``box`` [x0, y0, x1, y1] in pixels (inclusive-exclusive)
    kind="points" -> ``points`` (N, 2) float [x, y] and ``labels`` (N,) int (1=positive, 0=negative)
    """

    kind: PromptKind
    frame_idx: int = 0
    mask: np.ndarray | None = None
    box: np.ndarray | None = None
    points: np.ndarray | None = None
    labels: np.ndarray | None = None

    def describe(self) -> str:
        if self.kind == "mask":
            return f"mask@{self.frame_idx}"
        if self.kind == "box":
            return f"box@{self.frame_idx}"
        n = 0 if self.points is None else len(self.points)
        return f"{n}click@{self.frame_idx}"


@dataclass
class Sequence:
    """A video plus (optionally) its ground-truth masks."""

    name: str
    source: str  # "davis" | "sav" | "synthetic" | "unlabeled"
    frame_paths: list[Path]
    height: int
    width: int
    gt: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    fps: float = 24.0
    meta: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)

    @property
    def object_ids(self) -> list[int]:
        return sorted(self.gt.keys())

    @property
    def has_gt(self) -> bool:
        return bool(self.gt)

    def annotated_frames(self, obj_id: int) -> list[int]:
        return sorted(self.gt.get(obj_id, {}).keys())

    def first_visible_frame(self, obj_id: int) -> int | None:
        for f in self.annotated_frames(obj_id):
            if self.gt[obj_id][f].any():
                return f
        return None


@dataclass
class SegmentationResult:
    """Output of a video segmenter for one run.

    masks[obj_id][frame_idx]      -> (H, W) bool, frame_idx in *original* video indexing
    confidence[obj_id][frame_idx] -> mean foreground probability inside the predicted mask
    object_score[obj_id][frame_idx] -> SAM 2 "object present" logit (None if unavailable)
    frame_latency_s               -> wall-clock seconds per processed frame (propagation order)
    """

    masks: dict[int, dict[int, np.ndarray]]
    confidence: dict[int, dict[int, float]] = field(default_factory=dict)
    uncertain_fraction: dict[int, dict[int, float]] = field(default_factory=dict)
    object_score: dict[int, dict[int, float | None]] = field(default_factory=dict)
    frame_latency_s: list[float] = field(default_factory=list)
    init_latency_s: float = 0.0
    prompt_latency_s: float = 0.0
    peak_rss_mb: float = 0.0
    processed_frames: list[int] = field(default_factory=list)

    @property
    def total_latency_s(self) -> float:
        return self.init_latency_s + self.prompt_latency_s + float(sum(self.frame_latency_s))

    @property
    def fps(self) -> float:
        prop = float(sum(self.frame_latency_s))
        return len(self.frame_latency_s) / prop if prop > 0 else 0.0
