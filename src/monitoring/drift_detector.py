"""Ground-truth-free mask quality monitoring and drift detection.

Two layers:

1. :class:`MaskQualityMonitor` — *per request, per object* rules on the stability
   report produced by ``src/vos/stability.py``.  Each rule maps to a failure a
   client would notice (object lost, flicker, frame drops, low confidence).
   Default thresholds were calibrated on the Phase-1 benchmark (see
   ``results/phase1``): they sit below the 5th percentile of runs whose J&F > 0.8.

2. :class:`QualityDriftDetector` — *across requests*.  It keeps a reference window
   of a quality signal (e.g. per-request mean temporal IoU) and compares the recent
   window against it with a one-sided mean-shift test.  A sustained drop means the
   input distribution changed (new camera, night footage, motion blur …) even though
   no labels exist — the video equivalent of data drift.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_tiou: float = 0.70  # mean consecutive-mask IoU (fallback only)
    min_aligned_iou: float = 0.70  # centroid-aligned consecutive IoU (translation removed)
    min_warped_iou: float = 0.70  # flow-compensated (only when computed)
    max_empty_rate: float = 0.25  # share of frames with an empty mask
    max_isolated_dropouts: int = 0  # single-frame drop-outs = visible "flash" in a roto job
    max_area_jump_rate: float = 0.15  # share of transitions with > 30 % area change
    min_confidence: float = 0.80  # mean foreground probability inside the mask


class MaskQualityMonitor:
    def __init__(self, thresholds: QualityThresholds | None = None):
        self.t = thresholds or QualityThresholds()

    def evaluate(self, report: dict) -> list[str]:
        """Return the list of alert reasons for one object's stability report."""
        t = self.t
        alerts: list[str] = []

        def val(key):
            v = report.get(key)
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

        # Flicker = the mask's *shape* changes between frames. Raw temporal IoU also drops for fast
        # but perfectly tracked objects (false alarms on sports / kids footage), so use a
        # motion-compensated estimate. Centroid-aligned IoU comes first: in the Phase-1 benchmark,
        # flow-warped IoU collapsed (0.0–0.3) at large displacements where tracking was fine,
        # because dense optical flow itself fails there.
        for key, threshold in (
            ("aligned_iou_mean", t.min_aligned_iou),
            ("warped_iou_mean", t.min_warped_iou),
            ("tiou_mean", t.min_tiou),
        ):
            if val(key) is not None:
                if val(key) < threshold:
                    alerts.append("flicker")
                break
        if val("empty_rate") is not None and val("empty_rate") > t.max_empty_rate:
            alerts.append("object_lost")
        if (val("isolated_dropouts") or 0) > t.max_isolated_dropouts:
            alerts.append("frame_dropout")
        if val("area_jump_rate") is not None and val("area_jump_rate") > t.max_area_jump_rate:
            alerts.append("area_jump")
        if val("confidence_mean") is not None and val("confidence_mean") < t.min_confidence:
            alerts.append("low_confidence")
        return alerts


class QualityDriftDetector:
    """Rolling mean-shift detector on a scalar quality signal (higher = better)."""

    def __init__(self, reference_size: int = 50, window_size: int = 20, min_drop: float = 0.05, z: float = 3.0):
        self.reference: deque[float] = deque(maxlen=reference_size)
        self.window: deque[float] = deque(maxlen=window_size)
        self.min_drop = min_drop
        self.z = z

    def update(self, value: float | None) -> dict:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return {"drift": False, "reason": "no_value"}
        if len(self.reference) < self.reference.maxlen:
            self.reference.append(value)
            return {"drift": False, "reason": "warming_up", "reference_n": len(self.reference)}
        self.window.append(value)
        if len(self.window) < self.window.maxlen:
            return {"drift": False, "reason": "filling_window", "window_n": len(self.window)}
        ref = np.asarray(self.reference)
        cur = np.asarray(self.window)
        drop = float(ref.mean() - cur.mean())
        se = float(ref.std(ddof=1) / math.sqrt(len(cur))) if len(ref) > 1 else 0.0
        z = drop / se if se > 0 else (math.inf if drop > 0 else 0.0)
        drift = drop >= self.min_drop and z >= self.z
        return {
            "drift": bool(drift),
            "reference_mean": float(ref.mean()),
            "window_mean": float(cur.mean()),
            "drop": drop,
            "z": z,
        }
