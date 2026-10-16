import numpy as np
import pytest

from src.monitoring.drift_detector import MaskQualityMonitor, QualityDriftDetector
from src.vos import stability
from src.vos.prompts import interior_points, mask_to_box, prompt_from_mask


def _disk(h, w, cx, cy, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


# ---------------- stability -----------------------------------------------------------


def test_static_mask_is_perfectly_stable():
    m = _disk(64, 64, 32, 32, 10)
    rep = stability.object_stability(1, {f: m for f in range(10)})
    assert rep.tiou_mean == pytest.approx(1.0)
    assert rep.empty_frames == 0
    assert rep.isolated_dropouts == 0
    assert rep.area_jump_rate == 0.0


def test_dropout_and_recovery_are_counted():
    m = _disk(64, 64, 32, 32, 10)
    z = np.zeros_like(m)
    seq = {0: m, 1: m, 2: z, 3: m, 4: z, 5: z, 6: m}
    rep = stability.object_stability(1, seq)
    assert rep.isolated_dropouts == 1  # frame 2
    assert rep.longest_empty_run == 2  # frames 4-5
    assert rep.lost_events == 2 and rep.recovered_events == 2


def test_flow_compensation_removes_motion_penalty():
    """A textured disk translating 6 px/frame: raw tIoU drops, warped IoU stays high."""
    h, w = 96, 160
    rng = np.random.default_rng(0)
    tex = (rng.random((h, w + 80)) * 255).astype(np.uint8)
    frames, masks = {}, {}
    for t in range(5):
        img = np.full((h, w), 90, np.uint8)
        m = _disk(h, w, 40 + 6 * t, 48, 18)
        img[m] = tex[:, 6 * t : 6 * t + w][m]
        frames[t] = np.repeat(img[..., None], 3, axis=2)
        masks[t] = m
    rep = stability.object_stability(1, masks, frames=frames)
    assert rep.tiou_mean < 0.8
    assert rep.warped_iou_mean > rep.tiou_mean


# ---------------- prompts ------------------------------------------------------------


def test_click_lands_inside_non_convex_object():
    ring = _disk(100, 100, 50, 50, 40) & ~_disk(100, 100, 50, 50, 25)
    (x, y) = interior_points(ring, n=1)[0]
    assert ring[int(y), int(x)]  # a centroid (50, 50) would be outside the ring


def test_three_clicks_are_spread_out_and_inside():
    m = np.zeros((60, 200), bool)
    m[20:40, 10:190] = True
    pts = interior_points(m, n=3)
    assert len(pts) == 3
    assert all(m[int(y), int(x)] for x, y in pts)
    assert np.ptp(pts[:, 0]) > 100


def test_box_and_prompt_kinds():
    m = np.zeros((10, 10), bool)
    m[2:5, 3:8] = True
    assert mask_to_box(m).tolist() == [3, 2, 8, 5]
    assert prompt_from_mask(m, "mask").kind == "mask"
    assert prompt_from_mask(m, "box").kind == "box"
    p = prompt_from_mask(m, "click3")
    assert p.kind == "points" and len(p.points) <= 3 and p.labels.tolist() == [1] * len(p.points)
    with pytest.raises(ValueError):
        prompt_from_mask(m, "scribble")


# ---------------- monitoring ---------------------------------------------------------


def test_monitor_flags_expected_failures():
    mon = MaskQualityMonitor()
    good = {
        "tiou_mean": 0.95,
        "empty_rate": 0.0,
        "isolated_dropouts": 0,
        "area_jump_rate": 0.0,
        "confidence_mean": 0.97,
    }
    assert mon.evaluate(good) == []
    bad = {"tiou_mean": 0.4, "empty_rate": 0.5, "isolated_dropouts": 2, "area_jump_rate": 0.4, "confidence_mean": 0.6}
    assert set(mon.evaluate(bad)) == {"flicker", "object_lost", "frame_dropout", "area_jump", "low_confidence"}
    assert mon.evaluate({"tiou_mean": None}) == []


def test_drift_detector_fires_on_sustained_drop_only():
    det = QualityDriftDetector(reference_size=30, window_size=10)
    rng = np.random.default_rng(0)
    for v in rng.normal(0.92, 0.01, 30):
        assert det.update(float(v))["drift"] is False
    outs = [det.update(float(v)) for v in rng.normal(0.92, 0.01, 10)]
    assert outs[-1]["drift"] is False
    outs = [det.update(float(v)) for v in rng.normal(0.70, 0.01, 10)]
    assert outs[-1]["drift"] is True


def test_aligned_iou_removes_translation_but_not_shape_change():
    a = _disk(80, 160, 40, 40, 15)
    moved = _disk(80, 160, 110, 40, 15)
    assert stability._iou(a, moved) == 0.0  # raw temporal IoU: looks like total flicker
    assert stability.aligned_iou(a, moved) == pytest.approx(1.0)  # pure translation
    grown = _disk(80, 160, 110, 40, 25)
    assert stability.aligned_iou(a, grown) < 0.5  # shape change is still caught


def test_monitor_prefers_motion_compensated_iou():
    mon = MaskQualityMonitor()
    fast_but_clean = {"tiou_mean": 0.3, "aligned_iou_mean": 0.9, "empty_rate": 0.0, "confidence_mean": 0.97}
    assert "flicker" not in mon.evaluate(fast_but_clean)
    shape_noise = {"tiou_mean": 0.9, "aligned_iou_mean": 0.5}
    assert "flicker" in mon.evaluate(shape_noise)
