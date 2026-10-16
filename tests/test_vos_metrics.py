"""DAVIS metric re-implementation: edge cases + agreement with hand-computed values."""

import numpy as np
import pytest

from src.vos import metrics


def _rect(h=100, w=200, y0=20, y1=60, x0=30, x1=110):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_jaccard_identical_is_one():
    m = _rect()
    assert metrics.jaccard(m, m) == 1.0


def test_jaccard_both_empty_is_one():
    z = np.zeros((10, 10), bool)
    assert metrics.jaccard(z, z) == 1.0  # DAVIS convention: "correctly absent"


def test_jaccard_gt_empty_pred_nonempty_is_zero():
    z = np.zeros((10, 10), bool)
    assert metrics.jaccard(z, ~z) == 0.0


def test_jaccard_known_overlap():
    a = _rect(x0=0, x1=100)
    b = _rect(x0=50, x1=150)
    # overlap 50 cols, union 150 cols (same rows)
    assert metrics.jaccard(a, b) == pytest.approx(50 / 150)


def test_jaccard_void_pixels_ignored():
    a = _rect(x0=0, x1=100)
    b = _rect(x0=0, x1=120)
    void = np.zeros_like(a)
    void[:, 100:120] = True
    assert metrics.jaccard(a, b, void) == 1.0


def test_f_measure_perfect_and_tolerance():
    m = _rect()
    assert metrics.f_measure(m, m) == 1.0
    shifted = np.roll(m, 2, axis=1)  # 2 px shift < 8 px tolerance at this size? (diag≈224 → 2 px)
    assert metrics.boundary_tolerance_px(m.shape) == 2
    assert metrics.f_measure(m, shifted) == pytest.approx(1.0)
    far = np.roll(m, 30, axis=1)
    assert metrics.f_measure(m, far) < 0.5


def test_f_measure_empty_cases():
    z = np.zeros((50, 50), bool)
    m = _rect(50, 50, 10, 30, 10, 30)
    assert metrics.f_measure(z, z) == 1.0
    assert metrics.f_measure(m, z) == 0.0  # missed object
    assert metrics.f_measure(z, m) == 0.0  # hallucinated object


def test_seg2bmap_is_one_pixel_boundary():
    m = _rect(20, 20, 5, 15, 5, 15)
    b = metrics.seg2bmap(m)
    assert b.sum() > 0
    assert not b[8:12, 8:12].any()  # interior has no boundary


def test_statistics_decay_positive_when_degrading():
    v = np.linspace(1.0, 0.2, 40)
    s = metrics.statistics(v)
    assert s.decay > 0.5
    assert 0 < s.recall < 1


def test_statistics_long_sequence_no_uint8_overflow():
    """The official toolkit casts bin indices to uint8; > 255 frames gives a wrong decay."""
    v = np.concatenate([np.ones(300), np.zeros(300)])
    s = metrics.statistics(v)
    assert s.decay == pytest.approx(1.0)


def test_evaluate_sequence_skips_first_last_and_handles_missing():
    gt = {1: {f: _rect() for f in range(6)}}
    pred = {1: {f: _rect() for f in range(6) if f != 3}}  # frame 3 lost
    score = metrics.evaluate_sequence("s", gt, pred)
    obj = score.objects[0]
    assert obj.frames == [1, 2, 3, 4]
    assert obj.j[2] == 0.0
    assert score.summary()["J_mean"] == pytest.approx(0.75)


def test_evaluate_sequence_respects_eval_frames():
    gt = {1: {f: _rect() for f in range(10)}}
    pred = {1: {f: _rect() for f in range(0, 10, 2)}}
    score = metrics.evaluate_sequence("s", gt, pred, eval_frames=list(range(0, 10, 2)))
    assert score.objects[0].frames == [2, 4, 6]
    assert score.summary()["J&F"] == pytest.approx(1.0)
