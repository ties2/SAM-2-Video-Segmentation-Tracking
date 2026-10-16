"""The synthetic scenarios must contain the failure modes they claim to contain."""

import numpy as np
import pytest

from src.vos import synthetic


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    return {s: synthetic.generate(s, root, num_frames=48) for s in synthetic.SCENARIOS}


def test_frames_and_gt_are_aligned(scenarios):
    for seq in scenarios.values():
        assert seq.num_frames == 48
        for per in seq.gt.values():
            assert sorted(per) == list(range(48))
            assert all(m.shape == (seq.height, seq.width) and m.dtype == bool for m in per.values())


def test_occlusion_has_a_fully_hidden_window_and_a_clean_control(scenarios):
    seq = scenarios["occlusion"]
    hidden = synthetic.hidden_frames(seq, 1)
    assert len(hidden) >= 10 and hidden == list(range(hidden[0], hidden[-1] + 1))  # contiguous
    assert seq.gt[1][0].any() and seq.gt[1][47].any()  # visible before and after
    assert synthetic.hidden_frames(seq, 2) == []
    areas = [seq.gt[2][f].sum() for f in range(48)]
    assert np.ptp(areas) < 0.1 * np.mean(areas)  # control never occluded


def test_exit_reentry_leaves_and_returns(scenarios):
    hidden = synthetic.hidden_frames(scenarios["exit_reentry"], 1)
    assert 8 <= len(hidden) <= 14 and hidden[-1] < 47


def test_distractor_crosses_in_front_of_target(scenarios):
    seq = scenarios["distractor"]
    d = seq.meta["distractor_masks"]
    areas = np.array([seq.gt[1][f].sum() for f in range(48)])
    assert areas.min() < 0.6 * areas.max()  # target partly covered at the crossing
    assert all(not (seq.gt[1][f] & d[f]).any() for f in range(48))  # modal GT: no overlap


def test_fast_motion_moves_fast(scenarios):
    seq = scenarios["fast_motion"]
    cx = [np.nonzero(seq.gt[1][f])[1].mean() for f in range(48)]
    assert np.median(np.abs(np.diff(cx))) >= 40
