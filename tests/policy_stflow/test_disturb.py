"""The disturbance schedule: one draw per episode seed, the offset only from its start step on."""

import numpy as np

from policy.stflow.disturb import ACTION_DIM, draw

CFG = {"magnitude": 0.0005, "window": [10, 150]}


def test_the_same_seed_draws_the_same_disturbance():
    a, b = draw(CFG, 1000003), draw(CFG, 1000003)
    assert a.start == b.start and np.array_equal(a.offset, b.offset)


def test_starts_fall_inside_the_window_and_vary_across_seeds():
    starts = [draw(CFG, s).start for s in range(1000000, 1000200)]
    assert min(starts) >= 10 and max(starts) <= 150
    assert len(set(starts)) > 50


def test_the_offset_goes_to_the_gripper_by_default_and_only_from_the_start_on():
    d = draw(CFG, 1000007)
    assert not d.at(d.start - 1).any()
    expected = np.zeros(ACTION_DIM, dtype=np.float32)
    expected[-1] = 0.0005
    assert np.array_equal(d.at(d.start), expected) and np.array_equal(d.at(d.start + 100), expected)


def test_prob_zero_disturbs_no_episode():
    d = draw({**CFG, "prob": 0.0}, 1000000)
    assert d.start is None and not d.at(10_000).any()
