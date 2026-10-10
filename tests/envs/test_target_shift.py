"""The disturbance schedule of insert_hole_disturb: its own generator, a ramp that stops at the distance."""

import numpy as np

from envs.utils.target_shift import draw_schedule, schedule_rng


def test_schedule_is_deterministic_per_seed_and_independent_of_the_scene_generator():
    a = draw_schedule(schedule_rng(7), 1.0, (240, 320), (0.004, 0.004), 0.00025)
    b = draw_schedule(schedule_rng(7), 1.0, (240, 320), (0.004, 0.004), 0.00025)
    assert a == b
    assert 240 <= a.start <= 320
    assert abs(np.hypot(*a.direction) - 1.0) < 1e-9


def test_offset_ramps_at_speed_then_holds_the_distance():
    s = draw_schedule(schedule_rng(0), 1.0, (10, 10), (0.004, 0.004), 0.001)
    assert np.allclose(s.offset(9), 0.0)
    assert abs(np.linalg.norm(s.offset(12)) - 0.002) < 1e-9
    assert abs(np.linalg.norm(s.offset(100)) - 0.004) < 1e-9
    assert s.offset(100)[2] == 0.0


def test_prob_zero_draws_nothing_but_consumes_the_same_numbers():
    rng = schedule_rng(3)
    assert draw_schedule(rng, 0.0, (0, 10), (0.001, 0.002), 0.001) is None
    after_none = rng.random()
    rng2 = schedule_rng(3)
    draw_schedule(rng2, 1.0, (0, 10), (0.001, 0.002), 0.001)
    assert after_none == rng2.random()
