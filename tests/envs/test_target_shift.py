import numpy as np
import pytest

from envs.utils.target_shift import ShiftSchedule, draw_schedule, schedule_rng


def test_offset_ramps_at_the_speed_from_the_start_step_and_then_holds():
    s = ShiftSchedule(start=10, direction=(0.6, 0.8), distance=0.003, speed=0.0005)
    np.testing.assert_allclose(s.offset(0), [0, 0, 0])
    np.testing.assert_allclose(s.offset(10), [0, 0, 0])
    np.testing.assert_allclose(s.offset(12), [0.6 * 0.001, 0.8 * 0.001, 0])
    np.testing.assert_allclose(s.offset(16), [0.6 * 0.003, 0.8 * 0.003, 0])  # 6 steps would be 3 mm
    np.testing.assert_allclose(s.offset(500), [0.6 * 0.003, 0.8 * 0.003, 0])


def test_draws_are_reproducible_per_seed_and_respect_the_ranges():
    a = draw_schedule(schedule_rng(7), 1.0, (40, 120), (0.002, 0.004), 0.0005)
    b = draw_schedule(schedule_rng(7), 1.0, (40, 120), (0.002, 0.004), 0.0005)
    assert a == b
    starts, dists = [], []
    for seed in range(200):
        s = draw_schedule(schedule_rng(seed), 1.0, (40, 120), (0.002, 0.004), 0.0005)
        assert np.isclose(np.hypot(*s.direction), 1.0)
        starts.append(s.start)
        dists.append(s.distance)
    assert min(starts) >= 40 and max(starts) <= 120
    assert min(dists) >= 0.002 and max(dists) <= 0.004


def test_probability_zero_never_shifts_and_one_seed_uses_the_same_numbers_either_way():
    assert all(draw_schedule(schedule_rng(s), 0.0, (40, 120), (0.002, 0.004), 0.0005) is None for s in range(50))
    shifted = sum(draw_schedule(schedule_rng(s), 0.8, (40, 120), (0.002, 0.004), 0.0005) is not None for s in range(500))
    assert 350 < shifted < 450
    g1, g2 = schedule_rng(3), schedule_rng(3)
    draw_schedule(g1, 0.0, (40, 120), (0.002, 0.004), 0.0005)
    draw_schedule(g2, 1.0, (40, 120), (0.002, 0.004), 0.0005)
    assert g1.random() == g2.random()


def test_the_schedule_generator_is_independent_of_the_scene_generator():
    scene = np.random.default_rng(7)
    before = scene.random()
    draw_schedule(schedule_rng(7), 1.0, (40, 120), (0.002, 0.004), 0.0005)
    assert np.random.default_rng(7).random() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prob": 1.5}, {"window": (10, 5)}, {"window": (-1, 5)}, {"distance": (0.004, 0.002)}, {"speed": 0.0},
    ],
)
def test_bad_arguments_are_rejected(kwargs):
    args = {"prob": 1.0, "window": (40, 120), "distance": (0.002, 0.004), "speed": 0.0005}
    args.update(kwargs)
    with pytest.raises(ValueError):
        draw_schedule(schedule_rng(0), **args)


def test_metadata_is_json_friendly():
    s = ShiftSchedule(start=10, direction=(0.6, 0.8), distance=0.003, speed=0.0005)
    assert s.as_metadata() == {"start": 10, "direction": [0.6, 0.8], "distance": 0.003, "speed": 0.0005}
