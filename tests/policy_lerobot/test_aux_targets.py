"""The VRR auxiliary targets (ImplicitRDP, arXiv 2512.10946): adaptive stiffness and virtual target."""

import numpy as np
import pytest

from policy.lerobot.convert.aux_targets import (
    AUX_ACTION_NAMES,
    VirtualTargetParams,
    adaptive_stiffness,
    augment_actions,
    virtual_target,
)

PAPER = VirtualTargetParams()  # 0.5 N, 5 N, 200 N/m, 10000 N/m, force already in newtons


def test_defaults_are_the_papers_constants():
    assert (PAPER.f_min, PAPER.f_max, PAPER.k_min, PAPER.k_max, PAPER.force_scale) == (0.5, 5.0, 200.0, 10000.0, 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"f_min": 5.0, "f_max": 5.0}, {"f_min": -1.0}, {"k_min": 0.0}, {"k_min": 300.0, "k_max": 200.0}, {"force_scale": 0.0}],
)
def test_inconsistent_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        VirtualTargetParams(**kwargs)


def test_stiffness_is_k_max_below_f_min_and_k_min_above_f_max():
    k = adaptive_stiffness(np.array([0.0, 0.49, 5.01, 50.0]), PAPER)
    np.testing.assert_allclose(k, [10000.0, 10000.0, 200.0, 200.0])


def test_stiffness_falls_linearly_between_the_thresholds():
    k = adaptive_stiffness(np.array([0.5, 2.75, 5.0]), PAPER)
    np.testing.assert_allclose(k, [10000.0, 5100.0, 200.0])


def test_virtual_target_equals_the_pose_without_force():
    ee = np.array([[0.4, 0.0, 0.3], [0.5, 0.1, 0.2]])
    x_vt, k = virtual_target(ee, np.zeros((2, 3)), PAPER)
    np.testing.assert_allclose(x_vt, ee)
    np.testing.assert_allclose(k, [10000.0, 10000.0])


def test_virtual_target_is_the_pose_minus_force_over_stiffness():
    ee = np.array([[0.4, 0.0, 0.3]])
    force = np.array([[0.0, 0.0, 10.0]])  # above f_max: k = 200 N/m, offset 10 / 200 = 0.05 m
    x_vt, k = virtual_target(ee, force, PAPER)
    np.testing.assert_allclose(k, [200.0])
    np.testing.assert_allclose(x_vt, [[0.4, 0.0, 0.25]])


def test_force_scale_converts_the_recorded_unit_to_newtons_before_anything_else():
    ee = np.zeros((1, 3))
    force = np.array([[0.0, 0.0, 0.004]])
    scaled = VirtualTargetParams(force_scale=2500.0)  # 0.004 -> 10 N
    x_vt, k = virtual_target(ee, force, scaled)
    np.testing.assert_allclose(k, [200.0])
    np.testing.assert_allclose(x_vt, [[0.0, 0.0, -0.05]])


def test_virtual_target_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        virtual_target(np.zeros((3, 3)), np.zeros((2, 3)), PAPER)


def test_augment_actions_appends_the_targets_of_the_actions_own_frame():
    # T = 4 frames -> 3 transitions; action[t] is joint[t + 1], so its targets come from frame t + 1
    action = np.arange(24, dtype=np.float32).reshape(3, 8)
    ee = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    force = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    out = augment_actions(action, ee, force, PAPER)
    assert out.shape == (3, 12) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :8], action)
    np.testing.assert_allclose(out[:, 8:11], [[1.0, 0.0, -0.05], [2.0, 0.0, 0.0], [3.0, 0.0, -0.05]], atol=1e-6)
    np.testing.assert_allclose(out[:, 11], [200.0, 10000.0, 200.0])


def test_augment_actions_needs_one_more_frame_than_transitions():
    with pytest.raises(ValueError):
        augment_actions(np.zeros((3, 8), np.float32), np.zeros((3, 3)), np.zeros((3, 3)), PAPER)


def test_aux_action_names_cover_the_four_added_dims():
    assert AUX_ACTION_NAMES == ["virtual_target_x", "virtual_target_y", "virtual_target_z", "stiffness"]
