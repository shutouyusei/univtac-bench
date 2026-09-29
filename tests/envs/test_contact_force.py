import numpy as np
import pytest

from envs.sensors.contact_force import accumulate_vertex_forces, gel_vertex_forces, wrench_about


def test_gel_vertex_forces_keeps_only_the_gel_window_and_flips_the_gradient_sign():
    ids = np.array([0, 5, 6, 9, 10])
    grads = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3], [4, 0, 0], [0, 5, 0]], dtype=float)
    local, forces = gel_vertex_forces(ids, grads, offset=5, n_verts=5)
    np.testing.assert_array_equal(local, [0, 1, 4])
    np.testing.assert_allclose(forces, [[0, -2, 0], [0, 0, -3], [-4, 0, 0]])


def test_gel_vertex_forces_with_no_hit_returns_empty_arrays():
    local, forces = gel_vertex_forces(np.array([1, 2]), np.zeros((2, 3)), offset=10, n_verts=3)
    assert local.shape == (0,)
    assert forces.shape == (0, 3)


def test_accumulate_vertex_forces_sums_repeated_vertices():
    per_vertex = accumulate_vertex_forces(
        n_verts=3,
        contributions=[
            (np.array([0, 2]), np.array([[1.0, 0, 0], [0, 1.0, 0]])),
            (np.array([2]), np.array([[0, 1.0, 0]])),
        ],
    )
    np.testing.assert_allclose(per_vertex, [[1, 0, 0], [0, 0, 0], [0, 2, 0]])


def test_wrench_about_sums_forces_and_takes_moments_about_the_point():
    positions = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    forces = np.array([[0, 1.0, 0], [0, 0, 1.0]])
    force, torque = wrench_about(point=np.zeros(3), positions=positions, forces=forces)
    np.testing.assert_allclose(force, [0, 1, 1])
    # r x F: (1,0,0)x(0,1,0) = (0,0,1); (0,1,0)x(0,0,1) = (1,0,0)
    np.testing.assert_allclose(torque, [1, 0, 1])


def test_wrench_about_is_translation_aware():
    positions = np.array([[2.0, 0, 0]])
    forces = np.array([[0, 1.0, 0]])
    _, torque = wrench_about(point=np.array([1.0, 0, 0]), positions=positions, forces=forces)
    np.testing.assert_allclose(torque, [0, 0, 1])


def test_wrench_about_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        wrench_about(np.zeros(3), np.zeros((2, 3)), np.zeros((3, 3)))
