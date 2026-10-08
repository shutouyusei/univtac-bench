"""The insertion-phase oracle: the prism error in the target frame, and the joint offset whose hand motion
removes it."""

import numpy as np

from policy.stflow.oracle import correction, hand_twist, joint_offset, prism_error


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def pose(r=np.eye(3), p=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p
    return t


def test_an_aligned_prism_has_no_error():
    target = pose(rot_y(np.pi / 6), (0.5, 0.0, 0.1))
    v, w, tilt = prism_error(target, target.copy())
    assert np.allclose(v, 0) and np.allclose(w, 0) and tilt < 1e-6


def test_a_lateral_offset_is_undone_in_the_world_frame():
    target = pose(rot_y(np.pi / 6), (0.5, 0.0, 0.1))
    prism = target @ pose(p=(0.002, -0.001, -0.02))
    v, w, _ = prism_error(target, prism)
    moved = prism.copy()
    moved[:3, 3] += v
    rel = np.linalg.inv(target) @ moved
    assert np.allclose(rel[:2, 3], 0, atol=1e-9) and np.isclose(rel[2, 3], -0.02) and np.allclose(w, 0)


def test_rotating_by_the_returned_axis_angle_aligns_the_prism():
    target = pose(rot_y(np.pi / 6))
    prism = target @ pose(rot_y(np.radians(3.0)))
    _, w, tilt = prism_error(target, prism)
    assert np.isclose(tilt, 3.0)
    angle = np.linalg.norm(w)
    k = w / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    r = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
    aligned = r @ prism[:3, :3]
    assert np.allclose(aligned[:, 2], target[:3, 2], atol=1e-9)


def test_the_hand_twist_keeps_the_rigid_offset_and_the_joint_offset_realises_it():
    w = np.array([0.0, 0.02, 0.0])
    twist = hand_twist(np.zeros(3), w, np.zeros(3), np.array([0.0, 0.0, 0.1]))
    assert np.allclose(twist[:3], np.cross(w, [0, 0, 0.1]))
    jac = np.random.default_rng(0).normal(size=(6, 7))
    dq = joint_offset(jac, twist, damping=1e-6)
    assert np.allclose(jac @ dq, twist, atol=1e-6)


def test_no_correction_before_the_alignment_rotation():
    target = pose(rot_y(np.pi / 6))
    prism = pose()  # still upright: 30 degrees from the target axis
    dq, tilt = correction(target, prism, np.array([0, 0, 0.1]), np.eye(6, 7))
    assert np.isclose(tilt, 30.0) and not dq.any()
