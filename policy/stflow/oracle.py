"""A privileged correction for the insertion phase of a held-prism task (insert_hole): the joint offset that moves
the prism onto the target axis, from the simulator's poses the policy never sees.

With the prism's pose in the target frame (``rel``), the error is the lateral offset ``rel[:2, 3]`` and the tilt
of the prism axis ``rel[:3, 2]`` against the target axis ``e_z``. The prism should translate by ``(-x, -y, 0)``
and rotate by the axis-angle that brings its axis onto ``e_z`` (both turned into the world frame); the hand
holding it rigidly then needs the twist ``v_ee = v_prism + w x (p_ee - p_prism)``, ``w``, and the arm joints
the damped least-squares solution of ``J dq = [v_ee; w]``. The correction applies only once the alignment
rotation has been made (tilt below ``max_tilt_deg``): before it, the large tilt is the plan's to remove.
"""

from __future__ import annotations

import numpy as np


def prism_error(target: np.ndarray, prism: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """World-frame translation and rotation (axis * angle) that put the prism (4x4) on the target axis
    (4x4), and the tilt in degrees."""
    rel = np.linalg.inv(target) @ prism
    axis = rel[:3, 2] / np.linalg.norm(rel[:3, 2])
    cos = float(np.clip(axis[2], -1.0, 1.0))
    tilt = float(np.degrees(np.arccos(cos)))
    cross = np.cross(axis, [0.0, 0.0, 1.0])
    s = np.linalg.norm(cross)
    rot_t = np.zeros(3) if s < 1e-9 else cross / s * np.arccos(cos)
    trans_t = np.array([-rel[0, 3], -rel[1, 3], 0.0])
    r = target[:3, :3]
    return r @ trans_t, r @ rot_t, tilt


def hand_twist(v_prism: np.ndarray, w: np.ndarray, p_prism: np.ndarray, p_ee: np.ndarray) -> np.ndarray:
    """The 6-d twist ``[v; w]`` of the hand frame that moves a rigidly held prism by ``v_prism``, ``w``."""
    return np.concatenate([v_prism + np.cross(w, p_ee - p_prism), w])


def joint_offset(jacobian: np.ndarray, twist: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Damped least squares ``J^T (J J^T + d^2 I)^-1 twist`` for a (6, n) Jacobian."""
    jjt = jacobian @ jacobian.T + damping**2 * np.eye(6)
    return jacobian.T @ np.linalg.solve(jjt, twist)


def correction(
    target: np.ndarray,
    prism: np.ndarray,
    p_ee: np.ndarray,
    jacobian: np.ndarray,
    max_tilt_deg: float = 10.0,
) -> tuple[np.ndarray, float]:
    """Arm joint offset (n,) that removes the prism's lateral and tilt error, zero while the tilt is above
    ``max_tilt_deg``; and the tilt in degrees."""
    v, w, tilt = prism_error(target, prism)
    if tilt > max_tilt_deg:
        return np.zeros(jacobian.shape[1]), tilt
    return joint_offset(jacobian, hand_twist(v, w, prism[:3, 3], p_ee)), tilt
