"""Auxiliary action targets of VRR (ImplicitRDP, arXiv 2512.10946): virtual target and adaptive stiffness.

The action of a transition is widened from the joint target to
``[joint (8), virtual target (3), stiffness (1)]``. The policy generates and is
supervised on all of it; only the joints are executed. The virtual target is the
end-effector position displaced against the external contact force,

    x_vt = x_ee - f / k_adp(|f|)

with the stiffness falling linearly from ``k_max`` below ``f_min`` to ``k_min``
above ``f_max`` (the schedule of Adaptive Compliance Policy, arXiv 2410.09309):
in free motion the target stays on the end effector, in contact it moves away
from it by centimetres. Sign and frame follow ImplicitRDP's released code
(``post_process_data.py``), which subtracts the offset in the robot base frame.

The constants default to the paper's. The simulator's force is not in newtons,
so ``force_scale`` converts it first; it is the one quantity chosen from the
dataset (``force_stats``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

AUX_ACTION_NAMES = ["virtual_target_x", "virtual_target_y", "virtual_target_z", "stiffness"]


@dataclass(frozen=True)
class VirtualTargetParams:
    force_scale: float = 1.0  # newtons per unit of the recorded force
    f_min: float = 0.5  # N
    f_max: float = 5.0  # N
    k_min: float = 200.0  # N/m
    k_max: float = 10000.0  # N/m

    def __post_init__(self) -> None:
        if self.force_scale <= 0:
            raise ValueError(f"force_scale must be > 0, got {self.force_scale}")
        if not 0 <= self.f_min < self.f_max:
            raise ValueError(f"need 0 <= f_min < f_max, got {self.f_min} and {self.f_max}")
        if not 0 < self.k_min < self.k_max:
            raise ValueError(f"need 0 < k_min < k_max, got {self.k_min} and {self.k_max}")

    def to_dict(self) -> dict:
        return asdict(self)


def adaptive_stiffness(force_norm: np.ndarray, params: VirtualTargetParams) -> np.ndarray:
    """``k_adp`` [N/m] for force magnitudes in newtons."""
    norm = np.asarray(force_norm, dtype=np.float64)
    ratio = np.clip((norm - params.f_min) / (params.f_max - params.f_min), 0.0, 1.0)
    return params.k_max - (params.k_max - params.k_min) * ratio


def virtual_target(
    ee_pos: np.ndarray, force: np.ndarray, params: VirtualTargetParams
) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: virtual target ``(T, 3)`` [m] and stiffness ``(T,)`` [N/m].

    ``ee_pos`` and ``force`` share one frame of reference; ``force`` is in the
    recorded unit and is multiplied by ``params.force_scale`` first.
    """
    ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(-1, 3)
    newtons = params.force_scale * np.asarray(force, dtype=np.float64).reshape(-1, 3)
    if len(ee_pos) != len(newtons):
        raise ValueError(f"{len(ee_pos)} end-effector frames but {len(newtons)} force frames")
    stiffness = adaptive_stiffness(np.linalg.norm(newtons, axis=1), params)
    return ee_pos - newtons / stiffness[:, None], stiffness


def augment_actions(
    action: np.ndarray, ee_pos: np.ndarray, force: np.ndarray, params: VirtualTargetParams
) -> np.ndarray:
    """``(N, A)`` actions -> ``(N, A + 4)``: each action followed by its own frame's targets.

    The action of transition t is the joint state of frame t + 1, so its virtual
    target and stiffness are those of frame t + 1 as well; the per-frame inputs
    hold ``N + 1`` frames.
    """
    action = np.asarray(action, dtype=np.float32)
    if len(ee_pos) != len(action) + 1:
        raise ValueError(f"{len(action)} transitions need {len(action) + 1} frames, got {len(ee_pos)}")
    x_vt, stiffness = virtual_target(ee_pos, force, params)
    aux = np.concatenate([x_vt[1:], stiffness[1:, None]], axis=1).astype(np.float32)
    return np.concatenate([action, aux], axis=1)
