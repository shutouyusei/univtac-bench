"""A known disturbance added to the policy's joint targets, for measuring how the tactile reading and the
outcome respond to it.

Deploy key ``stflow_disturb`` (absent: no disturbance):

* ``dim``        action index the offset goes to; default -1, the gripper (finger joint, metres; positive opens)
* ``magnitude``  offset added from ``start`` to the end of the episode
* ``window``     ``[lo, hi]`` policy steps; ``start`` is drawn uniformly from it (inclusive)
* ``prob``       chance that an episode is disturbed at all; default 1

The draw depends on the episode's seed only, so the same seed is disturbed the same way in every evaluation
and the policy's own noise stream is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTION_DIM = 8


@dataclass(frozen=True)
class Disturbance:
    start: int | None  # policy step from which the offset applies; None: this episode is not disturbed
    offset: np.ndarray  # (ACTION_DIM,)

    def at(self, step: int) -> np.ndarray:
        if self.start is None or step < self.start:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        return self.offset


def draw(cfg: dict, episode_seed: int) -> Disturbance:
    rng = np.random.default_rng([int(episode_seed), 0x5D157])
    if rng.random() >= float(cfg.get("prob", 1.0)):
        return Disturbance(None, np.zeros(ACTION_DIM, dtype=np.float32))
    lo, hi = (int(v) for v in cfg["window"])
    offset = np.zeros(ACTION_DIM, dtype=np.float32)
    offset[int(cfg.get("dim", -1))] = float(cfg["magnitude"])
    return Disturbance(int(rng.integers(lo, hi + 1)), offset)
