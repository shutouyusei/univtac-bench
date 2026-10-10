"""A mid-episode shift of the target the scripted expert tracks (e.g. insert_hole's slot).

The schedule is drawn once per episode from its own generator, so the scene the task draws from ``self.rng``
stays the same as the unperturbed task on the same seed. The target then moves, from ``start`` steps (the
caller's clock) after the scripted pre-move, along a horizontal ``direction`` at ``speed`` metres per step until it
has moved ``distance``; it stays there. Nothing here touches the simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ShiftSchedule:
    start: int
    direction: tuple[float, float]
    distance: float
    speed: float

    def offset(self, step: int) -> np.ndarray:
        """World-frame displacement ``(3,)`` of the target ``step`` steps after the pre-move."""
        moved = min(max(step - self.start, 0) * self.speed, self.distance)
        dx, dy = self.direction
        return np.array([dx * moved, dy * moved, 0.0])

    def as_metadata(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def schedule_rng(seed: int) -> np.random.Generator:
    """The shift's own generator for an episode seed (independent of the task's scene generator)."""
    return np.random.default_rng([int(seed), 0x5EED])


def draw_schedule(
    rng: np.random.Generator,
    prob: float,
    window: tuple[int, int],
    distance: tuple[float, float],
    speed: float,
) -> ShiftSchedule | None:
    """``None`` with probability ``1 - prob``; otherwise a start step uniform in ``window`` (inclusive), a
    uniform horizontal direction and a distance uniform in ``distance`` (metres). Every draw is made either
    way, so one seed always consumes the same numbers."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    if window[0] < 0 or window[1] < window[0]:
        raise ValueError(f"window must be 0 <= lo <= hi, got {window}")
    if distance[0] < 0 or distance[1] < distance[0]:
        raise ValueError(f"distance must be 0 <= lo <= hi, got {distance}")
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    shifted = rng.random() < prob
    start = int(rng.integers(window[0], window[1] + 1))
    angle = rng.uniform(0.0, 2.0 * np.pi)
    dist = float(rng.uniform(distance[0], distance[1]))
    if not shifted:
        return None
    return ShiftSchedule(start, (float(np.cos(angle)), float(np.sin(angle))), dist, float(speed))
