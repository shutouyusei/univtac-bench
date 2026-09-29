"""Where the recorded contact force becomes a reliable direction.

VRR's stiffness schedule comes from Adaptive Compliance Policy (arXiv 2410.09309),
which lowers the stiffness with the force magnitude because "the low stiffness
direction is estimated from noisy force signal": below ``f_min`` the direction is
not trusted. That makes ``f_min`` measurable: it is the magnitude from which the
force keeps its direction from one frame to the next. The simulator's force unit
is then fixed by mapping that magnitude to the paper's ``f_min`` in newtons.
"""

from __future__ import annotations

import numpy as np


def direction_changes(forces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per consecutive frame pair: the smaller magnitude and the angle between the two forces [deg]."""
    forces = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
    norm = np.linalg.norm(forces, axis=1)
    unit = forces / np.clip(norm, 1e-30, None)[:, None]
    cosine = np.clip((unit[1:] * unit[:-1]).sum(axis=1), -1.0, 1.0)
    return np.minimum(norm[1:], norm[:-1]), np.degrees(np.arccos(cosine))


def log_edges(low: float, high: float, per_decade: int) -> np.ndarray:
    """Bin edges from ``low`` to ``high``, ``per_decade`` bins per factor of ten."""
    count = int(round(np.log10(high / low) * per_decade))
    return low * 10.0 ** (np.arange(count + 1) / per_decade)


def stability_table(magnitude: np.ndarray, angle: np.ndarray, edges: np.ndarray) -> list[dict]:
    """One row per non-empty magnitude bin: count, median and 90th percentile of the direction change."""
    magnitude, angle = np.asarray(magnitude), np.asarray(angle)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (magnitude >= low) & (magnitude < high)
        if inside.any():
            rows.append(
                {
                    "low": float(low),
                    "high": float(high),
                    "count": int(inside.sum()),
                    "median_deg": float(np.median(angle[inside])),
                    "p90_deg": float(np.percentile(angle[inside], 90)),
                }
            )
    return rows


def stable_force_threshold(
    magnitude: np.ndarray, angle: np.ndarray, edges: np.ndarray, max_median_deg: float, min_count: int
) -> float | None:
    """Lowest bin edge from which every populated bin changes direction by at most ``max_median_deg``.

    Bins with fewer than ``min_count`` samples do not decide. ``None`` when the
    highest populated bin is itself unstable.
    """
    rows = [r for r in stability_table(magnitude, angle, edges) if r["count"] >= min_count]
    threshold = None
    for row in reversed(rows):
        if row["median_deg"] > max_median_deg:
            break
        threshold = row["low"]
    return threshold
