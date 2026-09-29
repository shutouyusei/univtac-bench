"""What the recorded contact force looks like, and what VRR's targets make of it.

Two views of a dataset's summed fingertip force [N]:

* how much its direction changes from one frame to the next, per magnitude bin.
  The stiffness schedule VRR borrows from Adaptive Compliance Policy (arXiv
  2410.09309) exists because "the low stiffness direction is estimated from noisy
  force signal": below ``f_min`` the direction is not trusted. The table shows
  from which magnitude it can be;
* what a set of VRR constants does to the dataset: the share of frames below
  ``f_min``, in the linear range and above ``f_max``, and how far the virtual
  target leaves the end effector. The schedule is steep (0.05 mm of offset at
  0.5 N, 25 mm at 5 N), so this is worth looking at before training.
"""

from __future__ import annotations

import numpy as np

from .aux_targets import VirtualTargetParams, virtual_target


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


def summarize(episodes: list[np.ndarray], per_decade: int, max_median_deg: float, min_count: int) -> dict:
    """Distribution of the force magnitude and the stability of its direction.

    ``episodes`` holds one ``(T, 3)`` summed contact force per episode. Direction
    changes are taken within an episode only.
    """
    pairs = [direction_changes(forces) for forces in episodes]
    magnitude = np.concatenate([m for m, _ in pairs])
    angle = np.concatenate([a for _, a in pairs])
    norms = np.concatenate([np.linalg.norm(np.asarray(f, dtype=np.float64), axis=1) for f in episodes])
    positive = magnitude[magnitude > 0]
    low = 10.0 ** np.floor(np.log10(positive.min())) if positive.size else 1e-12
    high = 10.0 ** np.ceil(np.log10(max(norms.max(), low * 10.0)))
    edges = log_edges(low, high, per_decade)
    return {
        "episodes": len(episodes),
        "frames": int(len(norms)),
        "force_percentiles": {
            "p50": float(np.percentile(norms, 50)),
            "p90": float(np.percentile(norms, 90)),
            "p99": float(np.percentile(norms, 99)),
            "max": float(norms.max()),
        },
        "table": stability_table(magnitude, angle, edges),
        "criterion": {"max_median_deg": max_median_deg, "min_count": min_count, "per_decade": per_decade},
        "stable_force_threshold": stable_force_threshold(magnitude, angle, edges, max_median_deg, min_count),
    }


def describe_targets(forces: list[np.ndarray], ee_pos: list[np.ndarray], params: VirtualTargetParams) -> dict:
    """What ``params`` makes of the dataset: share of frames per regime and the virtual target's offset [mm]."""
    newtons = np.concatenate([np.linalg.norm(np.asarray(f, dtype=np.float64), axis=1) for f in forces])
    newtons = newtons * params.force_scale
    offsets = 1000.0 * np.concatenate(
        [np.linalg.norm(virtual_target(e, f, params)[0] - e, axis=1) for f, e in zip(forces, ee_pos)]
    )
    return {
        "params": params.to_dict(),
        "share_below_f_min": float((newtons < params.f_min).mean()),
        "share_between": float(((newtons >= params.f_min) & (newtons <= params.f_max)).mean()),
        "share_above_f_max": float((newtons > params.f_max).mean()),
        "offset_mm": {
            k: float(np.percentile(offsets, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))
        },
    }
