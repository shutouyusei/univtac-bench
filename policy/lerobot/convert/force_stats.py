"""Where the recorded contact force becomes a reliable direction.

VRR's stiffness schedule comes from Adaptive Compliance Policy (arXiv 2410.09309),
which lowers the stiffness with the force magnitude because "the low stiffness
direction is estimated from noisy force signal": below ``f_min`` the direction is
not trusted. That makes ``f_min`` measurable: it is the magnitude from which the
force keeps its direction from one frame to the next. The simulator's force unit
is then fixed by mapping that magnitude to the paper's ``f_min`` in newtons.

A second, independent proposal maps the dataset's 90th percentile of the force
magnitude to a reference value in newtons (ImplicitRDP's released episodes have
about 12 N there). The two need not agree: the schedule is steep, the virtual
target leaves the end effector by 0.05 mm at ``f_min`` and by 25 mm at ``f_max``,
so ``describe_scale`` reports what a scale does before one is chosen.
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


def summarize(
    episodes: list[np.ndarray],
    per_decade: int,
    max_median_deg: float,
    min_count: int,
    paper_f_min: float,
    reference_p90: float | None = None,
) -> dict:
    """Force statistics of a dataset and the force units they imply.

    ``episodes`` holds one ``(T, 3)`` summed contact force per episode. Direction
    changes are taken within an episode only. ``force_scale`` maps the stable
    magnitude to ``paper_f_min`` newtons (``None`` when no magnitude is stable);
    ``force_scale_p90`` maps the 90th percentile to ``reference_p90`` newtons.
    """
    pairs = [direction_changes(forces) for forces in episodes]
    magnitude = np.concatenate([m for m, _ in pairs])
    angle = np.concatenate([a for _, a in pairs])
    norms = np.concatenate([np.linalg.norm(np.asarray(f, dtype=np.float64), axis=1) for f in episodes])
    positive = magnitude[magnitude > 0]
    low = 10.0 ** np.floor(np.log10(positive.min())) if positive.size else 1e-12
    high = 10.0 ** np.ceil(np.log10(max(norms.max(), low * 10.0)))
    edges = log_edges(low, high, per_decade)
    threshold = stable_force_threshold(magnitude, angle, edges, max_median_deg, min_count)
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
        "stable_force_threshold": threshold,
        "paper_f_min": paper_f_min,
        "force_scale": None if threshold is None else paper_f_min / threshold,
        "reference_p90": reference_p90,
        "force_scale_p90": None if reference_p90 is None else reference_p90 / float(np.percentile(norms, 90)),
    }


def describe_scale(forces: list[np.ndarray], ee_pos: list[np.ndarray], params: VirtualTargetParams) -> dict:
    """What ``params`` does to the dataset: share of frames per regime and the virtual target's offset [mm]."""
    newtons = np.concatenate([np.linalg.norm(np.asarray(f, dtype=np.float64), axis=1) for f in forces])
    newtons = newtons * params.force_scale
    offsets = 1000.0 * np.concatenate(
        [np.linalg.norm(virtual_target(e, f, params)[0] - e, axis=1) for f, e in zip(forces, ee_pos)]
    )
    return {
        "force_scale": params.force_scale,
        "share_below_f_min": float((newtons < params.f_min).mean()),
        "share_between": float(((newtons >= params.f_min) & (newtons <= params.f_max)).mean()),
        "share_above_f_max": float((newtons > params.f_max).mean()),
        "offset_mm": {
            k: float(np.percentile(offsets, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))
        },
    }
