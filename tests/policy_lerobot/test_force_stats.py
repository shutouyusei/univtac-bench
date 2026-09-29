"""Force statistics that fix the force unit before training: where the force direction becomes reliable."""

import numpy as np
import pytest

from policy.lerobot.convert.force_stats import (
    direction_changes,
    log_edges,
    stability_table,
    stable_force_threshold,
)


def test_direction_changes_pairs_consecutive_frames():
    forces = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]])
    magnitude, angle = direction_changes(forces)
    np.testing.assert_allclose(magnitude, [1.0, 2.0])  # the smaller norm of each pair
    np.testing.assert_allclose(angle, [90.0, 0.0], atol=1e-6)


def test_direction_changes_of_a_zero_force_is_not_a_number_free():
    magnitude, angle = direction_changes(np.zeros((3, 3)))
    assert np.isfinite(angle).all() and np.isfinite(magnitude).all()


def test_log_edges_are_evenly_spaced_per_decade():
    edges = log_edges(1e-3, 1e-1, per_decade=2)
    np.testing.assert_allclose(edges, [1e-3, 10 ** -2.5, 1e-2, 10 ** -1.5, 1e-1])


def make_samples():
    rng = np.random.default_rng(0)
    # below 1e-2 the direction jumps by ~40 deg, above it by ~5 deg
    noisy = np.stack([rng.uniform(1e-3, 1e-2, 400), rng.normal(40.0, 5.0, 400)], axis=1)
    stable = np.stack([rng.uniform(1e-2, 1e-1, 400), rng.normal(5.0, 1.0, 400)], axis=1)
    samples = np.concatenate([noisy, stable])
    return samples[:, 0], samples[:, 1]


def test_stability_table_reports_each_bin():
    magnitude, angle = make_samples()
    rows = stability_table(magnitude, angle, log_edges(1e-3, 1e-1, per_decade=1))
    assert [r["count"] for r in rows] == [400, 400]
    assert rows[0]["median_deg"] > 30 and rows[1]["median_deg"] < 10


def test_threshold_is_the_lowest_edge_from_which_every_bin_is_stable():
    magnitude, angle = make_samples()
    threshold = stable_force_threshold(
        magnitude, angle, log_edges(1e-3, 1e-1, per_decade=2), max_median_deg=10.0, min_count=20
    )
    assert threshold == pytest.approx(1e-2)


def test_a_noisy_bin_above_resets_the_threshold():
    magnitude, angle = make_samples()
    # make the top half decade noisy again: only nothing above it is stable
    top = magnitude >= 10 ** -1.5
    angle = np.where(top, 45.0, angle)
    assert stable_force_threshold(magnitude, angle, log_edges(1e-3, 1e-1, per_decade=2), 10.0, 20) is None


def test_sparse_bins_do_not_decide():
    magnitude, angle = make_samples()
    magnitude = np.append(magnitude, [0.5] * 3)  # three wild samples in an otherwise empty bin
    angle = np.append(angle, [90.0] * 3)
    threshold = stable_force_threshold(magnitude, angle, log_edges(1e-3, 1.0, per_decade=2), 10.0, min_count=20)
    assert threshold == pytest.approx(1e-2)


def test_summarize_maps_the_stable_magnitude_to_the_papers_f_min():
    from policy.lerobot.convert.force_stats import summarize

    rng = np.random.default_rng(1)
    up = np.array([0.0, 0.0, 1.0])
    episodes = []
    for _ in range(4):
        noisy = rng.normal(size=(200, 3)) * 1e-3  # random direction, small
        stable = up * rng.uniform(2e-2, 8e-2, size=(200, 1)) + rng.normal(size=(200, 3)) * 1e-3
        episodes.append(np.concatenate([noisy, stable]))
    report = summarize(episodes, per_decade=2, max_median_deg=10.0, min_count=20, paper_f_min=0.5)
    assert report["frames"] == 1600 and report["episodes"] == 4
    assert 3e-3 <= report["stable_force_threshold"] <= 3e-2
    assert report["force_scale"] == pytest.approx(0.5 / report["stable_force_threshold"])
    assert set(report["force_percentiles"]) == {"p50", "p90", "p99", "max"}
    assert report["table"][0]["low"] < report["table"][-1]["low"]


def test_summarize_also_proposes_the_scale_that_matches_a_reference_p90():
    from policy.lerobot.convert.force_stats import summarize

    forces = [np.tile([0.0, 0.0, 1.0], (100, 1)) * np.linspace(0.0, 0.01, 100)[:, None]]
    report = summarize(forces, per_decade=2, max_median_deg=10.0, min_count=5, paper_f_min=0.5, reference_p90=12.0)
    assert report["force_scale_p90"] == pytest.approx(12.0 / report["force_percentiles"]["p90"])
    assert summarize(forces, 2, 10.0, 5, 0.5)["force_scale_p90"] is None


def test_regime_shares_and_offsets_describe_what_a_scale_does():
    from policy.lerobot.convert.aux_targets import VirtualTargetParams
    from policy.lerobot.convert.force_stats import describe_scale

    force = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.001], [0.0, 0.0, 0.01]])  # x1000: 0, 1, 10 N
    ee = np.zeros((3, 3))
    out = describe_scale([force], [ee], VirtualTargetParams(force_scale=1000.0))
    assert out["share_below_f_min"] == pytest.approx(1 / 3)
    assert out["share_between"] == pytest.approx(1 / 3)
    assert out["share_above_f_max"] == pytest.approx(1 / 3)
    assert out["offset_mm"]["max"] == pytest.approx(50.0)  # 10 N / 200 N/m


def test_direction_changes_never_pair_frames_of_different_episodes():
    from policy.lerobot.convert.force_stats import summarize

    a = np.tile([0.0, 0.0, 0.05], (50, 1))
    b = np.tile([0.05, 0.0, 0.0], (50, 1))  # 90 deg away from a, but a different episode
    report = summarize([a, b], per_decade=2, max_median_deg=10.0, min_count=20, paper_f_min=0.5)
    assert all(row["p90_deg"] < 1.0 for row in report["table"])
