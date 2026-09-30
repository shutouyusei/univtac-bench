"""Paired statistics of the sim summary (pure logic, no simulator)."""

import math

import pytest

from policy.lerobot.analysis import sim_summary


def ep(result, rotate=0.0, early=False):
    return {"result": result, "rotate": rotate, "early_stop": early}


def test_mcnemar_counts_discordant_pairs_on_shared_seeds_only():
    a = {"1": ep("success"), "2": ep("success"), "3": ep("failed"), "4": ep("failed"), "9": ep("success")}
    b = {"1": ep("success"), "2": ep("failed"), "3": ep("success"), "4": ep("failed")}
    n, both, na, nb, p = sim_summary.mcnemar(a, b)
    assert (n, both, na, nb) == (4, 1, 1, 1)
    assert p == pytest.approx(1.0)


def test_mcnemar_is_exact_and_two_sided():
    a = {str(i): ep("success") for i in range(6)}
    b = {str(i): ep("failed") for i in range(6)}
    n, both, na, nb, p = sim_summary.mcnemar(a, b)
    assert (n, both, na, nb) == (6, 0, 6, 0)
    assert p == pytest.approx(2 / 64)


def test_row_splits_by_hole_orientation_and_counts_early_stops():
    v = {"1": ep("success", 0.0), "2": ep("failed", math.pi, early=True), "3": ep("success", math.pi)}
    line = sim_summary.row("x", v)
    assert line.startswith("| x | 2/3 | 66.7% |")
    assert "| 1/1 | 1/2 | 1/3 |" in line


def test_wilson_interval_brackets_the_rate():
    lo, hi = sim_summary.wilson(25, 100)
    assert lo < 25 < hi and lo == pytest.approx(17.5, abs=0.1) and hi == pytest.approx(34.3, abs=0.1)
