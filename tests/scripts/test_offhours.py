"""scripts/offhours.py: run one queued unit only inside the off-hours window, stop and requeue it at the
window's end or when the pause file appears."""

import importlib.util
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "offhours", Path(__file__).resolve().parents[2] / "scripts" / "offhours.py")
oh = importlib.util.module_from_spec(SPEC)
sys.modules["offhours"] = oh
SPEC.loader.exec_module(oh)

CFG = oh.Config(weekday_start=21, weekday_end=8, weekend_all_day=True, holidays=["2026-11-03"],
                margin_min=10, grace_min=30)


def dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


# 2026-10-12 is a Monday, 2026-10-16 a Friday, 2026-10-17 a Saturday.
@pytest.mark.parametrize("when,allowed", [
    ("2026-10-12 20:59", False), ("2026-10-12 21:00", True), ("2026-10-13 07:59", True),
    ("2026-10-13 08:00", False), ("2026-10-13 12:00", False), ("2026-10-17 12:00", True),
    ("2026-10-18 23:00", True), ("2026-10-19 07:00", True), ("2026-10-19 09:00", False),
    ("2026-11-03 12:00", True),
])
def test_allowed(when, allowed):
    assert CFG.allowed(dt(when)) is allowed


def test_window_end_weeknight():
    assert oh.window_end(dt("2026-10-12 22:00"), CFG) == dt("2026-10-13 08:00")


def test_window_end_runs_through_the_weekend():
    assert oh.window_end(dt("2026-10-16 22:00"), CFG) == dt("2026-10-19 08:00")


def test_window_end_outside_is_none():
    assert oh.window_end(dt("2026-10-13 12:00"), CFG) is None


def test_next_window_start():
    assert oh.next_window_start(dt("2026-10-13 12:00"), CFG) == dt("2026-10-13 21:00")
    assert oh.next_window_start(dt("2026-10-13 22:00"), CFG) == dt("2026-10-13 22:00")


def test_can_start_needs_estimate_plus_margin_before_the_end():
    end = dt("2026-10-13 08:00")
    assert oh.can_start(dt("2026-10-13 05:00"), end, estimate_min=170, cfg=CFG)
    assert not oh.can_start(dt("2026-10-13 05:00"), end, estimate_min=171, cfg=CFG)
    assert not oh.can_start(dt("2026-10-13 05:00"), None, estimate_min=1, cfg=CFG)


def test_estimate_uses_the_longer_of_given_and_measured(tmp_path):
    store = oh.Durations(tmp_path / "d.json")
    assert store.estimate("act_train", 60) == 60
    store.record("act_train", 100.0)
    assert store.estimate("act_train", 60) == pytest.approx(110.0)
    assert oh.Durations(tmp_path / "d.json").estimate("act_train", 200) == 200


def test_requeue_command_repeats_the_invocation():
    argv = ["scripts/offhours.py", "--kind", "k", "--estimate-min", "5", "--", "bash", "x.sh", "a b"]
    cmd = oh.requeue_command("lbl", argv, python="/py")
    assert cmd == ["tsp", "-L", "lbl", "/py", "scripts/offhours.py", "--kind", "k", "--estimate-min", "5", "--",
                   "bash", "x.sh", "a b"]


def test_run_until_finishes_normally(tmp_path):
    status, code = oh.run_until(["bash", "-c", "exit 3"], deadline=datetime.now() + timedelta(seconds=30),
                                pause_file=tmp_path / "pause", poll_s=0.05)
    assert (status, code) == ("done", 3)


def test_run_until_cuts_at_the_deadline(tmp_path):
    t0 = time.monotonic()
    status, _ = oh.run_until(["sleep", "30"], deadline=datetime.now() + timedelta(seconds=0.3),
                             pause_file=tmp_path / "pause", poll_s=0.05, kill_after_s=1)
    assert status == "cut" and time.monotonic() - t0 < 5


def test_run_until_cuts_when_paused(tmp_path):
    pause = tmp_path / "pause"
    pause.touch()
    status, _ = oh.run_until(["sleep", "30"], deadline=datetime.now() + timedelta(hours=1),
                             pause_file=pause, poll_s=0.05, kill_after_s=1)
    assert status == "cut"


def test_run_until_kills_the_whole_process_group(tmp_path):
    marker = tmp_path / "child_alive"
    script = f"(sleep 2; touch {marker}) & wait"
    status, _ = oh.run_until(["bash", "-c", script], deadline=datetime.now() + timedelta(seconds=0.2),
                             pause_file=tmp_path / "pause", poll_s=0.05, kill_after_s=1)
    time.sleep(2.5)
    assert status == "cut" and not marker.exists()


def test_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("weekday_start: 22\nweekday_end: 7\nweekend_all_day: false\nholidays: []\n")
    cfg = oh.Config.load(p)
    assert (cfg.weekday_start, cfg.weekday_end, cfg.weekend_all_day) == (22, 7, False)
    assert cfg.allowed(dt("2026-10-17 12:00")) is False
    assert oh.Config.load(tmp_path / "missing.yaml") == oh.Config()


def test_gpu_busy_threshold():
    assert not oh.busy_from_smi("", 2048)
    assert not oh.busy_from_smi("1963610, 604\n", 2048)
    assert oh.busy_from_smi("1, 604\n2, 1500\n", 2048)
    assert oh.busy_from_smi("1, [N/A]\n", 2048)
