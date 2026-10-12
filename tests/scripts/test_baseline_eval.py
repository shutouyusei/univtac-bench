"""scripts/baseline_eval.py: the fixed-seed evaluation protocol shared with streaming-tactile-flow's runs
(one eval_policy.sh call over start..start+count-1, one more call per run of consecutive seeds without an
outcome, outcomes merged), and the success table."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "baseline_eval", Path(__file__).resolve().parents[2] / "scripts" / "baseline_eval.py")
be = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(be)


def test_seed_range():
    assert be.seed_range(1000000, 3) == [1000000, 1000001, 1000002]


def test_contiguous_ranges():
    assert be.contiguous_ranges([]) == []
    assert be.contiguous_ranges([5, 6, 7, 10, 12, 13]) == [(5, 3), (10, 1), (12, 2)]
    assert be.contiguous_ranges([7, 5, 6]) == [(5, 3)]


def test_load_outcomes_keeps_entries_with_a_result(tmp_path):
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps({"10": {"result": "success"}, "11": {"result": "fail", "x": 1}, "12": {"cost_step": 3}}))
    assert be.load_outcomes(p) == {10: {"result": "success"}, 11: {"result": "fail", "x": 1}}


def test_missing_seeds():
    assert be.missing_seeds({1: {"result": "success"}, 3: {"result": "fail"}}, [1, 2, 3, 4]) == [2, 4]


def test_merge_keeps_the_first_outcome_and_only_wanted_seeds():
    a = {1: {"result": "success"}, 9: {"result": "fail"}}
    b = {1: {"result": "fail"}, 2: {"result": "fail"}}
    assert be.merge([a, b], [1, 2, 3]) == {1: {"result": "success"}, 2: {"result": "fail"}}


def test_eval_command_matches_the_stflow_protocol(tmp_path):
    cmd = be.eval_command("lift_bottle", "clean51", tmp_path / "act_x.yml", 1000000, 100, "UniVTAC")
    assert cmd[:5] == ["conda", "run", "-n", "UniVTAC", "--no-capture-output"]
    assert cmd[5:10] == ["bash", "eval_policy.sh", "lift_bottle", "clean51", str(tmp_path / "act_x.yml")]
    tail = cmd[10:]
    assert tail[tail.index("--start_seed") + 1] == "1000000"
    assert tail[tail.index("--total_num") + 1] == "100"
    assert tail[tail.index("--max_seed") + 1] == "1000099"
    assert "--headless" in tail


def test_results_dir_follows_eval_policy_naming(tmp_path):
    yml = tmp_path / "act_train_config.yml"
    yml.write_text("policy_name: ACT\n")
    assert be.results_dir(Path("/r"), "lift_bottle", yml) == Path("/r/eval_result/ACT/lift_bottle/act_train_config")


def test_parse_env():
    assert be.parse_env(["TRAIN_CONFIG=train_config", "EP_NUM=50"]) == {"TRAIN_CONFIG": "train_config", "EP_NUM": "50"}
    with pytest.raises(ValueError):
        be.parse_env(["NOEQUALS"])


def test_wilson_interval():
    lo, hi = be.wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35
    lo, hi = be.wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3) and hi == pytest.approx(0.596, abs=1e-3)
    assert be.wilson(0, 0) == (0.0, 0.0)


def test_summary_line():
    outcomes = {1: {"result": "success"}, 2: {"result": "fail"}, 3: {"result": "success"}}
    line = be.summary_line("act lift_bottle", outcomes, [1, 2, 3, 4])
    assert line.startswith("act lift_bottle: 2/3 success")
    assert "without outcome: [4]" in line
