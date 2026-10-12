"""policy/ACT/baseline.sh in dry-run mode: which config, data, checkpoint and evaluation each stage uses."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SH = REPO / "policy" / "ACT" / "baseline.sh"


def run(*args, **env):
    full = {**os.environ, "DRY_RUN": "1", **{k: str(v) for k, v in env.items()}}
    p = subprocess.run(["bash", str(SH), *args], capture_output=True, text=True, env=full)
    return p.returncode, p.stdout + p.stderr


@pytest.mark.parametrize("task,cfg", [("lift_bottle", "train_config"), ("insert_hole", "train_config"),
                                      ("lift_can", "train_config_all"), ("insert_tube", "train_config_all")])
def test_tactile_arm_uses_the_task_camera_setting(task, cfg):
    code, out = run(task, "train")
    assert code == 0, out
    assert f"--config_path ./{cfg}.yml" in out
    assert f"--ckpt_dir ./act_ckpt/act-{task}/clean51-50/{cfg}" in out
    assert f"--task_name sim-{task}-clean51-50" in out


def test_vision_arm():
    code, out = run("lift_can", "train", ACT_ARM="vision")
    assert code == 0, out
    assert "--config_path ./train_config_vision_all.yml" in out


def test_eval_uses_the_shared_protocol_and_names_the_arm():
    code, out = run("lift_bottle", "eval", SEEDS=100, START_SEED=1000000)
    assert code == 0, out
    assert "scripts/baseline_eval.py --task lift_bottle" in out
    assert "act_train_config.yml" in out
    assert "--start 1000000 --count 100" in out
    assert "--env TRAIN_CONFIG=train_config --env EP_NUM=50" in out
    assert "eval_1000000_100.json" in out


def test_prepare_processes_50_demos():
    code, out = run("pull_out_key", "prepare")
    assert code == 0, out
    assert "process_data.py pull_out_key clean51 50" in out


@pytest.mark.parametrize("args", [(), ("lift_bottle",), ("lift_bottle", "fly"), ("no_such_task", "train")])
def test_bad_arguments_fail(args):
    code, _ = run(*args)
    assert code != 0


def test_unknown_arm_fails():
    code, _ = run("lift_bottle", "train", ACT_ARM="both")
    assert code != 0
