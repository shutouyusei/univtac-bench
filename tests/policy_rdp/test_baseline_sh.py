"""policy/RDP/baseline.sh in dry-run mode: the command each stage runs."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SH = REPO / "policy" / "RDP" / "baseline.sh"
TASKS = ["grasp_classify", "insert_HDMI", "insert_hole", "insert_tube", "lift_bottle", "lift_can", "pull_out_key",
         "put_bottle_in_shelf"]


def run(*args, **env):
    full = {**os.environ, "DRY_RUN": "1", **{k: str(v) for k, v in env.items()}}
    p = subprocess.run(["bash", str(SH), *args], capture_output=True, text=True, env=full)
    return p.returncode, p.stdout + p.stderr


def test_pca_pools_every_task():
    code, out = run("all", "pca")
    assert code == 0, out
    assert "rdp_univtac.prepare pca --tasks " + " ".join(TASKS) in out
    assert "--episodes 50 --out" in out and "data/marker_pca.npz" in out


def test_prepare_caches_one_task():
    code, out = run("lift_can", "prepare")
    assert code == 0, out
    assert "rdp_univtac.prepare cache --task lift_can --task-config clean51 --episodes 50" in out
    assert "data/lift_can-clean51-50" in out


@pytest.mark.parametrize("stage,cmd", [("train_at", "rdp_univtac.train at"), ("train_ldp", "rdp_univtac.train ldp")])
def test_training_stages(stage, cmd):
    code, out = run("lift_bottle", stage)
    assert code == 0, out
    assert cmd in out and "rdp_ckpt/lift_bottle-clean51-50" in out and "data/marker_pca.npz" in out


def test_eval_uses_the_shared_protocol():
    code, out = run("lift_bottle", "eval")
    assert code == 0, out
    assert "scripts/baseline_eval.py --task lift_bottle --task-config clean51" in out
    assert "rdp_ckpt/lift_bottle-clean51-50/rdp.yml" in out
    assert "--start 1000000 --count 100" in out and "eval_1000000_100.json" in out


def test_overridden_runs_get_their_own_folder():
    code, out = run("lift_bottle", "train_at", RDP_SET="at_training.num_epochs=2")
    assert code == 0, out
    assert "rdp_ckpt/lift_bottle-clean51-50-smoke" in out and "--set at_training.num_epochs=2" in out


def test_eval_stops_with_a_clear_message_when_it_cannot_run():
    code, out = run("lift_bottle", "eval", DRY_RUN=0)
    assert code != 0
    assert "baseline_eval.py" in out or "not trained" in out


@pytest.mark.parametrize("args", [(), ("lift_bottle",), ("lift_bottle", "fly"), ("nope", "prepare"), ("lift_bottle", "pca")])
def test_bad_arguments_fail(args):
    code, _ = run(*args)
    assert code != 0
