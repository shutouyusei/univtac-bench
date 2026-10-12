"""Evaluate one baseline policy on a fixed seed range, with the protocol streaming-tactile-flow's runs use, so
baselines and our methods are compared seed for seed.

    python scripts/baseline_eval.py --task lift_bottle --deploy policy/ACT/act_ckpt/.../act_train_config.yml \
        --start 1000000 --count 100 --out .../eval_1000000_100.json --env TRAIN_CONFIG=train_config --env EP_NUM=50

One ``eval_policy.sh`` call over ``start .. start+count-1`` (``--max_seed`` caps it there), then one call per run
of consecutive seeds that have no outcome (a simulator crash ends a call early), and the outcomes merged into
``--out`` as ``{seed: metadata entry}``. Seeds whose scripted expert fails are skipped by ``eval_policy.py`` and
stay without an outcome, as in the stflow runs. Results of each call stay where ``eval_policy.py`` writes them:
``eval_result/<policy_name>/<task>/<deploy file stem>/<time>``. Run evaluations one at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def seed_range(start: int, count: int) -> list[int]:
    return list(range(start, start + count))


def contiguous_ranges(seeds: list[int]) -> list[tuple[int, int]]:
    """``(start, count)`` per run of consecutive seeds."""
    out: list[tuple[int, int]] = []
    for s in sorted(seeds):
        if out and s == out[-1][0] + out[-1][1]:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((s, 1))
    return out


def load_outcomes(metadata: Path) -> dict[int, dict]:
    """The entries of an ``eval_policy.py`` metadata.json that carry a result."""
    data = json.loads(Path(metadata).read_text())
    return {int(k): v for k, v in data.items() if isinstance(v, dict) and "result" in v}


def missing_seeds(outcomes: dict[int, dict], seeds: list[int]) -> list[int]:
    return [s for s in seeds if s not in outcomes]


def merge(parts: list[dict[int, dict]], seeds: list[int]) -> dict[int, dict]:
    """Per seed of ``seeds``, the outcome of the first part that has one."""
    out: dict[int, dict] = {}
    for part in parts:
        for s in seeds:
            if s not in out and s in part:
                out[s] = part[s]
    return out


def eval_command(task: str, task_config: str, deploy: Path, start: int, count: int, conda_env: str) -> list[str]:
    return [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "bash", "eval_policy.sh", task, task_config, str(deploy), "0",
        "--start_seed", str(start), "--total_num", str(count), "--max_seed", str(start + count - 1), "--headless",
    ]


def results_dir(repo: Path, task: str, deploy: Path) -> Path:
    """Where ``eval_policy.py`` puts this deploy file's runs."""
    policy_name = yaml.safe_load(Path(deploy).read_text())["policy_name"]
    return Path(repo) / "eval_result" / policy_name / task / Path(deploy).stem


def parse_env(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env takes KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of a success rate."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summary_line(name: str, outcomes: dict[int, dict], seeds: list[int]) -> str:
    n = len(outcomes)
    k = sum(v["result"] == "success" for v in outcomes.values())
    lo, hi = wilson(k, n)
    missing = missing_seeds(outcomes, seeds)
    return (f"{name}: {k}/{n} success ({100 * k / max(n, 1):.0f} %, 95 % CI {100 * lo:.0f}-{100 * hi:.0f}); "
            f"without outcome: {missing or 'none'}")


def eval_once(args, deploy: Path, start: int, count: int, extra_env: dict, log: Path) -> dict[int, dict]:
    results = results_dir(REPO, args.task, deploy)
    before = set(results.glob("*")) if results.exists() else set()
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", **extra_env}
    with open(log, "a") as f:
        # The exit status is ignored: whether the call recorded outcomes decides.
        subprocess.run(eval_command(args.task, args.task_config, deploy, start, count, args.conda_env),
                       cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env, check=False)
    new = sorted(set(results.glob("*")) - before, key=lambda p: p.stat().st_mtime) if results.exists() else []
    if not new or not (new[-1] / "metadata.json").exists():
        raise SystemExit(f"evaluation of seeds {start}..{start + count - 1} recorded nothing; see {log}")
    return load_outcomes(new[-1] / "metadata.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--task-config", default="clean51")
    p.add_argument("--deploy", required=True, help="deploy yml; its stem names the result folder")
    p.add_argument("--start", type=int, default=1000000)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--out", required=True, help="merged outcomes json")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for the policy (repeatable)")
    p.add_argument("--conda-env", default="UniVTAC")
    args = p.parse_args(argv)

    deploy = Path(args.deploy).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    extra_env = parse_env(args.env)
    seeds = seed_range(args.start, args.count)
    parts = [eval_once(args, deploy, args.start, args.count, extra_env, log)]
    for start, count in contiguous_ranges(missing_seeds(parts[0], seeds)):
        parts.append(eval_once(args, deploy, start, count, extra_env, log))
    merged = merge(parts, seeds)
    out.write_text(json.dumps({str(k): v for k, v in sorted(merged.items())}, indent=1))
    print(summary_line(f"{args.task} {deploy.stem}", merged, seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
