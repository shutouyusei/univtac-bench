"""Eval-output helpers shared by the pod (eval_released.sh) and the collector. Stdlib only.

    python3 results.py train-config|task-config <checkpoint>/log.log
    python3 results.py run-limit <count>          (seconds; EPISODE_TIMEOUT_S from the environment)
    python3 results.py manifest --out manifest.json ...
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
RUN_OVERHEAD_S = 600
EPISODE_TIMEOUT_S = 400


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(UTC_FORMAT)


def read_json(path) -> dict:
    """The file's JSON, or {} if it does not exist."""
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json(path: Path, data) -> None:
    """Atomic write (tmp + rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def run_limit_s(count: int, episode_timeout_s: int = EPISODE_TIMEOUT_S) -> int:
    """Wall-clock limit of one eval run of count episodes."""
    return RUN_OVERHEAD_S + episode_timeout_s * count


def _eval_config(key: str, text: str) -> str:
    """A value of the eval config block in a released checkpoint's log.log (the authors' evaluation)."""
    match = re.search(rf'"{key}":\s*"([^"]+)"', text)
    if match is None:
        raise ValueError(f'no "{key}" in the checkpoint eval log')
    return match.group(1)


def train_config_from_log(text: str) -> str:
    return _eval_config("train_config", text)


def task_config_from_log(text: str) -> str:
    return _eval_config("task_config", text)


def recorded_seeds(metadata: dict) -> list:
    return [key for key in metadata if key.isdigit()]


def seed_coverage(metadata: dict, start: int, count: int) -> dict:
    """Coverage of seeds start..start+count-1; "error" episodes are not evaluated (as upstream counts)."""
    expected = set(range(start, start + count))
    seeds = {int(key) for key in recorded_seeds(metadata)}
    inside = [metadata[str(seed)].get("result") for seed in seeds & expected]
    return {
        "expected": count,
        "evaluated": sum(result in ("success", "failed") for result in inside),
        "success": inside.count("success"),
        "errors": inside.count("error"),
        "missing": sorted(expected - seeds),
        "outside": sorted(seeds - expected),
    }


def check_transfer(run_dir: Path) -> list:
    """Problems with an eval_result run (readable metadata, one video per seed). No paths, so pod and local compare."""
    run_dir = Path(run_dir)
    if not (run_dir / "metadata.json").is_file():
        return ["metadata.json missing"]
    try:
        metadata = read_json(run_dir / "metadata.json")
    except json.JSONDecodeError as error:
        return [f"metadata.json unreadable: {error}"]
    videos = {path.name.split("_")[0] for path in (run_dir / "video").glob("*.mp4")}
    no_video = [seed for seed in sorted(recorded_seeds(metadata)) if seed not in videos]
    return [f"no video for {len(no_video)} evaluated seeds (first: {no_video[:3]})"] if no_video else []


def gpu_info() -> list:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return [{"error": str(error)}]
    keys = ("name", "driver", "memory")
    return [dict(zip(keys, (part.strip() for part in line.split(",")))) for line in out.strip().splitlines() if line]


def checkpoint_hashes(sha_file: Path, task: str, arm: str) -> dict:
    """sha256 of one task/arm's checkpoint files and the encoder, from data/checkpoints.sha256."""
    prefix = f"checkpoints/{task}/{arm}/"
    hashes = {}
    if sha_file.is_file():
        for line in sha_file.read_text().splitlines():
            digest, _, name = line.partition("  ")
            if name.startswith(prefix) or name == "checkpoints/encoder.pth":
                hashes[name] = digest
    return hashes


def write_manifest(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir)
    manifest = {
        "job": {"task": args.task, "arm": args.arm, "start_seed": args.start_seed, "count": args.count,
                "task_config": args.task_config, "train_config": args.train_config},
        "render_overrides": os.environ.get("UNIVTAC45_RENDER_OVERRIDES", ""),
        "exit_code": args.exit_code,
        "started_utc": args.started,
        "finished_utc": utc_now(),
        "coverage": seed_coverage(read_json(run_dir / "metadata.json"), args.start_seed, args.count),
        "transfer_check_on_pod": check_transfer(run_dir),
        "gpu": gpu_info(),
        "image": os.environ.get("IMAGE_REF", ""),
        "pod_id": os.environ.get("RUNPOD_POD_ID", ""),
        "build": read_json(args.build_info),
        "checkpoint_sha256": checkpoint_hashes(Path(args.checkpoint_sha256), args.task, args.arm),
    }
    write_json(Path(args.out), manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train-config").add_argument("log")
    sub.add_parser("task-config").add_argument("log")
    sub.add_parser("run-limit").add_argument("count", type=int)
    mf = sub.add_parser("manifest")
    for name in ("out", "task", "arm", "task-config", "train-config", "started", "run-dir", "checkpoint-sha256"):
        mf.add_argument(f"--{name}", required=True)
    for name in ("start-seed", "count", "exit-code"):
        mf.add_argument(f"--{name}", type=int, required=True)
    mf.add_argument("--build-info", default="/workspace/build_info.json")
    args = parser.parse_args(argv)

    if args.command in ("train-config", "task-config"):
        try:
            print(_eval_config(args.command.replace("-", "_"), Path(args.log).read_text()))
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
    elif args.command == "run-limit":
        print(run_limit_s(args.count, int(os.environ.get("EPISODE_TIMEOUT_S") or EPISODE_TIMEOUT_S)))
    else:
        cov = write_manifest(args)["coverage"]
        print(f"{args.task} {args.arm}: {cov['success']}/{cov['evaluated']} success "
              f"({len(cov['missing'])} of {cov['expected']} seeds missing), exit {args.exit_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
