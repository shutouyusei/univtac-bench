"""One evaluation's output (upstream's eval_result/<run>/: metadata.json, video/) and its manifest.

Used on the pod (eval_released.sh: train config of a released checkpoint, manifest.json) and on the machine that
collects results (collect.py: is the pulled copy complete). Standard library only, so the pod's system python3
runs it without the Isaac environment.

    python3 results.py train-config <checkpoint>/log.log
    python3 results.py manifest --out <manifest.json> --task ... (see --help)
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TRAIN_CONFIG_RE = re.compile(r'"train_config":\s*"([^"]+)"')


def train_config_from_log(text: str) -> str:
    """Train config name in a released checkpoint's log.log (the authors' eval config block)."""
    match = TRAIN_CONFIG_RE.search(text)
    if match is None:
        raise ValueError('no "train_config" in the checkpoint eval log')
    return match.group(1)


def seed_coverage(metadata: dict, start: int, count: int) -> dict:
    """How much of seeds start..start+count-1 an eval's metadata.json covers, and how many succeeded.

    Seeds whose episode raised are recorded as result "error" and, as in upstream's own tally, are not evaluated.
    """
    expected = set(range(start, start + count))
    seeds = {int(key) for key in metadata if key.isdigit()}
    results_inside = [metadata[str(seed)].get("result") for seed in seeds & expected]
    return {
        "expected": count,
        "evaluated": sum(result in ("success", "failed") for result in results_inside),
        "success": results_inside.count("success"),
        "errors": results_inside.count("error"),
        "missing": sorted(expected - seeds),
        "outside": sorted(seeds - expected),
    }


def check_transfer(run_dir: Path) -> list:
    """Problems with an eval_result run: metadata.json readable, one video per recorded seed.

    Messages name no absolute path, so the check on the pod and on the pulled copy can be compared.
    """
    meta_path = Path(run_dir) / "metadata.json"
    if not meta_path.is_file():
        return ["metadata.json missing"]
    try:
        metadata = json.loads(meta_path.read_text())
    except json.JSONDecodeError as error:
        return [f"metadata.json unreadable: {error}"]
    videos = {path.name.split("_")[0] for path in (Path(run_dir) / "video").glob("*.mp4")}
    no_video = [seed for seed in sorted(key for key in metadata if key.isdigit()) if seed not in videos]
    if no_video:
        return [f"no video for {len(no_video)} evaluated seeds (first: {no_video[:3]})"]
    return []


def gpu_info() -> list:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return [{"error": str(error)}]
    rows = []
    for line in filter(None, out.strip().splitlines()):
        name, driver, memory = (part.strip() for part in line.split(","))
        rows.append({"name": name, "driver": driver, "memory": memory})
    return rows


def checkpoint_hashes(sha_file: Path, task: str, arm: str) -> dict:
    """sha256 of the checkpoint files of one task/arm, from the image's data/checkpoints.sha256."""
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
    metadata = {}
    if (run_dir / "metadata.json").is_file():
        metadata = json.loads((run_dir / "metadata.json").read_text())
    build_info = {}
    if Path(args.build_info).is_file():
        build_info = json.loads(Path(args.build_info).read_text())
    manifest = {
        "job": {"task": args.task, "arm": args.arm, "start_seed": args.start_seed, "count": args.count,
                "task_config": args.task_config, "train_config": args.train_config},
        "render_overrides": os.environ.get("UNIVTAC45_RENDER_OVERRIDES", ""),
        "exit_code": args.exit_code,
        "started_utc": args.started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coverage": seed_coverage(metadata, args.start_seed, args.count),
        "transfer_check_on_pod": check_transfer(run_dir),
        "gpu": gpu_info(),
        "image": os.environ.get("IMAGE_REF", ""),
        "pod_id": os.environ.get("RUNPOD_POD_ID", ""),
        "build": build_info,
        "checkpoint_sha256": checkpoint_hashes(Path(args.checkpoint_sha256), args.task, args.arm),
    }
    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    tc = sub.add_parser("train-config", help="print the train config of a released checkpoint")
    tc.add_argument("log")
    mf = sub.add_parser("manifest", help="write manifest.json for one evaluation")
    mf.add_argument("--out", required=True)
    mf.add_argument("--task", required=True)
    mf.add_argument("--arm", required=True)
    mf.add_argument("--start-seed", type=int, required=True)
    mf.add_argument("--count", type=int, required=True)
    mf.add_argument("--task-config", required=True)
    mf.add_argument("--train-config", required=True)
    mf.add_argument("--exit-code", type=int, required=True)
    mf.add_argument("--started", required=True)
    mf.add_argument("--run-dir", required=True)
    mf.add_argument("--build-info", default="/workspace/build_info.json")
    mf.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "train-config":
        try:
            print(train_config_from_log(Path(args.log).read_text()))
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
        return 0
    manifest = write_manifest(args)
    cov = manifest["coverage"]
    print(f"{args.task} {args.arm}: {cov['success']}/{cov['evaluated']} success "
          f"({len(cov['missing'])} of {cov['expected']} seeds missing), exit {args.exit_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
