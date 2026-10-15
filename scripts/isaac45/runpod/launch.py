"""Start one RunPod pod per evaluation job, each running the evaluation image's pod_start.sh.

    # jobs.csv: one row per pod (task, arm, start_seed, count, task_config)
    python3 launch.py jobs --tasks lift_can pull_out_key --arms univtac vision_only \\
        --start 1000000 --count 100 --chunk 50 --task-config demo > jobs.csv
    # what would be created, without creating anything
    python3 launch.py start --run-id 2026-10-16_released --jobs jobs.csv --image ghcr.io/<user>/univtac45-eval@sha256:... \\
        --registry-auth-id <id> --dry-run
    python3 launch.py start ... (same, without --dry-run)

start reads RUNPOD_API_KEY and OMNI_KIT_ACCEPT_EULA from the environment (e.g. `set -a; source .env; set +a`), passes
the local SSH public key so the collector can pull results, and records the pods in <records>/<run_id>/pods.json,
which collect.py works from. It refuses to start a run id that already has a pods.json.
"""

import argparse
import csv
import datetime
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_api  # noqa: E402

TASKS = ["lift_bottle", "pull_out_key", "lift_can", "put_bottle_in_shelf", "insert_hole", "insert_HDMI",
         "insert_tube", "grasp_classify"]
ARMS = ["univtac", "vision_only"]
# Pod volume (kept while a pod is stopped, unlike the container disk). Not /workspace, where the image's code is.
POD_OUT = "/runpod-out"
CUDA_VERSIONS = ["12.8", "12.9", "13.0"]  # the image's CUDA 12.8 needs a driver that supports it
DEFAULT_RECORDS = Path.home() / "Desktop" / "Robotics" / "records" / "runpod"
FIELDS = ["task", "arm", "start_seed", "count", "task_config"]
REPO = Path(__file__).resolve().parents[3]
EPISODE_TIMEOUT_S = 400  # eval_released.sh's run limit is 600 s + count x this
STARTUP_MARGIN_H = 1.0   # image pull and Isaac start-up before the run limit starts counting


def limit_hours(count: int, episode_timeout_s: int) -> float:
    """How long a pod may go without DONE before the collector gives up on it."""
    return (600 + episode_timeout_s * count) / 3600 + STARTUP_MARGIN_H


def task_config_names() -> set:
    return {path.stem for path in (REPO / "task_config").glob("*.yml")}


@dataclass(frozen=True)
class Job:
    task: str
    arm: str
    start_seed: int
    count: int
    task_config: str

    @property
    def key(self) -> str:
        return f"{self.task}_{self.arm}_{self.start_seed}_{self.count}"


def split_seeds(start: int, count: int, chunk: int) -> list:
    """(start, count) pieces of seeds start..start+count-1, at most chunk seeds each (chunk 0: one piece)."""
    if chunk <= 0 or chunk >= count:
        return [(start, count)]
    return [(s, min(chunk, start + count - s)) for s in range(start, start + count, chunk)]


def make_jobs(tasks, arms, start: int, count: int, chunk: int, task_config: str) -> list:
    return [Job(task, arm, s, n, task_config) for task in tasks for arm in arms for s, n in split_seeds(start, count, chunk)]


def jobs_to_csv(jobs) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for job in jobs:
        writer.writerow(asdict(job))
    return out.getvalue()


def parse_jobs(text: str, task_configs=None) -> list:
    """Jobs of a jobs.csv; task_configs, when given, is the set of valid task config names."""
    jobs = []
    for line, row in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        try:
            job = Job(row["task"].strip(), row["arm"].strip(), int(row["start_seed"]), int(row["count"]),
                      row["task_config"].strip())
        except (KeyError, ValueError, AttributeError) as error:
            raise ValueError(f"jobs line {line}: {error}") from error
        if job.task not in TASKS:
            raise ValueError(f"jobs line {line}: task must be one of {TASKS}, not {job.task!r}")
        if task_configs is not None and job.task_config not in task_configs:
            raise ValueError(f"jobs line {line}: no task config {job.task_config!r} (have {sorted(task_configs)})")
        if job.arm not in ARMS:
            raise ValueError(f"jobs line {line}: arm must be one of {ARMS}, not {job.arm!r}")
        if job.start_seed < 0 or job.count < 1:
            raise ValueError(f"jobs line {line}: start_seed >= 0 and count >= 1 required")
        for other in jobs:
            same = (other.task, other.arm) == (job.task, job.arm)
            if same and job.start_seed < other.start_seed + other.count and other.start_seed < job.start_seed + job.count:
                raise ValueError(f"jobs line {line}: seeds overlap with {other.key}")
        jobs.append(job)
    return jobs


def pod_spec(job: Job, run_id: str, image: str, gpu: str, cloud: str, registry_auth_id: str, public_key: str,
             extra_env: dict, container_disk_gb: int = 60, volume_gb: int = 10) -> dict:
    env = {
        "EVAL_JOB": f"{job.task} {job.arm} {job.start_seed} {job.count} {job.task_config}",
        "OUT_ROOT": POD_OUT,
        "PUBLIC_KEY": public_key,
        "IMAGE_REF": image,
        **extra_env,
    }
    spec = {
        "name": f"{run_id}-{job.task}-{job.arm}-{job.start_seed}",
        "imageName": image,
        "cloudType": cloud,
        "computeType": "GPU",
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "allowedCudaVersions": CUDA_VERSIONS,
        "containerDiskInGb": container_disk_gb,
        "volumeInGb": volume_gb,
        "volumeMountPath": POD_OUT,
        "ports": ["22/tcp"],
        "env": env,
        "interruptible": False,
    }
    if registry_auth_id:
        spec["containerRegistryAuthId"] = registry_auth_id
    if cloud == "COMMUNITY":
        spec["supportPublicIp"] = True
    return spec


def cmd_jobs(args) -> int:
    sys.stdout.write(jobs_to_csv(make_jobs(args.tasks, args.arms, args.start, args.count, args.chunk, args.task_config)))
    return 0


def cmd_start(args) -> int:
    try:
        jobs = parse_jobs(Path(args.jobs).read_text(), task_configs=task_config_names())
    except ValueError as error:
        print(f"{args.jobs}: {error}", file=sys.stderr)
        return 2
    run_dir = Path(args.records) / args.run_id
    pods_file = run_dir / "pods.json"
    if pods_file.exists() and not args.dry_run:
        print(f"{pods_file} exists: this run id was started already", file=sys.stderr)
        return 1
    public_key = Path(args.ssh_public_key).expanduser().read_text().strip()
    extra_env = {"OMNI_KIT_ACCEPT_EULA": os.environ.get("OMNI_KIT_ACCEPT_EULA", "")}
    if not extra_env["OMNI_KIT_ACCEPT_EULA"]:
        print("OMNI_KIT_ACCEPT_EULA is not set (Isaac Sim needs the EULA accepted)", file=sys.stderr)
        return 1
    if args.render_overrides:
        extra_env["UNIVTAC45_RENDER_OVERRIDES"] = args.render_overrides
    extra_env["EPISODE_TIMEOUT_S"] = str(args.episode_timeout_s)
    specs = [(job, pod_spec(job, args.run_id, args.image, args.gpu, args.cloud, args.registry_auth_id, public_key,
                            extra_env, args.container_disk_gb, args.volume_gb)) for job in jobs]
    if args.dry_run:
        for job, spec in specs:
            shown = {**spec, "env": {k: (v if k != "PUBLIC_KEY" else v[:20] + "...") for k, v in spec["env"].items()}}
            print(json.dumps(shown))
        print(f"{len(specs)} pods would be created (dry run)", file=sys.stderr)
        return 0

    client = runpod_api.RunPod(os.environ.get("RUNPOD_API_KEY", ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"run_id": args.run_id, "image": args.image, "gpu": args.gpu, "cloud": args.cloud,
              "render_overrides": args.render_overrides or "", "started_utc": utc_now(), "pods": []}
    for job, spec in specs:
        try:
            pod = client.create_pod(spec)
        except runpod_api.RunPodError as error:
            pod = find_created(client, spec["name"])
            if pod is None:
                print(f"[launch] {job.key}: {error}", file=sys.stderr)
                record["pods"].append({"job": asdict(job), "key": job.key, "pod_id": None, "status": "not_created",
                                       "error": str(error)})
                write_json(pods_file, record)
                continue
            print(f"[launch] {job.key}: create reported '{error}', but pod {pod['id']} exists; tracking it")
        print(f"[launch] {job.key}: pod {pod['id']} ({pod.get('costPerHr', '?')}/h)")
        record["pods"].append({"job": asdict(job), "key": job.key, "pod_id": pod["id"], "status": "launched",
                               "created_utc": utc_now(), "cost_per_hr": pod.get("costPerHr"),
                               "limit_h": round(limit_hours(job.count, args.episode_timeout_s), 2)})
        write_json(pods_file, record)
    write_json(pods_file, record)
    created = sum(p["pod_id"] is not None for p in record["pods"])
    print(f"[launch] {created}/{len(specs)} pods created; record {pods_file}")
    return 0 if created == len(specs) else 1


def find_created(client, name: str):
    """The pod of this name if a create that reported an error went through anyway (pod names carry the run id)."""
    try:
        pods = [p for p in client.list_pods(name=name) if p.get("desiredStatus") != "TERMINATED"]
    except runpod_api.RunPodError:
        return None
    return pods[0] if pods else None


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    jp = sub.add_parser("jobs", help="write jobs.csv to stdout")
    jp.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    jp.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    jp.add_argument("--start", type=int, default=1000000)
    jp.add_argument("--count", type=int, default=100)
    jp.add_argument("--chunk", type=int, default=0, help="seeds per pod (0: all of a task/arm on one pod)")
    jp.add_argument("--task-config", required=True)
    sp = sub.add_parser("start", help="create the pods of a jobs.csv")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--jobs", required=True)
    sp.add_argument("--image", required=True, help="image reference, preferably by digest")
    sp.add_argument("--registry-auth-id", default="", help="RunPod container registry auth id for a private image")
    sp.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    sp.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    sp.add_argument("--render-overrides", default="", help="UNIVTAC45_RENDER_OVERRIDES for every pod")
    sp.add_argument("--episode-timeout-s", type=int, default=EPISODE_TIMEOUT_S,
                    help="wall-clock budget per episode on the pod; also sets each pod's time limit in the collector")
    sp.add_argument("--container-disk-gb", type=int, default=60)
    sp.add_argument("--volume-gb", type=int, default=10)
    sp.add_argument("--ssh-public-key", default="~/.ssh/id_ed25519.pub")
    sp.add_argument("--records", default=str(DEFAULT_RECORDS))
    sp.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return cmd_jobs(args) if args.command == "jobs" else cmd_start(args)


if __name__ == "__main__":
    sys.exit(main())
