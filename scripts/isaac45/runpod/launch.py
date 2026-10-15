"""Start one RunPod pod per evaluation job (jobs.csv rows: task, arm, start_seed, count, task_config).

    python3 launch.py jobs --tasks lift_can --arms univtac vision_only --count 100 --chunk 50 --task-config demo > jobs.csv
    python3 launch.py start --run-id <id> --jobs jobs.csv --image <ref@sha256:...> [--registry-auth-id ID] [--dry-run]

start needs RUNPOD_API_KEY and OMNI_KIT_ACCEPT_EULA, gives each pod your SSH public key, and writes
<records>/<run_id>/pods.json for collect.py (an existing one is never overwritten).
"""

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_api  # noqa: E402
from results import EPISODE_TIMEOUT_S, run_limit_s, utc_now, write_json  # noqa: E402,F401

TASKS = ["lift_bottle", "pull_out_key", "lift_can", "put_bottle_in_shelf", "insert_hole", "insert_HDMI",
         "insert_tube", "grasp_classify"]
ARMS = ["univtac", "vision_only"]
POD_OUT = "/runpod-out"  # pod volume: survives a stop, unlike the container disk; /workspace holds the image's code
CUDA_VERSIONS = ["12.8", "12.9", "13.0"]  # drivers that run the image's CUDA 12.8
DEFAULT_RECORDS = Path.home() / "Desktop" / "Robotics" / "records" / "runpod"
REPO = Path(__file__).resolve().parents[3]
STARTUP_MARGIN_H = 1.0  # image pull + Isaac start-up before the run limit counts


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


FIELDS = [f.name for f in fields(Job)]


def limit_hours(count: int, episode_timeout_s: int) -> float:
    """How long a pod may go without DONE before the collector stops it."""
    return run_limit_s(count, episode_timeout_s) / 3600 + STARTUP_MARGIN_H


def task_config_names() -> set:
    return {path.stem for path in (REPO / "task_config").glob("*.yml")}


def split_seeds(start: int, count: int, chunk: int) -> list:
    """(start, count) pieces of at most chunk seeds (chunk 0: one piece)."""
    if chunk <= 0 or chunk >= count:
        return [(start, count)]
    return [(s, min(chunk, start + count - s)) for s in range(start, start + count, chunk)]


def make_jobs(tasks, arms, start: int, count: int, chunk: int, task_config: str) -> list:
    return [Job(t, a, s, n, task_config) for t in tasks for a in arms for s, n in split_seeds(start, count, chunk)]


def jobs_to_csv(jobs) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(job) for job in jobs)
    return out.getvalue()


def _parse_row(row: dict, earlier: list, task_configs) -> Job:
    job = Job(row["task"].strip(), row["arm"].strip(), int(row["start_seed"]), int(row["count"]),
              row["task_config"].strip())
    if job.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, not {job.task!r}")
    if task_configs is not None and job.task_config not in task_configs:
        raise ValueError(f"no task config {job.task_config!r} (have {sorted(task_configs)})")
    if job.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {job.arm!r}")
    if job.start_seed < 0 or job.count < 1:
        raise ValueError("start_seed >= 0 and count >= 1 required")
    for other in earlier:
        if (other.task, other.arm) == (job.task, job.arm) and \
                job.start_seed < other.start_seed + other.count and other.start_seed < job.start_seed + job.count:
            raise ValueError(f"seeds overlap with {other.key}")
    return job


def parse_jobs(text: str, task_configs=None) -> list:
    """Jobs of a jobs.csv; task_configs, if given, are the valid task config names."""
    jobs = []
    for line, row in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        try:
            jobs.append(_parse_row(row, jobs, task_configs))
        except (KeyError, ValueError, AttributeError) as error:
            raise ValueError(f"jobs line {line}: {error}") from error
    return jobs


def pod_spec(job: Job, run_id: str, image: str, gpu: str, cloud: str, registry_auth_id: str, public_key: str,
             extra_env: dict, container_disk_gb: int = 100, volume_gb: int = 10) -> dict:
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
        "env": {"EVAL_JOB": " ".join(str(getattr(job, f)) for f in FIELDS), "OUT_ROOT": POD_OUT,
                "PUBLIC_KEY": public_key, "IMAGE_REF": image, **extra_env},
        "interruptible": False,
    }
    if registry_auth_id:
        spec["containerRegistryAuthId"] = registry_auth_id
    if cloud == "COMMUNITY":
        spec["supportPublicIp"] = True
    return spec


def find_created(client, name: str):
    """The live pod named name, if a create that errored went through anyway."""
    try:
        pods = [p for p in client.list_pods(name=name) if p.get("desiredStatus") != "TERMINATED"]
    except runpod_api.RunPodError:
        return None
    return pods[0] if pods else None


def cmd_start(args) -> int:
    try:
        jobs = parse_jobs(Path(args.jobs).read_text(), task_configs=task_config_names())
    except ValueError as error:
        print(f"{args.jobs}: {error}", file=sys.stderr)
        return 2
    pods_file = Path(args.records) / args.run_id / "pods.json"
    if pods_file.exists() and not args.dry_run:
        print(f"{pods_file} exists: this run id was started already", file=sys.stderr)
        return 1
    eula = os.environ.get("OMNI_KIT_ACCEPT_EULA", "")
    if not eula:
        print("OMNI_KIT_ACCEPT_EULA is not set (Isaac Sim needs the EULA accepted)", file=sys.stderr)
        return 1
    extra_env = {"OMNI_KIT_ACCEPT_EULA": eula, "EPISODE_TIMEOUT_S": str(args.episode_timeout_s)}
    if args.render_overrides:
        extra_env["UNIVTAC45_RENDER_OVERRIDES"] = args.render_overrides
    public_key = Path(args.ssh_public_key).expanduser().read_text().strip()
    specs = [(job, pod_spec(job, args.run_id, args.image, args.gpu, args.cloud, args.registry_auth_id, public_key,
                            extra_env, args.container_disk_gb, args.volume_gb)) for job in jobs]
    if args.dry_run:
        for _, spec in specs:
            print(json.dumps({**spec, "env": {**spec["env"], "PUBLIC_KEY": spec["env"]["PUBLIC_KEY"][:20] + "..."}}))
        print(f"{len(specs)} pods would be created (dry run)", file=sys.stderr)
        return 0

    client = runpod_api.RunPod(os.environ.get("RUNPOD_API_KEY", ""))
    pods_file.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_id": args.run_id, "image": args.image, "gpu": args.gpu, "cloud": args.cloud,
              "render_overrides": args.render_overrides or "", "started_utc": utc_now(), "pods": []}
    write_json(pods_file, record)
    for job, spec in specs:
        entry = {"job": asdict(job), "key": job.key}
        try:
            pod = client.create_pod(spec)
        except runpod_api.RunPodError as error:
            pod = find_created(client, spec["name"])
            if pod is None:
                print(f"[launch] {job.key}: {error}", file=sys.stderr)
                entry.update(pod_id=None, status="not_created", error=str(error))
            else:
                print(f"[launch] {job.key}: create reported '{error}', but pod {pod['id']} exists; tracking it")
        if pod is not None:
            print(f"[launch] {job.key}: pod {pod['id']} ({pod.get('costPerHr', '?')}/h)")
            entry.update(pod_id=pod["id"], status="launched", created_utc=utc_now(), cost_per_hr=pod.get("costPerHr"),
                         limit_h=round(limit_hours(job.count, args.episode_timeout_s), 2))
        record["pods"].append(entry)
        write_json(pods_file, record)
    created = sum(p["pod_id"] is not None for p in record["pods"])
    print(f"[launch] {created}/{len(specs)} pods created; record {pods_file}")
    return 0 if created == len(specs) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    jp = sub.add_parser("jobs", help="write jobs.csv to stdout")
    jp.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    jp.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    jp.add_argument("--start", type=int, default=1000000)
    jp.add_argument("--count", type=int, default=100)
    jp.add_argument("--chunk", type=int, default=0, help="seeds per pod (0: a whole task/arm per pod)")
    jp.add_argument("--task-config", required=True)
    sp = sub.add_parser("start", help="create the pods of a jobs.csv")
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--jobs", required=True)
    sp.add_argument("--image", required=True, help="image reference, preferably by digest")
    sp.add_argument("--registry-auth-id", default="", help="RunPod registry auth id (private image)")
    sp.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    sp.add_argument("--cloud", default="COMMUNITY", choices=["SECURE", "COMMUNITY"])
    sp.add_argument("--render-overrides", default="", help="UNIVTAC45_RENDER_OVERRIDES for every pod")
    sp.add_argument("--episode-timeout-s", type=int, default=EPISODE_TIMEOUT_S,
                    help="per-episode budget on the pod; also sets each pod's limit in the collector")
    sp.add_argument("--container-disk-gb", type=int, default=100, help="the image alone is ~56 GB")
    sp.add_argument("--volume-gb", type=int, default=10)
    sp.add_argument("--ssh-public-key", default="~/.ssh/id_ed25519.pub")
    sp.add_argument("--records", default=str(DEFAULT_RECORDS))
    sp.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "jobs":
        sys.stdout.write(jobs_to_csv(make_jobs(args.tasks, args.arms, args.start, args.count, args.chunk,
                                               args.task_config)))
        return 0
    return cmd_start(args)


if __name__ == "__main__":
    sys.exit(main())
