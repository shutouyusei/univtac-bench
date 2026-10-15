"""Pull finished evaluations from a run's pods, verify the copies, delete the pods.

    python3 collect.py --run-id <run_id> [--once]

Tracks each pod in <records>/<run_id>/pods.json (from launch.py):
  wait              no DONE yet
  collected/failed  copy verified, pod deleted (failed: the eval exited non-zero)
  incomplete_copy   copy differs from the pod; pod stopped, its volume keeps the output
  timeout           past limit_h without DONE; finished episodes pulled, pod stopped
  exited/lost       stopped or gone before collection
Needs RUNPOD_API_KEY. --notify-cmd gets one message argument per problem and at the end.
"""

import argparse
import collections
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch  # noqa: E402
import results  # noqa: E402
import runpod_api  # noqa: E402

FINAL = {"collected", "failed", "incomplete_copy", "timeout", "exited", "lost", "not_created"}
STOPPED = (None, "TERMINATED", "EXITED")


def decide(desired_status, done: bool, hours: float, limit_h: float) -> str:
    if desired_status in (None, "TERMINATED"):
        return "lost"
    if desired_status == "EXITED":
        return "exited"
    if done:
        return "collect"
    return "timeout" if hours > limit_h else "wait"


def count_files(job_dir: Path) -> int:
    """Files in a job folder except DONE (same count as Connection.count_files on the pod)."""
    return sum(1 for p in Path(job_dir).rglob("*") if p.is_file() and p.name != "DONE")


def verify_collected(job_dir: Path, remote_files: int) -> list:
    """Ways the pulled job_dir differs from the pod's (a faithfully copied failed eval has none)."""
    problems, on_pod = [], None
    if not (job_dir / "manifest.json").is_file():
        problems.append("manifest.json missing")
    else:
        try:
            on_pod = set(results.read_json(job_dir / "manifest.json").get("transfer_check_on_pod", []))
        except json.JSONDecodeError as error:
            problems.append(f"manifest.json unreadable: {error}")
    problems += [p for p in results.check_transfer(job_dir / "eval_result") if on_pod is None or p not in on_pod]
    if count_files(job_dir) != remote_files:
        problems.append(f"{count_files(job_dir)} files pulled, {remote_files} on the pod")
    return problems


class Connection:
    """ssh / rsync to one pod, with the run's own known_hosts."""

    def __init__(self, ip: str, port: int, key: str, known_hosts: Path):
        self.target = f"root@{ip}"
        self.ssh = ["ssh", "-p", str(port), "-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                    "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}"]

    def _run(self, command: str) -> subprocess.CompletedProcess:
        return subprocess.run([*self.ssh, self.target, command], capture_output=True, text=True, timeout=120)

    def done(self, remote_job: str) -> bool:
        return self._run(f"test -f {remote_job}/DONE").returncode == 0

    def count_files(self, remote_job: str) -> int:
        out = self._run(f"find {remote_job} -type f ! -name DONE | wc -l")
        return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip().isdigit() else -1

    def pull(self, remote: str, local: Path) -> bool:
        """rsync remote (a folder if it ends in /) to local (folder or file path)."""
        target = str(local) + "/" if remote.endswith("/") else str(local)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        return subprocess.run(["rsync", "-a", "--partial", "-e", " ".join(self.ssh), f"{self.target}:{remote}", target],
                              capture_output=True, text=True, timeout=3600).returncode == 0


def hours_since(stamp: str) -> float:
    started = datetime.datetime.strptime(stamp, results.UTC_FORMAT).replace(tzinfo=datetime.timezone.utc)
    return (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() / 3600


def notify(cmd: str, message: str) -> None:
    print(message)
    if cmd:
        subprocess.run([*cmd.split(), message], check=False)


def _finish_collect(entry: dict, client, conn, local: Path, remote_job: str, pulled: bool, args) -> None:
    pod_id = entry["pod_id"]
    problems = verify_collected(local, conn.count_files(remote_job)) if pulled else ["rsync failed"]
    if problems:
        client.stop_pod(pod_id)
        entry.update(status="incomplete_copy", problems=problems)
        notify(args.notify_cmd, f"[runpod] {entry['key']}: copy incomplete, pod {pod_id} stopped: {problems}")
        return
    client.delete_pod(pod_id)
    manifest = results.read_json(local / "manifest.json")
    cov = manifest.get("coverage", {})
    entry.update(status="collected" if manifest.get("exit_code") == 0 else "failed", collected_utc=results.utc_now(),
                 summary=f"{cov.get('success')}/{cov.get('evaluated')} success, {cov.get('errors')} errors, "
                         f"{len(cov.get('missing', []))} seeds missing, exit {manifest.get('exit_code')}")
    notify(args.notify_cmd if entry["status"] == "failed" else "",
           f"[collect] {entry['key']}: {entry['status']}, {entry['summary']}; pod {pod_id} deleted")


def handle(entry: dict, client, run_dir: Path, args, connect=None) -> None:
    """One look at one pod: updates entry["status"], stopping or deleting the pod when due."""
    if connect is None:
        def connect(ip, port):
            return Connection(ip, port, args.ssh_key, run_dir / "known_hosts")
    pod_id, remote_job = entry["pod_id"], f"{launch.POD_OUT}/{entry['key']}"
    pod = client.get_pod(pod_id)
    status = pod.get("desiredStatus") if pod else None
    endpoint = runpod_api.ssh_endpoint(pod) if status not in STOPPED else None
    conn = connect(*endpoint) if endpoint else None
    action = decide(status, bool(conn) and conn.done(remote_job), hours_since(entry["created_utc"]),
                    entry.get("limit_h", args.max_hours))
    if action == "wait":
        return
    local = run_dir / entry["key"]
    if action in ("collect", "timeout") and conn:
        pulled = conn.pull(f"{remote_job}/", local)
        conn.pull(f"{launch.POD_OUT}/pod_stdout.log", run_dir / f"{entry['key']}.pod_stdout.log")
        if action == "collect":
            return _finish_collect(entry, client, conn, local, remote_job, pulled, args)
    if action in ("timeout", "exited"):
        client.stop_pod(pod_id)
    entry.update(status=action)
    notify(args.notify_cmd, f"[runpod] {entry['key']}: {action} (pod {pod_id}"
                            f"{', stopped' if action in ('timeout', 'exited') else ''})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--records", default=str(launch.DEFAULT_RECORDS))
    parser.add_argument("--max-hours", type=float, default=6.0, help="limit for pods whose record has no limit_h")
    parser.add_argument("--poll-s", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ssh-key", default=str(Path("~/.ssh/id_ed25519").expanduser()))
    parser.add_argument("--notify-cmd", default="")
    args = parser.parse_args(argv)

    run_dir = Path(args.records) / args.run_id
    pods_file = run_dir / "pods.json"
    record = json.loads(pods_file.read_text())
    client = runpod_api.RunPod(os.environ.get("RUNPOD_API_KEY", ""))
    while True:
        for entry in record["pods"]:
            if entry["status"] in FINAL:
                continue
            try:
                handle(entry, client, run_dir, args)
            except (runpod_api.RunPodError, subprocess.SubprocessError, OSError, ValueError) as error:
                print(f"[collect] {entry['key']}: {error} (retrying next pass)", file=sys.stderr)
            results.write_json(pods_file, record)
        open_pods = [e for e in record["pods"] if e["status"] not in FINAL]
        if not open_pods or args.once:
            break
        time.sleep(args.poll_s)

    counts = dict(collections.Counter(e["status"] for e in record["pods"]))
    notify(args.notify_cmd if not open_pods else "", f"[runpod] {args.run_id}: {counts}")
    return 0 if set(counts) <= {"collected"} else 1


if __name__ == "__main__":
    sys.exit(main())
