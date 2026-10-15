"""Pull finished evaluations from the pods of a run, check the copies, and delete the pods.

    python3 collect.py --run-id 2026-10-16_released             # loop until every pod is handled
    python3 collect.py --run-id 2026-10-16_released --once      # one pass

Works from <records>/<run_id>/pods.json (written by launch.py) and keeps each pod's status there. For each pod:
  wait     still evaluating (no DONE in its job folder yet)
  collect  rsync <pod>:/runpod-out/<job>/ to <records>/<run_id>/<job>/ (and the pod's stdout log next to it), check
           the copy is faithful (manifest present, nothing the pod had is missing, same file count); then delete the
           pod: status "collected", or "failed" when the evaluation itself exited non-zero. An unfaithful copy stops
           the pod instead (the pod volume keeps the output): status "incomplete_copy".
  timeout  past the pod's time limit (launch.py records it from the seed count) without DONE: pull what exists
           (finished episodes are on the pod volume), stop the pod
  exited / lost  stopped or gone without being collected
Reads RUNPOD_API_KEY from the environment. --notify-cmd is run with one message argument for every problem and at
the end (e.g. a chat notification script).
"""

import argparse
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


def decide(desired_status, done: bool, hours: float, limit_h: float) -> str:
    if desired_status in (None, "TERMINATED"):
        return "lost"
    if desired_status == "EXITED":
        return "exited"
    if done:
        return "collect"
    if hours > limit_h:
        return "timeout"
    return "wait"


def verify_collected(job_dir: Path, remote_files: int) -> list:
    """Problems with a pulled job folder, i.e. ways it differs from the pod's; remote_files counts the pod's files in
    the folder except DONE. A failed evaluation copied faithfully has no problems (its manifest says what failed)."""
    problems = []
    on_pod = None
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.is_file():
        problems.append("manifest.json missing")
    else:
        try:
            on_pod = set(json.loads(manifest_path.read_text()).get("transfer_check_on_pod", []))
        except json.JSONDecodeError as error:
            problems.append(f"manifest.json unreadable: {error}")
    local = results.check_transfer(job_dir / "eval_result")
    problems += [p for p in local if on_pod is None or p not in on_pod]
    local_files = sum(1 for p in job_dir.rglob("*") if p.is_file() and p.name != "DONE")
    if local_files != remote_files:
        problems.append(f"{local_files} files pulled, {remote_files} on the pod")
    return problems


class Connection:
    """ssh / rsync to one pod, with a known_hosts file of the run's own."""

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
        try:
            return int(out.stdout.strip()) if out.returncode == 0 else -1
        except ValueError:
            return -1

    def pull(self, remote: str, local: Path) -> bool:
        """rsync remote (a folder when it ends in /) to local (a folder, or a file path)."""
        target = str(local) + "/" if remote.endswith("/") else str(local)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(["rsync", "-a", "--partial", "-e", " ".join(self.ssh), f"{self.target}:{remote}",
                               target], capture_output=True, text=True, timeout=3600)
        return done.returncode == 0


def hours_since(stamp: str) -> float:
    started = datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    return (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() / 3600


def notify(cmd: str, message: str) -> None:
    print(message)
    if cmd:
        subprocess.run([*cmd.split(), message], check=False)


def handle(entry: dict, client, run_dir: Path, args, connect=None) -> None:
    """One look at one pod; updates entry["status"] (and stops / deletes the pod) when there is something to do."""
    if connect is None:
        def connect(ip, port):
            return Connection(ip, port, args.ssh_key, run_dir / "known_hosts")
    pod_id = entry["pod_id"]
    remote_job = f"{launch.POD_OUT}/{entry['key']}"
    pod = client.get_pod(pod_id)
    endpoint = runpod_api.ssh_endpoint(pod) if pod else None
    conn = connect(*endpoint) if endpoint else None
    done = bool(conn) and conn.done(remote_job)
    action = decide(pod.get("desiredStatus") if pod else None, done, hours_since(entry["created_utc"]),
                    entry.get("limit_h", getattr(args, "max_hours", None) or 6.0))
    if action == "wait":
        return
    local = run_dir / entry["key"]
    if action in ("collect", "timeout") and conn:
        pulled = conn.pull(f"{remote_job}/", local)
        conn.pull(f"{launch.POD_OUT}/pod_stdout.log", run_dir / f"{entry['key']}.pod_stdout.log")
        if action == "collect":
            problems = verify_collected(local, conn.count_files(remote_job)) if pulled else ["rsync failed"]
            if problems:
                client.stop_pod(pod_id)
                entry.update(status="incomplete_copy", problems=problems)
                notify(args.notify_cmd, f"[runpod] {entry['key']}: copy incomplete, pod {pod_id} stopped: {problems}")
                return
            client.delete_pod(pod_id)
            manifest = json.loads((local / "manifest.json").read_text())
            cov = manifest.get("coverage", {})
            entry.update(status="collected" if manifest.get("exit_code") == 0 else "failed",
                         collected_utc=launch.utc_now(),
                         summary=f"{cov.get('success')}/{cov.get('evaluated')} success, {cov.get('errors')} errors, "
                                 f"{len(cov.get('missing', []))} seeds missing, exit {manifest.get('exit_code')}")
            line = f"[collect] {entry['key']}: {entry['status']}, {entry['summary']}; pod {pod_id} deleted"
            if entry["status"] == "failed":
                notify(args.notify_cmd, line)
            else:
                print(line)
            return
    if action in ("timeout", "exited") and pod:
        client.stop_pod(pod_id)
    entry.update(status=action)
    notify(args.notify_cmd, f"[runpod] {entry['key']}: {action} (pod {pod_id}"
                            f"{', stopped' if action in ('timeout', 'exited') else ''})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--records", default=str(launch.DEFAULT_RECORDS))
    parser.add_argument("--max-hours", type=float, default=None,
                        help="time limit for pods whose record has none (default 6)")
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
            launch.write_json(pods_file, record)
        open_pods = [e for e in record["pods"] if e["status"] not in FINAL]
        if not open_pods or args.once:
            break
        time.sleep(args.poll_s)

    counts = {}
    for entry in record["pods"]:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    notify(args.notify_cmd if not open_pods else "", f"[runpod] {args.run_id}: {counts}")
    return 0 if set(counts) <= {"collected"} else 1


if __name__ == "__main__":
    sys.exit(main())
