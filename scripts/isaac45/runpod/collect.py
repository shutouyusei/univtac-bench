"""Pull finished evaluations from a run's pods, verify the copies, delete the pods.

    python3 collect.py --run-id <run_id> [--once]

Tracks each pod in <records>/<run_id>/pods.json (from launch.py):
  wait              no DONE yet
  collected/failed  copy verified, pod deleted (failed: the eval exited non-zero)
  incomplete_copy   copy differs from the pod; pod stopped, its volume keeps the output
  timeout           past limit_h without DONE; finished episodes pulled, pod stopped
  exited/lost       stopped or gone before collection
With --relaunch, each pass also creates pods for not_created jobs (no instance was free) and for jobs whose eval
failed with nothing evaluated (a broken host; the attempt's folder is kept as <job>.attemptN), within money guards:
at most --max-pods running, none created below --min-balance-usd, every running pod stopped below
--stop-balance-usd (status stopped_low_balance; the pod volume keeps its output), at most --max-attempts pods per
job, and no new pods after --relaunch-hours.
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

FINAL = {"collected", "failed", "incomplete_copy", "timeout", "exited", "lost", "not_created", "stopped_low_balance"}
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
                 evaluated=cov.get("evaluated"),
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


def launch_guard(running: int, max_pods: int, balance: float, min_balance: float) -> str:
    """Why no new pod may be created now ("" if one may)."""
    if running >= max_pods:
        return f"{running} pods running (max {max_pods})"
    if balance < min_balance:
        return f"balance ${balance:.2f} below ${min_balance:.2f}"
    return ""


def requeueable(entry: dict, max_attempts: int) -> bool:
    """An eval that failed with nothing evaluated (a broken host), with attempts left."""
    return (entry["status"] == "failed" and entry.get("evaluated") == 0 and "spec" in entry
            and entry.get("attempts", 1) < max_attempts)


def requeue_host_failures(record: dict, run_dir: Path, max_attempts: int) -> None:
    """Requeueable jobs go back to not_created, the failed attempt's folder kept as <job>.attemptN."""
    for entry in record["pods"]:
        if not requeueable(entry, max_attempts):
            continue
        attempts = entry.get("attempts", 1)
        for name in (entry["key"], f"{entry['key']}.pod_stdout.log"):
            if (run_dir / name).exists():
                (run_dir / name).rename(run_dir / name.replace(entry["key"], f"{entry['key']}.attempt{attempts}", 1))
        entry.update(status="not_created", attempts=attempts, requeued=f"attempt {attempts} failed with 0 evaluated")


def relaunch_pass(record: dict, client, args, run_dir: Path) -> None:
    """Create pods for waiting jobs within the money guards (see the module doc)."""
    balance = client.balance()
    running = [e for e in record["pods"] if e["status"] == "launched"]
    if balance < args.stop_balance_usd:
        for entry in running:
            client.stop_pod(entry["pod_id"])
            entry.update(status="stopped_low_balance")
        notify(args.notify_cmd, f"[runpod] balance ${balance:.2f} below ${args.stop_balance_usd:.2f}: "
                                f"{len(running)} pods stopped (volumes keep their output)")
        return
    requeue_host_failures(record, run_dir, args.max_attempts)
    for entry in record["pods"]:
        if entry["status"] != "not_created" or "spec" not in entry:
            continue
        if launch_guard(len(running), args.max_pods, balance, args.min_balance_usd):
            return
        if launch.try_create(client, entry):
            entry["attempts"] = entry.get("attempts", 0) + 1
            running.append(entry)
            print(f"[relaunch] {entry['key']}: pod {entry['pod_id']} ({entry['cost_per_hr']}/h, "
                  f"attempt {entry['attempts']})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--records", default=str(launch.DEFAULT_RECORDS))
    parser.add_argument("--max-hours", type=float, default=6.0, help="limit for pods whose record has no limit_h")
    parser.add_argument("--poll-s", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ssh-key", default=str(Path("~/.ssh/id_ed25519").expanduser()))
    parser.add_argument("--notify-cmd", default="")
    parser.add_argument("--relaunch", action="store_true", help="create pods for waiting / host-failed jobs")
    parser.add_argument("--max-pods", type=int, default=8)
    parser.add_argument("--min-balance-usd", type=float, default=4.0)
    parser.add_argument("--stop-balance-usd", type=float, default=1.5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--relaunch-hours", type=float, default=6.0)
    args = parser.parse_args(argv)

    run_dir = Path(args.records) / args.run_id
    pods_file = run_dir / "pods.json"
    record = json.loads(pods_file.read_text())
    client = runpod_api.RunPod(os.environ.get("RUNPOD_API_KEY", ""))
    deadline = time.time() + args.relaunch_hours * 3600
    while True:
        relaunching = args.relaunch and time.time() < deadline
        if relaunching:
            try:
                relaunch_pass(record, client, args, run_dir)
            except (runpod_api.RunPodError, OSError) as error:
                print(f"[relaunch] {error} (retrying next pass)", file=sys.stderr)
            results.write_json(pods_file, record)
        for entry in record["pods"]:
            if entry["status"] != "launched":
                continue
            try:
                handle(entry, client, run_dir, args)
            except (runpod_api.RunPodError, subprocess.SubprocessError, OSError, ValueError) as error:
                print(f"[collect] {entry['key']}: {error} (retrying next pass)", file=sys.stderr)
            results.write_json(pods_file, record)
        open_pods = [e for e in record["pods"] if e["status"] == "launched" or (relaunching and (
            (e["status"] == "not_created" and "spec" in e) or requeueable(e, args.max_attempts)))]
        if not open_pods or args.once:
            break
        time.sleep(args.poll_s)

    counts = dict(collections.Counter(e["status"] for e in record["pods"]))
    notify(args.notify_cmd if not open_pods else "", f"[runpod] {args.run_id}: {counts}")
    return 0 if set(counts) <= {"collected"} else 1


if __name__ == "__main__":
    sys.exit(main())
