"""Tests for scripts/isaac45/runpod and the pod-side job contract: python3 -m unittest discover -s scripts/isaac45/tests"""

import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import urllib.error
from pathlib import Path

ISAAC45 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC45 / "runpod"))
import collect  # noqa: E402
import launch  # noqa: E402
import results  # noqa: E402
import runpod_api  # noqa: E402

HEADER = ",".join(launch.FIELDS) + "\n"
OK_META, OK_VIDEOS = {"5": {"result": "success"}}, ["5_success.mp4"]


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))


class RunPodClient(unittest.TestCase):
    def test_create_pod_posts_json_with_bearer_key(self):
        seen = {}

        def opener(request, timeout):
            seen.update(url=request.full_url, method=request.get_method(), auth=request.get_header("Authorization"),
                        body=json.loads(request.data))
            seen["agent"] = request.get_header("User-agent")
            return io.BytesIO(b'{"id": "abc123"}')

        pod = runpod_api.RunPod("KEY", opener=opener).create_pod({"name": "x", "imageName": "img"})
        self.assertEqual(pod["id"], "abc123")
        agent = seen.pop("agent")
        self.assertEqual(seen, {"url": "https://rest.runpod.io/v1/pods", "method": "POST", "auth": "Bearer KEY",
                                "body": {"name": "x", "imageName": "img"}})
        # RunPod's Cloudflare front rejects urllib's default "Python-urllib/x.y" (HTTP 403, error code 1010)
        self.assertTrue(agent and not agent.startswith("Python-urllib"))

    def test_network_errors_become_runpod_errors(self):
        def opener(request, timeout):
            raise urllib.error.URLError("timed out")

        with self.assertRaises(runpod_api.RunPodError):
            runpod_api.RunPod("KEY", opener=opener).create_pod({})

    def test_balance_reads_the_account_through_graphql(self):
        seen = {}

        def opener(request, timeout):
            seen.update(url=request.full_url, body=json.loads(request.data))
            return io.BytesIO(b'{"data": {"myself": {"clientBalance": 12.5}}}')

        self.assertEqual(runpod_api.RunPod("KEY", opener=opener).balance(), 12.5)
        self.assertEqual(seen["url"], runpod_api.GRAPHQL_URL)
        self.assertIn("clientBalance", seen["body"]["query"])

    def test_delete_returns_none_on_empty_body(self):
        self.assertIsNone(runpod_api.RunPod("KEY", opener=lambda r, timeout: io.BytesIO(b"")).delete_pod("abc123"))

    def test_ssh_endpoint(self):
        self.assertEqual(runpod_api.ssh_endpoint({"publicIp": "1.2.3.4", "portMappings": {"22": 10341}}),
                         ("1.2.3.4", 10341))
        self.assertIsNone(runpod_api.ssh_endpoint({"publicIp": "", "portMappings": None}))
        self.assertIsNone(runpod_api.ssh_endpoint({"publicIp": "1.2.3.4", "portMappings": {"8888": 1}}))


class Jobs(unittest.TestCase):
    def test_split_seeds_in_chunks(self):
        self.assertEqual(launch.split_seeds(1000000, 100, 40), [(1000000, 40), (1000040, 40), (1000080, 20)])
        self.assertEqual(launch.split_seeds(5, 10, 0), [(5, 10)])

    def test_make_and_parse_jobs_round_trip(self):
        jobs = launch.make_jobs(["lift_can"], ["univtac", "vision_only"], 1000000, 100, 50, "demo")
        self.assertEqual(len(jobs), 4)
        self.assertEqual(launch.parse_jobs(launch.jobs_to_csv(jobs)), jobs)

    def test_parse_rejects_bad_rows(self):
        for row in ["lift_can,both,0,10,demo", "lift_can,univtac,x,10,demo", "lift_can,univtac,0,0,demo",
                    "lift_cam,univtac,0,10,demo", "lift_can,univtac,0,10,clen"]:
            with self.assertRaises(ValueError):
                launch.parse_jobs(HEADER + row + "\n", task_configs={"demo", "clean"})

    def test_auto_task_config_is_valid_and_clean_is_the_jobs_default(self):
        self.assertEqual(launch.parse_jobs(HEADER + "lift_can,univtac,0,10,auto\n", task_configs={"demo"})[0].task_config,
                         "auto")
        args = ["jobs", "--tasks", "lift_can", "--arms", "univtac", "--count", "10"]
        out = io.StringIO()
        with unittest.mock.patch("sys.stdout", out):
            launch.main(args)
        self.assertEqual(launch.parse_jobs(out.getvalue())[0].task_config, "clean")

    def test_parse_rejects_overlapping_seed_ranges(self):
        with self.assertRaises(ValueError):
            launch.parse_jobs(HEADER + "lift_can,univtac,0,10,demo\nlift_can,univtac,5,10,demo\n")

    def test_time_limit_is_the_pod_run_limit_plus_start_up(self):
        self.assertAlmostEqual(launch.limit_hours(count=90, episode_timeout_s=400), (600 + 400 * 90) / 3600 + 1.0)
        env = {k: v for k, v in os.environ.items() if k != "EPISODE_TIMEOUT_S"}
        shell = subprocess.run([sys.executable, str(ISAAC45 / "runpod/results.py"), "run-limit", "90"],
                               capture_output=True, text=True, env=env, check=True).stdout
        self.assertEqual(int(shell), results.run_limit_s(90))

    def test_pod_spec(self):
        job = launch.Job("lift_can", "univtac", 1000000, 50, "demo")
        spec = launch.pod_spec(job, run_id="r1", image="ghcr.io/u/img@sha256:ab", gpu="NVIDIA GeForce RTX 4090",
                               cloud="COMMUNITY", registry_auth_id="reg1", public_key="ssh-ed25519 AAA k",
                               extra_env={"OMNI_KIT_ACCEPT_EULA": "YES"})
        self.assertEqual(spec["name"], "r1-lift_can-univtac-1000000")
        self.assertEqual(spec["env"]["EVAL_JOB"], "lift_can univtac 1000000 50 demo")
        self.assertEqual(spec["env"]["OUT_ROOT"], launch.POD_OUT)
        self.assertEqual(spec["volumeMountPath"], launch.POD_OUT)
        self.assertNotEqual(spec["volumeMountPath"], "/workspace")  # the image's code lives under /workspace
        self.assertEqual(spec["env"]["PUBLIC_KEY"], "ssh-ed25519 AAA k")
        self.assertEqual(spec["env"]["IMAGE_REF"], "ghcr.io/u/img@sha256:ab")
        self.assertEqual(spec["ports"], ["22/tcp"])
        self.assertEqual(spec["containerRegistryAuthId"], "reg1")
        self.assertIn("12.8", spec["allowedCudaVersions"])
        self.assertNotIn("12.4", spec["allowedCudaVersions"])


class PodJobFolder(TempDir):
    """eval_released.sh writes where the collector looks: OUT_ROOT/<Job.key>/."""

    JOB = launch.Job("lift_can", "univtac", 1000000, 10, "no_such_config")

    def run_eval(self, job: launch.Job):
        env = {**os.environ, "OUT_ROOT": str(self.tmp), "OMNI_KIT_ACCEPT_EULA": "YES"}
        args = [str(getattr(job, f)) for f in launch.FIELDS]  # the same order launch puts into EVAL_JOB
        return subprocess.run(["bash", str(ISAAC45 / "docker/eval_released.sh"), *args], env=env,
                              capture_output=True, text=True)

    def test_early_failure_writes_done_under_the_job_key(self):
        self.assertEqual(self.run_eval(self.JOB).returncode, 2)
        self.assertEqual((self.tmp / self.JOB.key / "DONE").read_text().strip(), "2")
        self.assertTrue((self.tmp / self.JOB.key / "manifest.json").is_file())

    def test_finished_job_is_not_run_again(self):
        job = self.JOB
        (self.tmp / job.key).mkdir()
        (self.tmp / job.key / "DONE").write_text("0\n")
        out = self.run_eval(job)
        self.assertEqual(out.returncode, 0)
        self.assertIn("already done", out.stdout)
        self.assertEqual((self.tmp / job.key / "DONE").read_text(), "0\n")


class Decide(unittest.TestCase):
    def test_actions(self):
        for status, done, hours, wanted in [(None, False, 0.1, "lost"), ("TERMINATED", False, 0.1, "lost"),
                                            ("RUNNING", True, 1, "collect"), ("RUNNING", False, 1, "wait"),
                                            ("RUNNING", False, 7, "timeout"), ("EXITED", False, 1, "exited")]:
            self.assertEqual(collect.decide(status, done=done, hours=hours, limit_h=6), wanted)


def make_job(job: Path, metadata=None, videos=(), manifest=True, exit_code=0) -> Path:
    """A job folder as eval_released.sh leaves it (metadata None: upstream wrote no eval_result)."""
    job.mkdir(parents=True)
    if metadata is not None:
        (job / "eval_result" / "video").mkdir(parents=True)
        (job / "eval_result" / "metadata.json").write_text(json.dumps(metadata))
        for name in videos:
            (job / "eval_result" / "video" / name).write_bytes(b"\0")
    (job / "eval.log").write_text("log\n")
    if manifest:
        cov = results.seed_coverage(metadata or {}, 5, 1)
        pod_check = [] if metadata is not None else ["metadata.json missing"]
        (job / "manifest.json").write_text(json.dumps({"transfer_check_on_pod": pod_check, "exit_code": exit_code,
                                                       "coverage": cov}))
    (job / "DONE").write_text(f"{exit_code}\n")
    return job


class VerifyCollected(TempDir):
    def test_complete_copy(self):
        job = make_job(self.tmp / "j", OK_META, OK_VIDEOS)
        self.assertEqual(collect.verify_collected(job, collect.count_files(job)), [])

    def test_faithful_copy_of_a_failed_eval_is_complete(self):
        job = make_job(self.tmp / "j", None, exit_code=2)
        self.assertEqual(collect.verify_collected(job, collect.count_files(job)), [])

    def test_missing_manifest(self):
        job = make_job(self.tmp / "j", OK_META, OK_VIDEOS, manifest=False)
        self.assertTrue(any("manifest" in p for p in collect.verify_collected(job, collect.count_files(job))))

    def test_file_count_mismatch(self):
        job = make_job(self.tmp / "j", OK_META, OK_VIDEOS)
        self.assertTrue(any("files" in p for p in collect.verify_collected(job, remote_files=9)))

    def test_video_lost_in_transfer(self):
        job = make_job(self.tmp / "j", OK_META, [])
        self.assertTrue(any("video" in p for p in collect.verify_collected(job, collect.count_files(job))))


class FakeClient:
    def __init__(self, status="RUNNING"):
        self.status, self.calls = status, []

    def get_pod(self, pod_id):
        return None if self.status is None else {"desiredStatus": self.status, "publicIp": "1.2.3.4",
                                                 "portMappings": {"22": 2222}}

    def stop_pod(self, pod_id):
        self.calls.append(("stop", pod_id))

    def delete_pod(self, pod_id):
        self.calls.append(("delete", pod_id))


class FakeConnection:
    """A pod whose POD_OUT is a local folder; pull copies like rsync -a (minus the names in drop)."""

    def __init__(self, pod_out: Path, drop=()):
        self.pod_out, self.drop = pod_out, set(drop)

    def _remote(self, remote: str) -> Path:
        return self.pod_out / remote[len(launch.POD_OUT) + 1:].rstrip("/")

    def done(self, remote_job):
        return (self._remote(remote_job) / "DONE").is_file()

    def count_files(self, remote_job):
        return collect.count_files(self._remote(remote_job))

    def pull(self, remote, local: Path):
        src = self._remote(remote)
        if src.is_dir():
            shutil.copytree(src, local, dirs_exist_ok=True, ignore=lambda d, names: [n for n in names if n in self.drop])
        elif src.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local)
        return True


class Handle(TempDir):
    KEY = "lift_can_univtac_5_1"

    def run_handle(self, client, drop=(), limit_h=6.0, hours_old=0.5):
        created = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_old)
        entry = {"key": self.KEY, "pod_id": "p1", "status": "launched", "limit_h": limit_h,
                 "created_utc": created.strftime(results.UTC_FORMAT)}
        run_dir = self.tmp / "run"
        run_dir.mkdir()
        args = type("Args", (), {"notify_cmd": "", "ssh_key": "k", "max_hours": 6.0})()
        conn = FakeConnection(self.tmp / "pod", drop)
        collect.handle(entry, client, run_dir, args, connect=lambda ip, port: conn)
        return entry, run_dir

    def test_complete_job_is_pulled_and_the_pod_deleted(self):
        make_job(self.tmp / "pod" / self.KEY, OK_META, OK_VIDEOS)
        (self.tmp / "pod" / "pod_stdout.log").write_text("stdout\n")
        client = FakeClient()
        entry, run_dir = self.run_handle(client)
        self.assertEqual((entry["status"], client.calls), ("collected", [("delete", "p1")]))
        self.assertTrue((run_dir / self.KEY / "eval_result" / "video" / "5_success.mp4").is_file())
        self.assertTrue((run_dir / f"{self.KEY}.pod_stdout.log").is_file())

    def test_failed_eval_is_pulled_deleted_and_reported(self):
        make_job(self.tmp / "pod" / self.KEY, None, exit_code=2)
        client = FakeClient()
        entry, _ = self.run_handle(client)
        self.assertEqual((entry["status"], client.calls), ("failed", [("delete", "p1")]))

    def test_incomplete_copy_stops_instead_of_deleting(self):
        make_job(self.tmp / "pod" / self.KEY, OK_META, OK_VIDEOS)
        client = FakeClient()
        entry, _ = self.run_handle(client, drop={"5_success.mp4"})
        self.assertEqual((entry["status"], client.calls), ("incomplete_copy", [("stop", "p1")]))

    def test_running_job_waits(self):
        (self.tmp / "pod" / self.KEY / "deploy").mkdir(parents=True)
        client = FakeClient()
        entry, _ = self.run_handle(client)
        self.assertEqual((entry["status"], client.calls), ("launched", []))

    def test_overdue_job_is_pulled_and_stopped(self):
        (self.tmp / "pod" / self.KEY / "deploy" / "t").mkdir(parents=True)
        (self.tmp / "pod" / self.KEY / "deploy" / "t" / "metadata.json").write_text("{}")
        client = FakeClient()
        entry, run_dir = self.run_handle(client, limit_h=1.0, hours_old=2.0)
        self.assertEqual((entry["status"], client.calls), ("timeout", [("stop", "p1")]))
        self.assertTrue((run_dir / self.KEY / "deploy" / "t" / "metadata.json").is_file())

    def test_gone_pod_is_lost(self):
        client = FakeClient(status=None)
        entry, _ = self.run_handle(client)
        self.assertEqual((entry["status"], client.calls), ("lost", []))


class RelaunchClient(FakeClient):
    """FakeClient that can create pods (or have none available) and report a balance."""

    def __init__(self, balance=10.0, available=True):
        super().__init__()
        self.money, self.available, self.created = balance, available, 0

    def balance(self):
        return self.money

    def create_pod(self, spec):
        if not self.available:
            raise runpod_api.RunPodError("POST /pods: HTTP 500 no instances currently available")
        self.created += 1
        return {"id": f"new{self.created}", "costPerHr": 0.34}

    def list_pods(self, name=None):
        return []


class Relaunch(TempDir):
    ARGS = dict(max_pods=2, min_balance_usd=4.0, stop_balance_usd=1.5, max_attempts=3, notify_cmd="")

    def record(self, *statuses):
        pods = []
        for i, status in enumerate(statuses):
            entry = {"key": f"job{i}", "status": status, "spec": {"name": f"run-job{i}"}, "limit_h": 2.0}
            if status == "launched":
                entry.update(pod_id=f"old{i}", created_utc=results.utc_now())
            pods.append(entry)
        return {"pods": pods}

    def run_pass(self, record, client):
        collect.relaunch_pass(record, client, type("Args", (), self.ARGS)(), self.tmp)
        return [e["status"] for e in record["pods"]]

    def test_guard(self):
        self.assertEqual(collect.launch_guard(1, 2, 10.0, 4.0), "")
        self.assertIn("max", collect.launch_guard(2, 2, 10.0, 4.0))
        self.assertIn("balance", collect.launch_guard(0, 2, 3.0, 4.0))

    def test_creates_waiting_jobs_up_to_the_pod_cap(self):
        record = self.record("launched", "not_created", "not_created")
        client = RelaunchClient()
        self.assertEqual(self.run_pass(record, client), ["launched", "launched", "not_created"])
        self.assertEqual((record["pods"][1]["pod_id"], record["pods"][1]["attempts"]), ("new1", 1))

    def test_no_instances_keeps_the_job_waiting(self):
        record = self.record("not_created")
        self.assertEqual(self.run_pass(record, RelaunchClient(available=False)), ["not_created"])
        self.assertIn("no instances", record["pods"][0]["error"])

    def test_low_balance_creates_nothing(self):
        record = self.record("not_created")
        client = RelaunchClient(balance=3.0)
        self.assertEqual((self.run_pass(record, client), client.created), (["not_created"], 0))

    def test_balance_below_the_stop_line_stops_running_pods(self):
        record = self.record("launched", "not_created")
        client = RelaunchClient(balance=1.0)
        self.assertEqual(self.run_pass(record, client), ["stopped_low_balance", "not_created"])
        self.assertEqual((client.calls, client.created), ([("stop", "old0")], 0))

    def test_host_failure_is_moved_aside_and_requeued_until_the_attempt_cap(self):
        record = self.record("failed", "failed", "failed")
        for entry, attempts in zip(record["pods"], (1, 3, 1)):
            entry.update(evaluated=0, attempts=attempts)
        record["pods"][2]["evaluated"] = 40  # the eval itself ran: not a host failure
        (self.tmp / "job0").mkdir()
        (self.tmp / "job0" / "DONE").write_text("139\n")
        statuses = self.run_pass(record, RelaunchClient(available=False))
        self.assertEqual(statuses, ["not_created", "failed", "failed"])
        self.assertTrue((self.tmp / "job0.attempt1" / "DONE").is_file())
        self.assertFalse((self.tmp / "job0").exists())


if __name__ == "__main__":
    unittest.main()
