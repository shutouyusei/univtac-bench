"""Tests for scripts/isaac45/runpod/{runpod_api,launch,collect}.py: python3 -m unittest discover -s scripts/isaac45/tests"""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod"))
import collect  # noqa: E402
import launch  # noqa: E402
import runpod_api  # noqa: E402


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RunPodClient(unittest.TestCase):
    def test_create_pod_posts_json_with_bearer_key(self):
        seen = {}

        def opener(request, timeout):
            seen["url"], seen["method"] = request.full_url, request.get_method()
            seen["auth"] = request.get_header("Authorization")
            seen["body"] = json.loads(request.data)
            return FakeResponse(b'{"id": "abc123"}')

        client = runpod_api.RunPod("KEY", opener=opener)
        pod = client.create_pod({"name": "x", "imageName": "img"})
        self.assertEqual(pod["id"], "abc123")
        self.assertEqual(seen["url"], "https://rest.runpod.io/v1/pods")
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["auth"], "Bearer KEY")
        self.assertEqual(seen["body"], {"name": "x", "imageName": "img"})

    def test_network_errors_become_runpod_errors(self):
        import urllib.error

        def opener(request, timeout):
            raise urllib.error.URLError("timed out")

        with self.assertRaises(runpod_api.RunPodError):
            runpod_api.RunPod("KEY", opener=opener).create_pod({})

    def test_delete_returns_none_on_empty_body(self):
        client = runpod_api.RunPod("KEY", opener=lambda request, timeout: FakeResponse(b""))
        self.assertIsNone(client.delete_pod("abc123"))

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
        text = launch.jobs_to_csv(jobs)
        self.assertEqual(launch.parse_jobs(text), jobs)

    def test_parse_rejects_bad_rows(self):
        header = "task,arm,start_seed,count,task_config\n"
        for row in ["lift_can,both,0,10,demo", "lift_can,univtac,x,10,demo", "lift_can,univtac,0,0,demo"]:
            with self.assertRaises(ValueError):
                launch.parse_jobs(header + row + "\n")

    def test_parse_rejects_unknown_task_or_task_config(self):
        header = "task,arm,start_seed,count,task_config\n"
        with self.assertRaises(ValueError):
            launch.parse_jobs(header + "lift_cam,univtac,0,10,demo\n", task_configs={"demo"})
        with self.assertRaises(ValueError):
            launch.parse_jobs(header + "lift_can,univtac,0,10,clen\n", task_configs={"demo", "clean"})

    def test_time_limit_follows_the_seed_count(self):
        self.assertAlmostEqual(launch.limit_hours(count=90, episode_timeout_s=400), (600 + 400 * 90) / 3600 + 1.0)

    def test_parse_rejects_overlapping_seed_ranges(self):
        text = "task,arm,start_seed,count,task_config\nlift_can,univtac,0,10,demo\nlift_can,univtac,5,10,demo\n"
        with self.assertRaises(ValueError):
            launch.parse_jobs(text)

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


class Decide(unittest.TestCase):
    def test_actions(self):
        self.assertEqual(collect.decide(None, done=False, hours=0.1, limit_h=6), "lost")
        self.assertEqual(collect.decide("TERMINATED", done=False, hours=0.1, limit_h=6), "lost")
        self.assertEqual(collect.decide("RUNNING", done=True, hours=1, limit_h=6), "collect")
        self.assertEqual(collect.decide("RUNNING", done=False, hours=1, limit_h=6), "wait")
        self.assertEqual(collect.decide("RUNNING", done=False, hours=7, limit_h=6), "timeout")
        self.assertEqual(collect.decide("EXITED", done=False, hours=1, limit_h=6), "exited")


def make_job(job: Path, metadata=None, videos=(), manifest=True, exit_code=0):
    """A job folder as eval_released.sh leaves it (metadata None: upstream wrote no eval_result)."""
    job.mkdir(parents=True)
    pod_check = ["metadata.json missing"]
    if metadata is not None:
        (job / "eval_result" / "video").mkdir(parents=True)
        (job / "eval_result" / "metadata.json").write_text(json.dumps(metadata))
        for name in videos:
            (job / "eval_result" / "video" / name).write_bytes(b"\0")
        pod_check = []
    (job / "eval.log").write_text("log\n")
    if manifest:
        cov = {"expected": 1, "evaluated": 1 if metadata else 0, "success": 1 if metadata else 0, "errors": 0,
               "missing": [] if metadata else [5], "outside": []}
        (job / "manifest.json").write_text(json.dumps({"transfer_check_on_pod": pod_check, "exit_code": exit_code,
                                                       "coverage": cov}))
    (job / "DONE").write_text(f"{exit_code}\n")
    return job


def file_count(job: Path) -> int:
    return sum(1 for p in job.rglob("*") if p.is_file() and p.name != "DONE")


class VerifyCollected(unittest.TestCase):
    def test_complete_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "j", {"5": {"result": "success"}}, ["5_success.mp4"])
            self.assertEqual(collect.verify_collected(job, remote_files=file_count(job)), [])

    def test_faithful_copy_of_a_failed_eval_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "j", None, exit_code=2)
            self.assertEqual(collect.verify_collected(job, remote_files=file_count(job)), [])

    def test_missing_manifest_or_file_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "j", {"5": {"result": "success"}}, ["5_success.mp4"], manifest=False)
            self.assertTrue(any("manifest" in p for p in collect.verify_collected(job, file_count(job))))
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "j", {"5": {"result": "success"}}, ["5_success.mp4"])
            self.assertTrue(any("files" in p for p in collect.verify_collected(job, remote_files=9)))

    def test_video_lost_in_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "j", {"5": {"result": "success"}}, [])
            self.assertTrue(any("video" in p for p in collect.verify_collected(job, file_count(job))))


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
    """A pod whose POD_OUT is a local folder; pull copies like rsync -a."""

    def __init__(self, pod_out: Path, drop=()):
        self.pod_out, self.drop = pod_out, set(drop)

    def _remote(self, remote: str) -> Path:
        return self.pod_out / remote[len(launch.POD_OUT) + 1:].rstrip("/")

    def done(self, remote_job):
        return (self._remote(remote_job) / "DONE").is_file()

    def count_files(self, remote_job):
        return file_count(self._remote(remote_job))

    def pull(self, remote, local: Path):
        import shutil
        src = self._remote(remote)
        if src.is_dir():
            shutil.copytree(src, local, dirs_exist_ok=True,
                            ignore=lambda d, names: [n for n in names if n in self.drop])
        elif src.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local)
        return True


class Handle(unittest.TestCase):
    KEY = "lift_can_univtac_5_1"

    def run_handle(self, tmp: Path, client, drop=(), limit_h=6.0, hours_old=0.5):
        import datetime
        created = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours_old))
        entry = {"key": self.KEY, "pod_id": "p1", "status": "launched", "limit_h": limit_h,
                 "created_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ")}
        run_dir = tmp / "run"
        run_dir.mkdir()
        args = type("Args", (), {"notify_cmd": "", "ssh_key": "k"})()
        conn = FakeConnection(tmp / "pod", drop)
        collect.handle(entry, client, run_dir, args, connect=lambda ip, port: conn)
        return entry, run_dir

    def test_complete_job_is_pulled_and_the_pod_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_job(tmp / "pod" / self.KEY, {"5": {"result": "success"}}, ["5_success.mp4"])
            (tmp / "pod" / "pod_stdout.log").write_text("stdout\n")
            client = FakeClient()
            entry, run_dir = self.run_handle(tmp, client)
            self.assertEqual(entry["status"], "collected")
            self.assertEqual(client.calls, [("delete", "p1")])
            self.assertTrue((run_dir / self.KEY / "eval_result" / "video" / "5_success.mp4").is_file())
            self.assertTrue((run_dir / f"{self.KEY}.pod_stdout.log").is_file())

    def test_failed_eval_is_pulled_deleted_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_job(tmp / "pod" / self.KEY, None, exit_code=2)
            client = FakeClient()
            entry, _ = self.run_handle(tmp, client)
            self.assertEqual(entry["status"], "failed")
            self.assertEqual(client.calls, [("delete", "p1")])

    def test_incomplete_copy_stops_instead_of_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_job(tmp / "pod" / self.KEY, {"5": {"result": "success"}}, ["5_success.mp4"])
            client = FakeClient()
            entry, _ = self.run_handle(tmp, client, drop={"5_success.mp4"})
            self.assertEqual(entry["status"], "incomplete_copy")
            self.assertEqual(client.calls, [("stop", "p1")])

    def test_running_job_waits_and_overdue_job_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "pod" / self.KEY / "deploy").mkdir(parents=True)
            client = FakeClient()
            entry, _ = self.run_handle(tmp, client)
            self.assertEqual((entry["status"], client.calls), ("launched", []))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "pod" / self.KEY / "deploy" / "t").mkdir(parents=True)
            (tmp / "pod" / self.KEY / "deploy" / "t" / "metadata.json").write_text("{}")
            client = FakeClient()
            entry, run_dir = self.run_handle(tmp, client, limit_h=1.0, hours_old=2.0)
            self.assertEqual((entry["status"], client.calls), ("timeout", [("stop", "p1")]))
            self.assertTrue((run_dir / self.KEY / "deploy" / "t" / "metadata.json").is_file())

    def test_gone_pod_is_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(status=None)
            entry, _ = self.run_handle(Path(tmp), client)
            self.assertEqual((entry["status"], client.calls), ("lost", []))


if __name__ == "__main__":
    unittest.main()
