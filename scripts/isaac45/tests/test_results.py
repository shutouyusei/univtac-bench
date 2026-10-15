"""Tests for scripts/isaac45/runpod/results.py: python3 -m unittest discover -s scripts/isaac45/tests"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod"))
import results  # noqa: E402

RELEASED_LOG_HEAD = """[2026-03-24 11:09:26] Task Name: lift_can
[2026-03-24 11:09:26] Eval Config: {
    "policy_name": "ACT",
    "task_config": "demo",
    "train_config": "train_config_all"
}
"""


class TrainConfigFromLog(unittest.TestCase):
    def test_reads_the_released_eval_config(self):
        self.assertEqual(results.train_config_from_log(RELEASED_LOG_HEAD), "train_config_all")

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            results.train_config_from_log("[2026-03-24] Task Name: lift_can\n")


class SeedCoverage(unittest.TestCase):
    def test_complete_run(self):
        meta = {"10": {"result": "success"}, "11": {"result": "failed"}, "12": {"result": "success"}}
        cov = results.seed_coverage(meta, start=10, count=3)
        self.assertEqual(cov, {"expected": 3, "evaluated": 3, "success": 2, "errors": 0, "missing": [],
                               "outside": []})

    def test_errored_seeds_leave_the_denominator_as_upstream_does(self):
        meta = {"10": {"result": "success"}, "11": {"result": "error"}, "12": {"result": "failed"}}
        cov = results.seed_coverage(meta, start=10, count=3)
        self.assertEqual((cov["evaluated"], cov["success"], cov["errors"], cov["missing"]), (2, 1, 1, []))

    def test_missing_and_outside_seeds(self):
        meta = {"10": {"result": "success"}, "13": {"result": "success"}, "note": {"result": "success"}}
        cov = results.seed_coverage(meta, start=10, count=3)
        self.assertEqual(cov["evaluated"], 1)
        self.assertEqual(cov["success"], 1)
        self.assertEqual(cov["missing"], [11, 12])
        self.assertEqual(cov["outside"], [13])


class CheckTransfer(unittest.TestCase):
    def make_run(self, root: Path, seeds: dict, videos: list) -> Path:
        run = root / "eval_result"
        (run / "video").mkdir(parents=True)
        (run / "metadata.json").write_text(json.dumps(seeds))
        for name in videos:
            (run / "video" / name).write_bytes(b"\0")
        return run

    def test_complete_transfer_has_no_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp), {"5": {"result": "success"}, "6": {"result": "failed"}},
                                ["5_success.mp4", "6_failed.mp4"])
            self.assertEqual(results.check_transfer(run), [])

    def test_missing_video_is_a_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp), {"5": {"result": "success"}, "6": {"result": "failed"}},
                                ["5_success.mp4"])
            problems = results.check_transfer(run)
            self.assertEqual(len(problems), 1)
            self.assertIn("video", problems[0])

    def test_missing_or_broken_metadata_is_a_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            # messages carry no absolute path, so the pod's and the local check can be compared
            self.assertEqual(results.check_transfer(Path(tmp) / "nothing"), ["metadata.json missing"])
            run = self.make_run(Path(tmp), {}, [])
            (run / "metadata.json").write_text("{not json")
            self.assertIn("unreadable", results.check_transfer(run)[0])


if __name__ == "__main__":
    unittest.main()
