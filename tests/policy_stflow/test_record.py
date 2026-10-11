"""stflow_record: a per-step record of what the policy saw and did and of the simulator state, one npz per
episode, without changing the actions."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

stflow = pytest.importorskip("stflow")

from stflow import checkpoint  # noqa: E402
from stflow.config import Config, MethodConfig, ModelConfig  # noqa: E402
from stflow.model.policy import FlowPolicy  # noqa: E402

from policy.stflow.deploy_policy import Policy  # noqa: E402
from policy.stflow.record import controller_state  # noqa: E402

SIZE = 64
CHUNK = 6
NO_DOWNLOAD = {"pretrained_targets": False}


METHODS = {
    "chunk": MethodConfig("chunk", {}),
    "sfp": MethodConfig("sfp", {"n_action_steps": CHUNK}),
    "sfp_tr": MethodConfig("sfp_tr", {"interval": 2, "n_action_steps": CHUNK}),
}


def save_ckpt(path: Path, method: str = "chunk") -> Path:
    model = ModelConfig(
        image_size=SIZE, pretrained_backbone=False, d_model=32, n_heads=4, n_layers=1,
        dim_feedforward=64, dropout=0.0, chunk_size=CHUNK, num_inference_steps=2,
    )
    torch.manual_seed(0)
    return checkpoint.save(path, FlowPolicy(model), Config(model=model, method=METHODS[method]))


@pytest.fixture
def ckpt(tmp_path):
    return save_ckpt(tmp_path / "ckpt")


def observation(rng) -> dict:
    def frame(h, w):
        return torch.from_numpy(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))

    return {
        "observation": {"head": {"rgb": frame(270, 480)}},
        "tactile": {"left_tactile": {"rgb_marker": frame(240, 320)}, "right_tactile": {"rgb_marker": frame(240, 320)}},
        "embodiment": {"joint": torch.arange(9, dtype=torch.float32)},
    }


class FakePose:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class FakeActor:
    def __init__(self, task, offset):
        self.task, self.offset = task, offset

    def get_pose(self):
        # Moves with the step count so rows can be told apart.
        return FakePose([self.offset + self.task.take_action_cnt, 0, 0, 1, 0, 0, 0])


class RecordTask:
    """The parts of BaseTask the adapter and the recorder touch."""

    device = "cpu"

    def __init__(self, save_root: Path, seed: int, actors=("prism", "slot"), step_lim: int = 600):
        self.save_root = save_root
        self.cfg = SimpleNamespace(seed=seed, step_lim=step_lim)
        self.eval_success = False
        self.metadata = {}
        self.take_action_cnt = 0
        self.actions = []
        self._robot_manager = SimpleNamespace(
            get_ee_pose=lambda: FakePose([0.1, 0.2, 0.3, 1, 0, 0, 0]),
            get_gripper_center_pose=lambda: FakePose([0.1, 0.2, 0.2, 1, 0, 0, 0]),
        )
        self._actor_manager = SimpleNamespace(actors={name: FakeActor(self, 10 * i) for i, name in enumerate(actors)})
        self.target_pose = FakePose([0.5, 0, 0, 1, 0, 0, 0])

    def new_episode(self, seed: int):
        """What ``task.reset`` does to the parts the recorder reads."""
        self.cfg.seed = seed
        self.metadata = {}
        self.take_action_cnt = 0
        self.eval_success = False

    def take_action(self, action, action_type):
        assert action_type == "qpos"
        self.actions.append(action.clone())
        self.take_action_cnt += 1
        return True, False


def make_policy(ckpt, **extra):
    return Policy({"stflow_ckpt_dir": str(ckpt), "stflow_device": "cpu", "seed": 0, **extra})


def run_episode(policy, task, steps: int, rng) -> None:
    policy.reset()
    for _ in range(steps):
        policy.eval(task, observation(rng))


def test_recording_is_off_by_default(ckpt, tmp_path):
    task = RecordTask(tmp_path / "res", seed=7)
    policy = make_policy(ckpt)
    run_episode(policy, task, 3, np.random.default_rng(0))
    policy.reset()
    policy.close()
    assert "record" not in task.metadata
    assert not (tmp_path / "res" / "record").exists()


@pytest.mark.parametrize("method", sorted(METHODS))
def test_recording_does_not_change_the_actions(method, tmp_path):
    ckpt = save_ckpt(tmp_path / "ckpt", method)
    plain_task, rec_task = RecordTask(tmp_path / "a", seed=7), RecordTask(tmp_path / "b", seed=7)
    run_episode(make_policy(ckpt), plain_task, 8, np.random.default_rng(1))
    run_episode(make_policy(ckpt, stflow_record=NO_DOWNLOAD), rec_task, 8, np.random.default_rng(1))
    assert len(plain_task.actions) == len(rec_task.actions) == 8
    assert all(torch.equal(a, b) for a, b in zip(plain_task.actions, rec_task.actions))


def test_one_npz_per_episode_with_one_row_per_step(ckpt, tmp_path):
    task = RecordTask(tmp_path / "res", seed=1000003)
    policy = make_policy(ckpt, stflow_record=NO_DOWNLOAD)
    run_episode(policy, task, 5, np.random.default_rng(0))
    assert task.metadata["record"] == "record/1000003.npz"
    policy.reset()  # the next episode starts: the finished one is written
    rec = np.load(tmp_path / "res" / "record" / "1000003.npz")
    assert int(rec["seed"]) == 1000003
    assert rec["action_count"].tolist() == [0, 1, 2, 3, 4]
    assert rec["command"].shape == (5, 8)
    assert torch.equal(torch.from_numpy(rec["command"][2]), task.actions[2].float())
    assert rec["joint"].shape == (5, 9)
    assert rec["ee_pose"].shape == rec["gripper_center_pose"].shape == (5, 7)
    assert rec["actor_prism"][:, 0].tolist() == [0, 1, 2, 3, 4]
    assert rec["actor_slot"][:, 0].tolist() == [10, 11, 12, 13, 14]
    assert rec["task_target_pose"].tolist() == [0.5, 0, 0, 1, 0, 0, 0]
    assert rec["infer_ms"].shape == (5,) and (rec["infer_ms"] > 0).all()
    # ResNet18 tokens of both fingertips on the default 3 x 3 grid, as the demonstration targets.
    assert rec["tactile_tokens"].shape == (5, 2, 9, 512) and rec["tactile_tokens"].dtype == np.float16
    assert rec["fingertips"].tolist() == ["left", "right"]


def test_each_episode_keeps_its_own_seed(ckpt, tmp_path):
    task = RecordTask(tmp_path / "res", seed=1)
    policy = make_policy(ckpt, stflow_record=NO_DOWNLOAD)
    rng = np.random.default_rng(0)
    run_episode(policy, task, 3, rng)
    task.new_episode(seed=2)  # eval_policy resets the task before the policy
    run_episode(policy, task, 4, rng)
    policy.close()  # the last episode is written when the evaluation closes the policy
    assert len(np.load(tmp_path / "res" / "record" / "1.npz")["action_count"]) == 3
    assert len(np.load(tmp_path / "res" / "record" / "2.npz")["action_count"]) == 4


def test_fingertip_frames_every_k_steps_as_jpeg(ckpt, tmp_path):
    task = RecordTask(tmp_path / "res", seed=5)
    policy = make_policy(ckpt, stflow_record={**NO_DOWNLOAD, "frames_every": 2, "tactile_tokens": False})
    rng = np.random.default_rng(0)
    policy.reset()
    raws = []
    for _ in range(5):
        obs = observation(rng)
        raws.append(obs["tactile"]["left_tactile"]["rgb_marker"].numpy())
        policy.eval(task, obs)
    policy.close()
    rec = np.load(tmp_path / "res" / "record" / "5.npz")
    assert "tactile_tokens" not in rec
    assert rec["frame_steps"].tolist() == [0, 2, 4]
    data, offsets = rec["frames_left_jpeg"], rec["frames_left_offsets"]
    assert len(offsets) == 4
    second = cv2.imdecode(data[offsets[1] : offsets[2]], cv2.IMREAD_COLOR)
    assert second.shape == (240, 320, 3)
    # JPEG of the step-2 frame: closer to it than to any other step's frame (random frames compress poorly).
    errors = [np.abs(second.astype(int) - raw.astype(int)).mean() for raw in raws]
    assert int(np.argmin(errors)) == 2


def test_missing_simulator_parts_become_nan(ckpt, tmp_path):
    task = RecordTask(tmp_path / "res", seed=9, actors=())
    del task._robot_manager
    del task.target_pose
    policy = make_policy(ckpt, stflow_record={**NO_DOWNLOAD, "tactile_tokens": False})
    run_episode(policy, task, 2, np.random.default_rng(0))
    policy.close()
    rec = np.load(tmp_path / "res" / "record" / "9.npz")
    assert np.isnan(rec["ee_pose"]).all()
    assert not any(key.startswith(("actor_", "task_")) for key in rec.files)


@pytest.mark.parametrize(
    "controller, expected",
    [
        # streaming flow (SFP / SFP-TR): ``step`` counts actions in the chunk after ``act``
        (SimpleNamespace(step=1, reading_step=0, n_action_steps=6), {"new_chunk": 1, "reread": 0}),
        (SimpleNamespace(step=4, reading_step=3, n_action_steps=6), {"new_chunk": 0, "reread": 1}),
        (SimpleNamespace(step=4, reading_step=2, n_action_steps=6), {"new_chunk": 0, "reread": 0}),
        # chunk sampling: a fresh chunk leaves n_action_steps - 1 actions queued
        (SimpleNamespace(queue=[0] * 5, n_action_steps=6), {"new_chunk": 1, "reread": 0}),
        (SimpleNamespace(queue=[0] * 2, n_action_steps=6), {"new_chunk": 0, "reread": 0}),
    ],
)
def test_controller_state_flags(controller, expected):
    state = controller_state(controller)
    assert {k: state[k] for k in expected} == expected


def test_flags_of_a_real_sfp_tr_controller(tmp_path):
    """interval 2, chunk 6: a chunk starts at steps 0, 6, 12 and the tactile is read again at 2, 4, 8, 10."""
    task = RecordTask(tmp_path / "res", seed=3)
    policy = make_policy(save_ckpt(tmp_path / "ckpt", "sfp_tr"), stflow_record={**NO_DOWNLOAD, "tactile_tokens": False})
    run_episode(policy, task, 13, np.random.default_rng(0))
    policy.close()
    rec = np.load(tmp_path / "res" / "record" / "3.npz")
    assert np.flatnonzero(rec["new_chunk"]).tolist() == [0, 6, 12]
    assert np.flatnonzero(rec["reread"]).tolist() == [2, 4, 8, 10]


def test_final_state_after_the_last_action(ckpt, tmp_path):
    task = RecordTask(tmp_path / "res", seed=4)
    policy = make_policy(ckpt, stflow_record={**NO_DOWNLOAD, "tactile_tokens": False})
    run_episode(policy, task, 5, np.random.default_rng(0))
    policy.close()
    rec = np.load(tmp_path / "res" / "record" / "4.npz")
    assert int(rec["final_action_count"]) == 5
    assert rec["final_actor_prism"][0] == 5  # the pose after the fifth action, which the rows never see
    assert rec["final_ee_pose"].shape == (7,)


@pytest.mark.parametrize("ending", ["success", "step_lim"])
def test_written_as_soon_as_the_episode_ends(ckpt, tmp_path, ending):
    task = RecordTask(tmp_path / "res", seed=6, step_lim=3 if ending == "step_lim" else 600)
    policy = make_policy(ckpt, stflow_record={**NO_DOWNLOAD, "tactile_tokens": False})
    rng = np.random.default_rng(0)
    policy.reset()
    for i in range(3):
        if ending == "success" and i == 2:
            task.eval_success = True  # the simulator would set it inside take_action
        policy.eval(task, observation(rng))
    # no reset / close yet: a crash in the simulator teardown cannot lose the episode
    assert len(np.load(tmp_path / "res" / "record" / "6.npz")["action_count"]) == 3


class BrokenActor:
    def get_pose(self):
        raise RuntimeError("soft body has no pose")


def test_a_failing_record_never_stops_the_evaluation(ckpt, tmp_path):
    task = RecordTask(tmp_path / "res", seed=8)
    task._actor_manager.actors["cloth"] = BrokenActor()
    policy = make_policy(ckpt, stflow_record={**NO_DOWNLOAD, "tactile_tokens": False})
    run_episode(policy, task, 4, np.random.default_rng(0))
    assert len(task.actions) == 4  # every action still executed
    assert "soft body has no pose" in task.metadata["record_error"]
    policy.reset()
    assert not (tmp_path / "res" / "record" / "8.npz").exists()
    # the next episode records normally
    del task._actor_manager.actors["cloth"]
    task.new_episode(seed=9)
    run_episode(policy, task, 2, np.random.default_rng(1))
    policy.close()
    assert (tmp_path / "res" / "record" / "9.npz").exists()


def test_a_failing_write_never_stops_the_evaluation(ckpt, tmp_path):
    blocker = tmp_path / "res"
    blocker.mkdir()
    (blocker / "record").write_text("not a folder")  # mkdir of the record folder fails
    task = RecordTask(blocker, seed=10)
    policy = make_policy(ckpt, stflow_record={**NO_DOWNLOAD, "tactile_tokens": False})
    run_episode(policy, task, 2, np.random.default_rng(0))
    policy.reset()  # must not raise
    task.new_episode(seed=11)
    run_episode(policy, task, 2, np.random.default_rng(1))
    policy.close()
    assert len(task.actions) == 4


@pytest.mark.parametrize("spec", [1, "yes", ["tactile_tokens"]])
def test_a_malformed_record_key_is_rejected(ckpt, spec):
    with pytest.raises(ValueError, match="stflow_record"):
        make_policy(ckpt, stflow_record=spec)
