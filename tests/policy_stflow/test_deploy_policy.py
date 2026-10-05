"""The stflow adapter feeds the policy what the training cache holds and executes chunks in order."""

import dataclasses

import cv2
import numpy as np
import pytest
import torch

stflow = pytest.importorskip("stflow")

from stflow import checkpoint  # noqa: E402
from stflow.config import Config, MethodConfig, ModelConfig  # noqa: E402
from stflow.data.univtac_hdf5 import decode  # noqa: E402
from stflow.model.policy import FlowPolicy  # noqa: E402

from policy.stflow.deploy_policy import Policy  # noqa: E402

SIZE = 64
CHUNK = 6


@pytest.fixture
def ckpt(tmp_path):
    model = ModelConfig(
        image_size=SIZE, pretrained_backbone=False, d_model=32, n_heads=4, n_layers=1,
        dim_feedforward=64, dropout=0.0, chunk_size=CHUNK, num_inference_steps=2,
    )
    torch.manual_seed(0)
    return checkpoint.save(tmp_path / "ckpt", FlowPolicy(model), Config(model=model))


def observation(rng) -> dict:
    def frame(h, w):
        return torch.from_numpy(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))

    return {
        "observation": {"head": {"rgb": frame(270, 480)}},
        "tactile": {"left_tactile": {"rgb_marker": frame(240, 320)}, "right_tactile": {"rgb_marker": frame(240, 320)}},
        "embodiment": {"joint": torch.arange(9, dtype=torch.float32)},
    }


class FakeTask:
    device = "cpu"

    def __init__(self):
        self.actions = []

    def take_action(self, action, action_type):
        assert action_type == "qpos"
        self.actions.append(action.clone())
        return True, False


def make_policy(ckpt, **extra):
    return Policy({"stflow_ckpt_dir": str(ckpt), "stflow_device": "cpu", "seed": 0, **extra})


def test_adapter_frames_equal_the_training_cache_frames(ckpt):
    policy = make_policy(ckpt)
    obs = observation(np.random.default_rng(0))
    raw = obs["observation"]["head"]["rgb"].numpy()
    ok, buf = cv2.imencode(".jpg", raw)
    cached = decode(buf.tobytes(), SIZE)  # what build_cache stores for this frame
    live = policy.encode_obs(obs)["images"]["head"][0]
    assert torch.equal((live * 255).round().byte().permute(1, 2, 0), torch.from_numpy(cached))


def test_state_is_the_first_eight_joints(ckpt):
    state = make_policy(ckpt).encode_obs(observation(np.random.default_rng(0)))["state"]
    assert state.tolist() == [list(range(8))]


def count_samples(policy, monkeypatch) -> list:
    calls = []
    real = policy.model.sample

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(policy.model, "sample", counting)
    return calls


def test_executes_n_action_steps_before_sampling_again(ckpt, monkeypatch):
    policy = make_policy(ckpt, stflow_method_args={"n_action_steps": 4})
    calls = count_samples(policy, monkeypatch)
    task = FakeTask()
    obs = observation(np.random.default_rng(1))
    for _ in range(9):
        policy.eval(task, obs)
    assert len(calls) == 3 and len(task.actions) == 9
    policy.reset()
    policy.eval(task, obs)
    assert len(calls) == 4


def test_uses_the_checkpoint_method_and_lets_the_deploy_file_override_it(tmp_path, monkeypatch):
    model = ModelConfig(
        image_size=SIZE, pretrained_backbone=False, d_model=32, n_heads=4, n_layers=1,
        dim_feedforward=64, dropout=0.0, chunk_size=CHUNK, num_inference_steps=2,
    )
    cfg = Config(model=model, method=MethodConfig(name="chunk", args={"n_action_steps": 2}))
    path = checkpoint.save(tmp_path / "ckpt", FlowPolicy(model), cfg)
    obs = observation(np.random.default_rng(2))
    for extra, expected_samples in (({}, 3), ({"stflow_method_args": {"n_action_steps": 3}}, 2)):
        policy = make_policy(path, **extra)
        calls = count_samples(policy, monkeypatch)
        for _ in range(6):
            policy.eval(FakeTask(), obs)
        assert len(calls) == expected_samples, extra


def test_rejects_an_execution_horizon_beyond_the_chunk(ckpt):
    with pytest.raises(ValueError):
        make_policy(ckpt, stflow_method_args={"n_action_steps": CHUNK + 1})


@pytest.mark.parametrize("key", ["stflow_n_action_steps", "stflow_num_steps"])
def test_rejects_the_replaced_deploy_keys(ckpt, key):
    with pytest.raises(ValueError, match="stflow_method_args"):
        make_policy(ckpt, **{key: 4})


def first_action(policy, obs):
    task = FakeTask()
    policy.eval(task, obs)
    return task.actions[0]


def test_an_episode_does_not_depend_on_the_episodes_before_it(ckpt):
    """Re-running one seed alone must sample what it sampled inside a full evaluation."""
    obs = observation(np.random.default_rng(3))
    alone = first_action(make_policy(ckpt), obs)

    policy = make_policy(ckpt)
    other = observation(np.random.default_rng(4))
    for _ in range(3):  # earlier episodes draw noise
        policy.reset()
        for _ in range(CHUNK + 2):
            policy.eval(FakeTask(), other)
    policy.reset()
    assert torch.equal(first_action(policy, obs), alone)


def test_actions_are_eight_joint_targets(ckpt):
    action = first_action(make_policy(ckpt), observation(np.random.default_rng(5)))
    assert action.shape == (8,) and action.dtype == torch.float32


def test_reads_gsmini_fingertips_like_the_training_reader(ckpt):
    obs = observation(np.random.default_rng(6))
    renamed = {**obs, "tactile": {
        "left_gsmini": obs["tactile"]["left_tactile"],
        "right_gsmini": obs["tactile"]["right_tactile"],
    }}
    policy = make_policy(ckpt)
    a = policy.encode_obs(obs)["tactile"]["left"]
    b = policy.encode_obs(renamed)["tactile"]["left"]
    assert torch.equal(a, b)


def test_a_checkpoint_without_a_marker_target_reads_no_marker_observation(ckpt):
    obs = observation(np.random.default_rng(7))  # no "marker" entries: reading one would fail
    assert "tactile_marker" not in make_policy(ckpt).encode_obs(obs)


def marker_grid() -> torch.Tensor:
    rows, cols = torch.meshgrid(torch.arange(7.0), torch.arange(9.0), indexing="ij")
    return torch.stack([44.14 + 28.97 * cols.flatten(), 35.26 + 28.25 * rows.flatten()], 1)


def marker_ckpt(tmp_path, tactile_names):
    if "tactile_target" not in {f.name for f in dataclasses.fields(ModelConfig)}:
        pytest.skip("this stflow has no tactile targets")
    pytest.importorskip("stflow.methods.sfp")
    model = ModelConfig(
        image_size=SIZE, pretrained_backbone=False, d_model=32, n_heads=4, n_layers=1, dim_feedforward=64,
        dropout=0.0, chunk_size=CHUNK, tactile_names=tactile_names, tactile_target="marker", tactile_target_dim=252,
    )
    torch.manual_seed(0)
    policy = FlowPolicy(model)
    if policy.tactile_target is not None:
        policy.tactile_target.grid.copy_(marker_grid().expand(2, -1, -1))
    cfg = Config(model=model, method=MethodConfig(name="sfp", args={"n_action_steps": 2}))
    return checkpoint.save(tmp_path / "ckpt", policy, cfg)


def with_markers(obs: dict, rng) -> dict:
    """The fingertip entries with UniVTAC's (2, 63, 2) marker motion, markers listed in a random order."""
    grid = marker_grid()
    for side in ("left_tactile", "right_tactile"):
        perm = torch.from_numpy(rng.permutation(63))
        current = grid + torch.from_numpy(rng.normal(size=(63, 2))).float()
        obs["tactile"][side]["marker"] = torch.stack([grid[perm], current[perm]])
    return obs


def test_a_marker_target_checkpoint_gets_the_raw_marker_observation_and_measures_it(tmp_path):
    policy = make_policy(marker_ckpt(tmp_path, ["left", "right"]))
    obs = with_markers(observation(np.random.default_rng(8)), np.random.default_rng(9))
    encoded = policy.encode_obs(obs)
    for name in ("left", "right"):
        marker = encoded["tactile_marker"][name]
        assert marker.shape == (1, 2, 63, 2) and marker.dtype == torch.float32
        assert torch.equal(marker[0], obs["tactile"][f"{name}_tactile"]["marker"])
    measured = policy.model.measure_tactile_target(encoded)
    task = FakeTask()
    policy.eval(task, obs)
    assert task.actions[0].shape == (8,) and torch.isfinite(task.actions[0]).all()
    # The chunk started from the measured target: one Euler step moved it by velocity / H only.
    assert torch.allclose(policy.controller.flow_state[0, 8:], measured[0], atol=1.0)
    assert measured.abs().sum() > 0


def test_a_no_tactile_marker_target_checkpoint_reads_no_tactile_observation(tmp_path):
    policy = make_policy(marker_ckpt(tmp_path, []))
    obs = observation(np.random.default_rng(10))
    del obs["tactile"]  # nothing tactile may be read
    task = FakeTask()
    for _ in range(3):
        policy.eval(task, obs)
    assert len(task.actions) == 3 and all(torch.isfinite(a).all() for a in task.actions)


def test_trace_records_the_prism_error_and_the_joints_of_every_step(ckpt):
    from envs.utils.transforms import Pose

    from policy.stflow.deploy_policy import TRACE_COLUMNS

    class Holder:
        def __init__(self, pose):
            self.pose = pose

        def get_pose(self):
            return self.pose

        get_gripper_center_pose = get_pose

    def task_with_prism():
        task = FakeTask()
        task.metadata, task.take_action_cnt = {}, 7
        task.prism = Holder(Pose([0.61, 0.02, 0.05]))
        task.target_pose = Pose([0.6, 0.0, 0.1])
        task._robot_manager = Holder(Pose([0.61, 0.02, 0.15]))
        task.origin_inhand_pose = Pose([0.0, 0.0, -0.08])  # the prism now hangs 0.10 below the gripper centre
        return task

    obs = observation(np.random.default_rng(11))
    task = task_with_prism()
    policy = make_policy(ckpt, stflow_trace=True)
    policy.eval(task, obs)
    policy.eval(task, obs)
    trace = task.metadata["trace"]
    assert trace["columns"] == TRACE_COLUMNS and len(trace["rows"]) == 2
    row = dict(zip(trace["columns"], trace["rows"][0]))
    assert row["action_count"] == 7
    assert (row["prism_x"], row["prism_y"], row["prism_z"]) == pytest.approx((0.01, 0.02, -0.05), abs=1e-5)
    assert row["prism_axis_dot"] == pytest.approx(1.0) and row["inhand_slip"] == pytest.approx(0.02, abs=1e-5)
    assert [row[f"command_{i}"] for i in range(8)] == pytest.approx(task.actions[0].tolist(), abs=1e-5)
    assert [row[f"joint_{i}"] for i in range(8)] == pytest.approx(list(range(8)))

    untraced = task_with_prism()
    make_policy(ckpt).eval(untraced, obs)
    assert "trace" not in untraced.metadata  # off by default

    bare = FakeTask()  # a task without a held prism still gets the joints
    bare.metadata, bare.take_action_cnt = {}, 0
    policy.eval(bare, obs)
    row = dict(zip(TRACE_COLUMNS, bare.metadata["trace"]["rows"][0]))
    assert np.isnan(row["prism_x"]) and np.isnan(row["inhand_slip"]) and row["joint_3"] == 3.0
