"""The stflow adapter feeds the policy what the training cache holds and executes chunks in order."""

import cv2
import numpy as np
import pytest
import torch

stflow = pytest.importorskip("stflow")

from stflow import checkpoint  # noqa: E402
from stflow.config import Config, ModelConfig  # noqa: E402
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


def test_executes_n_action_steps_before_sampling_again(ckpt, monkeypatch):
    policy = make_policy(ckpt, stflow_n_action_steps=4)
    calls = []
    real = policy.model.sample

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(policy.model, "sample", counting)
    task = FakeTask()
    obs = observation(np.random.default_rng(1))
    for _ in range(9):
        policy.eval(task, obs)
    assert len(calls) == 3 and len(task.actions) == 9
    policy.reset()
    policy.eval(task, obs)
    assert len(calls) == 4


def test_rejects_an_execution_horizon_beyond_the_chunk(ckpt):
    with pytest.raises(ValueError):
        make_policy(ckpt, stflow_n_action_steps=CHUNK + 1)


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
