"""policy/RDP/deploy_policy.py: UniVTAC observation -> controller inputs -> take_action('qpos')."""

import numpy as np
import pytest
import torch

from policy.RDP import deploy_policy as dp
from rdp_univtac import marker


class FakeTask:
    device = "cpu"

    def __init__(self):
        self.actions = []

    def take_action(self, action, action_type):
        self.actions.append((action.clone(), action_type))
        return True, False


def observation(k, side_key="tactile"):
    rng = np.random.default_rng(k)
    m = np.empty((2, 63, 2), np.float32)
    m[0] = rng.uniform(0, 300, (63, 2))
    m[1] = m[0] + 1.5
    still = m.copy()
    still[1] = still[0]
    head = np.zeros((270, 480, 3), np.uint8)
    head[..., 0] = 200
    return {
        "observation": {"head": {"rgb": torch.from_numpy(head)}},
        "embodiment": {"joint": torch.arange(9, dtype=torch.float32) + k},
        "tactile": {f"left_{side_key}": {"marker": torch.from_numpy(m)},
                    f"right_{side_key}": {"marker": torch.from_numpy(still)}},
    }


def make_policy(bundle):
    return dp.Policy({"rdp_bundle": str(bundle), "rdp_device": "cpu", "task_name": "x", "task_config": "y"})


def test_encode_matches_the_training_preprocessing(rdp_bundle):
    policy = make_policy(rdp_bundle)
    obs = observation(3)
    frames, joint, tactile = policy.encode(obs)
    assert frames["cam_head"].shape == (240, 320, 3) and frames["cam_head"][..., 0].mean() > 190
    assert frames["cam_head"][..., 2].max() < 10  # channel order kept
    assert np.allclose(joint, np.arange(9) + 3)
    disp = marker.displacement(obs["tactile"]["left_tactile"]["marker"].numpy()[None])
    assert np.allclose(tactile["tactile_left"], policy.controller.pca.embed(disp)[0], atol=1e-5)
    assert tactile["tactile_right"].shape == (15,)


def test_eval_takes_one_qpos_action_per_call_and_reset_restarts(rdp_bundle):
    policy = make_policy(rdp_bundle)
    task = FakeTask()
    for k in range(3):
        policy.eval(task, observation(k, side_key="gsmini"))
    assert len(task.actions) == 3
    assert all(t == "qpos" and a.shape == (8,) for a, t in task.actions)
    policy.reset()
    assert policy.controller.step == 0


def test_default_bundle_path_follows_task_config_and_episodes():
    args = {"task_name": "lift_bottle", "task_config": "clean51", "expert_data_num": "50"}
    assert dp.default_bundle(args).parts[-3:] == ("rdp_ckpt", "lift_bottle-clean51-50", "rdp_policy.pt")


def test_missing_bundle_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_policy(tmp_path / "none.pt")
