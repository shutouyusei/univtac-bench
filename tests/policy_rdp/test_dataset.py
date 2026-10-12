"""RDP's training windows on UniVTAC arrays: the official RealImageTactileDataset semantics (32-step window,
pad_before 3 / pad_after 28 with edge padding, observation frames at window indices 1 and 3, tactile over the
whole window, action[t] = state[t+1], arm joints relative to the last observed frame, gripper absolute) and the
official normalizers."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "policy" / "RDP"))

from rdp_univtac import dataset as ds  # noqa: E402


def episode_arrays(lengths=(40, 35), cams=("cam_head",)):
    joint, tl, tr, imgs = [], [], [], {c: [] for c in cams}
    for e, T in enumerate(lengths):
        t = np.arange(T, dtype=np.float32)
        j = np.stack([t + 100 * e + k for k in range(8)], 1)  # joint k at step t = t + 100 e + k
        joint.append(j)
        tl.append(np.repeat(t[:, None], 15, 1) + 1000 * e)
        tr.append(-np.repeat(t[:, None], 15, 1))
        for c in cams:
            imgs[c].append(np.full((T, 6, 8, 3), e * 50 + (t[:, None, None, None] % 50), np.uint8))
    return ds.concat_episodes(joint=joint, tactile={"tactile_left": tl, "tactile_right": tr}, images=imgs)


def test_concat_episodes_builds_next_state_actions_with_the_last_repeated():
    arrays = episode_arrays()
    assert list(arrays.episode_ends) == [40, 75]
    a, j = arrays.data["action"], arrays.data["joint"]
    assert np.allclose(a[:39], j[1:40]) and np.allclose(a[39], j[39])
    assert np.allclose(a[40:74], j[41:75]) and np.allclose(a[74], j[74])


def test_window_count_matches_the_official_sampler():
    d = ds.UniVTACImageTactileDataset(episode_arrays(), cameras=["cam_head"])
    # per episode: T + pad_before + pad_after - horizon + 1 = T windows
    assert len(d) == 40 + 35


def test_sample_layout_and_relative_actions():
    arrays = episode_arrays()
    d = ds.UniVTACImageTactileDataset(arrays, cameras=["cam_head"])
    s = d[10]  # episode 0, window starting at step 10 - 3 = 7
    obs, act, ext = s["obs"], s["action"], s["extended_obs"]
    assert obs["cam_head"].shape == (2, 3, 6, 8) and obs["cam_head"].dtype == torch.float32
    # observation frames: window indices 1 and 3 -> steps 8 and 10
    assert obs["cam_head"][0, 0, 0, 0] == pytest.approx(8 / 255) and obs["cam_head"][1, 0, 0, 0] == pytest.approx(10 / 255)
    # arm joints relative to the last observed frame (step 10), gripper absolute
    assert torch.allclose(obs["robot_joint"][1], torch.zeros(7))
    assert torch.allclose(obs["robot_joint"][0], torch.full((7,), -2.0))
    assert obs["gripper_width"][:, 0].tolist() == [8 + 7, 10 + 7]
    assert obs["tactile_left"][:, 0].tolist() == [8, 10]
    # action k = state at step 7 + k + 1; arm relative to step 10's arm joints, gripper absolute
    assert act.shape == (32, 8)
    assert act[0, 0] == pytest.approx(8 - 10) and act[3, 0] == pytest.approx(11 - 10)
    assert act[3, 7] == pytest.approx(11 + 7)
    # tactile over the whole window, same index as the action
    assert ext["tactile_left"].shape == (32, 15) and ext["tactile_left"][3, 0] == 10


def test_edge_padding_at_the_episode_start_and_end():
    d = ds.UniVTACImageTactileDataset(episode_arrays(), cameras=["cam_head"])
    first = d[0]  # window starts 3 steps before the episode: first frame repeated
    assert first["extended_obs"]["tactile_left"][:4, 0].tolist() == [0, 0, 0, 0]
    assert first["obs"]["gripper_width"][:, 0].tolist() == [7, 7]
    last = d[39]  # last window of episode 0 ends 28 steps after its last step: last frame repeated
    assert last["extended_obs"]["tactile_left"][-1, 0] == 39
    assert last["extended_obs"]["tactile_left"][3, 0] == 39


def test_normalizers_follow_the_official_choice():
    d = ds.UniVTACImageTactileDataset(episode_arrays(), cameras=["cam_head"])
    n = d.get_normalizer()
    for key in ("action", "robot_joint", "gripper_width", "tactile_left", "tactile_right", "cam_head"):
        assert key in n.params_dict
    acts = torch.stack([d[i]["action"] for i in range(len(d))])
    na = n["action"].normalize(acts)
    assert na.min() >= -1 - 1e-5 and na.max() <= 1 + 1e-5
    img = n["cam_head"].normalize(torch.ones(1, 3, 2, 2))
    assert torch.allclose(img, torch.ones_like(img))


def test_two_cameras_and_shape_meta():
    arrays = episode_arrays(cams=("cam_head", "cam_wrist"))
    d = ds.UniVTACImageTactileDataset(arrays, cameras=["cam_head", "cam_wrist"])
    meta = d.shape_meta
    assert meta["obs"]["cam_wrist"] == {"shape": [3, 6, 8], "type": "rgb"}
    assert meta["obs"]["robot_joint"]["shape"] == [7] and meta["action"]["shape"] == [8]
    assert sorted(meta["extended_obs"]) == ["tactile_left", "tactile_right"]
    assert d[5]["obs"]["cam_wrist"].shape == (2, 3, 6, 8)
