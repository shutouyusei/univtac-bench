"""RDP's training samples built from UniVTAC demonstrations, with the semantics of the official
``RealImageTactileDataset`` (``dataset/real_image_tactile_dataset.py``) and its ``get_normalizer``:

* windows of ``horizon`` = 32 steps from the official ``SequenceSampler`` with pad_before 3 / pad_after 28 (edge
  padding by repeating the first / last frame), one window per step of every episode;
* observation keys keep only the first ``n_obs_steps * ratio`` = 4 frames and use ``[:4][::-2][::-1]``, i.e.
  window indices 1 and 3 (``t-2`` and ``t``); the tactile embeddings are also "extended" observations, given to
  the tokenizer's decoder over the whole window at the same index as the action;
* ``action[t] = state[t+1]`` with the episode's last frame repeated (``post_process_data.py``);
* relative actions: the arm part of the action and of the arm observation is expressed relative to the arm
  state at the last observed frame, the gripper stays absolute.

UniVTAC's robot is commanded in joint space (``take_action(..., 'qpos')``), so RDP's TCP pose is replaced by
the 7 arm joints (``robot_joint``) and its gripper width by the first finger joint (``gripper_width``); the
action is these 8 numbers. For 3-D TCP positions the official relative transform is a subtraction, which is
what is applied to the joints here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from reactive_diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_action_normalizer,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from reactive_diffusion_policy.common.sampler import SequenceSampler
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

HORIZON = 32
N_OBS_STEPS = 2
OBS_RATIO = 2
DATASET_OBS_STEPS = N_OBS_STEPS * OBS_RATIO  # 4
PAD_BEFORE = DATASET_OBS_STEPS - 1  # 3
PAD_AFTER = HORIZON - DATASET_OBS_STEPS  # 28
ARM = 7
TACTILE_KEYS = ("tactile_left", "tactile_right")


@dataclass
class ArrayBuffer:
    """What ``SequenceSampler`` needs from the official zarr ReplayBuffer: concatenated per-step arrays and the
    end index of every episode."""

    data: dict[str, np.ndarray]
    episode_ends: np.ndarray
    meta: dict = field(default_factory=dict)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]


def concat_episodes(joint: list[np.ndarray], tactile: dict[str, list[np.ndarray]],
                    images: dict[str, list[np.ndarray]]) -> ArrayBuffer:
    """Per-episode arrays -> one buffer. ``joint`` (T, >= 8): 7 arm joints then the finger; ``tactile`` 15-D
    embeddings per step; ``images`` (T, H, W, 3) uint8 per camera."""
    lengths = [len(j) for j in joint]
    data = {"joint": np.concatenate([j[:, :ARM + 1] for j in joint]).astype(np.float32)}
    data["action"] = np.concatenate([np.concatenate([j[1:, :ARM + 1], j[-1:, :ARM + 1]]) for j in joint]).astype(np.float32)
    data["robot_joint"] = data["joint"][:, :ARM]
    data["gripper_width"] = data["joint"][:, ARM:ARM + 1]
    for key, eps in tactile.items():
        assert [len(e) for e in eps] == lengths, key
        data[key] = np.concatenate(eps).astype(np.float32)
    for key, eps in images.items():
        assert [len(e) for e in eps] == lengths, key
        data[key] = np.concatenate(eps)
    return ArrayBuffer(data=data, episode_ends=np.cumsum(lengths).astype(np.int64))


def shape_meta(cameras: list[str], image_hw: tuple[int, int], tactile_dim: int = 15) -> dict:
    obs = {c: {"shape": [3, *image_hw], "type": "rgb"} for c in cameras}
    obs["robot_joint"] = {"shape": [ARM], "type": "low_dim"}
    obs["gripper_width"] = {"shape": [1], "type": "low_dim"}
    for k in TACTILE_KEYS:
        obs[k] = {"shape": [tactile_dim], "type": "low_dim"}
    return {
        "obs": obs,
        "extended_obs": {k: {"shape": [tactile_dim], "type": "low_dim"} for k in TACTILE_KEYS},
        "action": {"shape": [ARM + 1]},
    }


def to_relative(action: np.ndarray, robot_joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arm part of the action and of the observed arm joints relative to the last observed arm joints."""
    base = robot_joint[-1]
    action = action.copy()
    action[:, :ARM] -= base
    return action, robot_joint - base


class UniVTACImageTactileDataset(torch.utils.data.Dataset):
    def __init__(self, buffer: ArrayBuffer, cameras: list[str], horizon: int = HORIZON,
                 n_obs_steps: int = N_OBS_STEPS, obs_ratio: int = OBS_RATIO, images: bool = True):
        """``images=False`` leaves the camera frames out of the samples (the tokenizer never reads them)."""
        self.buffer = buffer
        self.all_cameras = list(cameras)
        self.cameras = list(cameras) if images else []
        self.horizon = horizon
        self.obs_ratio = obs_ratio
        self.dataset_obs_steps = n_obs_steps * obs_ratio
        self.lowdim_keys = ["robot_joint", "gripper_width", *TACTILE_KEYS]
        key_first_k = {k: self.dataset_obs_steps for k in self.cameras + ["robot_joint", "gripper_width"]}
        self.sampler = SequenceSampler(
            replay_buffer=buffer, sequence_length=horizon, pad_before=self.dataset_obs_steps - 1,
            pad_after=horizon - self.dataset_obs_steps, keys=self.cameras + self.lowdim_keys + ["action"],
            key_first_k=key_first_k)
        h, w = buffer[self.all_cameras[0]].shape[1:3]
        self.shape_meta = shape_meta(self.all_cameras, (h, w), buffer[TACTILE_KEYS[0]].shape[1])

    def __len__(self) -> int:
        return len(self.sampler)

    def _obs_frames(self, x: np.ndarray) -> np.ndarray:
        return x[:self.dataset_obs_steps][::-self.obs_ratio][::-1]

    def __getitem__(self, idx: int) -> dict:
        data = self.sampler.sample_sequence(idx)
        obs = {c: np.moveaxis(self._obs_frames(data[c]), -1, 1).astype(np.float32) / 255.0 for c in self.cameras}
        for key in self.lowdim_keys:
            obs[key] = self._obs_frames(data[key]).astype(np.float32)
        extended = {k: data[k].astype(np.float32) for k in TACTILE_KEYS}
        action, obs["robot_joint"] = to_relative(data["action"].astype(np.float32), obs["robot_joint"])
        return {
            "obs": {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in obs.items()},
            "action": torch.from_numpy(action),
            "extended_obs": {k: torch.from_numpy(v) for k, v in extended.items()},
        }

    def relative_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Every window's relative action (N, H, 8) and relative observed arm joints (N, 2, 7), without touching
        the images (the official normalizer iterates the whole dataset for these)."""
        acts, joints = [], []
        for i in range(len(self.sampler)):
            b0, b1, s0, s1 = self.sampler.indices[i]
            seq = {}
            for key in ("action", "robot_joint"):
                arr = self.buffer[key][b0:b1]
                full = np.empty((self.horizon,) + arr.shape[1:], arr.dtype)
                full[:s0] = arr[0]
                full[s1:] = arr[-1]
                full[s0:s1] = arr
                seq[key] = full
            a, j = to_relative(seq["action"], self._obs_frames(seq["robot_joint"]))
            acts.append(a)
            joints.append(j)
        return np.stack(acts), np.stack(joints)

    def get_normalizer(self) -> LinearNormalizer:
        n = LinearNormalizer()
        acts, joints = self.relative_arrays()
        n["action"] = get_action_normalizer(acts)
        # official: relative TCP observations use the action helper (per-dim [-1, 1] range); for 7 joints the
        # helper has no case, so its range normalizer is applied directly
        n["robot_joint"] = get_range_normalizer_from_stat(array_to_stats(joints.reshape(-1, ARM)))
        for key in ["gripper_width", *TACTILE_KEYS]:
            n[key] = SingleFieldLinearNormalizer.create_fit(self.buffer[key])
        for c in self.all_cameras:
            n[c] = get_image_range_normalizer()
        return n
