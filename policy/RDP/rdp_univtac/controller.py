"""RDP's control loop for a synchronous simulator, with the index logic of the official ``real_runner.py``.

Official (24 fps, asynchronous): the slow policy samples a latent action from the two newest observations two
frames apart; its 29 identical rows are tagged with the base TCP pose and ``ext_step = 4 + k`` and pushed every
16 control steps after dropping ``latency_step`` rows; every control step the fast thread re-runs the tokenizer's
GRU decoder from scratch over the newest ``ext`` tactile frames (padding by repeating the oldest frame) and
executes its last output, converted from relative to absolute with the stored base pose.

Here the simulator waits for the policy, so ``latency_step`` is 0 (see rdp_config.yml): at control step
``now`` inside a chunk that started at ``chunk_start``, the decoder sees the tactile of steps
``chunk_start - 3 .. now`` (``ext = 4 + i``), i.e. "now" sits at window position 3 + i as in training, and its
last output is the action for that position. The arm part is relative to the arm joints observed at the chunk
start; the gripper is absolute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer

from . import build
from .dataset import ARM, TACTILE_KEYS
from .marker import MarkerPCA


def obs_indices(now: int, n_obs_steps: int = 2, ratio: int = 2) -> list[int]:
    """History indices of the slow policy's observation frames, oldest first, padded with frame 0."""
    return [max(0, now - ratio * (n_obs_steps - 1 - j)) for j in range(n_obs_steps)]


def tactile_indices(chunk_start: int, now: int, before: int = 3) -> list[int]:
    """History indices of the decoder's tactile window: ``chunk_start - before .. now``, padded with frame 0."""
    return [max(0, s) for s in range(chunk_start - before, now + 1)]


class RDPController:
    def __init__(self, policy, cfg: dict, cameras: list[str], pca: MarkerPCA, device: str):
        self.policy = policy
        self.cfg = cfg
        self.cameras = list(cameras)
        self.pca = pca
        self.device = device
        self.interval = cfg["deploy"]["chunk_interval"]
        self.latency = cfg["deploy"]["latency_step"]
        if self.latency != 0:
            raise NotImplementedError("latency_step > 0 is for asynchronous real robots")
        self.ratio = cfg["dataset_obs_temporal_downsample_ratio"]
        self.n_obs_steps = cfg["n_obs_steps"]
        self.before = self.n_obs_steps * self.ratio - 1  # 3: "now" is window position 3
        self.reset()

    @classmethod
    def load(cls, bundle_path: Path, device: str = "cuda:0") -> RDPController:
        b = torch.load(bundle_path, map_location="cpu", weights_only=False)
        cfg = b["config"]
        at = build.build_at(cfg, b["shape_meta"], device)
        at.load_state_dict(b["at"])
        at.eval()
        policy = build.build_policy(cfg, b["shape_meta"], at)
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(b["normalizer"])
        policy.set_normalizer(normalizer)
        policy.load_state_dict(b["policy"])
        policy.to(device)
        policy.eval()
        at.normalizer.to(device)
        return cls(policy, cfg, b["cameras"], MarkerPCA(b["pca_mean"], b["pca_components"]), device)

    def reset(self) -> None:
        self.step = 0
        self.history = {k: [] for k in [*self.cameras, "joint", *TACTILE_KEYS]}
        self.latent = None
        self.base = None
        self.chunk_start = 0

    def _stack(self, key: str, idx: list[int]) -> np.ndarray:
        return np.stack([self.history[key][i] for i in idx])

    @torch.no_grad()
    def _new_latent(self) -> None:
        idx = obs_indices(self.step, self.n_obs_steps, self.ratio)
        joint = self._stack("joint", idx)
        obs = {c: torch.from_numpy(np.moveaxis(self._stack(c, idx), -1, 1).astype(np.float32) / 255.0)
               for c in self.cameras}
        obs["robot_joint"] = torch.from_numpy(joint[:, :ARM] - joint[-1, :ARM])
        obs["gripper_width"] = torch.from_numpy(joint[:, ARM:ARM + 1])
        for k in TACTILE_KEYS:
            obs[k] = torch.from_numpy(self._stack(k, idx))
        obs = {k: v[None].to(self.device) for k, v in obs.items()}
        out = self.policy.predict_action(obs, dataset_obs_temporal_downsample_ratio=self.ratio,
                                         return_latent_action=True)
        self.latent = out["action"][:, 0]  # the 29 rows are identical copies of the decoded latent
        self.base = joint[-1, :ARM].copy()
        self.chunk_start = self.step

    @torch.no_grad()
    def act(self, frames: dict[str, np.ndarray], joint: np.ndarray, tactile: dict[str, np.ndarray]) -> np.ndarray:
        """``frames`` per camera (240, 320, 3) uint8, ``joint`` the measured (>= 8,) joints, ``tactile`` the 15-D
        embedding per fingertip; returns the 8 absolute joint targets for this step."""
        for c in self.cameras:
            self.history[c].append(frames[c])
        self.history["joint"].append(np.asarray(joint, dtype=np.float32)[:ARM + 1])
        for k in TACTILE_KEYS:
            self.history[k].append(np.asarray(tactile[k], dtype=np.float32))
        if self.latent is None or self.step - self.chunk_start >= self.interval:
            self._new_latent()
        idx = tactile_indices(self.chunk_start, self.step, self.before)
        ext = {k: torch.from_numpy(self._stack(k, idx))[None].to(self.device) for k in TACTILE_KEYS}
        out = self.policy.predict_from_latent_action(self.latent, ext, len(idx), self.ratio)
        rel = out["action_pred"][0, -1].float().cpu().numpy()
        action = rel.copy()
        action[:ARM] = rel[:ARM] + self.base
        self.step += 1
        return action

    def embed(self, displacement: np.ndarray) -> np.ndarray:
        return self.pca.embed(displacement[None])[0]
