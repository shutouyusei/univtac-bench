"""UniVTAC policy running a stflow checkpoint (``stflow`` on PYTHONPATH); the checkpoint's method
controller gives one joint target per control step.

deploy yml: stflow_ckpt_dir (absolute), stflow_method_args ({}), stflow_device (cuda:0),
stflow_jpeg_roundtrip (true).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .._base_policy import BasePolicy


def to_uint8_hwc(frame) -> np.ndarray:
    """Simulator frame (torch or numpy, (H, W, 3+) in 0..255) -> contiguous uint8 (H, W, 3)."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame[..., :3])


def jpeg_roundtrip(frame: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class Policy(BasePolicy):
    def __init__(self, args: dict):
        from stflow import checkpoint
        from stflow.methods import build_method

        ckpt = Path(args["stflow_ckpt_dir"]).expanduser()
        self.device = torch.device(args.get("stflow_device", "cuda:0"))
        self.model, cfg = checkpoint.load(ckpt, self.device)
        self.cfg = cfg.model
        method_args = {**cfg.method.args, **(args.get("stflow_method_args") or {})}
        self.controller = build_method(cfg.method.name, method_args).controller(self.model, seed=int(args.get("seed", 0)))
        self.jpeg = bool(args.get("stflow_jpeg_roundtrip", True))
        print(f"stflow policy from {ckpt}: method {cfg.method.name} {method_args}")

    def _frame(self, frame) -> torch.Tensor:
        from stflow.batch import image_to_float
        from stflow.data.univtac_hdf5 import resize_rgb

        img = to_uint8_hwc(frame)
        if self.jpeg:
            img = jpeg_roundtrip(img)
        img = resize_rgb(img, self.cfg.image_size)
        return image_to_float(torch.from_numpy(img)[None].to(self.device))

    def encode_obs(self, observation: dict) -> dict:
        """The batch the training cache holds: camera and fingertip frames, joint[:state_dim]."""
        images = {c: self._frame(observation["observation"][c]["rgb"]) for c in self.cfg.camera_names}
        tactile = {t: self._frame(observation["tactile"][f"{t}_tactile"]["rgb_marker"]) for t in self.cfg.tactile_names}
        joint = torch.as_tensor(observation["embodiment"]["joint"], dtype=torch.float32, device=self.device)
        return {"images": images, "tactile": tactile, "state": joint[: self.cfg.state_dim][None]}

    def eval(self, task, observation):
        with torch.inference_mode():
            action = self.controller.act(self.encode_obs(observation))
        return task.take_action(action.to(task.device).float(), action_type="qpos")

    def reset(self):
        """Before every episode: the controller forgets its chunk and restarts its noise stream."""
        self.controller.reset()
