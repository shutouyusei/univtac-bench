"""Reactive Diffusion Policy (RDP) in UniVTAC's evaluation loop.

Deploy keys: ``rdp_bundle`` (path to ``rdp_policy.pt``; default ``policy/RDP/rdp_ckpt/<task>-<task_config>-
<EP_NUM>/rdp_policy.pt``), ``rdp_device`` (``cuda:0``), ``rdp_jpeg_roundtrip`` (true: camera frames are JPEG
encoded and decoded like the demonstrations before resizing).

Per step: the cameras the bundle was trained with (``cam_head`` -> ``observation['observation']['head']``),
resized to 240 x 320; the measured joints; each fingertip's marker motion ``[reference, current]`` as a
normalised displacement embedded by the bundle's marker PCA. The controller returns 8 absolute joint targets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

RDP_DIR = Path(__file__).resolve().parent
if str(RDP_DIR) not in sys.path:
    sys.path.insert(0, str(RDP_DIR))  # the vendored reactive_diffusion_policy package and rdp_univtac

from .._base_policy import BasePolicy  # noqa: E402
from rdp_univtac.controller import RDPController  # noqa: E402
from rdp_univtac.dataset import TACTILE_KEYS  # noqa: E402
from rdp_univtac.marker import displacement  # noqa: E402
from rdp_univtac.prepare import resize_frame  # noqa: E402


def default_bundle(args: dict) -> Path:
    episodes = args.get("expert_data_num") or os.environ.get("EP_NUM", "50")
    return RDP_DIR / "rdp_ckpt" / f"{args['task_name']}-{args['task_config']}-{episodes}" / "rdp_policy.pt"


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def to_uint8_hwc(frame) -> np.ndarray:
    frame = to_numpy(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame[..., :3])


def jpeg_roundtrip(frame: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def tactile_entry(observation: dict, side: str) -> dict:
    for key in (f"{side}_tactile", f"{side}_gsmini"):
        if key in observation["tactile"]:
            return observation["tactile"][key]
    raise KeyError(f"no {side}_tactile or {side}_gsmini in the tactile observation: {list(observation['tactile'])}")


class Policy(BasePolicy):
    def __init__(self, args):
        super().__init__(args)
        bundle = Path(args["rdp_bundle"]) if args.get("rdp_bundle") else default_bundle(args)
        if not bundle.exists():
            raise FileNotFoundError(f"RDP bundle {bundle} does not exist (train with policy/RDP/baseline.sh)")
        device = args.get("rdp_device", "cuda:0")
        self.jpeg = bool(args.get("rdp_jpeg_roundtrip", True))
        self.controller = RDPController.load(bundle, device=device)
        print(f"RDP policy loaded from {bundle}")

    def _frame(self, frame) -> np.ndarray:
        img = to_uint8_hwc(frame)
        if self.jpeg:
            img = jpeg_roundtrip(img)
        return resize_frame(img)

    def encode(self, observation: dict):
        frames = {c: self._frame(observation["observation"][c.removeprefix("cam_")]["rgb"])
                  for c in self.controller.cameras}
        joint = to_numpy(observation["embodiment"]["joint"]).astype(np.float32)
        tactile = {}
        for key in TACTILE_KEYS:
            m = to_numpy(tactile_entry(observation, key.removeprefix("tactile_"))["marker"]).astype(np.float32)
            tactile[key] = self.controller.embed(displacement(m[None])[0])
        return frames, joint, tactile

    def eval(self, task, observation):
        frames, joint, tactile = self.encode(observation)
        action = self.controller.act(frames, joint, tactile)
        return task.take_action(torch.as_tensor(action, dtype=torch.float32, device=task.device), action_type="qpos")

    def reset(self):
        self.controller.reset()
