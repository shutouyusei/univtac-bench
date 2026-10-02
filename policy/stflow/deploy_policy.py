"""UniVTAC ``BasePolicy`` adapter for a ``stflow`` FlowPolicy checkpoint, run inside the Isaac process.

The model core comes from the ``stflow`` package (repo ``streaming-tactile-flow``,
installed editable in the ``UniVTAC`` env); nothing here imports lerobot. Per step
the adapter builds the same observation the training cache holds:

* camera ``rgb`` and fingertip ``rgb_marker`` frames, JPEG-encoded and decoded
  again the way UniVTAC's HDF5 writer stored the demonstrations, then resized
  with ``stflow.data.univtac_hdf5.resize_rgb``
* ``embodiment/joint[:8]`` as the state

It samples a chunk, executes ``stflow_n_action_steps`` of it as joint targets
(``take_action(..., "qpos")``) and samples again.

deploy yml keys:

* ``stflow_ckpt_dir``          checkpoint directory (``config.yaml`` + ``model.safetensors``);
                               relative paths resolve against the univtac-bench root
* ``stflow_n_action_steps``    actions executed per chunk (default: the whole chunk)
* ``stflow_num_steps``         flow integration steps (default: the checkpoint's)
* ``stflow_device``            default ``cuda:0``
* ``stflow_jpeg_roundtrip``    default true
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from policy._base_policy import BasePolicy  # noqa: E402


def to_uint8_hwc(frame) -> np.ndarray:
    """Simulator frame (torch or numpy, ``(H, W, 3)`` in 0..255) -> contiguous uint8 array."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame[..., :3])


def tactile_entry(observation: dict, side: str) -> dict:
    """Fingertip ``side``'s observation under ``<side>_tactile`` or ``<side>_gsmini``, the names the
    training reader (``stflow.data.univtac_hdf5.stream_key``) also accepts."""
    for key in (f"{side}_tactile", f"{side}_gsmini"):
        if key in observation["tactile"]:
            return observation["tactile"][key]
    raise KeyError(f"no {side}_tactile or {side}_gsmini in the tactile observation: {list(observation['tactile'])}")


def jpeg_roundtrip(frame: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class Policy(BasePolicy):
    def __init__(self, args: dict):
        from stflow import checkpoint

        ckpt = Path(args["stflow_ckpt_dir"]).expanduser()
        if not ckpt.is_absolute():
            ckpt = _REPO_ROOT / ckpt
        self.device = torch.device(args.get("stflow_device", "cuda:0"))
        self.model, cfg = checkpoint.load(ckpt, self.device)
        self.cfg = cfg.model
        self.n_action_steps = int(args.get("stflow_n_action_steps") or self.cfg.chunk_size)
        if not 1 <= self.n_action_steps <= self.cfg.chunk_size:
            raise ValueError(f"stflow_n_action_steps must be in 1..{self.cfg.chunk_size}")
        self.num_steps = args.get("stflow_num_steps") or self.cfg.num_inference_steps
        self.jpeg = bool(args.get("stflow_jpeg_roundtrip", True))
        self.seed = int(args.get("seed", 0))
        self.generator = torch.Generator()
        self.reset()
        print(f"stflow policy from {ckpt}: executing {self.n_action_steps}/{self.cfg.chunk_size}, "
              f"{self.num_steps} flow steps")

    def _frame(self, frame) -> torch.Tensor:
        from stflow.batch import image_to_float
        from stflow.data.univtac_hdf5 import resize_rgb

        img = to_uint8_hwc(frame)
        if self.jpeg:
            img = jpeg_roundtrip(img)
        img = resize_rgb(img, self.cfg.image_size)
        return image_to_float(torch.from_numpy(img)[None].to(self.device))

    def encode_obs(self, observation: dict) -> dict:
        images = {c: self._frame(observation["observation"][c]["rgb"]) for c in self.cfg.camera_names}
        tactile = {t: self._frame(tactile_entry(observation, t)["rgb_marker"]) for t in self.cfg.tactile_names}
        joint = observation["embodiment"]["joint"]
        if isinstance(joint, torch.Tensor):
            joint = joint.detach().cpu().numpy()
        state = torch.as_tensor(np.asarray(joint, dtype=np.float32)[: self.cfg.state_dim])[None].to(self.device)
        return {"images": images, "tactile": tactile, "state": state}

    def eval(self, task, observation):
        if not self.queue:
            chunk = self.model.sample(self.encode_obs(observation), generator=self.generator, num_steps=self.num_steps)
            self.queue = list(chunk[0, : self.n_action_steps].cpu().numpy())
        action = torch.from_numpy(self.queue.pop(0)).to(task.device).float()
        return task.take_action(action, action_type="qpos")

    def reset(self):
        """Called before every episode: drop the unexecuted actions and restart the noise stream, so an
        episode's sampled chunks depend on its own observations only, not on how many episodes ran before."""
        self.queue: list[np.ndarray] = []
        self.generator.manual_seed(self.seed)

    def close(self):
        pass
