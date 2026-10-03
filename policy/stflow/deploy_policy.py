"""UniVTAC ``BasePolicy`` adapter for a ``stflow`` FlowPolicy checkpoint, run inside the Isaac process.

The model core comes from the ``stflow`` package (repo ``streaming-tactile-flow``,
installed editable in the ``UniVTAC`` env); nothing here imports lerobot. Per step
the adapter builds the same observation the training cache holds:

* camera ``rgb`` and fingertip ``rgb_marker`` frames, JPEG-encoded and decoded
  again the way UniVTAC's HDF5 writer stored the demonstrations, then resized
  with ``stflow.data.univtac_hdf5.resize_rgb``
* ``embodiment/joint[:8]`` as the state
* for a checkpoint whose model measures a ``marker`` tactile target (``model.tactile_target``, with
  fingertips), each fingertip's raw ``marker`` observation (TacEx marker motion ``(2, M, 2)``:
  reference and current positions) as ``tactile_marker``; the model matches and differences it itself.
  The task config must record ``marker`` in ``observation_settings.tactile`` (clean51 does).

How actions are produced from those observations belongs to the checkpoint's method
(``stflow.methods``): every control step the adapter hands the observation to the method's
controller and executes the one action it returns as a joint target (``take_action(..., "qpos")``);
``reset()`` before an episode resets the controller. A new method therefore needs no change here.

deploy yml keys:

* ``stflow_ckpt_dir``          checkpoint directory (``config.yaml`` + ``model.safetensors``);
                               relative paths resolve against the univtac-bench root
* ``stflow_method_args``       overrides of the checkpoint's method arguments for this evaluation,
                               e.g. ``{n_action_steps: 25, num_steps: 5}`` for the chunk method
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


REMOVED_KEYS = {
    "stflow_n_action_steps": "stflow_method_args: {n_action_steps: N}",
    "stflow_num_steps": "stflow_method_args: {num_steps: S}",
}


class Policy(BasePolicy):
    def __init__(self, args: dict):
        from stflow import checkpoint
        from stflow.methods import build_method

        for key, replacement in REMOVED_KEYS.items():
            if key in args:
                raise ValueError(f"deploy key {key} was replaced by {replacement}")
        ckpt = Path(args["stflow_ckpt_dir"]).expanduser()
        if not ckpt.is_absolute():
            ckpt = _REPO_ROOT / ckpt
        self.device = torch.device(args.get("stflow_device", "cuda:0"))
        self.model, cfg = checkpoint.load(ckpt, self.device)
        self.cfg = cfg.model
        method_args = {**cfg.method.args, **(args.get("stflow_method_args") or {})}
        self.controller = build_method(cfg.method.name, method_args).controller(self.model, seed=int(args.get("seed", 0)))
        self.jpeg = bool(args.get("stflow_jpeg_roundtrip", True))
        # Checkpoints from before tactile targets have no such field and never read the marker observation.
        self.markers = getattr(self.cfg, "tactile_target", "none") == "marker" and bool(self.cfg.tactile_names)
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
        images = {c: self._frame(observation["observation"][c]["rgb"]) for c in self.cfg.camera_names}
        tactile = {t: self._frame(tactile_entry(observation, t)["rgb_marker"]) for t in self.cfg.tactile_names}
        joint = observation["embodiment"]["joint"]
        if isinstance(joint, torch.Tensor):
            joint = joint.detach().cpu().numpy()
        state = torch.as_tensor(np.asarray(joint, dtype=np.float32)[: self.cfg.state_dim])[None].to(self.device)
        obs = {"images": images, "tactile": tactile, "state": state}
        if self.markers:
            obs["tactile_marker"] = {t: self._marker(tactile_entry(observation, t), t) for t in self.cfg.tactile_names}
        return obs

    def _marker(self, entry: dict, side: str) -> torch.Tensor:
        if "marker" not in entry:
            raise KeyError(
                f"the checkpoint measures a marker tactile target but the {side} fingertip observation has no "
                "'marker'; add 'marker' to observation_settings.tactile in the task config"
            )
        marker = entry["marker"]
        if isinstance(marker, torch.Tensor):
            marker = marker.detach().cpu().numpy()
        return torch.as_tensor(np.asarray(marker, dtype=np.float32))[None].to(self.device)

    def eval(self, task, observation):
        # Controllers may call the policy's encoder and expert directly; no graph is ever needed here.
        with torch.inference_mode():
            action = self.controller.act(self.encode_obs(observation)).to(task.device).float()
        return task.take_action(action, action_type="qpos")

    def reset(self):
        """Called before every episode; the controller drops what it carries and restarts its noise
        stream, so an episode depends on its own observations only."""
        self.controller.reset()

    def close(self):
        pass
