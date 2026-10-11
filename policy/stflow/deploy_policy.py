"""UniVTAC ``BasePolicy`` adapter for a ``stflow`` FlowPolicy checkpoint, run inside the Isaac process.

The model core comes from the ``stflow`` package (repo ``streaming-tactile-flow``,
installed editable in the ``UniVTAC`` env); nothing here imports lerobot. Per step
the adapter builds the same observation the training cache holds:

* camera ``rgb`` and fingertip ``rgb_marker`` frames, JPEG-encoded and decoded
  again the way UniVTAC's HDF5 writer stored the demonstrations, then resized
  with ``stflow.data.univtac_hdf5.resize_rgb``
* ``embodiment/joint[:8]`` as the state

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
* ``stflow_record``            default off; true or a mapping writes a per-step record of every episode to
                               ``<result folder>/record/<seed>.npz`` (``policy.stflow.record``) without changing
                               the actions. Mapping keys: ``tactile_tokens`` (default true: frozen ResNet18
                               tokens of the fingertip frames), ``frames_every`` (default 10: raw fingertip
                               frames as JPEG every k steps, 0 = none), ``grid`` (token grid, default the
                               checkpoint's ``tactile_target_grid`` or 3), ``pretrained_targets`` (default true)
"""

from __future__ import annotations

import sys
import time
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
        self.recorder = self._recorder(args.get("stflow_record"))
        print(f"stflow policy from {ckpt}: method {cfg.method.name} {method_args}")

    def _recorder(self, spec):
        """``stflow_record``: None / false = no record, true = defaults, a mapping overrides them."""
        if not spec:
            return None
        from policy.stflow.record import EpisodeRecorder

        if spec is True:
            spec = {}
        elif not isinstance(spec, dict):
            raise ValueError(f"stflow_record must be true, false or a mapping, got {spec!r}")
        unknown = set(spec) - {"tactile_tokens", "frames_every", "grid", "pretrained_targets"}
        if unknown:
            raise ValueError(f"unknown stflow_record keys {sorted(unknown)}")
        encoder = None
        if spec.get("tactile_tokens", True) and self.cfg.tactile_names:
            from stflow.data.tactile_targets import TargetEncoder

            grid = int(spec.get("grid") or getattr(self.cfg, "tactile_target_grid", 0) or 3)
            encoder = TargetEncoder(grid, pretrained=bool(spec.get("pretrained_targets", True))).to(self.device)
        return EpisodeRecorder(self.cfg.tactile_names, encoder, int(spec.get("frames_every", 10)))

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
        # Controllers may call the policy's encoder and expert directly; no graph is ever needed here.
        with torch.inference_mode():
            obs = self.encode_obs(observation)
            start = self._clock()
            action = self.controller.act(obs)
            infer_ms = (self._clock() - start) * 1e3
            if self.recorder is not None:
                self._record(task, lambda: self.recorder.step(
                    task, obs, observation["embodiment"]["joint"],
                    {t: tactile_entry(observation, t)["rgb_marker"] for t in self.cfg.tactile_names},
                    action, self.controller, infer_ms,
                ))
            action = action.to(task.device).float()
        result = task.take_action(action, action_type="qpos")
        if self.recorder is not None:
            self._record(task, lambda: self.recorder.after_action(task))
            # Write the episode as soon as it is over (success or step limit), so a crash in the simulator's
            # teardown cannot lose it; an early stop is only known to the loop and is written at the next reset.
            if getattr(task, "eval_success", False) or task.take_action_cnt >= task.cfg.step_lim:
                self._record(task, self.recorder.flush)
        return result

    def _record(self, task, call) -> None:
        """Run a recorder call; a failure drops the episode's record but never stops the evaluation."""
        try:
            call()
        except Exception as error:  # recording must not change which seeds get evaluated
            self.recorder.fail(task, error)

    def _clock(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def reset(self):
        """Called before every episode; the controller drops what it carries and restarts its noise
        stream, so an episode depends on its own observations only. A previous episode's record not yet
        written (an early stop, which only the evaluation loop sees) is written here."""
        self.controller.reset()
        if self.recorder is not None:
            self._record(None, self.recorder.flush)

    def close(self):
        if self.recorder is not None:
            self._record(None, self.recorder.flush)
