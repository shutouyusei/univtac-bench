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
* ``stflow_disturb``           a known offset added to the executed joint targets (``policy.stflow.disturb``)
* ``stflow_oracle``            ``{gain, max_tilt_deg, beta}``: the privileged insertion-phase correction
                               (``policy.stflow.oracle``) recorded as the label of each row; ``beta`` > 0 also
                               executes it (mixed with the head's as ``(1 - beta) head + beta oracle``)
* ``stflow_disturb_log``       jsonl file receiving one line per step of a disturbed evaluation: the offset,
                               the commanded and measured gripper, and each fingertip's mean absolute pixel
                               change against the previous step and against the step before the offset began
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
        self.disturb_cfg = args.get("stflow_disturb")
        self.disturb_log = Path(args["stflow_disturb_log"]) if args.get("stflow_disturb_log") else None
        self.head = None
        if args.get("stflow_correction"):
            from stflow.correction import load as load_head

            self.head = load_head(args["stflow_correction"], self.device)
        self.head_noise = bool(args.get("stflow_correction_tactile_noise", False))
        self.record_dir = Path(args["stflow_correction_record"]) if args.get("stflow_correction_record") else None
        self.oracle_cfg = args.get("stflow_oracle")
        self.step = 0
        self.disturbance = None
        self.tactile_prev: dict = {}
        self.tactile_ref: dict = {}
        self.previous = None
        self.rows: list = []
        self.seed = None
        print(f"stflow policy from {ckpt}: method {cfg.method.name} {method_args}"
              + (f", disturbance {self.disturb_cfg}" if self.disturb_cfg else "")
              + (f", correction {args['stflow_correction']}" if self.head is not None else "")
              + (" with noise tactile" if self.head_noise else "")
              + (f", recording to {self.record_dir}" if self.record_dir else "")
              + (f", oracle {self.oracle_cfg}" if self.oracle_cfg else ""))

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
            obs = self.encode_obs(observation)
            plan = self.controller.act(obs).to(task.device).float()
            correcting = self.head is not None or self.record_dir is not None or bool(self.oracle_cfg)
            if correcting:
                plan_n, tactile, previous_n, correction = self._correction(obs, plan)
        oracle, tilt = None, float("nan")
        if self.oracle_cfg:
            oracle, tilt = self._oracle(task, plan)
            beta = float(self.oracle_cfg.get("beta", 0.0))
            if beta > 0:
                correction = (1.0 - beta) * correction + beta * oracle
        action = plan + correction if correcting else plan
        offset = np.zeros(plan.shape[-1], dtype=np.float32)
        if self.disturb_cfg:
            action, offset = self._disturb(task, observation, action)
        if self.record_dir is not None:
            self._record(task, plan_n, tactile, previous_n, offset, oracle, tilt)
        if correcting:
            self.previous = correction
        self.step += 1
        return task.take_action(action, action_type="qpos")

    def _oracle(self, task, plan):
        """The privileged insertion-phase correction (``policy.stflow.oracle``) as a total correction in action
        units: the correction executed one step before plus ``gain`` times the joint offset that puts the
        prism on the target axis; zero before the alignment rotation (tilt above ``max_tilt_deg``)."""
        from policy.stflow.oracle import correction as oracle_correction

        rm = task._robot_manager
        gain = float(self.oracle_cfg.get("gain", 0.5))
        max_tilt = float(self.oracle_cfg.get("max_tilt_deg", 10.0))
        target = task.target_pose.to_transformation_matrix()
        prism = task.prism.get_pose("matrix")
        p_ee = rm.robot.data.body_link_pos_w[0, rm._body_idx].detach().cpu().numpy()
        jac = rm.robot.root_physx_view.get_jacobians()[0, rm._jacobi_body_idx][:, rm._arm_ids]
        dq, tilt = oracle_correction(target, prism, p_ee, jac.detach().cpu().numpy(), max_tilt)
        total = torch.zeros_like(plan)
        if tilt <= max_tilt:
            previous = self.previous if self.previous is not None else torch.zeros_like(plan)
            total[:7] = previous[:7] + gain * torch.as_tensor(dq, dtype=plan.dtype, device=plan.device)
        return total, tilt

    def _correction(self, obs, plan):
        """The head's correction for this step (zero without a head) and the inputs it was computed from,
        in correction units; the previous correction carries across chunk boundaries until ``reset``."""
        from stflow.correction import action_scale, tactile_features

        scale = action_scale(self.model).to(plan)
        plan_n = ((plan - self.model.normalizer.action_mean.to(plan)) / scale)[None]
        tactile = tactile_features(self.model, obs["tactile"]).float()
        if self.head_noise:
            gen = torch.Generator().manual_seed(int(self.step) + 7919 * int(self.seed or 0))
            tactile = torch.randn(tactile.shape, generator=gen).to(tactile)
        previous = self.previous if self.previous is not None else torch.zeros_like(plan)
        previous_n = (previous / scale)[None]
        if self.head is None:
            return plan_n, tactile, previous_n, torch.zeros_like(plan)
        correction = self.head(plan_n, tactile, previous_n)[0] * scale
        return plan_n, tactile, previous_n, correction

    def _record(self, task, plan_n, tactile, previous_n, offset, oracle=None, tilt=float("nan")):
        from stflow.correction import action_scale

        self.seed = int(task.cfg.seed)
        scale = action_scale(self.model).cpu().numpy()
        oracle_n = np.full(len(scale), np.nan, np.float32) if oracle is None else oracle.cpu().numpy() / scale
        self.rows.append((plan_n[0].cpu().numpy(), tactile[0].cpu().numpy().astype(np.float16),
                          previous_n[0].cpu().numpy(), offset / scale, oracle_n, np.float32(tilt)))
        if len(self.rows) % 50 == 0:
            self._flush()

    def _flush(self):
        """Write the episode's rows so far to ``<record dir>/<seed>.npz`` (overwritten as it grows): plan,
        tactile, previous correction, offset (``delta``), the oracle's total correction (NaN without
        ``stflow_oracle``) and the prism tilt it saw, all corrections in action-std units."""
        if self.record_dir is None or not self.rows or self.seed is None:
            return
        self.record_dir.mkdir(parents=True, exist_ok=True)
        plan, tactile, previous, delta, oracle, tilt = (np.stack(c) for c in zip(*self.rows))
        np.savez(self.record_dir / f"{self.seed}.npz", plan=plan.astype(np.float32), tactile=tactile,
                 previous=previous.astype(np.float32), delta=delta.astype(np.float32),
                 oracle=oracle.astype(np.float32), tilt=tilt.astype(np.float32))

    def _disturb(self, task, observation, action):
        from policy.stflow.disturb import draw

        seed = int(task.cfg.seed)
        self.seed = seed
        if self.step == 0:
            self.disturbance = draw(self.disturb_cfg, seed)
        offset = self.disturbance.at(self.step)
        action = action + torch.from_numpy(offset).to(action)
        if self.disturb_log is not None:
            self._log_step(seed, observation, action, offset)
        return action, offset

    def _log_step(self, seed, observation, action, offset):
        import json

        line = {"seed": seed, "step": self.step, "start": self.disturbance.start, "offset": float(offset[-1]),
                "cmd_gripper": float(action[-1])}
        line["gripper"] = float(observation["embodiment"]["joint"][7])
        for side in self.cfg.tactile_names:
            frame = to_uint8_hwc(tactile_entry(observation, side)["rgb_marker"]).astype(np.int16)
            if side in self.tactile_prev:
                line[f"{side}_change"] = float(np.abs(frame - self.tactile_prev[side]).mean())
            if self.disturbance.start is not None and self.step == self.disturbance.start - 1:
                self.tactile_ref[side] = frame
            if side in self.tactile_ref:
                line[f"{side}_from_ref"] = float(np.abs(frame - self.tactile_ref[side]).mean())
            self.tactile_prev[side] = frame
        self.disturb_log.parent.mkdir(parents=True, exist_ok=True)
        with self.disturb_log.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def reset(self):
        """Called before every episode; the controller drops what it carries and restarts its noise
        stream, so an episode depends on its own observations only."""
        self._flush()
        self.controller.reset()
        self.step = 0
        self.disturbance = None
        self.tactile_prev = {}
        self.tactile_ref = {}
        self.previous = None
        self.rows = []
        self.seed = None

    def close(self):
        self._flush()
