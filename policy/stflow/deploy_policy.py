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
* ``stflow_trace``             default false; true records one row per control step under ``trace`` in the
                               episode's ``metadata.json`` entry (see ``trace_row``): the commanded and
                               measured joints and, for tasks with a held prism and a target pose, the
                               prism's pose error in the target frame and its slip in the hand, read from
                               the simulator state the policy does not see
* ``stflow_prediction_log``    default off; ``{dir: <path>, frames_every: N}`` for a checkpoint whose model has
                               ``tactile_prediction``: one ``<dir>/<seed>.npz`` per episode (see
                               ``PredictionLog``) with, every step, the slow side's tactile read-out for the
                               query just executed and the pooled reading of the fingertips, plus the
                               policy's fingertip input frames every ``N`` steps (default 5)
* ``stflow_rl``                default off; ``{dir, mode: train|eval, config: {SACConfig fields}, seed, load,
                               wandb: {project, name}}``: SAC on the fast side of a slow-fast SFP checkpoint,
                               rewarded by agreement with the slow side's tactile forecast (``policy/stflow/sac.py``,
                               ``RLSession``); learns between episodes and resumes from ``dir``
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

TRACE_COLUMNS = (
    ["action_count", "prism_x", "prism_y", "prism_z", "prism_axis_dot", "inhand_slip"]
    + [f"command_{i}" for i in range(8)]
    + [f"joint_{i}" for i in range(8)]
)


def trace_row(task, joint: np.ndarray, action: np.ndarray) -> list[float]:
    """One ``TRACE_COLUMNS`` row for the step about to be executed. ``prism_x/y/z`` is the held prism's
    position in the task's target frame (success needs |x|, |y| < 0.01 and z < -0.04),
    ``prism_axis_dot`` the cosine between the prism's axis and the target's (success > 0.99) and
    ``inhand_slip`` the prism's displacement along the gripper axis since the grasp (early stop and
    failure at 0.04). They are NaN for a task without ``prism`` / ``target_pose`` / ``origin_inhand_pose``."""
    privileged = [float("nan")] * 5
    if all(hasattr(task, name) for name in ("prism", "target_pose", "origin_inhand_pose")):
        prism = task.prism.get_pose()
        in_target = prism.rebase(task.target_pose)
        in_hand = prism.rebase(task._robot_manager.get_gripper_center_pose())
        privileged = [
            *(float(v) for v in in_target[:3]),
            float(in_target.to_transformation_matrix()[2, 2]),
            float(abs(task.origin_inhand_pose[2] - in_hand[2])),
        ]
    row = [float(task.take_action_cnt), *privileged, *(float(v) for v in action[:8]), *(float(v) for v in joint[:8])]
    return [round(v, 5) for v in row]


class PredictionLog:
    """What the slow side of a ``tactile_prediction`` model expects the fingertips to read, next to what they
    read, step by step.

    A forward pre-hook keeps the arguments of the controller's last expert call. After each step the same
    call is repeated without tactile (the slow side, as in training) and the read-out of the executed
    query's action token is ``predicted``: training teaches query ``i`` the reading of the first frame of
    its block, from the chunk start's observation. ``reading`` is the current fingertips' tokens, mean-pooled
    per fingertip and layer-normed: the space the read-out was trained in. ``query`` is the executed query
    (the step inside the chunk); ``action_count`` the task's action counter. A file is written when the next
    episode starts and when the policy closes."""

    def __init__(self, model, directory: Path, frames_every: int = 5):
        if getattr(model, "tactile_predictor", None) is None:
            raise ValueError("stflow_prediction_log needs a checkpoint whose model has tactile_prediction")
        self.model = model
        self.dir = Path(directory)
        self.frames_every = int(frames_every)
        self.last_call = None
        model.expert.register_forward_pre_hook(self._keep, with_kwargs=True)
        self.clear()

    def _keep(self, module, args, kwargs):
        self.last_call = (args, kwargs)

    def clear(self) -> None:
        self.seed = None
        self.rows = {"query": [], "action_count": [], "predicted": [], "reading": []}
        self.frames: dict[str, list] = {}
        self.frame_steps: list[int] = []

    def record(self, task, obs: dict) -> None:
        import torch.nn.functional as F

        model = self.model
        (obs_tokens, x, t), kwargs = self.last_call[0][:3], self.last_call[1]
        horizon = x.shape[1]
        query = int(round(float(t[0, 0]) * horizon))
        _, hidden = model.expert(obs_tokens, x, t, mask=kwargs.get("mask"), return_hidden=True)
        names = model.tactile.names
        d = hidden.shape[-1]
        tokens = model.tactile_tokens(obs["tactile"])
        reading = F.layer_norm(tokens.view(1, len(names), -1, d).mean(2).float(), (d,))[0]
        self.seed = int(task.cfg.seed)
        step = len(self.rows["query"])
        self.rows["query"].append(query)
        self.rows["action_count"].append(int(task.take_action_cnt))
        self.rows["predicted"].append(model.tactile_predictor(hidden[:, query]).float().view(len(names), d).cpu())
        self.rows["reading"].append(reading.cpu())
        if step % self.frames_every == 0:
            self.frame_steps.append(step)
            for name in names:
                frame = (obs["tactile"][name][0] * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                self.frames.setdefault(name, []).append(frame)

    def flush(self) -> None:
        if self.seed is not None and self.rows["query"]:
            self.dir.mkdir(parents=True, exist_ok=True)
            arrays = {
                "query": np.asarray(self.rows["query"], dtype=np.int32),
                "action_count": np.asarray(self.rows["action_count"], dtype=np.int64),
                "predicted": torch.stack(self.rows["predicted"]).numpy().astype(np.float16),
                "reading": torch.stack(self.rows["reading"]).numpy().astype(np.float16),
                "frame_steps": np.asarray(self.frame_steps, dtype=np.int32),
                **{f"frames_{name}": np.stack(frames) for name, frames in self.frames.items()},
            }
            np.savez_compressed(self.dir / f"{self.seed}.npz", **arrays)
        self.clear()


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
        self.trace = bool(args.get("stflow_trace", False))
        log = args.get("stflow_prediction_log")
        self.prediction_log = None
        if log:
            self.prediction_log = PredictionLog(self.model, Path(log["dir"]).expanduser(), log.get("frames_every", 5))
        self.rl = None
        if args.get("stflow_rl"):
            from policy.stflow.sac import RLSession

            if self.prediction_log is not None:
                raise ValueError("stflow_rl and stflow_prediction_log cannot be combined (the RL log has the reward)")
            self.rl = RLSession(self.model, method_args, args["stflow_rl"])
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
            obs = self.encode_obs(observation)
            controller = self.rl if self.rl is not None else self.controller
            action = controller.act(obs).to(task.device).float()
            if self.prediction_log is not None:
                self.prediction_log.record(task, obs)
        if self.trace:
            # ``task.metadata`` is cleared at every episode reset and written by the task at its end.
            trace = task.metadata.setdefault("trace", {"columns": TRACE_COLUMNS, "rows": []})
            trace["rows"].append(trace_row(task, obs["state"][0].cpu().numpy(), action.cpu().numpy()))
        result = task.take_action(action, action_type="qpos")
        if self.rl is not None:
            early_stop = task.check_early_stop() if hasattr(task, "check_early_stop") else False
            self.rl.observe(getattr(task, "eval_success", False), early_stop)
        return result

    def reset(self):
        """Called before every episode; the controller drops what it carries and restarts its noise
        stream, so an episode depends on its own observations only. Under ``stflow_rl`` the episode
        before is stored and learnt from here, outside inference mode."""
        self.controller.reset()
        if self.prediction_log is not None:
            self.prediction_log.flush()
        if self.rl is not None:
            self.rl.end_episode()

    def close(self):
        if self.prediction_log is not None:
            self.prediction_log.flush()
        if self.rl is not None:
            self.rl.close()
