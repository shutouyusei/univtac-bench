"""Per-step record of a stflow evaluation episode (deploy key ``stflow_record``).

An evaluation run cannot be repeated exactly (identical runs flip some seeds), so whatever a later analysis
may need is recorded in the run that produced the outcome. One ``record/<seed>.npz`` per episode sits next
to the run's ``metadata.json``; the episode's metadata entry names it under ``record``. The npz holds, one
row per control step, read before that step's action is executed:

* ``action_count``         the task's action counter before the step
* ``command``              the joint target the policy returned ``(T, action_dim)``
* ``joint``                the measured joints the policy was given ``(T, 9)``
* ``ee_pose``, ``gripper_center_pose``   ``(T, 7)`` position + wxyz quaternion, NaN if the task has no robot manager
* ``actor_<name>``         every actor's pose ``(T, 7)``, from the simulator state the policy does not see
* ``controller_step``, ``reading_step``, ``new_chunk``, ``reread``   the controller's position in its chunk
  (``controller_state``)
* ``infer_ms``             wall time of the controller's ``act`` (encoding the frames is not included)
* ``tactile_tokens``       ``(T, fingertips, grid * grid, 512)`` float16: the fingertip frames the policy was
  given, encoded by the frozen ResNet18 of the future-tactile targets (``stflow.data.tactile_targets``), so
  rollouts and demonstrations share one embedding
* ``frames_<fingertip>_jpeg`` / ``frames_<fingertip>_offsets`` / ``frame_steps``   the raw ``rgb_marker``
  frames every ``frames_every`` steps, JPEG bytes concatenated with their start offsets; ``cv2.imdecode``
  returns them in the simulator's RGB channel order (they were encoded without a channel swap)

and once per episode:

* ``seed``, ``fingertips``
* ``task_<name>``          the pose attributes of ``TASK_POSES`` the task has (targets, the grasp's in-hand
  pose), read at the first step; an actor that moves during the episode is followed by ``actor_<name>``
* ``final_action_count``, ``final_ee_pose``, ``final_gripper_center_pose``, ``final_actor_<name>``   the
  simulator state after the last executed action, the state the task's success and early-stop checks judge

A failure while recording never stops the evaluation: the episode's record is dropped and its metadata
entry gets ``record_error`` instead.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

TASK_POSES = ("target_pose", "hole_pose", "noise_pose", "origin_inhand_pose")
NAN_POSE = [float("nan")] * 7


def controller_state(controller) -> dict:
    """Where the controller stands after ``act``. Streaming flow controllers (SFP, SFP-TR) count the actions
    of the current chunk in ``step`` and keep the step of the last tactile reading in ``reading_step``;
    the chunk controller keeps the actions still queued. ``new_chunk`` = this step started a chunk,
    ``reread`` = this step read the tactile again inside a chunk. -1 = the controller has no such field."""
    step = getattr(controller, "step", None)
    queue = getattr(controller, "queue", None)
    n = getattr(controller, "n_action_steps", None)
    reading = getattr(controller, "reading_step", None)
    if step is not None:
        new_chunk = int(step == 1)
    elif queue is not None and n is not None:
        new_chunk = int(len(queue) == n - 1)
    else:
        new_chunk = -1
    reread = int(reading is not None and step is not None and reading > 0 and reading == step - 1)
    return {
        "controller_step": -1 if step is None else int(step),
        "reading_step": -1 if reading is None else int(reading),
        "new_chunk": new_chunk,
        "reread": reread,
    }


def _pose(obj) -> list[float]:
    values = obj.tolist() if hasattr(obj, "tolist") else list(obj)
    return np.asarray(values, dtype=np.float64).reshape(-1).tolist()


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _poses(task) -> dict[str, list[float]]:
    """The robot's and every actor's pose now."""
    robot = getattr(task, "_robot_manager", None)
    out = {
        "ee_pose": _pose(robot.get_ee_pose()) if robot is not None else NAN_POSE,
        "gripper_center_pose": _pose(robot.get_gripper_center_pose()) if robot is not None else NAN_POSE,
    }
    manager = getattr(task, "_actor_manager", None)
    for name, actor in (getattr(manager, "actors", None) or {}).items():
        out[f"actor_{name}"] = _pose(actor.get_pose())
    return out


class EpisodeRecorder:
    """Collects one episode's rows and writes them when the episode is over (``flush``). ``encoder`` maps
    the policy's ``(F, 3, S, S)`` fingertip frames to ``(F, grid * grid, 512)`` tokens; None records no
    tokens. ``frames_every`` 0 records no raw frames."""

    def __init__(self, fingertips: list[str], encoder=None, frames_every: int = 10):
        if frames_every < 0:
            raise ValueError(f"frames_every must be >= 0, got {frames_every}")
        self.fingertips = list(fingertips)
        self.encoder = encoder
        self.frames_every = frames_every
        self._clear()

    def _clear(self) -> None:
        self.path: Path | None = None
        self.seed: int | None = None
        self.failed = False
        self.static: dict[str, np.ndarray] = {}
        self.final: dict[str, np.ndarray] = {}
        self.rows: dict[str, list] = {}
        self.frames: dict[str, list[bytes]] = {f: [] for f in self.fingertips}
        self.frame_steps: list[int] = []
        self.n = 0

    @property
    def active(self) -> bool:
        return self.path is not None

    def start(self, task) -> None:
        """First step of an episode: fix the output file from the task's result folder and seed (the next
        episode's reset changes both) and point the episode's metadata at it."""
        self.seed = int(task.cfg.seed)
        self.path = Path(task.save_root) / "record" / f"{self.seed}.npz"
        task.metadata["record"] = f"record/{self.seed}.npz"
        for name in TASK_POSES:
            value = getattr(task, name, None)
            if value is not None:
                self.static[f"task_{name}"] = np.asarray(_pose(value), dtype=np.float64)

    def _append(self, key: str, value) -> None:
        self.rows.setdefault(key, []).append(value)

    def step(self, task, obs: dict, joint, raw_tactile: dict, action, controller, infer_ms: float) -> None:
        """One row: ``obs`` is the policy's encoded batch, ``joint`` the simulator's joint reading and
        ``raw_tactile`` each fingertip's ``rgb_marker`` frame as the simulator gave it."""
        if self.failed:
            return
        if not self.active:
            self.start(task)
        self._append("action_count", int(task.take_action_cnt))
        self._append("command", _numpy(action).astype(np.float32).reshape(-1))
        self._append("joint", _numpy(joint).astype(np.float32).reshape(-1))
        for key, value in _poses(task).items():
            self._append(key, value)
        for key, value in controller_state(controller).items():
            self._append(key, value)
        self._append("infer_ms", float(infer_ms))
        if self.encoder is not None:
            frames = torch.cat([obs["tactile"][f] for f in self.fingertips])
            self._append("tactile_tokens", _numpy(self.encoder(frames)).astype(np.float16))
        if self.frames_every and self.n % self.frames_every == 0:
            self.frame_steps.append(self.n)
            for f in self.fingertips:
                frame = np.clip(np.rint(_numpy(raw_tactile[f])), 0, 255).astype(np.uint8)
                ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(frame[..., :3]))
                if not ok:
                    raise ValueError("JPEG encoding of a fingertip frame failed")
                self.frames[f].append(buf.tobytes())
        self.n += 1

    def after_action(self, task) -> None:
        """The state after the step's action; the last call of the episode is kept as ``final_*``."""
        if self.failed or not self.active:
            return
        self.final = {f"final_{k}": np.asarray(v, dtype=np.float64) for k, v in _poses(task).items()}
        self.final["final_action_count"] = np.asarray(int(task.take_action_cnt))

    def fail(self, task, error: BaseException) -> None:
        """Drop the episode's record and say why in its metadata; later steps of the episode record nothing."""
        message = "".join(traceback.format_exception_only(type(error), error)).strip()
        print(f"stflow_record: episode record dropped: {message}")
        if getattr(task, "metadata", None) is not None:
            task.metadata["record_error"] = message
            task.metadata.pop("record", None)
        self._clear()
        self.failed = True

    def flush(self) -> Path | None:
        """Write the episode (if one ran) and forget it, whether or not the write succeeds."""
        try:
            if not self.active or self.n == 0 or self.failed:
                return None
            out = {key: np.asarray(values) for key, values in self.rows.items()}
            out["infer_ms"] = out["infer_ms"].astype(np.float32)
            for key in [k for k in out if k.endswith("_pose") or k.startswith("actor_")]:
                out[key] = out[key].astype(np.float64)
            out.update(self.static)
            out.update(self.final)
            out["seed"] = np.asarray(self.seed)
            out["fingertips"] = np.asarray(self.fingertips)
            if self.frames_every:
                out["frame_steps"] = np.asarray(self.frame_steps, dtype=np.int32)
                for f, blobs in self.frames.items():
                    sizes = [len(b) for b in blobs]
                    out[f"frames_{f}_offsets"] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
                    out[f"frames_{f}_jpeg"] = np.frombuffer(b"".join(blobs), dtype=np.uint8)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, **out)
            return self.path
        finally:
            self._clear()
