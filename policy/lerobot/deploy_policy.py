"""UniVTAC ``BasePolicy`` adapter that drives a lerobot policy through ``server.py``.

This file runs inside the Isaac process (``UniVTAC`` env) and imports nothing
from lerobot: the checkpoint, its processors and the FTP-1 tactile encoder
live in the server started with the ``lerobot`` env's python. Per step the
adapter ships the head/wrist RGB frames, both fingertip ``rgb_marker`` frames
and ``joint[:8]`` to the server and executes the returned joint target with
``take_action(..., action_type="qpos")``.

deploy_*.yml keys (all optional except ``lerobot_ckpt_dir``):

* ``lerobot_ckpt_dir``          ``pretrained_model`` directory of a lerobot-train run
* ``lerobot_python``            interpreter of the lerobot env (default ``$LEROBOT_PYTHON``
                                or ``~/miniforge3/envs/lerobot/bin/python``)
* ``lerobot_port``              0 starts a server on a free port; otherwise connect to a running one
* ``lerobot_tactile_embedding`` ``cls`` or ``proj``, must match the training dataset
* ``lerobot_image_size``        camera frame size the dataset was converted with
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from policy._base_policy import BasePolicy  # noqa: E402
from policy.lerobot.wire import from_wire, to_wire  # noqa: E402

DEFAULT_LEROBOT_PYTHON = "~/miniforge3/envs/lerobot/bin/python"
TACTILE_ALIASES = {"left": ("left_tactile", "left_gsmini"), "right": ("right_tactile", "right_gsmini")}


def _as_uint8_image(img) -> np.ndarray:
    """HWC image as the simulator hands it (torch or numpy, uint8 or float) -> HWC uint8 numpy."""
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"expected an HWC image, got shape {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def select_observation(observation: dict, cameras: tuple[str, ...]) -> dict:
    """Keep only what the server needs: camera RGB, fingertip rgb_marker, joint[:8]."""
    images = {}
    for cam in cameras:
        cam_obs = observation.get("observation", {}).get(cam)
        if cam_obs is None or "rgb" not in cam_obs:
            raise KeyError(f"observation missing observation.{cam}.rgb")
        images[cam] = _as_uint8_image(cam_obs["rgb"])

    tactile = {}
    tactile_obs = observation.get("tactile", {})
    for side, aliases in TACTILE_ALIASES.items():
        for alias in aliases:
            if alias in tactile_obs and "rgb_marker" in tactile_obs[alias]:
                tactile[side] = _as_uint8_image(tactile_obs[alias]["rgb_marker"])
                break
        if side not in tactile:
            raise KeyError(f"observation missing tactile.{aliases[0]}.rgb_marker")

    joint = observation["embodiment"]["joint"]
    if hasattr(joint, "detach"):
        joint = joint.detach().cpu().numpy()
    joint = np.asarray(joint, dtype=np.float32).reshape(-1)[:8]
    return {"images": images, "tactile": tactile, "joint": joint}


def cameras_for_task(task_name: str, task_settings: Path = _REPO_ROOT / "policy" / "task_settings.json") -> tuple[str, ...]:
    camera_type = "head"
    if task_settings.exists():
        with open(task_settings) as f:
            camera_type = json.load(f).get(task_name, {}).get("camera_type", "head")
    return ("head", "wrist") if camera_type == "all" else (camera_type,)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ServerClient:
    def __init__(self, args: dict):
        self.host = str(args.get("lerobot_host", os.environ.get("LEROBOT_HOST", "127.0.0.1")))
        self.port = int(args.get("lerobot_port", os.environ.get("LEROBOT_PORT", 0)) or 0)
        self.python = Path(
            str(args.get("lerobot_python") or os.environ.get("LEROBOT_PYTHON") or DEFAULT_LEROBOT_PYTHON)
        ).expanduser()
        self.startup_timeout = float(args.get("lerobot_startup_timeout", 600))
        self.request_timeout = float(args.get("lerobot_request_timeout", 300))
        self.authkey = str(args.get("lerobot_authkey", os.environ.get("LEROBOT_AUTHKEY", "")))
        self._external = self.port != 0
        self._process: subprocess.Popen | None = None
        self._closed = False

        if not self._external:
            self.port = _find_free_port()
            self.authkey = secrets.token_hex(16)
            self._start_server()
        self.base_url = f"http://{self.host}:{self.port}"
        self._wait_until_ready()
        self._post("/init", {"authkey": self.authkey, "args": to_wire(args)})

    def _start_server(self) -> None:
        if not self.python.exists():
            raise FileNotFoundError(f"lerobot python not found: {self.python} (set lerobot_python or $LEROBOT_PYTHON)")
        self._process = subprocess.Popen(
            [str(self.python), str(_THIS_DIR / "server.py"), "--host", self.host, "--port", str(self.port), "--authkey", self.authkey],
            cwd=str(_REPO_ROOT),
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                if self._request("GET", "/health").get("status") == "ok":
                    return
            except BaseException as exc:
                last_error = exc
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(f"lerobot server exited with code {self._process.returncode}") from exc
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for the lerobot server: {last_error}")

    def _request(self, method: str, path: str, data: bytes | None = None) -> Any:
        headers = {"X-Lerobot-Auth": self.authkey}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"lerobot server HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc

    def _post(self, path: str, payload: Any) -> Any:
        response = self._request("POST", path, json.dumps(payload).encode("utf-8"))
        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(f"lerobot server error:\n{response.get('error')}\n{response.get('traceback')}")
        return response

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        response = self._post("/act", {"authkey": self.authkey, "instruction": instruction, "observation": to_wire(observation)})
        return np.asarray(from_wire(response["action"]), dtype=np.float32)

    def reset(self) -> None:
        self._post("/reset", {"authkey": self.authkey})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._post("/reset" if self._external else "/shutdown", {"authkey": self.authkey})
        except Exception:
            pass
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()


class Policy(BasePolicy):
    """Deploy-time adapter: UniVTAC observation -> server -> ``take_action('qpos')``."""

    def __init__(self, args: dict):
        self.task_name = args["task_name"]
        self.cameras = cameras_for_task(self.task_name)
        self.model = _ServerClient(args)
        self._instruction: str | None = None

    def encode_obs(self, observation: dict) -> dict:
        return select_observation(observation, self.cameras)

    def eval(self, task, observation):
        import torch

        if self._instruction is None:
            self._instruction = getattr(task, "instruction", None) or self.task_name.replace("_", " ")
        action = self.model.act(self.encode_obs(observation), self._instruction).reshape(-1)
        action_t = torch.from_numpy(action).to(task.device).float()
        return task.take_action(action_t, action_type="qpos")

    def reset(self):
        self._instruction = None
        self.model.reset()

    def close(self):
        self.model.close()
