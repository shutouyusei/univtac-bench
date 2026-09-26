"""UniVTAC ``BasePolicy`` adapter that drives a lerobot policy through ``server.py``.

Runs inside the Isaac process (``UniVTAC`` env) and imports nothing from
lerobot: the checkpoint, its processors and the FTP-1 tactile encoder live in
the server started with the ``lerobot`` env's python. Per step the adapter
ships the camera RGB frames, both fingertip ``rgb_marker`` frames and
``joint[:8]`` (``bridge/observation.py``) to the server (``bridge/client.py``)
and executes the returned joint target with ``take_action(..., "qpos")``.

deploy_*.yml keys (all optional except ``lerobot_ckpt_dir``):

* ``lerobot_ckpt_dir``          ``pretrained_model`` directory of a lerobot-train run;
                                the ``LEROBOT_CKPT_DIR`` environment variable overrides it,
                                so one deploy yml serves every run of the same policy type
* ``lerobot_python``            interpreter of the lerobot env (default ``$LEROBOT_PYTHON``
                                or ``~/miniforge3/envs/lerobot/bin/python``)
* ``lerobot_port``              0 starts a server on a free port; otherwise connect to a running one
* ``lerobot_tactile_embedding`` ``cls`` or ``proj``, must match the training dataset
* ``lerobot_image_size``        camera frame size the dataset was converted with
* ``lerobot_flip_cameras``      swap the camera channel order before the policy (default false);
                                ``LEROBOT_FLIP_CAMERAS`` overrides
* ``lerobot_flip_tactile``      swap the fingertip channel order before the tactile encoder
                                (default true); ``LEROBOT_FLIP_TACTILE`` overrides
* ``lerobot_retries``, ``lerobot_request_timeout``, ``lerobot_startup_timeout``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from policy._base_policy import BasePolicy  # noqa: E402
from policy.lerobot.bridge.client import BridgeError, connect  # noqa: E402
from policy.lerobot.bridge.observation import cameras_for_task, select_observation  # noqa: E402


class Policy(BasePolicy):
    """UniVTAC observation -> server -> ``take_action('qpos')``; a dead server fails every call fast."""

    def __init__(self, args: dict):
        self.task_name = args["task_name"]
        self.cameras = cameras_for_task(self.task_name)
        for env, key in (
            ("LEROBOT_CKPT_DIR", "lerobot_ckpt_dir"),
            ("LEROBOT_FLIP_CAMERAS", "lerobot_flip_cameras"),
            ("LEROBOT_FLIP_TACTILE", "lerobot_flip_tactile"),
        ):
            if os.environ.get(env):
                args = {**args, key: os.environ[env]}
                print(f"[lerobot-bridge] {env} overrides {key}: {args[key]}")
        self.model = connect(args)
        self._instruction: str | None = None

    def encode_obs(self, observation: dict) -> dict:
        return select_observation(observation, self.cameras)

    def eval(self, task, observation):
        import torch

        if self.model.dead:
            raise BridgeError(f"lerobot server is down, fix it and rerun the eval: {self.model.dead}")
        if self._instruction is None:
            self._instruction = getattr(task, "instruction", None) or self.task_name.replace("_", " ")
        action = self.model.act(self.encode_obs(observation), self._instruction)
        action_t = torch.from_numpy(action).to(task.device).float()
        return task.take_action(action_t, action_type="qpos")

    def reset(self):
        self._instruction = None
        self.model.reset()

    def close(self):
        self.model.close()
