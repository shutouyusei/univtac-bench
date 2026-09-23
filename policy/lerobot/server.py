"""FastAPI inference server for lerobot policies (SmolVLA, tacforcing) evaluated in UniVTAC.

Runs with the ``lerobot`` conda env's python; ``policy/lerobot/deploy_policy.py``
starts it from the Isaac process and talks to it over HTTP. One request is one
``policy.select_action`` call: SmolVLA executes its chunk from an action
queue, tacforcing streams the chunk block by block, each block conditioned on
the tactile embedding of the frame that asked for it.

The checkpoint's own config decides the policy class (``type``) and the
saved preprocessor decides normalisation and any key renaming from training
(for example ``observation.images.head`` -> ``camera1`` for smolvla_base).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.lerobot.process_data import CAMERA_KEYS, STATE_DIM, TACTILE_CAMERAS  # noqa: E402
from policy.lerobot.tactile_encoder import FTP1GelSightEncoder, rgb_to_bgr  # noqa: E402
from policy.lerobot.wire import from_wire, to_wire  # noqa: E402


def build_batch(
    observation: dict,
    embed: Callable[[np.ndarray], np.ndarray],
    instruction: str,
    image_size: int,
) -> dict:
    """Wire observation (see deploy_policy.select_observation) -> un-batched lerobot frame.

    Images become ``(3, image_size, image_size)`` float in [0, 1] (what a
    LeRobot dataset yields), the state is ``joint[:8]``, and the fingertip
    ``rgb_marker`` frames (RGB from the simulator) are embedded in BGR like the
    dataset converter did.
    """
    import torch

    batch: dict = {}
    for cam, img in observation["images"].items():
        img = np.asarray(img, dtype=np.uint8)
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        batch[CAMERA_KEYS[cam]] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    batch["observation.state"] = torch.as_tensor(
        np.asarray(observation["joint"], dtype=np.float32).reshape(-1)[:STATE_DIM]
    )
    tactile = np.stack([np.asarray(observation["tactile"][cam], dtype=np.uint8) for cam in TACTILE_CAMERAS])
    batch["observation.environment_state"] = torch.from_numpy(embed(rgb_to_bgr(tactile)).reshape(-1)).float()
    batch["task"] = instruction
    return batch


class LocalPolicy:
    """Checkpoint + processors + frozen tactile encoder, all in this process."""

    def __init__(self, args: dict):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()

        ckpt = Path(str(args["lerobot_ckpt_dir"])).expanduser()
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        self.device = str(args.get("lerobot_device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = int(args.get("lerobot_image_size", 256))

        config = PreTrainedConfig.from_pretrained(str(ckpt))
        config.device = self.device
        print(f"[lerobot-server] loading {config.type} from {ckpt}")
        self.policy = get_policy_class(config.type).from_pretrained(str(ckpt), config=config)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

        embedding = str(args.get("lerobot_tactile_embedding", "cls"))
        self.encoder = FTP1GelSightEncoder.from_pretrained(embedding, device=self.device)
        print(f"[lerobot-server] FTP-1 encoder '{embedding}' ({self.encoder.embedding_dim} per fingertip)")

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        import torch

        batch = build_batch(observation, self.encoder.embed, instruction, self.image_size)
        batch = self.pre(batch)
        with torch.no_grad():
            action = self.policy.select_action(batch)
        action = self.post(action)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        self.policy.reset()


def create_app(authkey: str):
    from fastapi import FastAPI, Header, HTTPException, Request

    app = FastAPI(title="UniVTAC lerobot inference server")
    app.state.authkey = authkey
    app.state.model = None
    app.state.server = None

    def check_auth(header_auth: str | None, body_auth: str | None = None) -> None:
        expected = app.state.authkey
        if expected and header_auth != expected and body_auth != expected:
            raise HTTPException(status_code=401, detail="authentication failed")

    @app.get("/health")
    def health(x_lerobot_auth: str | None = Header(default=None)):
        check_auth(x_lerobot_auth)
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.post("/init")
    async def init(request: Request, x_lerobot_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_lerobot_auth, payload.get("authkey"))
        try:
            if app.state.model is None:
                app.state.model = LocalPolicy(from_wire(payload["args"]))
            return {"type": "ok"}
        except BaseException as exc:
            return {"type": "error", "error": repr(exc), "traceback": traceback.format_exc()}

    @app.post("/act")
    async def act(request: Request, x_lerobot_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_lerobot_auth, payload.get("authkey"))
        if app.state.model is None:
            raise HTTPException(status_code=409, detail="model is not initialized")
        try:
            action = app.state.model.act(from_wire(payload["observation"]), payload.get("instruction") or "")
            return {"type": "ok", "action": to_wire(action)}
        except BaseException as exc:
            return {"type": "error", "error": repr(exc), "traceback": traceback.format_exc()}

    @app.post("/reset")
    async def reset(request: Request, x_lerobot_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_lerobot_auth, payload.get("authkey"))
        if app.state.model is not None:
            app.state.model.reset()
        return {"type": "ok"}

    @app.post("/shutdown")
    async def shutdown(request: Request, x_lerobot_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_lerobot_auth, payload.get("authkey"))
        asyncio.create_task(_shutdown_soon())
        return {"type": "ok"}

    async def _shutdown_soon() -> None:
        await asyncio.sleep(0.1)
        if app.state.server is not None:
            app.state.server.should_exit = True

    return app


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="lerobot policy inference server for UniVTAC")
    parser.add_argument("--host", default=os.environ.get("LEROBOT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=os.environ.get("LEROBOT_PORT"))
    parser.add_argument("--authkey", default=os.environ.get("LEROBOT_AUTHKEY", ""))
    args = parser.parse_args()
    if args.port is None:
        raise SystemExit("--port or LEROBOT_PORT is required")

    app = create_app(args.authkey)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=int(args.port), log_level="info", access_log=False))
    app.state.server = server
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
