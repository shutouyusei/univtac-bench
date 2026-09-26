"""lerobot side: checkpoint + processors + frozen tactile encoder in one process."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..tactile import FTP1GelSightEncoder
from .batch import build_batch

REPO_ROOT = Path(__file__).resolve().parents[3]


def as_bool(value) -> bool:
    """yml booleans arrive as bool, environment overrides as strings like "1" / "false"."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def resolve_checkpoint(ckpt_dir: str) -> Path:
    ckpt = Path(ckpt_dir).expanduser()
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    return ckpt


def load_policy(ckpt: Path, device: str):
    """Policy of the checkpoint's own type (plugins registered) and its saved pre/post processors."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    config = PreTrainedConfig.from_pretrained(str(ckpt))
    config.device = device
    print(f"[lerobot-server] loading {config.type} from {ckpt}")
    policy = get_policy_class(config.type).from_pretrained(str(ckpt), config=config)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, pre, post


class LocalPolicy:
    def __init__(self, args: dict):
        import torch

        self.device = str(args.get("lerobot_device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = int(args.get("lerobot_image_size", 256))
        self.policy, self.pre, self.post = load_policy(resolve_checkpoint(str(args["lerobot_ckpt_dir"])), self.device)
        self.flip_cameras = as_bool(args.get("lerobot_flip_cameras", False))
        self.flip_tactile = as_bool(args.get("lerobot_flip_tactile", True))
        embedding = str(args.get("lerobot_tactile_embedding", "cls"))
        self.encoder = FTP1GelSightEncoder.from_pretrained(embedding, device=self.device)
        print(f"[lerobot-server] FTP-1 encoder '{embedding}' ({self.encoder.embedding_dim} per fingertip)")
        print(f"[lerobot-server] channel flips: cameras={self.flip_cameras} tactile={self.flip_tactile}")

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        import torch

        batch = self.pre(
            build_batch(
                observation,
                self.encoder.embed,
                instruction,
                self.image_size,
                flip_cameras=self.flip_cameras,
                flip_tactile=self.flip_tactile,
            )
        )
        with torch.no_grad():
            action = self.policy.select_action(batch)
        return np.asarray(self.post(action), dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        self.policy.reset()
