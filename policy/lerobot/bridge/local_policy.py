"""lerobot side: checkpoint + processors + frozen tactile encoder in one process."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..tactile import FTP1GelSightEncoder
from .batch import build_batch

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_checkpoint(ckpt_dir: str) -> Path:
    ckpt = Path(ckpt_dir).expanduser()
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    return ckpt


def apply_steps_per_block(config, steps_per_block: int | None) -> None:
    """Set a block-streaming policy's sampling steps per block (S) for this run; None keeps the checkpoint's.

    S is inference-only, so it is chosen at deploy time rather than baked into a checkpoint copy.
    ``num_steps`` follows as blocks x S, the way the plugin derives it.
    """
    if steps_per_block is None:
        return
    if not hasattr(config, "steps_per_block"):
        raise ValueError(f"{config.type} has no steps_per_block; the override applies to block-streaming policies")
    if steps_per_block < 1:
        raise ValueError(f"steps_per_block must be >= 1, got {steps_per_block}")
    config.steps_per_block = steps_per_block
    config.num_steps = config.chunk_size // config.block_size * steps_per_block


def apply_n_action_steps(config, n_action_steps: int | None) -> int | None:
    """Set how many actions of each chunk run before the next inference; None keeps the checkpoint's.

    Like S, the execution horizon is inference-only. A chunking policy takes it in its config. A
    block-streaming policy streams its whole chunk by design, so its config is left alone and the
    returned horizon tells the bridge to start a new chunk itself after that many actions.
    """
    if n_action_steps is None:
        return None
    if not 1 <= n_action_steps <= config.chunk_size:
        raise ValueError(f"n_action_steps must be in [1, {config.chunk_size}], got {n_action_steps}")
    if hasattr(config, "steps_per_block"):
        return n_action_steps
    config.n_action_steps = n_action_steps
    return None


class Replanner:
    """Counts executed actions and says when a new chunk is due, every ``every`` actions (None: never).

    Resetting a policy clears its action queue and stream, so its next call starts a chunk from the
    current observation.
    """

    def __init__(self, every: int | None):
        self.every = every
        self.count = 0

    def executed(self) -> bool:
        """Record one executed action; True when the policy should be reset before the next one."""
        if self.every is None:
            return False
        self.count += 1
        if self.count < self.every:
            return False
        self.count = 0
        return True

    def reset(self) -> None:
        self.count = 0


def load_policy(
    ckpt: Path, device: str, steps_per_block: int | None = None, n_action_steps: int | None = None
):
    """Policy of the checkpoint's own type (plugins registered), its saved pre/post processors, and
    the horizon after which the bridge must start a new chunk (None when the policy handles it)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    config = PreTrainedConfig.from_pretrained(str(ckpt))
    config.device = device
    apply_steps_per_block(config, steps_per_block)
    replan_every = apply_n_action_steps(config, n_action_steps)
    print(f"[lerobot-server] loading {config.type} from {ckpt}"
          + (f" at steps_per_block={steps_per_block}" if steps_per_block else "")
          + (f" executing {n_action_steps} of {config.chunk_size} actions" if n_action_steps else ""))
    policy = get_policy_class(config.type).from_pretrained(str(ckpt), config=config)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, pre, post, replan_every


class LocalPolicy:
    def __init__(self, args: dict):
        import torch

        self.device = str(args.get("lerobot_device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = int(args.get("lerobot_image_size", 256))
        steps = args.get("lerobot_steps_per_block")
        executed = args.get("lerobot_n_action_steps")
        self.policy, self.pre, self.post, replan_every = load_policy(
            resolve_checkpoint(str(args["lerobot_ckpt_dir"])),
            self.device,
            int(steps) if steps else None,
            int(executed) if executed else None,
        )
        self.replanner = Replanner(replan_every)
        embedding = str(args.get("lerobot_tactile_embedding", "cls"))
        self.encoder = FTP1GelSightEncoder.from_pretrained(embedding, device=self.device)
        print(f"[lerobot-server] FTP-1 encoder '{embedding}' ({self.encoder.embedding_dim} per fingertip)")

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        import torch

        batch = self.pre(build_batch(observation, self.encoder.embed, instruction, self.image_size))
        with torch.no_grad():
            action = self.policy.select_action(batch)
        if self.replanner.executed():
            self.policy.reset()
        return np.asarray(self.post(action), dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        self.policy.reset()
        self.replanner.reset()
