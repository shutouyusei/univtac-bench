"""Shared pieces of the offline checks: load a checkpoint with its preprocessor, draw a fixed
set of training chunks, stream one chunk the way eval does, and score it per block.

Everything runs on demo observations with the same noise for every condition, so two rows
of a report differ only in what they are meant to differ in. ``root``, ``repo_id`` and
``n_chunks`` are set by the calling script before ``make_loader``.
"""

import torch
from torch.utils.data import default_collate

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins

try:
    from lerobot.datasets.factory import resolve_delta_timestamps
except ImportError:
    from lerobot.datasets.utils import resolve_delta_timestamps

register_third_party_plugins()
device = "cuda"
torch.manual_seed(0)
root, repo_id, n_chunks = None, None, 128  # set by main or by an importing script


def load(ckpt):
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(str(ckpt), config=cfg).to(device).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(ckpt),
                                      preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre


def make_loader(cfg):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta = resolve_delta_timestamps(cfg, meta)
    ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n_chunks].tolist()
    return ds, idx


def batches(ds, idx, bs=16):
    for s in range(0, len(idx), bs):
        yield default_collate([ds[i] for i in idx[s : s + bs]])


def block_mse(pred, target, is_pad, block):
    """(B, H, A) -> per-block MSE over valid entries, (K,) and overall."""
    err = ((pred - target) ** 2).mean(-1)  # (B, H)
    valid = ~is_pad
    k = err.shape[1] // block
    per = torch.stack([(err[:, b * block:(b + 1) * block] * valid[:, b * block:(b + 1) * block]).sum()
                       / valid[:, b * block:(b + 1) * block].sum().clamp_min(1) for b in range(k)])
    return per, (err * valid).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def run_tacforcing(policy, pre, ds, idx, tactile_mode, steps_per_block):
    cfg = policy.config
    s0, n0 = cfg.steps_per_block, cfg.num_steps
    cfg.steps_per_block, cfg.num_steps = steps_per_block, cfg.num_blocks * steps_per_block
    policy.streamer.state.steps_per_block = steps_per_block
    per, tot, n = 0, 0, 0
    perm_cache = None
    for b in batches(ds, idx):
        b = pre(b)
        tac = b["observation.environment_state"]  # (B, K, C) normalised
        if tactile_mode == "zero":
            tac = torch.zeros_like(tac)
        elif tactile_mode == "shuffle":
            tac = tac[torch.randperm(tac.shape[0], generator=torch.Generator().manual_seed(1))]
        bsz = tac.shape[0]
        g = torch.Generator(device=device).manual_seed(123)
        noise = torch.randn((bsz, cfg.chunk_size, cfg.max_action_dim), generator=g, device=device)
        policy.streamer.reset()
        policy.streamer.begin(policy.prepare_prefix(b), tac[:, 0], noise)
        blocks = [policy.streamer.next_block(tac[:, k]) for k in range(cfg.num_blocks)]
        pred = policy._unpad_actions(torch.cat(blocks, dim=1))
        target = b["action"]
        p, t = block_mse(pred.float(), target.float(), b["action_is_pad"], cfg.block_size)
        per, tot, n = per + p * bsz, tot + t * bsz, n + bsz
    cfg.steps_per_block, cfg.num_steps = s0, n0
    policy.streamer.state.steps_per_block = s0
    return per / n, tot / n


@torch.no_grad()
def run_base(policy, pre, ds, idx):
    cfg = policy.config
    per, tot, n = 0, 0, 0
    for b in batches(ds, idx):
        b = pre(b)
        bsz = b["action"].shape[0]
        g = torch.Generator(device=device).manual_seed(123)
        noise = torch.randn((bsz, cfg.chunk_size, cfg.max_action_dim), generator=g, device=device)
        images, img_masks = policy.prepare_images(b)
        state = policy.prepare_state(b)
        pred = policy.model.sample_actions(images, img_masks, b["observation.language.tokens"],
                                          b["observation.language.attention_mask"], state, noise=noise)
        pred = pred[:, :, : b["action"].shape[-1]]
        p, t = block_mse(pred.float(), b["action"].float(), b["action_is_pad"], 5)
        per, tot, n = per + p * bsz, tot + t * bsz, n + bsz
    return per / n, tot / n


def fmt(tag, per, tot):
    print(f"| {tag} | {tot:.4f} | " + " | ".join(f"{v:.3f}" for v in per.tolist()) + " |")


