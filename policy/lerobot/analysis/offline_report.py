"""Offline report of one tacforcing checkpoint: MSE per 5-action block on training chunks.

Rows (normalised action space, same chunks and noise for every row and checkpoint):
  free-running     the eval procedure on demo observations: finished blocks are the model's output
  tactile=zero     free-running with the tactile token at the dataset mean
  teacher-forced   finished blocks replaced by the demo actions (as the model sees them in training)
each at the requested steps per block. A rising free-running error over the blocks with a flat
teacher-forced one is the compounding of the finished-block context.

Usage: python -m policy.lerobot.analysis.offline_report <pretrained_model> <dataset root> [--steps 1 5] [--chunks 128]
"""

import argparse
from pathlib import Path

import torch

from . import offline_common as oc

BLOCK = 5


@torch.no_grad()
def stream(policy, pre, ds, idx, tactile_mode, steps_per_block, teacher_forced):
    cfg = policy.config
    s0, n0 = cfg.steps_per_block, cfg.num_steps
    cfg.steps_per_block, cfg.num_steps = steps_per_block, cfg.num_blocks * steps_per_block
    policy.streamer.state.steps_per_block = steps_per_block
    # a finished block enters the context scaled to the mean of its training-time interpolation
    context_scale = policy.model.context_scale() if hasattr(policy.model, "context_scale") else 1.0
    per, tot, n = 0, 0, 0
    for b in oc.batches(ds, idx):
        b = pre(b)
        tac = b["observation.environment_state"]
        if tactile_mode == "zero":
            tac = torch.zeros_like(tac)
        bsz = tac.shape[0]
        target = b["action"]
        g = torch.Generator(device=oc.device).manual_seed(123)
        noise = torch.randn((bsz, cfg.chunk_size, cfg.max_action_dim), generator=g, device=oc.device)
        policy.streamer.reset()
        policy.streamer.begin(policy.prepare_prefix(b), tac[:, 0], noise)
        blocks = []
        for k in range(cfg.num_blocks):
            blocks.append(policy.streamer.next_block(tac[:, k]).clone())
            if teacher_forced:
                st = policy.streamer.state
                end = (k + 1) * cfg.block_size
                st.latent = st.latent.clone()
                st.latent[:, :end, : target.shape[-1]] = context_scale * target[:, :end].to(st.latent.dtype)
        pred = policy._unpad_actions(torch.cat(blocks, dim=1))
        p, t = oc.block_mse(pred.float(), target.float(), b["action_is_pad"], BLOCK)
        per, tot, n = per + p * bsz, tot + t * bsz, n + bsz
    cfg.steps_per_block, cfg.num_steps = s0, n0
    policy.streamer.state.steps_per_block = s0
    return per / n, tot / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pretrained_model", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--chunks", type=int, default=128)
    args = parser.parse_args()
    oc.root, oc.repo_id, oc.n_chunks = args.dataset_root, "local/" + args.dataset_root.name, args.chunks
    policy, pre = oc.load(args.pretrained_model)
    ds, idx = oc.make_loader(policy.config)
    cfg = policy.config
    k = cfg.num_blocks
    options = {
        name: getattr(cfg, name)
        for name in ("hide_finished_blocks", "block_schedule_prob", "finished_block_time", "loss_reduction")
        if hasattr(cfg, name)
    }
    print(f"checkpoint {args.pretrained_model.parents[2].name}: K={k} B={cfg.block_size} {options}; "
          f"chunks {len(idx)}; MSE per 5-action block")
    print("| condition | overall | " + " | ".join(f"b{i + 1}" for i in range(cfg.chunk_size // BLOCK)) + " |")
    print("|---|---|" + "---|" * (cfg.chunk_size // BLOCK))
    for s in args.steps:
        for mode, tf in (("recorded", False), ("zero", False), ("recorded", True)):
            if tf and k == 1:
                continue
            per, tot = stream(policy, pre, ds, idx, mode, s, tf)
            kind = "teacher-forced" if tf else "free-running"
            oc.fmt(f"{kind} tactile={mode} S={s} steps={k * s}", per, tot)


if __name__ == "__main__":
    main()
