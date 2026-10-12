"""Train RDP's two stages on one UniVTAC task, as the official workspaces do.

    python -m rdp_univtac.train at  --cache policy/RDP/data/lift_bottle-clean51-50 --pca policy/RDP/data/marker_pca.npz \
        --out policy/RDP/rdp_ckpt/lift_bottle-clean51-50
    python -m rdp_univtac.train ldp --cache ... --pca ... --out policy/RDP/rdp_ckpt/lift_bottle-clean51-50

* ``at`` (TrainATWorkspace): the asymmetric tokenizer, AdamW 1e-3 / 1e-4, cosine with 100 warm-up steps,
  601 epochs, batch 64, no EMA; loss = L1 reconstruction + 1e-6 KL (``VAE.compute_loss_and_metric``).
* ``ldp`` (TrainDiffusionUnetImageWorkspace): the latent diffusion policy on the frozen tokenizer's last
  checkpoint, AdamW 1e-4 (0.95, 0.999) / 1e-6, cosine with 500 warm-up steps, 401 epochs, batch 64, EMA
  (power 0.75, max 0.9999); the latent normalizer is fit by encoding every window (posterior samples).
  At the end it writes ``rdp_policy.pt``, everything evaluation needs: the EMA policy (the official evaluation
  uses the EMA model), the tokenizer, the normalizer, the marker PCA, the shape meta and the config.

Both stages save ``<stage>_latest.pt`` every ``checkpoint_every`` epochs and resume from it (the official
workspaces resume by default), so an interrupted run loses at most that many epochs. A finished stage is
skipped. Values come from rdp_config.yml; ``--set key.sub=value`` overrides them (tests, smoke runs).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from reactive_diffusion_policy.model.common.lr_scheduler import get_scheduler
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer
from reactive_diffusion_policy.model.diffusion.ema_model import EMAModel

from . import build
from .dataset import UniVTACImageTactileDataset
from .marker import MarkerPCA
from .prepare import load_buffer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)


def loader(dataset, cfg: dict, seed: int) -> torch.utils.data.DataLoader:
    workers = cfg["num_workers"]
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0, generator=torch.Generator().manual_seed(seed))


def log(out: Path, row: dict) -> None:
    with open(out / "train_log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def load_dataset(cache: Path, pca: Path, images: bool) -> UniVTACImageTactileDataset:
    buffer = load_buffer(cache, MarkerPCA.load(pca))
    return UniVTACImageTactileDataset(buffer, cameras=buffer.meta["cameras"], images=images)


def run_epochs(stage: str, out: Path, cfg_t: dict, n_batches: int, state: dict, step_fn, save_fn) -> None:
    for epoch in range(state["epoch"], cfg_t["num_epochs"]):
        t0, losses = time.time(), []
        for loss in step_fn():
            losses.append(loss)
        state["epoch"] = epoch + 1
        log(out, {"stage": stage, "epoch": epoch + 1, "loss": float(np.mean(losses)), "batches": n_batches,
                  "seconds": round(time.time() - t0, 1)})
        if state["epoch"] % cfg_t["checkpoint_every"] == 0 or state["epoch"] == cfg_t["num_epochs"]:
            save_fn(done=state["epoch"] == cfg_t["num_epochs"])


def train_at(cfg: dict, cache: Path, pca: Path, out: Path, device: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "at_latest.pt"
    previous = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.exists() else None
    if previous is not None and previous["done"]:
        print(f"tokenizer already trained: {ckpt}")
        return ckpt
    t = cfg["at_training"]
    seed_everything(t["seed"])
    dataset = load_dataset(cache, pca, images=False)
    normalizer = dataset.get_normalizer()
    at = build.build_at(cfg, dataset.shape_meta, device)
    at.set_normalizer(normalizer)
    at.normalizer.to(device)  # same numbers; avoids a CPU round trip per batch
    data = loader(dataset, t, t["seed"])
    optimizer = torch.optim.AdamW(at.optim_params, lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = get_scheduler(t["lr_scheduler"], optimizer=optimizer, num_warmup_steps=t["lr_warmup_steps"],
                              num_training_steps=len(data) * t["num_epochs"])
    state = {"epoch": 0}
    if previous is not None:
        at.load_state_dict(previous["at"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        state["epoch"] = previous["epoch"]
        print(f"resuming the tokenizer at epoch {state['epoch']}")

    def steps():
        at.train()
        for batch in data:
            loss = at.compute_loss_and_metric(to_device(batch, device))["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            yield loss.item()

    def save(done: bool):
        torch.save({"at": at.state_dict(), "normalizer": normalizer.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": state["epoch"], "done": done,
                    "shape_meta": dataset.shape_meta, "config": cfg}, ckpt)

    run_epochs("at", out, t, len(data), state, steps, save)
    return ckpt


def load_tokenizer(cfg: dict, shape_meta: dict, at_ckpt: Path, device: str):
    saved = torch.load(at_ckpt, map_location="cpu", weights_only=False)
    if not saved["done"]:
        raise RuntimeError(f"the tokenizer in {at_ckpt} has not finished training")
    at = build.build_at(cfg, shape_meta, device)
    at.load_state_dict(saved["at"])
    at.eval()
    return at


def train_ldp(cfg: dict, cache: Path, pca: Path, out: Path, device: str) -> Path:
    bundle = out / "rdp_policy.pt"
    ckpt = out / "ldp_latest.pt"
    if bundle.exists():
        print(f"policy already trained: {bundle}")
        return bundle
    t = cfg["ldp_training"]
    seed_everything(t["seed"])
    dataset = load_dataset(cache, pca, images=True)
    normalizer = dataset.get_normalizer()
    at = load_tokenizer(cfg, dataset.shape_meta, out / "at_latest.pt", device)
    actions, _ = dataset.relative_arrays()
    previous = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.exists() else None
    if previous is None:
        normalizer["latent_action"] = build.latent_normalizer(at, normalizer, actions)
    else:  # the latent normalizer is a posterior-sample fit; keep the one training started with
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(previous["normalizer"])
    policy = build.build_policy(cfg, dataset.shape_meta, at)
    policy.set_normalizer(normalizer)
    policy.to(device)
    ema_model = copy.deepcopy(policy)
    ema = EMAModel(ema_model, **t["ema"])
    data = loader(dataset, t, t["seed"])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=t["lr"], betas=tuple(t["betas"]), eps=t["eps"],
                                  weight_decay=t["weight_decay"])
    scheduler = get_scheduler(t["lr_scheduler"], optimizer=optimizer, num_warmup_steps=t["lr_warmup_steps"],
                              num_training_steps=len(data) * t["num_epochs"])
    state = {"epoch": 0}
    if previous is not None:
        policy.load_state_dict(previous["model"])
        ema_model.load_state_dict(previous["ema_model"])
        ema.optimization_step = previous["ema_step"]
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        state["epoch"] = previous["epoch"]
        print(f"resuming the latent diffusion policy at epoch {state['epoch']}")

    def steps():
        policy.train()
        for batch in data:
            loss = policy.compute_loss(to_device(batch, device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            ema.step(policy)
            yield loss.item()

    def save(done: bool):
        torch.save({"model": policy.state_dict(), "ema_model": ema_model.state_dict(), "ema_step": ema.optimization_step,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": state["epoch"],
                    "normalizer": normalizer.state_dict(), "done": done}, ckpt)
        if done:
            pca_saved = MarkerPCA.load(pca)
            torch.save({"policy": ema_model.state_dict(), "at": at.state_dict(), "normalizer": normalizer.state_dict(),
                        "pca_mean": pca_saved.mean, "pca_components": pca_saved.components,
                        "shape_meta": dataset.shape_meta, "cameras": dataset.all_cameras, "config": cfg}, bundle)

    run_epochs("ldp", out, t, len(data), state, steps, save)
    return bundle


def parse_set(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        key, value = item.split("=", 1)
        node = out
        *parents, last = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(value)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["at", "ldp"])
    p.add_argument("--cache", required=True)
    p.add_argument("--pca", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--config", default=str(build.CONFIG))
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args(argv)
    cfg = build.load_config(Path(args.config), parse_set(args.set))
    fn = train_at if args.stage == "at" else train_ldp
    fn(cfg, Path(args.cache), Path(args.pca), Path(args.out), args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
