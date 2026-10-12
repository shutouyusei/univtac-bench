"""The official RDP components built from rdp_config.yml, without Hydra (same classes and arguments as the
official workspace configs)."""

from __future__ import annotations

import copy
from pathlib import Path

import einops
import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf

from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.model.vision.model_getter import get_resnet
from reactive_diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy import LatentDiffusionUnetImagePolicy

CONFIG = Path(__file__).resolve().parents[1] / "rdp_config.yml"


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path = CONFIG, override: dict | None = None) -> dict:
    return deep_update(yaml.safe_load(Path(path).read_text()), override or {})


def build_at(cfg: dict, shape_meta: dict, device: str) -> VAE:
    a = cfg["at"]
    return VAE(horizon=cfg["horizon"], shape_meta=copy.deepcopy(shape_meta), n_latent_dims=a["n_latent_dims"],
               mlp_layer_num=a["mlp_layer_num"], use_conv_encoder=a["use_conv_encoder"],
               conv_latent_dims=a["conv_latent_dims"], conv_layer_num=a["conv_layer_num"],
               use_rnn_decoder=a["use_rnn_decoder"], rnn_latent_dims=a["rnn_latent_dims"],
               rnn_layer_num=a["rnn_layer_num"], use_vq=a["use_vq"], n_embed=a["n_embed"],
               vqvae_groups=a["vqvae_groups"], kl_multiplier=a["kl_multiplier"], eval=False, device=device,
               act_scale=a["act_scale"])


def n_action_steps(cfg: dict) -> int:
    """horizon - n_obs_steps * ratio + 1 (official ``n_action_steps``)."""
    return cfg["horizon"] - cfg["n_obs_steps"] * cfg["dataset_obs_temporal_downsample_ratio"] + 1


def build_policy(cfg: dict, shape_meta: dict, at: VAE) -> LatentDiffusionUnetImagePolicy:
    p, e = cfg["ldp"], cfg["ldp"]["obs_encoder"]
    meta = copy.deepcopy(shape_meta)
    encoder = MultiImageObsEncoder(
        shape_meta=meta, rgb_model=get_resnet(e["rgb_model"], weights=e["weights"]),
        resize_shape=list(e["resize_shape"]),
        random_transforms=OmegaConf.create([{"type": "RandomCrop", "ratio": e["random_crop_ratio"]}]),
        use_group_norm=e["use_group_norm"], share_rgb_model=e["share_rgb_model"], imagenet_norm=e["imagenet_norm"])
    return LatentDiffusionUnetImagePolicy(
        at=at, use_latent_action_before_vq=False, shape_meta=meta,
        noise_scheduler=DDIMScheduler(**p["noise_scheduler"]), obs_encoder=encoder, horizon=cfg["horizon"],
        n_action_steps=n_action_steps(cfg), n_obs_steps=cfg["n_obs_steps"],
        num_inference_steps=p["num_inference_steps"], obs_as_global_cond=True,
        diffusion_step_embed_dim=p["diffusion_step_embed_dim"], down_dims=list(p["down_dims"]),
        n_groups=p["n_groups"], cond_predict_scale=p["cond_predict_scale"])


@torch.no_grad()
def latent_normalizer(at: VAE, normalizer: LinearNormalizer, actions: np.ndarray,
                      batch_size: int = 1024) -> SingleFieldLinearNormalizer:
    """Official RealImageTactileLatentDiffusionDataset.get_normalizer: every training window's action through
    the frozen tokenizer (encoder -> quant -> posterior sample), per-channel min-max over all latent tokens."""
    at.eval()
    out = []
    for i in range(0, len(actions), batch_size):
        a = torch.as_tensor(actions[i:i + batch_size], dtype=torch.float32, device=at.device)
        a = normalizer["action"].normalize(a)
        latent = at.encoder(at.preprocess(a / at.act_scale))
        latent, _ = at.quant_state_without_vq(latent)
        latent = einops.rearrange(latent, "N (T A) -> N T A", T=at.downsampled_input_h)
        out.append(latent.reshape(-1, latent.shape[-1]).cpu().numpy())
    return SingleFieldLinearNormalizer.create_fit(np.concatenate(out))
