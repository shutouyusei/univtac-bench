"""The vendored official RDP modules import in the UniVTAC env (after the import patches) and run end to end at
toy sizes: tokenizer loss, latent diffusion loss, latent sampling, per-step decoding with tactile."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "policy" / "RDP"))

from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from reactive_diffusion_policy.common.normalize_util import get_image_range_normalizer  # noqa: E402
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer  # noqa: E402
from reactive_diffusion_policy.model.vae.model import VAE  # noqa: E402
from reactive_diffusion_policy.model.vision.model_getter import get_resnet  # noqa: E402
from reactive_diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder  # noqa: E402
from reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy import LatentDiffusionUnetImagePolicy  # noqa: E402

A, H, B = 8, 32, 4
SHAPE_META = {
    "obs": {
        "cam_head": {"shape": [3, 24, 32], "type": "rgb"},
        "joint": {"shape": [7], "type": "low_dim"},
        "tactile_left": {"shape": [15], "type": "low_dim"},
    },
    "extended_obs": {"tactile_left": {"shape": [15], "type": "low_dim"}},
    "action": {"shape": [A]},
}


def normalizer():
    n = LinearNormalizer()
    n["action"] = SingleFieldLinearNormalizer.create_fit(torch.randn(100, A))
    n["joint"] = SingleFieldLinearNormalizer.create_fit(torch.randn(100, 7))
    n["tactile_left"] = SingleFieldLinearNormalizer.create_fit(torch.randn(100, 15))
    n["cam_head"] = get_image_range_normalizer()
    return n


def tokenizer():
    at = VAE(horizon=H, shape_meta=SHAPE_META, n_latent_dims=8, use_conv_encoder=True, conv_latent_dims=32,
             conv_layer_num=1, use_rnn_decoder=True, rnn_latent_dims=32, rnn_layer_num=1, kl_multiplier=1e-6,
             n_embed=16, use_vq=False, eval=False, device="cpu")
    at.set_normalizer(normalizer())
    return at


def test_tokenizer_downsamples_32_steps_to_8_tokens_and_trains():
    at = tokenizer()
    assert at.downsampled_input_h == 8
    batch = {"action": torch.randn(B, H, A), "extended_obs": {"tactile_left": torch.randn(B, H, 15)}}
    out = at.compute_loss_and_metric(batch)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])


def test_latent_diffusion_policy_loss_sample_and_decode():
    at = tokenizer()
    encoder = MultiImageObsEncoder(shape_meta=SHAPE_META, rgb_model=get_resnet("resnet18", weights=None),
                                   resize_shape=[24, 32], random_transforms=OmegaConf.create([{"type": "RandomCrop", "ratio": 0.9}]),
                                   use_group_norm=True, share_rgb_model=False, imagenet_norm=True)
    sched = DDIMScheduler(num_train_timesteps=100, beta_start=1e-4, beta_end=0.02, beta_schedule="squaredcos_cap_v2",
                          clip_sample=True, set_alpha_to_one=True, steps_offset=0, prediction_type="epsilon")
    policy = LatentDiffusionUnetImagePolicy(at=at, use_latent_action_before_vq=False, shape_meta=SHAPE_META,
                                            noise_scheduler=sched, obs_encoder=encoder, horizon=H, n_action_steps=29,
                                            n_obs_steps=2, num_inference_steps=8, diffusion_step_embed_dim=16,
                                            down_dims=[16, 32, 64], n_groups=8)
    norm = normalizer()
    norm["latent_action"] = SingleFieldLinearNormalizer.create_fit(torch.randn(100, 16))
    policy.set_normalizer(norm)
    obs = {"cam_head": torch.rand(B, 2, 3, 24, 32), "joint": torch.randn(B, 2, 7), "tactile_left": torch.randn(B, 2, 15)}
    loss = policy.compute_loss({"obs": obs, "action": torch.randn(B, H, A)})
    loss.backward()
    assert torch.isfinite(loss)
    policy.eval()
    with torch.no_grad():
        latent = policy.predict_action(obs, dataset_obs_temporal_downsample_ratio=2, return_latent_action=True)["action"]
        assert latent.shape == (B, 29, 8 * 8)
        out = policy.predict_from_latent_action(latent[:, 0], {"tactile_left": torch.randn(B, 10, 15)}, 10, 2)
    assert out["action_pred"].shape == (B, 10, A)
    assert out["action"].shape == (B, 10 - 3, A)
