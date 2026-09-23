"""Frozen FTP-1 tactile encoder for GelSight Mini images.

FTP-1 (arXiv 2606.13102, ``michaelyuancb/ftp1-policy``) tokenises an image-type
tactile sensor with a sensor-specific ViT (3 blocks) followed by a shared T3
trunk (9 blocks + LayerNorm); the trunk's [CLS] token is the per-fingertip
representation and ``image_proj`` maps it into the policy's token width.
This module rebuilds exactly that stack from ``timm`` blocks and loads the
released ``hpt_tokenizer`` safetensors, so the embedding can be computed in
the ``lerobot`` env without FTP-1's openpi/JAX dependencies.

The encoder is frozen: it turns each fingertip image into one vector that the
dataset converter stores as ``observation.environment_state`` and the
inference server recomputes on live frames.

Input convention (matches FTP-1's UniVTAC pipeline): the ``rgb_marker`` frame
in **BGR** channel order (FTP-1 decodes its training zarr with cv2), resized
to 224x224, scaled to [-1, 1].
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

FTP1_REPO_ID = "MJJJJ1064/ftp1_v0426_50kstep"
SENSOR_WEIGHTS = "hpt_tokenizer/GelSightMini_image_224_224_3.safetensors"
SHARED_WEIGHTS = "hpt_tokenizer/shared_image_chunk_encoder.safetensors"

IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768
NUM_HEADS = 12
MLP_RATIO = 4
SENSOR_DEPTH = 3
SHARED_DEPTH = 9
PROJ_DIM = 512

EMBEDDINGS = ("cls", "proj")


def _blocks(depth: int, norm_layer) -> nn.ModuleList:
    from timm.models.vision_transformer import Block

    return nn.ModuleList(
        Block(EMBED_DIM, NUM_HEADS, MLP_RATIO, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)
    )


class _SensorViT(nn.Module):
    """FTP-1's ``ViTEncoder``: patch embedding, [CLS], position embedding, 3 blocks, no final norm."""

    def __init__(self) -> None:
        super().__init__()
        from timm.models.vision_transformer import PatchEmbed

        self.patch_embed = PatchEmbed(
            img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_chans=3, embed_dim=EMBED_DIM
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, EMBED_DIM))
        self.blocks = _blocks(SENSOR_DEPTH, partial(nn.LayerNorm, eps=1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return x


class _SharedTrunk(nn.Module):
    """FTP-1's ``TransformerTrunk`` (T3 shared chunk encoder): 9 blocks and a final LayerNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = _blocks(SHARED_DEPTH, nn.LayerNorm)
        self.norm = nn.LayerNorm(EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class FTP1GelSightEncoder(nn.Module):
    """GelSight Mini image -> fixed vector, weights from the FTP-1 pretrain checkpoint.

    ``embedding="cls"`` returns the 768-d [CLS] token after the shared trunk's
    LayerNorm (the T3 representation); ``"proj"`` returns the 512-d output of
    FTP-1's ``image_proj`` on top of it.
    """

    def __init__(self, embedding: str = "cls") -> None:
        super().__init__()
        if embedding not in EMBEDDINGS:
            raise ValueError(f"embedding must be one of {EMBEDDINGS}, got {embedding!r}")
        self.embedding = embedding
        # Attribute names mirror the safetensors key prefixes so the released
        # files load with strict=True.
        self.vit_encoder = _SensorViT()
        self.shared_chunk_encoder = _SharedTrunk()
        self.image_proj = nn.Linear(EMBED_DIM, PROJ_DIM)

    @property
    def embedding_dim(self) -> int:
        return EMBED_DIM if self.embedding == "cls" else PROJ_DIM

    @classmethod
    def from_pretrained(
        cls,
        embedding: str = "cls",
        sensor_weights: str | Path | None = None,
        shared_weights: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> "FTP1GelSightEncoder":
        """Build the encoder, load both FTP-1 files (downloaded from HF if not given), freeze it."""
        from safetensors.torch import load_file

        sensor_weights = sensor_weights or _download(SENSOR_WEIGHTS)
        shared_weights = shared_weights or _download(SHARED_WEIGHTS)
        encoder = cls(embedding)
        state = {**load_file(str(sensor_weights)), **load_file(str(shared_weights))}
        encoder.load_state_dict(state, strict=True)
        encoder.requires_grad_(False)
        return encoder.to(device).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 3, 224, 224)`` float in [-1, 1] -> ``(N, embedding_dim)``."""
        tokens = self.shared_chunk_encoder(self.vit_encoder(x))
        cls = tokens[:, 0]
        return cls if self.embedding == "cls" else self.image_proj(cls)

    @torch.no_grad()
    def embed(self, images_bgr: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """``(N, H, W, 3)`` uint8 BGR frames -> ``(N, embedding_dim)`` float32."""
        device = next(self.parameters()).device
        out = []
        for start in range(0, len(images_bgr), batch_size):
            x = preprocess(images_bgr[start : start + batch_size]).to(device)
            out.append(self(x).float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.embedding_dim), np.float32)


def _download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(FTP1_REPO_ID, filename)


def preprocess(images_bgr: np.ndarray) -> torch.Tensor:
    """``(N, H, W, 3)`` uint8 BGR -> ``(N, 3, 224, 224)`` float32 in [-1, 1], as FTP-1 feeds its tokenizer."""
    images_bgr = np.asarray(images_bgr)
    if images_bgr.ndim != 4 or images_bgr.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) images, got {images_bgr.shape}")
    if images_bgr.shape[1:3] != (IMAGE_SIZE, IMAGE_SIZE):
        images_bgr = np.stack([cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in images_bgr])
    x = torch.from_numpy(np.ascontiguousarray(images_bgr)).float() / 255.0 * 2.0 - 1.0
    return x.permute(0, 3, 1, 2).contiguous()


def rgb_to_bgr(images: np.ndarray) -> np.ndarray:
    """Swap the channel order of ``(..., 3)`` images; RGB frames from the simulator become FTP-1's BGR."""
    return np.ascontiguousarray(np.asarray(images)[..., ::-1])
