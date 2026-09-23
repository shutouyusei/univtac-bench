"""FTP-1 encoder: preprocessing contract, output shapes, and strict loading of the released weights."""

import numpy as np
import pytest
import torch

from policy.lerobot.tactile import ftp1_encoder as te


def test_preprocess_resizes_scales_and_permutes():
    imgs = np.zeros((2, 240, 320, 3), np.uint8)
    imgs[0] = 255
    x = te.preprocess(imgs)
    assert x.shape == (2, 3, 224, 224)
    assert x.dtype == torch.float32
    assert torch.allclose(x[0], torch.ones_like(x[0]))
    assert torch.allclose(x[1], -torch.ones_like(x[1]))


def test_preprocess_rejects_non_image_input():
    with pytest.raises(ValueError):
        te.preprocess(np.zeros((224, 224, 3), np.uint8))


def test_rgb_to_bgr_swaps_channels():
    rgb = np.zeros((1, 2, 2, 3), np.uint8)
    rgb[..., 0] = 10
    rgb[..., 2] = 30
    bgr = te.rgb_to_bgr(rgb)
    assert bgr[..., 0].max() == 30 and bgr[..., 2].max() == 10
    assert bgr.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("embedding,dim", [("cls", 768), ("proj", 512)])
def test_forward_shape_per_embedding(embedding, dim):
    enc = te.FTP1GelSightEncoder(embedding).eval()
    assert enc.embedding_dim == dim
    with torch.no_grad():
        out = enc(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, dim)


def test_unknown_embedding_rejected():
    with pytest.raises(ValueError):
        te.FTP1GelSightEncoder("mean")


def test_embed_batches_and_returns_float32():
    enc = te.FTP1GelSightEncoder("cls").eval()
    imgs = np.random.randint(0, 255, (5, 240, 320, 3), np.uint8)
    out = enc.embed(imgs, batch_size=2)
    assert out.shape == (5, 768) and out.dtype == np.float32
    assert enc.embed(imgs[:0]).shape == (0, 768)


def _cached(filename):
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(te.FTP1_REPO_ID, filename, local_files_only=True)
    except Exception:
        return None


@pytest.mark.skipif(
    _cached(te.SENSOR_WEIGHTS) is None or _cached(te.SHARED_WEIGHTS) is None,
    reason="FTP-1 weights not in the HF cache",
)
def test_released_weights_load_strictly_and_are_frozen():
    enc = te.FTP1GelSightEncoder.from_pretrained("cls")
    assert not any(p.requires_grad for p in enc.parameters())
    assert not enc.training
    a = np.random.randint(0, 255, (1, 240, 320, 3), np.uint8)
    b = np.full((1, 240, 320, 3), 128, np.uint8)
    ea, eb = enc.embed(a), enc.embed(b)
    assert np.isfinite(ea).all()
    assert not np.allclose(ea, eb)
    assert np.allclose(enc.embed(a), ea)
