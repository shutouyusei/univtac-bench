"""Marker displacement -> 15-D PCA embedding, as RDP's official GelSight pipeline (offset = current - reference,
normalised by the image size, flattened x0, y0, x1, y1, ..., emb = (x - mean) @ W)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "policy" / "RDP"))

from rdp_univtac import marker  # noqa: E402


def frames(T=5, n=63, seed=0):
    rng = np.random.default_rng(seed)
    init = rng.uniform(0, 300, (n, 2))
    m = np.empty((T, 2, n, 2), np.float32)
    m[:, 0] = init
    m[:, 1] = init + rng.normal(0, 3, (T, n, 2))
    return m


def test_displacement_is_normalised_current_minus_reference_flattened_x_y():
    m = frames()
    d = marker.displacement(m, image_size=(320, 240))
    assert d.shape == (5, 126) and d.dtype == np.float32
    assert d[2, 0] == pytest.approx((m[2, 1, 0, 0] - m[2, 0, 0, 0]) / 320, rel=1e-5)
    assert d[2, 1] == pytest.approx((m[2, 1, 0, 1] - m[2, 0, 0, 1]) / 240, rel=1e-5)
    assert d[2, 3] == pytest.approx((m[2, 1, 1, 1] - m[2, 0, 1, 1]) / 240, rel=1e-5)


def test_displacement_rejects_other_layouts():
    with pytest.raises(ValueError):
        marker.displacement(np.zeros((5, 63, 2)), image_size=(320, 240))


def test_pca_matches_the_principal_subspace():
    rng = np.random.default_rng(1)
    basis = np.linalg.qr(rng.normal(size=(126, 3)))[0]
    x = rng.normal(size=(2000, 3)) * np.array([5.0, 2.0, 1.0]) @ basis.T + 0.3 + rng.normal(0, 1e-3, (2000, 126))
    pca = marker.fit_pca(x, n_components=15)
    assert pca.mean.shape == (126,) and pca.components.shape == (126, 15)
    assert np.allclose(pca.components.T @ pca.components, np.eye(15), atol=1e-5)
    # first three components span the generating subspace, in decreasing variance
    proj = basis.T @ pca.components[:, :3]
    assert np.allclose(np.abs(np.linalg.det(proj)), 1.0, atol=1e-3)
    emb = pca.embed(x)
    var = emb.var(0)
    assert emb.shape == (2000, 15) and np.all(np.diff(var[:3]) < 0)
    assert np.allclose(emb, (x - pca.mean) @ pca.components, atol=1e-5)


def test_pca_save_load_roundtrip(tmp_path):
    x = np.random.default_rng(2).normal(size=(300, 126))
    pca = marker.fit_pca(x, n_components=15)
    pca.save(tmp_path / "pca.npz")
    back = marker.MarkerPCA.load(tmp_path / "pca.npz")
    assert np.allclose(back.embed(x), pca.embed(x))
