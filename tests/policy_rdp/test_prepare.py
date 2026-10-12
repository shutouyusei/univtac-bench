"""Demonstration hdf5 -> RDP cache -> buffer, and the pooled marker PCA."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "policy" / "RDP"))

from envs.utils.data import HDF5Handler  # noqa: E402
from rdp_univtac import marker, prepare  # noqa: E402


def write_demo(path: Path, T: int, seed: int, cams=("head",)):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f["embodiment/joint"] = rng.normal(size=(T, 9)).astype(np.float32)
        for side in ("left", "right"):
            m = np.empty((T, 2, 63, 2), np.float32)
            m[:, 0] = rng.uniform(0, 300, (63, 2))
            m[:, 1] = m[:, 0] + rng.normal(0, 2, (T, 63, 2))
            f[f"tactile/{side}_tactile/marker"] = m
        for cam in cams:
            imgs = np.zeros((T, 270, 480, 3), np.uint8)
            imgs[..., 0] = 200  # red channel only: catches a channel swap
            data, max_len = HDF5Handler.img_to_stream(imgs)
            f.create_dataset(f"observation/{cam}/rgb", data=np.array(data, dtype=f"S{max_len}"))


@pytest.fixture
def demos(tmp_path):
    d = tmp_path / "hdf5"
    d.mkdir()
    for i, T in enumerate((12, 9, 15)):
        write_demo(d / f"{i}.hdf5", T, i, cams=("head", "wrist"))
    return d


def test_resize_frame_keeps_rgb_and_gives_240x320():
    img = np.zeros((270, 480, 3), np.uint8)
    img[..., 0] = 200
    out = prepare.resize_frame(img)
    assert out.shape == (240, 320, 3) and out[..., 0].mean() == 200 and out[..., 2].max() == 0


def test_demo_paths_are_sorted_numerically_and_cut_to_n(demos):
    paths = prepare.demo_paths(demos, 2)
    assert [p.stem for p in paths] == ["0", "1"]
    with pytest.raises(ValueError):
        prepare.demo_paths(demos, 4)


def test_cache_roundtrip(demos, tmp_path):
    out = tmp_path / "cache"
    prepare.build_cache(prepare.demo_paths(demos, 3), cameras=["head", "wrist"], out=out)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["lengths"] == [12, 9, 15] and meta["cameras"] == ["cam_head", "cam_wrist"]
    pca = marker.fit_pca(np.random.default_rng(0).normal(size=(500, 126)))
    buf = prepare.load_buffer(out, pca)
    assert list(buf.episode_ends) == [12, 21, 36]
    assert buf["cam_head"].shape == (36, 240, 320, 3) and buf["cam_head"][..., 0].mean() > 190
    assert buf["tactile_left"].shape == (36, 15)
    with h5py.File(demos / "1.hdf5") as f:
        j = f["embodiment/joint"][:]
        m = f["tactile/right_tactile/marker"][:]
    assert np.allclose(buf["joint"][12:21], j[:, :8])
    assert np.allclose(buf["tactile_right"][12:21], pca.embed(marker.displacement(m)), atol=1e-5)


def test_pooled_pca_uses_both_fingertips_of_every_demo(demos):
    pca = prepare.fit_pooled_pca([prepare.demo_paths(demos, 3)], n_components=15)
    assert pca.components.shape == (126, 15)
    x = []
    for p in prepare.demo_paths(demos, 3):
        with h5py.File(p) as f:
            x += [marker.displacement(f[f"tactile/{s}_tactile/marker"][:]) for s in ("left", "right")]
    assert np.allclose(pca.mean, np.concatenate(x).mean(0), atol=1e-6)
