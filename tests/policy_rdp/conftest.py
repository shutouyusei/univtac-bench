import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "policy" / "RDP"))

from envs.utils.data import HDF5Handler  # noqa: E402


def write_demo(path: Path, T: int, seed: int, cams=("head",)):
    """A UniVTAC-like demonstration: 9 joints, both fingertips' markers, JPEG camera frames (red only)."""
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
            imgs[..., 0] = 200
            data, max_len = HDF5Handler.img_to_stream(imgs)
            f.create_dataset(f"observation/{cam}/rgb", data=np.array(data, dtype=f"S{max_len}"))


TOY = {
    "ldp": {"down_dims": [16, 32, 64], "diffusion_step_embed_dim": 16, "obs_encoder": {"resize_shape": [24, 32]}},
    "at_training": {"num_epochs": 2, "checkpoint_every": 1, "num_workers": 0, "batch_size": 16},
    "ldp_training": {"num_epochs": 2, "checkpoint_every": 1, "num_workers": 0, "batch_size": 16},
}


@pytest.fixture(scope="session")
def rdp_bundle(tmp_path_factory):
    """A toy RDP trained for 2 + 2 epochs on CPU, written as an evaluation bundle."""
    from rdp_univtac import build, prepare, train

    tmp = tmp_path_factory.mktemp("rdp")
    demos = tmp / "hdf5"
    demos.mkdir()
    for i, T in enumerate((30, 26)):
        write_demo(demos / f"{i}.hdf5", T, i)
    cache = tmp / "cache"
    prepare.build_cache(prepare.demo_paths(demos, 2), cameras=["head"], out=cache)
    pca = tmp / "pca.npz"
    prepare.fit_pooled_pca([prepare.demo_paths(demos, 2)]).save(pca)
    cfg = build.load_config(override=TOY)
    train.train_at(cfg, cache, pca, tmp / "out", "cpu")
    return train.train_ldp(cfg, cache, pca, tmp / "out", "cpu")
