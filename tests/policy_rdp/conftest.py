import sys
from pathlib import Path

import h5py
import numpy as np

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
