"""Reading UniVTAC HDF5 episodes without ``envs.utils`` (which needs scipy/transforms3d)."""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np

from .schema import TACTILE_CAMERAS


def episode_files(root: Path) -> list[Path]:
    """All ``*.hdf5`` under ``root`` in numeric order of their stem (0, 1, 2, ..., 10)."""
    files = list(Path(root).rglob("*.hdf5"))
    try:
        return sorted(files, key=lambda p: int(p.stem))
    except ValueError:
        return sorted(files)


def select_episode_files(root: Path, episode_num: int) -> list[Path]:
    files = episode_files(root)
    if len(files) < episode_num:
        raise ValueError(f"requested {episode_num} episodes but found {len(files)} under {root}")
    return files[:episode_num]


def decode_frames(buffers) -> np.ndarray:
    """Encoded (JPEG/PNG) byte strings -> ``(N, H, W, 3)`` uint8.

    UniVTAC's writer runs ``cv2.imencode`` on the simulator's RGB array without
    converting it, so decoding returns that array: the frames are RGB, not BGR.
    """
    frames = []
    for buf in np.asarray(buffers).ravel():
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("failed to decode an image buffer")
        frames.append(img)
    return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), np.uint8)


def tactile_stream_key(f: h5py.File, cam: str) -> str:
    for prefix in (f"tactile/{cam}_tactile", f"tactile/{cam}_gsmini"):
        if f"{prefix}/rgb_marker" in f:
            return f"{prefix}/rgb_marker"
    raise KeyError(f"no tactile/{cam}_tactile/rgb_marker or tactile/{cam}_gsmini/rgb_marker in {f.filename}")


def read_episode(path: Path, cameras: tuple[str, ...]) -> dict:
    """One HDF5 episode -> joints, RGB camera frames per camera, RGB rgb_marker frames per fingertip."""
    with h5py.File(str(path), "r") as f:
        return {
            "joint": np.asarray(f["embodiment/joint"][()], dtype=np.float32),
            "cameras": {cam: decode_frames(f[f"observation/{cam}/rgb"][()]) for cam in cameras},
            "tactile": {cam: decode_frames(f[tactile_stream_key(f, cam)][()]) for cam in TACTILE_CAMERAS},
        }
