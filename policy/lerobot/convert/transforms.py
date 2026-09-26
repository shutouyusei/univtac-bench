"""Frame and joint transforms shared by the converter and the inference server."""

from __future__ import annotations

import cv2
import numpy as np

from .schema import STATE_DIM


def split_transitions(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(T, >=8)`` joint trajectory -> state ``joint[:-1, :8]`` and action ``joint[1:, :8]``.

    Same pairing as UniVTAC's HDF5Handler: the action of frame t is the joint
    reading of frame t+1, so the last frame has no action and is dropped.
    """
    joint = np.asarray(joint, dtype=np.float32)
    if joint.ndim != 2 or joint.shape[0] < 2 or joint.shape[1] < STATE_DIM:
        raise ValueError(f"expected (T>=2, >={STATE_DIM}) joints, got {joint.shape}")
    return joint[:-1, :STATE_DIM], joint[1:, :STATE_DIM]


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """``(N, H, W, 3)`` -> ``(N, size, size, 3)`` with bilinear resampling (UniVTAC's visual_transform)."""
    frames = np.asarray(frames)
    if frames.shape[1:3] == (size, size):
        return frames
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in frames]).astype(np.uint8)


def stack_fingertips(embeddings: list[np.ndarray]) -> np.ndarray:
    """Per-fingertip ``(N, D)`` embeddings -> one ``(N, len * D)`` environment_state per frame."""
    return np.concatenate([np.asarray(e, dtype=np.float32) for e in embeddings], axis=1)
