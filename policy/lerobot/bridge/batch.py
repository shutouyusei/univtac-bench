"""lerobot side: a wire observation -> the un-batched frame the policy preprocessor expects."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ..convert.schema import CAMERA_KEYS, ENV_STATE_KEY, STATE_DIM, STATE_KEY, TACTILE_CAMERAS

Embed = Callable[[np.ndarray], np.ndarray]


def build_batch(observation: dict, embed: Embed, instruction: str, image_size: int) -> dict:
    """Images become ``(3, image_size, image_size)`` float in [0, 1] (what a LeRobot dataset yields),
    the state is ``joint[:8]``, and the fingertip ``rgb_marker`` frames are embedded as the
    simulator hands them, which is also what the dataset converter fed the encoder."""
    import torch

    batch: dict = {}
    for cam, img in observation["images"].items():
        img = np.asarray(img, dtype=np.uint8)
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        batch[CAMERA_KEYS[cam]] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    batch[STATE_KEY] = torch.as_tensor(np.asarray(observation["joint"], dtype=np.float32).reshape(-1)[:STATE_DIM])
    tactile = np.stack([np.asarray(observation["tactile"][cam], dtype=np.uint8) for cam in TACTILE_CAMERAS])
    batch[ENV_STATE_KEY] = torch.from_numpy(embed(tactile).reshape(-1)).float()
    batch["task"] = instruction
    return batch
