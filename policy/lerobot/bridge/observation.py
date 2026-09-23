"""Isaac side: reduce a UniVTAC observation to what the server needs (numpy only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..convert.schema import STATE_DIM, TASK_SETTINGS

TACTILE_ALIASES = {"left": ("left_tactile", "left_gsmini"), "right": ("right_tactile", "right_gsmini")}


def as_uint8_image(img) -> np.ndarray:
    """HWC image as the simulator hands it (torch or numpy, uint8 or float) -> HWC uint8 numpy."""
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"expected an HWC image, got shape {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def select_observation(observation: dict, cameras: tuple[str, ...]) -> dict:
    """Keep only camera RGB, fingertip rgb_marker and joint[:8]; raise KeyError on a missing stream."""
    images = {}
    for cam in cameras:
        cam_obs = observation.get("observation", {}).get(cam)
        if cam_obs is None or "rgb" not in cam_obs:
            raise KeyError(f"observation missing observation.{cam}.rgb")
        images[cam] = as_uint8_image(cam_obs["rgb"])

    tactile = {}
    tactile_obs = observation.get("tactile", {})
    for side, aliases in TACTILE_ALIASES.items():
        for alias in aliases:
            if alias in tactile_obs and "rgb_marker" in tactile_obs[alias]:
                tactile[side] = as_uint8_image(tactile_obs[alias]["rgb_marker"])
                break
        if side not in tactile:
            raise KeyError(f"observation missing tactile.{aliases[0]}.rgb_marker")

    joint = observation["embodiment"]["joint"]
    if hasattr(joint, "detach"):
        joint = joint.detach().cpu().numpy()
    joint = np.asarray(joint, dtype=np.float32).reshape(-1)[:STATE_DIM]
    return {"images": images, "tactile": tactile, "joint": joint}


def cameras_for_task(task_name: str, task_settings: Path = TASK_SETTINGS) -> tuple[str, ...]:
    camera_type = "head"
    if Path(task_settings).exists():
        with open(task_settings) as f:
            camera_type = json.load(f).get(task_name, {}).get("camera_type", "head")
    return ("head", "wrist") if camera_type == "all" else (camera_type,)
