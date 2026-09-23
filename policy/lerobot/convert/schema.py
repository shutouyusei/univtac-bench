"""Dataset contract of the bridge: keys, dimensions, feature schema, task text, cameras."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
INSTRUCTIONS_DIR = REPO_ROOT / "instructions"
TASK_SETTINGS = REPO_ROOT / "policy" / "task_settings.json"

STATE_DIM = 8  # joint[:8]: 7 arm joints + gripper; joint[8] mirrors the gripper
TACTILE_CAMERAS = ("left", "right")
CAMERA_KEYS = {"head": "observation.images.head", "wrist": "observation.images.wrist"}
TACTILE_IMAGE_KEYS = {"left": "observation.images.tactile_left", "right": "observation.images.tactile_right"}
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ENV_STATE_KEY = "observation.environment_state"


def instruction_for(task_name: str, instructions_dir: Path = INSTRUCTIONS_DIR) -> str:
    """The first "seen" instruction of ``instructions/<task>.json``, else the task name as words."""
    path = Path(instructions_dir) / f"{task_name}.json"
    if path.exists():
        with open(path) as f:
            seen = json.load(f).get("seen", [])
        if seen:
            return str(seen[0])
    return task_name.replace("_", " ")


def visual_cameras(task_name: str, task_settings: Path = TASK_SETTINGS) -> tuple[str, ...]:
    """Cameras the task's ACT recipe uses (``camera_type`` in task_settings.json): head, or head + wrist."""
    camera_type = "head"
    if Path(task_settings).exists():
        with open(task_settings) as f:
            camera_type = json.load(f).get(task_name, {}).get("camera_type", "head")
    return ("head", "wrist") if camera_type == "all" else (camera_type,)


def build_features(
    image_shapes: dict[str, tuple[int, int]],
    tactile_channels: int,
    use_videos: bool,
    state_dim: int = STATE_DIM,
    action_dim: int = STATE_DIM,
) -> dict:
    """LeRobot feature schema: one image feature per camera, state, action, environment_state."""
    features: dict = {}
    for key, (h, w) in image_shapes.items():
        features[key] = {
            "dtype": "video" if use_videos else "image",
            "shape": (3, int(h), int(w)),
            "names": ["channel", "height", "width"],
        }
    features[STATE_KEY] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": [f"joint_{i}" for i in range(state_dim)],
    }
    features[ACTION_KEY] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": [f"joint_{i}" for i in range(action_dim)],
    }
    features[ENV_STATE_KEY] = {
        "dtype": "float32",
        "shape": (int(tactile_channels),),
        "names": [f"tactile_{i}" for i in range(tactile_channels)],
    }
    return features
