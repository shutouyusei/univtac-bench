"""Convert UniVTAC HDF5 episodes into a LeRobot v3 dataset for the lerobot bridge.

Run with the ``lerobot`` conda env's python::

    python policy/lerobot/process_data.py insert_hole clean51 50 [--fps 60] [--embedding cls]

Per frame the dataset holds

* ``observation.images.<camera>``   RGB, ``--image-size`` square (uint8, image files)
* ``observation.state``             joint[:8] (7 arm + gripper), the same slice ACT uses
* ``action``                        joint[:8] of the next frame
* ``observation.environment_state`` FTP-1 embedding of the left and right
                                    fingertip ``rgb_marker`` frames, concatenated
* ``task``                          the task's "seen" instruction

Raw tactile images are written only with ``--tactile-images``: lerobot turns
every image feature of a dataset into a policy camera when a policy is built
from ``--policy.type``, and the tacforcing plugin must see tactile only as the
environment_state vector.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.lerobot.tactile_encoder import EMBEDDINGS, FTP1GelSightEncoder  # noqa: E402

DATA_ROOT = REPO_ROOT / "data"
INSTRUCTIONS_DIR = REPO_ROOT / "instructions"
TASK_SETTINGS = REPO_ROOT / "policy" / "task_settings.json"

STATE_DIM = 8  # joint[:8]: 7 arm joints + gripper; joint[8] mirrors the gripper
TACTILE_CAMERAS = ("left", "right")
CAMERA_KEYS = {"head": "observation.images.head", "wrist": "observation.images.wrist"}
TACTILE_IMAGE_KEYS = {"left": "observation.images.tactile_left", "right": "observation.images.tactile_right"}


# --------------------------------------------------------------------------- pure helpers


def episode_files(root: Path) -> list[Path]:
    """All ``*.hdf5`` under ``root`` in numeric order of their stem (0, 1, 2, ..., 10)."""
    files = list(Path(root).rglob("*.hdf5"))
    try:
        return sorted(files, key=lambda p: int(p.stem))
    except ValueError:
        return sorted(files)


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


def split_transitions(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(T, >=8)`` joint trajectory -> state ``joint[:-1, :8]`` and action ``joint[1:, :8]``.

    Same pairing as UniVTAC's HDF5Handler: the action of frame t is the joint
    reading of frame t+1, so the last frame has no action and is dropped.
    """
    joint = np.asarray(joint, dtype=np.float32)
    if joint.ndim != 2 or joint.shape[0] < 2 or joint.shape[1] < STATE_DIM:
        raise ValueError(f"expected (T>=2, >={STATE_DIM}) joints, got {joint.shape}")
    return joint[:-1, :STATE_DIM], joint[1:, :STATE_DIM]


def decode_frames(buffers) -> np.ndarray:
    """Encoded (JPEG/PNG) byte strings -> ``(N, H, W, 3)`` uint8 in cv2's BGR order."""
    frames = []
    for buf in np.asarray(buffers).ravel():
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("failed to decode an image buffer")
        frames.append(img)
    return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), np.uint8)


def bgr_to_rgb(frames: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(frames)[..., ::-1])


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """``(N, H, W, 3)`` -> ``(N, size, size, 3)`` with bilinear resampling (UniVTAC's visual_transform)."""
    frames = np.asarray(frames)
    if frames.shape[1:3] == (size, size):
        return frames
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in frames]).astype(np.uint8)


def stack_fingertips(embeddings: list[np.ndarray]) -> np.ndarray:
    """Per-fingertip ``(N, D)`` embeddings -> one ``(N, len * D)`` environment_state per frame."""
    return np.concatenate([np.asarray(e, dtype=np.float32) for e in embeddings], axis=1)


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
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": [f"joint_{i}" for i in range(state_dim)],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": [f"joint_{i}" for i in range(action_dim)],
    }
    features["observation.environment_state"] = {
        "dtype": "float32",
        "shape": (int(tactile_channels),),
        "names": [f"tactile_{i}" for i in range(tactile_channels)],
    }
    return features


# --------------------------------------------------------------------------- HDF5 access


def tactile_stream_key(f: h5py.File, cam: str) -> str:
    for prefix in (f"tactile/{cam}_tactile", f"tactile/{cam}_gsmini"):
        if f"{prefix}/rgb_marker" in f:
            return f"{prefix}/rgb_marker"
    raise KeyError(f"no tactile/{cam}_tactile/rgb_marker or tactile/{cam}_gsmini/rgb_marker in {f.filename}")


def read_episode(path: Path, cameras: tuple[str, ...]) -> dict:
    """One HDF5 episode -> joints, BGR camera frames per camera, BGR rgb_marker frames per fingertip."""
    with h5py.File(str(path), "r") as f:
        episode = {
            "joint": np.asarray(f["embodiment/joint"][()], dtype=np.float32),
            "cameras": {cam: decode_frames(f[f"observation/{cam}/rgb"][()]) for cam in cameras},
            "tactile": {cam: decode_frames(f[tactile_stream_key(f, cam)][()]) for cam in TACTILE_CAMERAS},
        }
    return episode


# --------------------------------------------------------------------------- conversion


def convert(
    task_name: str,
    task_config: str,
    episode_num: int,
    out_root: Path,
    repo_id: str,
    fps: int,
    embedding: str,
    device: str,
    image_size: int,
    tactile_images: bool,
    use_videos: bool,
    overwrite: bool,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw_root = DATA_ROOT / task_name / task_config
    files = episode_files(raw_root)
    if len(files) < episode_num:
        raise ValueError(f"requested {episode_num} episodes but found {len(files)} under {raw_root}")
    files = files[:episode_num]

    out_root = Path(out_root).expanduser().resolve()
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"LeRobot dataset already exists: {out_root}")
        shutil.rmtree(out_root)

    cameras = visual_cameras(task_name)
    encoder = FTP1GelSightEncoder.from_pretrained(embedding, device=device)
    tactile_channels = len(TACTILE_CAMERAS) * encoder.embedding_dim
    instruction = instruction_for(task_name)

    first = read_episode(files[0], cameras)
    image_shapes = {CAMERA_KEYS[cam]: (image_size, image_size) for cam in cameras}
    if tactile_images:
        for cam in TACTILE_CAMERAS:
            h, w = first["tactile"][cam].shape[1:3]
            image_shapes[TACTILE_IMAGE_KEYS[cam]] = (h, w)
    features = build_features(image_shapes, tactile_channels, use_videos)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_root,
        robot_type="vitac_arm",
        use_videos=use_videos,
    )

    lengths = []
    for idx, path in enumerate(tqdm(files, desc="episodes")):
        ep = first if idx == 0 else read_episode(path, cameras)
        state, action = split_transitions(ep["joint"])
        n = len(state)
        images = {
            CAMERA_KEYS[cam]: resize_frames(bgr_to_rgb(ep["cameras"][cam][:n]), image_size) for cam in cameras
        }
        if tactile_images:
            for cam in TACTILE_CAMERAS:
                images[TACTILE_IMAGE_KEYS[cam]] = bgr_to_rgb(ep["tactile"][cam][:n])
        env_state = stack_fingertips([encoder.embed(ep["tactile"][cam][:n]) for cam in TACTILE_CAMERAS])
        for t in range(n):
            frame = {
                "observation.state": state[t],
                "action": action[t],
                "observation.environment_state": env_state[t],
                "task": instruction,
            }
            for key, frames in images.items():
                frame[key] = frames[t]
            dataset.add_frame(frame)
        dataset.save_episode()
        lengths.append(n)
    dataset.finalize()

    metadata = {
        "task_name": task_name,
        "task_config": task_config,
        "repo_id": repo_id,
        "dataset_dir": str(dataset.root),
        "fps": fps,
        "instruction": instruction,
        "cameras": list(cameras),
        "tactile_images": tactile_images,
        "tactile_embedding": embedding,
        "tactile_channels": tactile_channels,
        "state": "joint[:8]",
        "action": "joint[1:, :8]",
        "num_episodes": len(files),
        "num_frames": int(sum(lengths)),
        "episodes": {i: {"source": str(p), "length": int(n)} for i, (p, n) in enumerate(zip(files, lengths))},
    }
    with open(dataset.root / "source_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_name")
    p.add_argument("task_config")
    p.add_argument("episode_num", type=int)
    p.add_argument("--out", type=Path, default=None, help="dataset root (default policy/lerobot/data/<repo-id>)")
    p.add_argument("--repo-id", default=None, help="default local/<task>-<config>-<n>")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--embedding", choices=EMBEDDINGS, default="cls")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--tactile-images", action="store_true", help="also store raw fingertip images")
    p.add_argument("--videos", action="store_true", help="store images as MP4 instead of image files")
    p.add_argument("--no-overwrite", action="store_true")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    repo_id = args.repo_id or f"local/{args.task_name}-{args.task_config}-{args.episode_num}"
    out_root = args.out or (REPO_ROOT / "policy" / "lerobot" / "data" / repo_id)
    metadata = convert(
        task_name=args.task_name,
        task_config=args.task_config,
        episode_num=args.episode_num,
        out_root=out_root,
        repo_id=repo_id,
        fps=args.fps,
        embedding=args.embedding,
        device=args.device,
        image_size=args.image_size,
        tactile_images=args.tactile_images,
        use_videos=args.videos,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
