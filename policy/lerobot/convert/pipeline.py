"""The conversion: episodes in, one LeRobot dataset with FTP-1 embeddings out."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from ..tactile import FTP1GelSightEncoder
from .hdf5 import read_episode, select_episode_files
from .schema import (
    ACTION_KEY,
    CAMERA_KEYS,
    DATA_ROOT,
    ENV_STATE_KEY,
    STATE_KEY,
    TACTILE_CAMERAS,
    TACTILE_IMAGE_KEYS,
    build_features,
    instruction_for,
    visual_cameras,
)
from .transforms import bgr_to_rgb, resize_frames, split_transitions, stack_fingertips

Embed = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ConvertOptions:
    fps: int = 60
    embedding: str = "cls"
    device: str = "cuda"
    image_size: int = 256
    tactile_images: bool = False
    use_videos: bool = False
    overwrite: bool = True


@dataclass(frozen=True)
class EpisodeFrames:
    """Everything of one episode in dataset layout, one row per transition."""

    state: np.ndarray  # (N, 8)
    action: np.ndarray  # (N, 8)
    env_state: np.ndarray  # (N, tactile_channels)
    images: dict[str, np.ndarray]  # key -> (N, H, W, 3) RGB uint8

    def __len__(self) -> int:
        return len(self.state)


def prepare_output_dir(out_root: Path, overwrite: bool) -> Path:
    out_root = Path(out_root).expanduser().resolve()
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"LeRobot dataset already exists: {out_root}")
        shutil.rmtree(out_root)
    return out_root


def image_shapes_for(episode: dict, cameras: tuple[str, ...], image_size: int, tactile_images: bool) -> dict:
    """Image feature key -> (h, w): cameras are resized square, raw fingertip images keep their size."""
    shapes = {CAMERA_KEYS[cam]: (image_size, image_size) for cam in cameras}
    if tactile_images:
        for cam in TACTILE_CAMERAS:
            h, w = episode["tactile"][cam].shape[1:3]
            shapes[TACTILE_IMAGE_KEYS[cam]] = (h, w)
    return shapes


def encode_episode(
    episode: dict, cameras: tuple[str, ...], embed: Embed, image_size: int, tactile_images: bool
) -> EpisodeFrames:
    """Pair frames into transitions, resize the RGB cameras, embed both fingertips (BGR in)."""
    state, action = split_transitions(episode["joint"])
    n = len(state)
    images = {CAMERA_KEYS[cam]: resize_frames(bgr_to_rgb(episode["cameras"][cam][:n]), image_size) for cam in cameras}
    if tactile_images:
        for cam in TACTILE_CAMERAS:
            images[TACTILE_IMAGE_KEYS[cam]] = bgr_to_rgb(episode["tactile"][cam][:n])
    env_state = stack_fingertips([embed(episode["tactile"][cam][:n]) for cam in TACTILE_CAMERAS])
    return EpisodeFrames(state=state, action=action, env_state=env_state, images=images)


def write_episode(dataset, frames: EpisodeFrames, instruction: str) -> None:
    for t in range(len(frames)):
        frame = {
            STATE_KEY: frames.state[t],
            ACTION_KEY: frames.action[t],
            ENV_STATE_KEY: frames.env_state[t],
            "task": instruction,
        }
        for key, images in frames.images.items():
            frame[key] = images[t]
        dataset.add_frame(frame)
    dataset.save_episode()


def conversion_metadata(
    task_name: str,
    task_config: str,
    repo_id: str,
    dataset_root: Path,
    options: ConvertOptions,
    cameras: tuple[str, ...],
    tactile_channels: int,
    instruction: str,
    files: list[Path],
    lengths: list[int],
) -> dict:
    """What train.sh and the server need to know about the dataset (source_metadata.json)."""
    return {
        "task_name": task_name,
        "task_config": task_config,
        "repo_id": repo_id,
        "dataset_dir": str(dataset_root),
        "fps": options.fps,
        "instruction": instruction,
        "cameras": list(cameras),
        "image_size": options.image_size,
        "tactile_images": options.tactile_images,
        "tactile_embedding": options.embedding,
        "tactile_channels": tactile_channels,
        "state": "joint[:8]",
        "action": "joint[1:, :8]",
        "num_episodes": len(files),
        "num_frames": int(sum(lengths)),
        "episodes": {i: {"source": str(p), "length": int(n)} for i, (p, n) in enumerate(zip(files, lengths))},
    }


def convert(
    task_name: str, task_config: str, episode_num: int, out_root: Path, repo_id: str, options: ConvertOptions
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = select_episode_files(DATA_ROOT / task_name / task_config, episode_num)
    out_root = prepare_output_dir(out_root, options.overwrite)
    cameras = visual_cameras(task_name)
    instruction = instruction_for(task_name)
    encoder = FTP1GelSightEncoder.from_pretrained(options.embedding, device=options.device)
    tactile_channels = len(TACTILE_CAMERAS) * encoder.embedding_dim

    first = read_episode(files[0], cameras)
    features = build_features(
        image_shapes_for(first, cameras, options.image_size, options.tactile_images),
        tactile_channels,
        options.use_videos,
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=options.fps, features=features, root=out_root, robot_type="vitac_arm",
        use_videos=options.use_videos,
    )

    lengths = []
    for idx, path in enumerate(tqdm(files, desc="episodes")):
        episode = first if idx == 0 else read_episode(path, cameras)
        frames = encode_episode(episode, cameras, encoder.embed, options.image_size, options.tactile_images)
        write_episode(dataset, frames, instruction)
        lengths.append(len(frames))
    dataset.finalize()

    metadata = conversion_metadata(
        task_name, task_config, repo_id, dataset.root, options, cameras, tactile_channels, instruction, files, lengths
    )
    with open(dataset.root / "source_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata
