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
environment_state vector. The work is in ``policy/lerobot/convert/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.lerobot.convert.pipeline import ConvertOptions, convert  # noqa: E402
from policy.lerobot.tactile import EMBEDDINGS  # noqa: E402


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
    options = ConvertOptions(
        fps=args.fps,
        embedding=args.embedding,
        device=args.device,
        image_size=args.image_size,
        tactile_images=args.tactile_images,
        use_videos=args.videos,
        overwrite=not args.no_overwrite,
    )
    metadata = convert(args.task_name, args.task_config, args.episode_num, out_root, repo_id, options)
    print(json.dumps({k: v for k, v in metadata.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
