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

from policy.lerobot.convert.aux_targets import VirtualTargetParams  # noqa: E402
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
    defaults = VirtualTargetParams()
    p.add_argument("--aux-target", choices=("none", "virtual_target"), default="none",
                   help="virtual_target appends VRR's [virtual target 3, stiffness 1] to every action")
    p.add_argument("--force-scale", type=float, default=None,
                   help="newtons per unit of the recorded force (policy/lerobot/force_stats.py prints it)")
    p.add_argument("--f-min", type=float, default=defaults.f_min, help="N")
    p.add_argument("--f-max", type=float, default=defaults.f_max, help="N")
    p.add_argument("--k-min", type=float, default=defaults.k_min, help="N/m")
    p.add_argument("--k-max", type=float, default=defaults.k_max, help="N/m")
    return p.parse_args(argv)


def virtual_target_params(args: argparse.Namespace) -> VirtualTargetParams | None:
    if args.aux_target == "none":
        return None
    if args.force_scale is None:
        raise SystemExit(
            "--aux-target virtual_target needs --force-scale: the recorded force is not in newtons. "
            "Run policy/lerobot/force_stats.py on the dataset first."
        )
    return VirtualTargetParams(
        force_scale=args.force_scale, f_min=args.f_min, f_max=args.f_max, k_min=args.k_min, k_max=args.k_max
    )


def default_repo_id(args: argparse.Namespace) -> str:
    """``local/<task>-<config>-<n>``, with ``-vt`` when the action carries the virtual target."""
    suffix = "-vt" if args.aux_target == "virtual_target" else ""
    return f"local/{args.task_name}-{args.task_config}-{args.episode_num}{suffix}"


def build_options(args: argparse.Namespace) -> ConvertOptions:
    return ConvertOptions(
        fps=args.fps,
        embedding=args.embedding,
        device=args.device,
        image_size=args.image_size,
        tactile_images=args.tactile_images,
        use_videos=args.videos,
        overwrite=not args.no_overwrite,
        virtual_target=virtual_target_params(args),
    )


def main() -> None:
    args = parse_args()
    options = build_options(args)
    repo_id = args.repo_id or default_repo_id(args)
    out_root = args.out or (REPO_ROOT / "policy" / "lerobot" / "data" / repo_id)
    metadata = convert(args.task_name, args.task_config, args.episode_num, out_root, repo_id, options)
    print(json.dumps({k: v for k, v in metadata.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
