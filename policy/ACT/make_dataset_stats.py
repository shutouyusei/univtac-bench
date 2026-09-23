"""Write the dataset_stats.pkl a released ACT checkpoint needs at deploy time.

The released checkpoints (data/checkpoints/<task>/{univtac,vision_only}) ship
policy_last.ckpt without the normalisation statistics that act_policy.py reads
from dataset_stats.pkl. This computes them the way process_data.py + utils
get_norm_stats would: over the first N raw episodes of a task config, with
qpos = joint[:-1, :8] and action = joint[1:, :8] (7 arm + 1 gripper, same
frame pairing as envs/utils/data.py batch_gather_hdf5 at downsample 1).

    python policy/ACT/make_dataset_stats.py insert_hole clean51 50 <out_dir>
"""
import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIM = 8


def raw_episode_paths(task_name: str, task_config: str) -> list[Path]:
    # resolve() so a data/ symlink (worktrees) is followed by rglob
    root = (PROJECT_ROOT / "data" / task_name / task_config).resolve()
    return sorted(root.rglob("*.hdf5"), key=lambda p: int(p.stem))


def compute_stats(paths: list[Path]) -> dict:
    qpos, action = [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            joints = f["embodiment/joint"][()][:, :STATE_DIM].astype(np.float32)
        qpos.append(joints[:-1])
        action.append(joints[1:])
    qpos = np.concatenate(qpos)
    action = np.concatenate(action)
    return {
        "action_mean": action.mean(0),
        "action_std": np.clip(action.std(0), 1e-2, np.inf),
        "qpos_mean": qpos.mean(0),
        "qpos_std": np.clip(qpos.std(0), 1e-2, np.inf),
        "example_qpos": qpos[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_name")
    ap.add_argument("task_config")
    ap.add_argument("episode_num", type=int)
    ap.add_argument("out_dir", type=Path, help="checkpoint directory that gets dataset_stats.pkl")
    args = ap.parse_args()

    paths = raw_episode_paths(args.task_name, args.task_config)
    if len(paths) < args.episode_num:
        raise SystemExit(f"{len(paths)} episodes under data/{args.task_name}/{args.task_config}, need {args.episode_num}")
    stats = compute_stats(paths[: args.episode_num])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "dataset_stats.pkl"
    with open(out, "wb") as f:
        pickle.dump(stats, f)
    print(f"wrote {out}")
    for k in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        print(f"{k}: {np.array2string(stats[k], precision=3)}")


if __name__ == "__main__":
    main()
