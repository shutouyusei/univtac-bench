"""Force statistics of a collected dataset and what VRR's targets make of it.

    python policy/lerobot/force_stats.py insert_hole clean51_force

Reads ``data/<task>/<config>/hdf5/*.hdf5`` (episodes that hold the fingertip
``force``) and prints

* the percentiles of the summed contact force [N];
* how its direction changes from frame to frame per magnitude bin, and the
  lowest magnitude from which the direction is stable;
* for the given VRR constants (default: ImplicitRDP's): the share of frames
  below ``f_min``, in the linear range and above ``f_max``, and the percentiles of
  the virtual target's offset from the end effector.

Run it before converting with ``--aux-target virtual_target``; ``--json`` writes
the report, so the numbers the targets were judged by stay on record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from policy.lerobot.convert.aux_targets import VirtualTargetParams  # noqa: E402
from policy.lerobot.convert.force_stats import describe_targets, summarize  # noqa: E402
from policy.lerobot.convert.hdf5 import episode_files, read_contact  # noqa: E402
from policy.lerobot.convert.schema import DATA_ROOT  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    defaults = VirtualTargetParams()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_name")
    p.add_argument("task_config")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="directory holding the episodes (default data/<task_name>/<task_config>)")
    p.add_argument("--max-median-deg", type=float, default=10.0,
                   help="a magnitude bin is stable when its median frame-to-frame direction change is at most this")
    p.add_argument("--min-count", type=int, default=30, help="bins with fewer frame pairs do not decide")
    p.add_argument("--per-decade", type=int, default=4, help="magnitude bins per factor of ten")
    p.add_argument("--force-scale", type=float, default=defaults.force_scale, help="newtons per recorded unit")
    p.add_argument("--f-min", type=float, default=defaults.f_min, help="N")
    p.add_argument("--f-max", type=float, default=defaults.f_max, help="N")
    p.add_argument("--k-min", type=float, default=defaults.k_min, help="N/m")
    p.add_argument("--k-max", type=float, default=defaults.k_max, help="N/m")
    p.add_argument("--json", type=Path, default=None, help="write the report here")
    return p.parse_args(argv)


def load_forces(dataset_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    forces, ee = [], []
    for path in episode_files(dataset_dir):
        with h5py.File(path, "r") as f:
            contact = read_contact(f)
        if contact["force"] is None:
            raise ValueError(f"{path} holds no fingertip force")
        forces.append(contact["force"])
        ee.append(contact["ee_pos"])
    return forces, ee


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or (DATA_ROOT / args.task_name / args.task_config)
    params = VirtualTargetParams(
        force_scale=args.force_scale, f_min=args.f_min, f_max=args.f_max, k_min=args.k_min, k_max=args.k_max
    )
    forces, ee = load_forces(dataset_dir)
    report = summarize(forces, args.per_decade, args.max_median_deg, args.min_count)
    report["dataset"] = str(dataset_dir)
    report["targets"] = describe_targets(forces, ee, params)

    print(f"{report['episodes']} episodes, {report['frames']} frames from {dataset_dir}")
    pct = report["force_percentiles"]
    print(f"|f| recorded: p50 {pct['p50']:.3g}  p90 {pct['p90']:.3g}  p99 {pct['p99']:.3g}  max {pct['max']:.3g}\n")
    print("| |f| from | to | frame pairs | median change [deg] | p90 [deg] |")
    print("|---|---|---|---|---|")
    for row in report["table"]:
        print(f"| {row['low']:.3g} | {row['high']:.3g} | {row['count']} | {row['median_deg']:.1f} | {row['p90_deg']:.1f} |")
    threshold = report["stable_force_threshold"]
    stable = "no magnitude is stable" if threshold is None else f"stable from |f| = {threshold:.3g}"
    print(f"\ndirection: {stable} (median change <= {args.max_median_deg:g} deg, bins of >= {args.min_count} pairs)")

    t = report["targets"]
    off = t["offset_mm"]
    print(f"\nvirtual target with force_scale {params.force_scale:g}, f_min {params.f_min:g} N, "
          f"f_max {params.f_max:g} N, k {params.k_max:g} -> {params.k_min:g} N/m:")
    print(f"  frames below f_min {t['share_below_f_min']:.0%}, between {t['share_between']:.0%}, "
          f"above f_max {t['share_above_f_max']:.0%}")
    print(f"  offset from the end effector [mm]: p50 {off['p50']:.3g}  p90 {off['p90']:.3g}"
          f"  p99 {off['p99']:.3g}  max {off['max']:.3g}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
