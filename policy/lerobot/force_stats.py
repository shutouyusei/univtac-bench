"""Force statistics of a collected dataset and the force unit VRR should use.

    python policy/lerobot/force_stats.py insert_hole clean51_force

Reads ``data/<task>/<config>/hdf5/*.hdf5`` (episodes that hold the fingertip
``force``), prints how the direction of the summed contact force changes from
frame to frame per magnitude bin and the lowest magnitude from which it is
stable. It then proposes two force scales and shows what each does to the
virtual target:

* the scale that maps the stable magnitude to the paper's ``f_min``;
* the scale that maps the dataset's 90th percentile to ``--reference-p90`` newtons.

Run it before converting with ``--aux-target virtual_target`` and pass the chosen
scale on as ``--force-scale``; ``--json`` writes the full report.
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
from policy.lerobot.convert.force_stats import describe_scale, summarize  # noqa: E402
from policy.lerobot.convert.hdf5 import episode_files, read_contact  # noqa: E402
from policy.lerobot.convert.schema import DATA_ROOT  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_name")
    p.add_argument("task_config")
    p.add_argument("--max-median-deg", type=float, default=10.0,
                   help="a magnitude bin is stable when its median frame-to-frame direction change is at most this")
    p.add_argument("--min-count", type=int, default=30, help="bins with fewer frame pairs do not decide")
    p.add_argument("--per-decade", type=int, default=4, help="magnitude bins per factor of ten")
    p.add_argument("--paper-f-min", type=float, default=VirtualTargetParams().f_min)
    p.add_argument("--reference-p90", type=float, default=12.0,
                   help="newtons at the 90th percentile of |f| in ImplicitRDP's released episodes (10.6 and 12.8)")
    p.add_argument("--json", type=Path, default=None, help="write the report here")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="directory holding the episodes (default data/<task_name>/<task_config>)")
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


def print_scale(label: str, description: dict) -> None:
    off = description["offset_mm"]
    print(f"\n{label}: --force-scale {description['force_scale']:.6g}")
    print(f"  frames below f_min {description['share_below_f_min']:.0%}, between "
          f"{description['share_between']:.0%}, above f_max {description['share_above_f_max']:.0%}")
    print(f"  virtual target offset [mm]: p50 {off['p50']:.3g}  p90 {off['p90']:.3g}"
          f"  p99 {off['p99']:.3g}  max {off['max']:.3g}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or (DATA_ROOT / args.task_name / args.task_config)
    forces, ee = load_forces(dataset_dir)
    report = summarize(
        forces, args.per_decade, args.max_median_deg, args.min_count, args.paper_f_min, args.reference_p90
    )
    report["dataset"] = str(dataset_dir)

    print(f"{report['episodes']} episodes, {report['frames']} frames from {dataset_dir}")
    pct = report["force_percentiles"]
    print(f"|f| p50 {pct['p50']:.3g}  p90 {pct['p90']:.3g}  p99 {pct['p99']:.3g}  max {pct['max']:.3g}\n")
    print("| |f| from | to | frame pairs | median change [deg] | p90 [deg] |")
    print("|---|---|---|---|---|")
    for row in report["table"]:
        print(f"| {row['low']:.3g} | {row['high']:.3g} | {row['count']} | {row['median_deg']:.1f} | {row['p90_deg']:.1f} |")
    report["scales"] = {}
    if report["force_scale"] is None:
        print("\nno magnitude is stable under this criterion")
    else:
        print(f"\ndirection stable from |f| = {report['stable_force_threshold']:.3g}"
              f" (median change <= {args.max_median_deg:g} deg, bins of >= {args.min_count} pairs)")
        report["scales"]["stable_direction"] = describe_scale(
            forces, ee, VirtualTargetParams(force_scale=report["force_scale"])
        )
        print_scale(f"stable magnitude -> f_min = {args.paper_f_min:g} N", report["scales"]["stable_direction"])
    report["scales"]["reference_p90"] = describe_scale(
        forces, ee, VirtualTargetParams(force_scale=report["force_scale_p90"])
    )
    print_scale(f"p90 of |f| -> {args.reference_p90:g} N", report["scales"]["reference_p90"])
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
