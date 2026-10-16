"""Rename observations/images/cam_head -> cam_high in our processed ACT episodes (policy/ACT/data/sim-<task>/release45-50).

    rename_cam_key.py <checkout> <task>...

Upstream policy/_base_data_preprocessor.py (added 2026-04-29) saves the head camera as 'cam_head', while ACT
training (train_config*.yml camera_names) and deployment (deploy_policy.py) use 'cam_high'. Only our derived
files are changed; the released demos are untouched. Idempotent.
"""
import sys
from pathlib import Path

import h5py

ROOT = Path(sys.argv[1]) / "policy" / "ACT" / "data"
for task in sys.argv[2:]:
    files = sorted((ROOT / f"sim-{task}" / "release45-50").glob("episode_*.hdf5"))
    n = 0
    for p in files:
        with h5py.File(p, "r+") as f:
            g = f["observations/images"]
            if "cam_head" in g and "cam_high" not in g:
                g.move("cam_head", "cam_high")
                n += 1
            assert "cam_high" in g, p
    print(task, "episodes", len(files), "renamed", n, "keys", list(h5py.File(files[0], "r")["observations/images"].keys()))
