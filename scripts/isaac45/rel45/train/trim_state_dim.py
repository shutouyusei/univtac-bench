"""Trim observations/qpos and action from 9 to 8 dims in our processed ACT episodes (release45-50).

Upstream policy/_base_data_preprocessor.py stores the full joint vector (7 arm + 2 finger joints, the fingers
equal), but ACT is configured with state_dim 8 and deployment feeds joint[:8] (deploy_policy.py:90); the released
checkpoints' dataset_stats.pkl are 8-dim too. Only our derived files change; idempotent.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/home/yusei/Desktop/Robotics/univtac-isaac45/policy/ACT/data")
for task in sys.argv[1:]:
    files = sorted((ROOT / f"sim-{task}" / "release45-50").glob("episode_*.hdf5"))
    n = 0
    for p in files:
        with h5py.File(p, "r+") as f:
            for key in ("observations/qpos", "action"):
                d = f[key][:]
                if d.shape[1] == 9:
                    # the two finger joints differ by at most ~1.4e-5 m in the release; keep the first (as deploy does)
                    assert np.allclose(d[:, 7], d[:, 8], atol=1e-4), (p, key)
                    del f[key]
                    f.create_dataset(key, data=d[:, :8].astype(np.float32))
                    n += 1
                assert f[key].shape[1] == 8, (p, key)
    print(task, "episodes", len(files), "datasets trimmed", n)
