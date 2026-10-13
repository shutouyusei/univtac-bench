"""Fetch the first 50 released isaac45 demos (hdf5/0-49 + metadata.json) of one task from HF byml/UniVTAC into the
4.5 checkout's data/isaac45/<task>/ (same layout as data/download.sh), and link data/<task>/release45 -> ../isaac45/<task>
so that process_data.py <task> release45 50 reads them (first 50 by numeric stem). Episodes 50-99 are not needed for
the stage-1 training and the disk is 95 % full. Idempotent; exits non-zero unless exactly the 50 files are present.

    fetch_release.py <task>
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path("/home/yusei/Desktop/Robotics/univtac-isaac45/data")
task = sys.argv[1]
patterns = [f"isaac45/{task}/hdf5/{i}.hdf5" for i in range(50)] + [f"isaac45/{task}/metadata.json"]
snapshot_download(repo_id="byml/UniVTAC", repo_type="dataset", local_dir=str(ROOT), allow_patterns=patterns,
                  max_workers=8)
have = sorted(int(p.stem) for p in (ROOT / "isaac45" / task / "hdf5").glob("*.hdf5"))
assert have == list(range(50)), f"{task}: have {len(have)} files, missing {sorted(set(range(50)) - set(have))}"
link = ROOT / task / "release45"
link.parent.mkdir(exist_ok=True)
if not link.exists():
    link.symlink_to(Path("..") / "isaac45" / task)
assert (link / "hdf5" / "49.hdf5").exists(), link
print(task, "release demos 0-49 present,", link, "->", link.resolve())
