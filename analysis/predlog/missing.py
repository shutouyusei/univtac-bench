"""missing.py <result dir> <first> <last> [--summary]: the seeds in first..last without an outcome in any
<result dir>/*/metadata.json (space separated), or with --summary 'S/N success, M without outcome'."""
import json
import sys
from pathlib import Path

root, first, last = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
outcomes = {}
for meta in sorted(root.glob("*/metadata.json"), key=lambda p: p.stat().st_mtime):
    for key, entry in json.loads(meta.read_text()).items():
        if key.isdigit() and isinstance(entry, dict) and entry.get("result") in ("success", "fail", "failed"):
            outcomes[int(key)] = entry["result"]
seeds = range(first, last + 1)
missing = [s for s in seeds if s not in outcomes]
if "--summary" in sys.argv:
    done = [outcomes[s] for s in seeds if s in outcomes]
    print(f"{sum(r == 'success' for r in done)}/{len(done)} success, {len(missing)} without outcome")
else:
    print(" ".join(map(str, missing)))
