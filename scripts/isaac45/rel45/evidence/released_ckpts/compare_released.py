"""Released ACT checkpoint, our Isaac Sim 4.5 run vs the authors' (data/checkpoints/<task>/<arm>/metadata.json),
on the seeds both have. Descriptive only.

Per seed: the recorded initial condition (rotate) equal, outcome equal, and whether the episode was cut by the
in-hand slip early stop (insert_hole writes early_stop / inhand_bias only when the prism slips > 4 cm, so they
are a failure kind, not an initial condition); totals. Usage:
python compare_released.py <task> <arm> <our eval_result dir with metadata.json>
"""

import json
import sys
from pathlib import Path

task, arm, ours_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
ROOT = Path("/home/yusei/Desktop/Robotics/univtac-isaac45")
ref = json.load(open(ROOT / "data/checkpoints" / task / arm / "metadata.json"))
ours = json.load(open(ours_dir / "metadata.json"))
seeds = sorted(s for s in ours if isinstance(ours[s], dict) and isinstance(ref.get(s), dict))


def kind(d):
    if d.get("result") == "success":
        return "success"
    return "slip-stop" if d.get("early_stop") else "failed"


same_init = same_out = same_kind = 0
print(f"{task} {arm}: {len(seeds)} shared seeds")
print("| seed | rotate equal | authors | ours | steps authors | steps ours |")
print("|---|---|---|---|---|---|")
for s in seeds:
    a, b = ref[s], ours[s]
    has_rotate = "rotate" in a and "rotate" in b  # only insert_hole records an initial condition
    init_ok = has_rotate and abs(float(a["rotate"]) - float(b["rotate"])) <= 1e-9
    same_init += init_ok
    same_out += a.get("result") == b.get("result")
    same_kind += kind(a) == kind(b)
    mark = ("yes" if init_ok else "NO") if has_rotate else "-"
    print(f"| {s} | {mark} | {kind(a)} | {kind(b)} | {a.get('cost_step')} | {b.get('cost_step')} |")
succ = lambda d: sum(d[s].get("result") == "success" for s in seeds)
stops = lambda d: sum(bool(d[s].get("early_stop")) for s in seeds)
print(f"\nrotate equal: {same_init}/{len(seeds)}; outcome equal: {same_out}/{len(seeds)}; "
      f"outcome kind equal: {same_kind}/{len(seeds)}")
print(f"slip early stops: authors {stops(ref)}, ours {stops(ours)}")
print(f"successes: authors {succ(ref)}/{len(seeds)}, ours {succ(ours)}/{len(seeds)}")
