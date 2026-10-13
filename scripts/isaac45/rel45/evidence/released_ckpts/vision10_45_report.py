"""Released vision_only ACT checkpoints on Isaac Sim 4.5 (RTX 5090, balanced preset with sampled direct lighting
and the DL denoiser off), seeds 1000000-1000009 per task, against (a) the authors' shipped 4.5 evaluation of the
same checkpoints on the same 10 seeds and over 100 seeds, and (b) UniVTAC Table I "ACT". Fixed before the runs;
descriptive only, the user judges. insert_hole comes from run 1 of the 20-seed check (its first 10 seeds).

Per task: our k/10 with the 95 % Wilson interval, the authors' k/10 on the same seeds, the authors' 100-seed rate,
the paper's rate, and whether each reference lies inside our interval. Pooled over the eight tasks (80 episodes)
with its interval; Spearman rank correlation of our per-task rates with each reference.
"""

import json
import math
from pathlib import Path

ROOT = Path("/home/yusei/Desktop/Robotics/univtac-isaac45")
OUT = Path("/home/yusei/.claude/jobs/7fcd22c8/tmp/isaac45")
PAPER = {"lift_bottle": 42, "pull_out_key": 28, "lift_can": 20, "put_bottle_in_shelf": 28, "insert_hole": 19,
         "insert_HDMI": 15, "insert_tube": 45, "grasp_classify": 50}
SEEDS = [str(s) for s in range(1000000, 1000010)]
IH_RUN1 = ROOT / "eval_result/ACT/insert_hole/deploy/2026-10-13_09:03:56"


def our_dir(task):
    if task == "insert_hole":
        return IH_RUN1
    f = OUT / f"rundir_{task}_vision_only_nosample.txt"
    return Path(f.read_text().strip()) if f.exists() else None


def outcomes(path):
    if path is None or not (path / "metadata.json").exists():
        return None
    d = json.load(open(path / "metadata.json"))
    if not all(s in d and isinstance(d[s], dict) for s in SEEDS):
        return None
    return [d[s]["result"] == "success" for s in SEEDS]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


print("| task | ours /10 | 95 % Wilson | authors same 10 | authors 100-seed | paper | authors-100 in | paper in |")
print("|---|---|---|---|---|---|---|---|")
ours_r, a100_r, paper_r, tot, tot_a10 = [], [], [], 0, 0
for task, paper in PAPER.items():
    ref = json.load(open(ROOT / "data/checkpoints" / task / "vision_only/metadata.json"))
    keys = sorted(k for k in ref if isinstance(ref[k], dict))
    a100 = 100 * sum(ref[k]["result"] == "success" for k in keys) / len(keys)
    a10 = sum(ref[s]["result"] == "success" for s in SEEDS)
    out = outcomes(our_dir(task))
    if out is None:
        print(f"| {task} | - | - | {a10}/10 | {a100:.0f} | {paper} | - | - |")
        continue
    k = sum(out)
    lo, hi = wilson(k, 10)
    print(f"| {task} | {k}/10 | {lo:.0f}-{hi:.0f} | {a10}/10 | {a100:.0f} | {paper} | "
          f"{'yes' if lo <= a100 <= hi else 'no'} | {'yes' if lo <= paper <= hi else 'no'} |")
    ours_r.append(10 * k)
    a100_r.append(a100)
    paper_r.append(paper)
    tot += k
    tot_a10 += a10
if ours_r:
    n = 10 * len(ours_r)
    lo, hi = wilson(tot, n)
    print(f"\npooled over {len(ours_r)} tasks: ours {tot}/{n} = {100 * tot / n:.1f} % (95 % {lo:.0f}-{hi:.0f}); "
          f"authors same seeds {tot_a10}/{n}; authors 100-seed mean {sum(a100_r) / len(a100_r):.1f} %; "
          f"paper mean {sum(paper_r) / len(paper_r):.1f} %")
if len(ours_r) >= 3:
    print(f"Spearman ours vs authors-100: {spearman(ours_r, a100_r):.2f}; ours vs paper: {spearman(ours_r, paper_r):.2f}; "
          f"authors-100 vs paper: {spearman(a100_r, paper_r):.2f}")
