"""Released univtac (ACT + UniVTAC encoder) checkpoints on Isaac Sim 4.5 (RTX 5090, balanced preset with sampled
direct lighting and the DL denoiser off), seeds 1000000-1000009 per task, all eight tasks. Fixed before the runs;
descriptive only, the user judges whether the trend matches the paper.

References: (a) the authors' shipped 4.5 evaluation of the same checkpoints (metadata.json: same 10 seeds and all
100 seeds), (b) UniVTAC Table I "Ours (ACT + UniVTAC Encoder)" (arXiv 2602.10093v1).

Section 1, tactile arm alone: our k/10 with the 95 % Wilson interval vs the references, pooled rate, Spearman of
per-task rates.
Section 2, tactile effect: our univtac minus our vision_only on the same 10 seeds (vision runs from
vision10_45_report.py, insert_hole = first 10 seeds of run 1), per task and pooled (paired, exact McNemar);
next to the authors' shipped 100-seed difference and the paper's Table I difference; sign agreement per task.
"""

import json
import math
from pathlib import Path

ROOT = Path("/home/yusei/Desktop/Robotics/univtac-isaac45")
OUT = Path("/home/yusei/.claude/jobs/7fcd22c8/tmp/isaac45")
TASKS = ["lift_bottle", "pull_out_key", "lift_can", "put_bottle_in_shelf", "insert_hole", "insert_HDMI",
         "insert_tube", "grasp_classify"]
PAPER_TAC = {"lift_bottle": 71, "pull_out_key": 46, "lift_can": 29, "put_bottle_in_shelf": 31, "insert_hole": 24,
             "insert_HDMI": 28, "insert_tube": 56, "grasp_classify": 99}
PAPER_VIS = {"lift_bottle": 42, "pull_out_key": 28, "lift_can": 20, "put_bottle_in_shelf": 28, "insert_hole": 19,
             "insert_HDMI": 15, "insert_tube": 45, "grasp_classify": 50}
SEEDS = [str(s) for s in range(1000000, 1000010)]
IH_VIS_RUN1 = ROOT / "eval_result/ACT/insert_hole/deploy/2026-10-13_09:03:56"


def our_dir(task, arm):
    if task == "insert_hole" and arm == "vision_only":
        return IH_VIS_RUN1
    f = OUT / f"rundir_{task}_{arm}_nosample.txt"
    return Path(f.read_text().strip()) if f.exists() else None


def outcomes(path):
    if path is None or not (path / "metadata.json").exists():
        return None
    d = json.load(open(path / "metadata.json"))
    if not all(s in d and isinstance(d[s], dict) for s in SEEDS):
        return None
    return [d[s]["result"] == "success" for s in SEEDS]


def authors(task, arm):
    ref = json.load(open(ROOT / "data/checkpoints" / task / arm / "metadata.json"))
    keys = [k for k in ref if isinstance(ref[k], dict)]
    a100 = 100 * sum(ref[k]["result"] == "success" for k in keys) / len(keys)
    a10 = sum(ref[s]["result"] == "success" for s in SEEDS)
    return a10, a100


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


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


def sign(x):
    return "+" if x > 0 else "-" if x < 0 else "0"


print("## 1. Tactile arm (univtac), 10 seeds per task\n")
print("| task | ours /10 | 95 % Wilson | authors same 10 | authors 100-seed | paper | authors-100 in | paper in |")
print("|---|---|---|---|---|---|---|---|")
rows = {}
for task in TASKS:
    a10, a100 = authors(task, "univtac")
    out = outcomes(our_dir(task, "univtac"))
    if out is None:
        print(f"| {task} | - | - | {a10}/10 | {a100:.0f} | {PAPER_TAC[task]} | - | - |")
        continue
    k = sum(out)
    lo, hi = wilson(k, 10)
    print(f"| {task} | {k}/10 | {lo:.0f}-{hi:.0f} | {a10}/10 | {a100:.0f} | {PAPER_TAC[task]} | "
          f"{'yes' if lo <= a100 <= hi else 'no'} | {'yes' if lo <= PAPER_TAC[task] <= hi else 'no'} |")
    rows[task] = dict(out=out, k=k, a10=a10, a100=a100)
if rows:
    n = 10 * len(rows)
    tot = sum(r["k"] for r in rows.values())
    lo, hi = wilson(tot, n)
    print(f"\npooled over {len(rows)} tasks: ours {tot}/{n} = {100 * tot / n:.1f} % (95 % {lo:.0f}-{hi:.0f}); "
          f"authors same seeds {sum(r['a10'] for r in rows.values())}/{n}; "
          f"authors 100-seed mean {sum(r['a100'] for r in rows.values()) / len(rows):.1f} %; "
          f"paper mean {sum(PAPER_TAC[t] for t in rows) / len(rows):.1f} %")
if len(rows) >= 3:
    ours_r = [10 * r["k"] for r in rows.values()]
    a100_r = [r["a100"] for r in rows.values()]
    paper_r = [PAPER_TAC[t] for t in rows]
    print(f"Spearman ours vs authors-100: {spearman(ours_r, a100_r):.2f}; ours vs paper: {spearman(ours_r, paper_r):.2f}; "
          f"authors-100 vs paper: {spearman(a100_r, paper_r):.2f}")

print("\n## 2. Tactile effect (univtac minus vision_only), percentage points\n")
print("| task | ours vis /10 | ours tac /10 | ours diff | tac-only / vis-only seeds | McNemar p | "
      "authors-100 vis | authors-100 tac | authors-100 diff | paper diff | sign ours/authors/paper |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
pb = pc = 0
diffs = []
for task in TASKS:
    _, av100 = authors(task, "vision_only")
    _, at100 = authors(task, "univtac")
    ad, pd = at100 - av100, PAPER_TAC[task] - PAPER_VIS[task]
    vis, tac = outcomes(our_dir(task, "vision_only")), rows.get(task, {}).get("out")
    if vis is None or tac is None:
        print(f"| {task} | - | - | - | - | - | {av100:.0f} | {at100:.0f} | {ad:+.0f} | {pd:+.0f} | ?/{sign(ad)}/{sign(pd)} |")
        continue
    b = sum(t and not v for t, v in zip(tac, vis))
    c = sum(v and not t for t, v in zip(tac, vis))
    pb, pc = pb + b, pc + c
    od = 10 * (sum(tac) - sum(vis))
    diffs.append((od, ad, pd))
    print(f"| {task} | {sum(vis)}/10 | {sum(tac)}/10 | {od:+d} | {b} / {c} | {mcnemar_exact(b, c):.2f} | "
          f"{av100:.0f} | {at100:.0f} | {ad:+.0f} | {pd:+.0f} | {sign(od)}/{sign(ad)}/{sign(pd)} |")
if diffs:
    n = len(diffs)
    print(f"\npooled over {n} tasks: ours diff {sum(d[0] for d in diffs) / n:+.1f} pt "
          f"(tac-only {pb} / vis-only {pc} seeds, McNemar p {mcnemar_exact(pb, pc):.3f}); "
          f"authors-100 diff {sum(d[1] for d in diffs) / n:+.1f} pt; paper diff {sum(d[2] for d in diffs) / n:+.1f} pt")
    print(f"sign agreement ours vs paper {sum(sign(d[0]) == sign(d[2]) for d in diffs)}/{n}; "
          f"ours vs authors-100 {sum(sign(d[0]) == sign(d[1]) for d in diffs)}/{n}; "
          f"authors-100 vs paper {sum(sign(d[1]) == sign(d[2]) for d in diffs)}/{n}")
    if n >= 3:
        print(f"Spearman of per-task diffs: ours vs paper {spearman([d[0] for d in diffs], [d[2] for d in diffs]):.2f}; "
              f"ours vs authors-100 {spearman([d[0] for d in diffs], [d[1] for d in diffs]):.2f}")
