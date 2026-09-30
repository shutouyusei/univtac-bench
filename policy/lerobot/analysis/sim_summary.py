"""Success table of several eval runs on the same seeds, paired against the first.

Each run is ``label=<path to metadata.json>``. Prints, per run, successes, rate, Wilson 95%
interval, the split by hole orientation (rotate 0 / pi) and early stops; then McNemar's exact
test of every run against the first on the seeds they share.

Usage: python -m policy.lerobot.analysis.sim_summary "clean=eval_result/.../metadata.json" "hide=..."
"""

import json
import math
import sys
from math import comb


def wilson(s, n, z=1.96):
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load(path):
    m = json.load(open(path))
    return {k: x for k, x in m.items() if isinstance(x, dict) and x.get("result") in ("success", "failed")}


def mcnemar(a, b):
    """(paired n, both succeed, only a, only b, exact two-sided p) on the seeds a and b share."""
    common = [k for k in a if k in b]
    both = sum(a[k]["result"] == "success" and b[k]["result"] == "success" for k in common)
    na = sum(a[k]["result"] == "success" and b[k]["result"] != "success" for k in common)
    nb = sum(a[k]["result"] != "success" and b[k]["result"] == "success" for k in common)
    d = na + nb
    p = min(1.0, sum(comb(d, i) for i in range(0, min(na, nb) + 1)) / 2**d * 2) if d else 1.0
    return len(common), both, na, nb, p


def row(label, v):
    n = len(v)
    s = sum(x["result"] == "success" for x in v.values())
    lo, hi = wilson(s, n) if n else (0.0, 0.0)
    r0 = [x for x in v.values() if abs(x["rotate"]) < 1e-3]
    r1 = [x for x in v.values() if abs(x["rotate"] - math.pi) < 1e-3]
    es = sum(bool(x.get("early_stop")) for x in v.values())
    return (f"| {label} | {s}/{n} | {100 * s / max(n, 1):.1f}% | [{lo:.1f}, {hi:.1f}] | "
            f"{sum(x['result'] == 'success' for x in r0)}/{len(r0)} | "
            f"{sum(x['result'] == 'success' for x in r1)}/{len(r1)} | {es}/{n} |")


def main(specs):
    runs = [(spec.split("=", 1)[0], load(spec.split("=", 1)[1])) for spec in specs]
    print("| run | success | rate | 95% Wilson | rotate=0 | rotate=pi | early stop |")
    print("|---|---|---|---|---|---|---|")
    for label, v in runs:
        print(row(label, v))
    print()
    first_label, first = runs[0]
    for label, v in runs[1:]:
        n, both, na, nb, p = mcnemar(first, v)
        print(f"{first_label} vs {label}: paired {n}, both {both}, only {first_label} {na}, only {label} {nb}, "
              f"McNemar p = {p:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
