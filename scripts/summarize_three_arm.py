"""Success-rate table for the SmolVLA three-arm comparison on one task.

Fixed before the runs. One row per arm, success rate with a 95% Wilson interval
over the seeds that produced a verdict (``error`` seeds, e.g. an env reset
timeout, are skipped by eval_policy.py itself and are skipped here too), and the
TacForcing paper's insert_hole ablation numbers for reference (pi0.5 base, 100
rollouts, so a different backbone and a different sim version).

    python scripts/summarize_three_arm.py \
        base=eval_result/lerobot/insert_hole/deploy_smolvla/<run>/metadata.json \
        fixedtactile=eval_result/lerobot/insert_hole/deploy_fixedtactile/<run>/metadata.json \
        tacforcing=eval_result/lerobot/insert_hole/deploy_tacforcing/<run>/metadata.json \
        [--extra name=metadata.json ...] [--markdown]
"""
import argparse
import json
import math
from pathlib import Path

PAPER_INSERT_HOLE = {"base": 39, "fixedtactile": 34, "tacforcing": 69}
PAPER_LABEL = {"base": "Base", "fixedtactile": "Fixed Tactile", "tacforcing": "TacForcing"}


def load_verdicts(path: Path) -> dict[str, bool]:
    meta = json.loads(path.read_text())
    return {
        seed: v["result"] == "success"
        for seed, v in meta.items()
        if v.get("result") in ("success", "failed")
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", metavar="NAME=metadata.json")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=metadata.json",
                    help="additional rows (e.g. the ACT baselines) shown without a paper reference")
    ap.add_argument("--markdown", action="store_true", help="markdown table only, no seed line")
    args = ap.parse_args()

    rows = []
    for item in args.arms + args.extra:
        name, path = item.split("=", 1)
        verdicts = load_verdicts(Path(path))
        k, n = sum(verdicts.values()), len(verdicts)
        lo, hi = wilson(k, n)
        seeds = sorted(verdicts)
        rows.append((name, k, n, lo, hi, seeds))

    if not args.markdown:
        for name, _, n, _, _, seeds in rows:
            span = f"{seeds[0]}..{seeds[-1]}" if seeds else "none"
            print(f"{name}: {n} seeds ({span})")
        print()
    print("| arm | success | rate | 95% Wilson | paper insert_hole (pi0.5, n=100) |")
    print("|---|---|---|---|---|")
    for name, k, n, lo, hi, _ in rows:
        rate = f"{100 * k / n:.1f}%" if n else "n/a"
        ci = f"[{100 * lo:.1f}, {100 * hi:.1f}]" if n else "n/a"
        paper = f"{PAPER_LABEL[name]} {PAPER_INSERT_HOLE[name]}%" if name in PAPER_INSERT_HOLE else "-"
        print(f"| {name} | {k}/{n} | {rate} | {ci} | {paper} |")


if __name__ == "__main__":
    main()
