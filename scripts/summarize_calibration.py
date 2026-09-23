"""Success-rate table for the ACT calibration runs against the released numbers.

Fixed before the runs: one row per (checkpoint, environment), success rate with a
95% Wilson interval, restricted to the seed set the local runs covered so the
Isaac 4.5 reference is compared on the same seeds.

    python scripts/summarize_calibration.py \
        --local univtac=eval_result/ACT/insert_hole/deploy/<run>/metadata.json \
        --local vision_only=eval_result/ACT/insert_hole/deploy/<run>/metadata.json
"""
import argparse
import json
import math
from pathlib import Path

REFERENCE = {
    "univtac": Path("data/checkpoints/insert_hole/univtac/metadata.json"),
    "vision_only": Path("data/checkpoints/insert_hole/vision_only/metadata.json"),
}


def load_results(path: Path) -> dict[str, bool]:
    meta = json.loads(path.read_text())
    return {seed: v.get("result") == "success" for seed, v in meta.items() if "result" in v}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def row(name: str, env: str, results: dict[str, bool], seeds: list[str]) -> str:
    hits = [results[s] for s in seeds if s in results]
    k, n = sum(hits), len(hits)
    lo, hi = wilson(k, n)
    rate = f"{100 * k / n:5.1f}%" if n else "  n/a"
    return f"| {name:<12} | {env:<9} | {k:>3}/{n:<3} | {rate} | [{100 * lo:4.1f}, {100 * hi:4.1f}] |"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="append", default=[], metavar="NAME=metadata.json")
    args = ap.parse_args()

    local = {}
    for item in args.local:
        name, path = item.split("=", 1)
        local[name] = load_results(Path(path))
    seeds = sorted(set().union(*local.values())) if local else sorted(load_results(REFERENCE["univtac"]))

    print(f"seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})" if seeds else "no seeds")
    print("| checkpoint   | env       | succ    | rate   | 95% Wilson    |")
    print("|--------------|-----------|---------|--------|---------------|")
    for name, path in REFERENCE.items():
        print(row(name, "isaac45", load_results(path), seeds))
    for name, results in local.items():
        print(row(name, "isaac51", results, seeds))


if __name__ == "__main__":
    main()
