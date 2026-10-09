"""Slow-side tactile forecast on failing seeds vs the expert's reading in the same scene (definitions: spec.md,
fixed before the runs). Outputs into <out>: table1.md, table2.md, figA.png, figB.png, per_step.csv, summary.json.

Usage (UniVTAC env, stflow-slowfast src on PYTHONPATH):
  python analyze.py --policy_results <eval_result/.../predlog_k10_tp_st> --policy_npz <dir> \
      --expert_hdf5 <dir> --ckpt <checkpoint dir> --demo_joint <cache joint.npy> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

BLOCK = 5
OFFSETS_A = (-20, -10, 0, 10, 20, 30)
BIN = 10
BIN_RANGE = (-60, 60)
rng_boot = np.random.default_rng(0)


def rms(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """RMS difference over the last two axes (fingertips x d_model)."""
    return np.sqrt(((u.astype(np.float64) - v.astype(np.float64)) ** 2).mean(axis=(-2, -1)))


def ratio(p, a, d) -> np.ndarray:
    dpd, dpa = rms(p, d), rms(p, a)
    return (dpd - dpa) / np.maximum(dpd + dpa, 1e-12)


def load_outcomes(root: Path) -> dict[int, dict]:
    """seed -> metadata entry (result, trace); later files win."""
    out = {}
    for meta in sorted(root.glob("*/metadata.json"), key=lambda p: p.stat().st_mtime):
        for key, entry in json.loads(meta.read_text()).items():
            if key.isdigit() and isinstance(entry, dict) and "result" in entry:
                out[int(key)] = entry
    return out


def expert_seeds(hdf5_dir: Path) -> set[int]:
    """Seeds the expert solved: collect mode writes an hdf5 only for a success."""
    return {int(p.stem) for p in hdf5_dir.glob("*.hdf5") if p.stem.isdigit()}


@torch.no_grad()
def readings(model, frames: dict[str, np.ndarray], device) -> np.ndarray:
    """``(T, n_tips, d)`` pooled, layer-normed tactile tokens of uint8 frames ``{name: (T, S, S, 3)}``."""
    from stflow.batch import image_to_float

    names = model.tactile.names
    n = len(next(iter(frames.values())))
    out = []
    for lo in range(0, n, 64):
        batch = {k: image_to_float(torch.from_numpy(frames[k][lo:lo + 64]).to(device)) for k in names}
        tokens = model.tactile_tokens(batch)
        d = tokens.shape[-1]
        pooled = tokens.view(tokens.shape[0], len(names), -1, d).mean(2).float()
        out.append(F.layer_norm(pooled, (d,)).cpu().numpy())
    return np.concatenate(out)


def read_expert(path: Path, size: int, names: list[str]) -> dict:
    from stflow.data.univtac_hdf5 import decode, stream_key

    with h5py.File(str(path), "r") as f:
        frames = {n: np.stack([decode(b, size) for b in np.asarray(f[stream_key(f, "tactile", n)][()]).ravel()])
                  for n in names}
        joint = np.asarray(f["embodiment/joint"][()], dtype=np.float64)[:, :8]
    return {"frames": frames, "joint": joint}


def open_end_dtw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every row of ``a`` ``(n, k)`` the index into ``b`` ``(m, k)`` it is aligned with: DTW with steps
    (1,0), (0,1), (1,1), Euclidean cost, start at (0, 0), end anywhere on the last row of ``a``."""
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
    acc = np.full((n, m), np.inf)
    acc[0, 0] = cost[0, 0]
    for j in range(1, m):
        acc[0, j] = acc[0, j - 1] + cost[0, j]
    for i in range(1, n):
        acc[i, 0] = acc[i - 1, 0] + cost[i, 0]
        for j in range(1, m):
            acc[i, j] = cost[i, j] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j = n - 1, int(np.argmin(acc[n - 1]))
    cells = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
            i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
        cells.append((i, j))
    match = np.zeros(n, dtype=int)
    for row in range(n):
        js = [c[1] for c in cells if c[0] == row]
        match[row] = int(np.median(js))
    return match


def boot_ci(values: list[float]) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), np.nan, np.nan
    means = rng_boot.choice(v, size=(2000, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_results", type=Path, required=True)
    ap.add_argument("--policy_npz", type=Path, required=True)
    ap.add_argument("--expert_hdf5", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--demo_joint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from stflow import checkpoint

    model, cfg = checkpoint.load(args.ckpt, args.device)
    names = list(model.tactile.names)
    size = cfg.model.image_size
    joint_std = np.load(args.demo_joint)[:, :8].astype(np.float64).std(0)

    outcomes = load_outcomes(args.policy_results)
    solved = expert_seeds(args.expert_hdf5)
    seeds = sorted(s for s in outcomes if s in solved and (args.policy_npz / f"{s}.npz").exists())
    left_out = sorted(set(outcomes) - set(seeds))

    episodes = {}
    expert_cache = {}
    for seed in seeds:
        rec = np.load(args.policy_npz / f"{seed}.npz")
        trace = outcomes[seed]["trace"]
        cols = trace["columns"]
        rows = np.asarray(trace["rows"], dtype=np.float64)
        n = min(len(rows), len(rec["query"]))
        query = rec["query"][:n]
        joint = rows[:n, [cols.index(f"joint_{i}") for i in range(8)]]
        exp = read_expert(args.expert_hdf5 / f"{seed}.hdf5", size, names)
        exp_read = readings(model, exp["frames"], args.device)
        expert_cache[seed] = {"frames": exp["frames"], "reading": exp_read}
        match = open_end_dtw(joint / joint_std, exp["joint"] / joint_std)
        analysed = np.flatnonzero(query % BLOCK == 0)
        chunk_start = np.array([max(k for k in range(i + 1) if query[k] == 0) for i in range(n)])
        A = rec["reading"][:n].astype(np.float32)
        P = rec["predicted"][:n].astype(np.float32)
        D = exp_read[match]
        C = A[chunk_start]
        frame_steps = rec["frame_steps"].tolist()
        episodes[seed] = {
            "success": outcomes[seed]["result"] == "success",
            "steps": analysed, "chunk_start": chunk_start[analysed], "match": match[analysed],
            "A": A[analysed], "P": P[analysed], "D": D[analysed], "C": C[analysed],
            "frames": {nm: rec[f"frames_{nm}"] for nm in names}, "frame_steps": frame_steps,
            "inhand_slip": rows[:n, cols.index("inhand_slip")][analysed],
        }

    for ep in episodes.values():
        ep["dAD"] = rms(ep["A"], ep["D"])
        ep["dPA"], ep["dPD"], ep["dCA"] = rms(ep["P"], ep["A"]), rms(ep["P"], ep["D"]), rms(ep["C"], ep["A"])
        ep["r"] = ratio(ep["P"], ep["A"], ep["D"])
        ep["r_copy"] = ratio(ep["C"], ep["A"], ep["D"])

    succ = {s: e for s, e in episodes.items() if e["success"]}
    fail = {s: e for s, e in episodes.items() if not e["success"]}
    tau = float(np.percentile(np.concatenate([e["dAD"] for e in succ.values()]), 95))
    for ep in fail.values():
        above = ep["dAD"] > tau
        hits = [k for k in range(len(above) - 1) if above[k] and above[k + 1]]
        ep["onset"] = int(ep["steps"][hits[0]]) if hits else None
    with_onset = {s: e for s, e in fail.items() if e["onset"] is not None}

    # Table 1
    lines = ["| episodes | n | d(p, A) | d(C, A) | d(D, A) |", "|---|---|---|---|---|"]
    for label, group in (("success", succ), ("failure", fail)):
        cells = []
        for key in ("dPA", "dCA", "dAD"):
            m, lo, hi = boot_ci([e[key].mean() for e in group.values()])
            cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {label} | {len(group)} | " + " | ".join(cells) + " |")
    note = ("Per-episode mean over block-start steps, then mean over episodes [95 % bootstrap CI]. p = slow-side "
            "read-out, A = reading, C = chunk-start reading (copy), D = expert reading at the DTW match.")
    (args.out / "table1.md").write_text("\n".join(lines) + "\n\n" + note + "\n")

    # Table 2
    lines = ["| failure steps | episodes | steps | mean r [95 % CI] | mean r_copy |", "|---|---|---|---|---|"]
    classes = {
        "before onset": lambda e, k: e["steps"][k] < e["onset"],
        "after onset, chunk started before onset": lambda e, k: e["steps"][k] >= e["onset"]
        and e["chunk_start"][k] < e["onset"],
        "after onset, chunk started at/after onset": lambda e, k: e["steps"][k] >= e["onset"]
        and e["chunk_start"][k] >= e["onset"],
    }
    table2 = {}
    for label, pick in classes.items():
        per_ep, per_ep_copy, n_steps = [], [], 0
        for e in with_onset.values():
            ks = [k for k in range(len(e["steps"])) if pick(e, k)]
            if ks:
                per_ep.append(e["r"][ks].mean())
                per_ep_copy.append(e["r_copy"][ks].mean())
                n_steps += len(ks)
        m, lo, hi = boot_ci(per_ep)
        mc = boot_ci(per_ep_copy)[0]
        table2[label] = {"episodes": len(per_ep), "steps": n_steps, "r": m, "ci": [lo, hi], "r_copy": mc}
        lines.append(f"| {label} | {len(per_ep)} | {n_steps} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {mc:.3f} |")
    m, lo, hi = boot_ci([e["r"].mean() for e in succ.values()])
    lines.append(f"| success episodes, all steps | {len(succ)} | {sum(len(e['r']) for e in succ.values())} | "
                 f"{m:.3f} [{lo:.3f}, {hi:.3f}] | {boot_ci([e['r_copy'].mean() for e in succ.values()])[0]:.3f} |")
    note = ("r = (d(p,D) - d(p,A)) / (d(p,D) + d(p,A)): -1 = forecast at the expert's (demo-like) reading, "
            "+1 = at the actual reading. Per-episode mean, then mean over episodes.")
    (args.out / "table2.md").write_text("\n".join(lines) + "\n\n" + note + "\n")

    # Fig B
    edges = np.arange(BIN_RANGE[0], BIN_RANGE[1] + BIN, BIN)
    centers = edges[:-1] + BIN / 2

    def binned(key):
        stats = []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            vals = []
            for e in with_onset.values():
                rel = e["steps"] - e["onset"]
                sel = (rel >= lo_e) & (rel < hi_e)
                if sel.any():
                    vals.append(e[key][sel].mean())
            stats.append(boot_ci(vals) + (len(vals),))
        return np.array(stats)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
    s = binned("dAD")
    axes[0].plot(centers, s[:, 0], "o-", color="C3", label=f"failures (n={len(with_onset)})")
    axes[0].fill_between(centers, s[:, 1], s[:, 2], color="C3", alpha=0.2)
    axes[0].axhline(tau, color="k", ls=":", label=r"$\tau$ (95th pct. in successes)")
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set_ylabel("d(actual, expert)")
    axes[0].legend(fontsize=8)
    for key, style, label in (("r", "o-", "forecast p"), ("r_copy", "s--", "copy of chunk-start reading")):
        s = binned(key)
        axes[1].plot(centers, s[:, 0], style, color="C0" if key == "r" else "C7", label=label)
        axes[1].fill_between(centers, s[:, 1], s[:, 2], color="C0" if key == "r" else "C7", alpha=0.15)
    m, lo, hi = boot_ci([e["r"].mean() for e in succ.values()])
    axes[1].axhspan(lo, hi, color="C2", alpha=0.2, label=f"forecast in successes (n={len(succ)})")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].set_ylim(-1, 1)
    axes[1].set_ylabel("r  (-1: expert-like, +1: actual)")
    axes[1].set_xlabel("steps from onset (60 Hz)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "figB.png", dpi=200)
    plt.close(fig)

    # Fig A
    summary = {"tau": tau, "n_success": len(succ), "n_fail": len(fail), "n_fail_with_onset": len(with_onset),
               "left_out_seeds": left_out, "table2": table2}
    if with_onset:
        onsets = {s: e["onset"] for s, e in with_onset.items()}
        median = float(np.median(list(onsets.values())))
        seed_a = min(onsets, key=lambda s: (abs(onsets[s] - median), s))
        e = with_onset[seed_a]
        bank_vec, bank_ref = [], []
        for s2, cache in expert_cache.items():
            bank_vec.append(cache["reading"])
            bank_ref += [("expert", s2, j) for j in range(len(cache["reading"]))]
        for s2, e2 in episodes.items():
            if s2 == seed_a:
                continue
            bank_vec.append(e2["A"])
            label = "policy success" if e2["success"] else "policy failure"
            bank_ref += [(label, s2, k) for k in range(len(e2["A"]))]
        bank = np.concatenate(bank_vec).reshape(-1, len(names) * e["A"].shape[-1])

        def frame_of(ref):
            kind, s2, j = ref
            if kind == "expert":
                return np.concatenate([expert_cache[s2]["frames"][nm][j] for nm in names], axis=1)
            e2 = episodes[s2]
            idx = e2["frame_steps"].index(int(e2["steps"][j]))
            return np.concatenate([e2["frames"][nm][idx] for nm in names], axis=1)

        cols = []
        for off in OFFSETS_A:
            k = int(np.argmin(np.abs(e["steps"] - (e["onset"] + off))))
            if k not in cols:
                cols.append(k)
        fig, axes = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 4.4))
        axes = np.atleast_2d(axes).reshape(3, len(cols))
        for c, k in enumerate(cols):
            step = int(e["steps"][k])
            fi = e["frame_steps"].index(step)
            actual = np.concatenate([e["frames"][nm][fi] for nm in names], axis=1)
            j = int(e["match"][k])
            expert = np.concatenate([expert_cache[seed_a]["frames"][nm][j] for nm in names], axis=1)
            nn = int(np.argmin(((bank - e["P"][k].reshape(1, -1)) ** 2).sum(1)))
            images = (actual, expert, frame_of(bank_ref[nn]))
            for r_i, img in enumerate(images):
                axes[r_i, c].imshow(img)
                axes[r_i, c].set_xticks([])
                axes[r_i, c].set_yticks([])
            axes[0, c].set_title(f"{step - e['onset']:+d} steps", fontsize=9)
            axes[2, c].set_xlabel(
                f"NN: {bank_ref[nn][0]}\nd(p,A)={e['dPA'][k]:.2f} d(p,D)={e['dPD'][k]:.2f}\nr={e['r'][k]:+.2f}",
                fontsize=7,
            )
        for r_i, label in enumerate(("actual", "expert\n(same seed)", "nearest to\nforecast")):
            axes[r_i, 0].set_ylabel(label, fontsize=8)
        fig.suptitle(f"seed {seed_a} (failure), onset at step {e['onset']}; left | right fingertip", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out / "figA.png", dpi=200)
        plt.close(fig)
        summary["figA_seed"] = seed_a
        summary["figA_onset"] = e["onset"]
        summary["onsets"] = onsets

    with open(args.out / "per_step.csv", "w") as f:
        f.write("seed,success,step,chunk_start,onset,dAD,dPA,dPD,dCA,r,r_copy,inhand_slip\n")
        for s2, e2 in episodes.items():
            for k in range(len(e2["steps"])):
                f.write(f"{s2},{int(e2['success'])},{e2['steps'][k]},{e2['chunk_start'][k]},{e2.get('onset')},"
                        f"{e2['dAD'][k]:.5f},{e2['dPA'][k]:.5f},{e2['dPD'][k]:.5f},{e2['dCA'][k]:.5f},"
                        f"{e2['r'][k]:.5f},{e2['r_copy'][k]:.5f},{e2['inhand_slip'][k]:.5f}\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "onsets"}, indent=1))


if __name__ == "__main__":
    main()
