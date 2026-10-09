"""Checks (ii) and (iii) of spec2.md on the predlog data. Same inputs as analyze.py, plus --out for
check2_table.md and check2_fig.png. Episodes, p / A / D, tau and onset are built exactly as in analyze.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import (  # noqa: E402
    BLOCK, boot_ci, expert_seeds, load_outcomes, open_end_dtw, read_expert, readings, rms,
)

GROUPS = ("failure, after onset", "failure, before onset", "success")


def contact_levels(path: Path, names) -> np.ndarray:
    """``(T,)`` mean over fingertips of the per-frame maximum press_depth."""
    with h5py.File(str(path), "r") as f:
        per_tip = []
        for n in names:
            d = f[f"tactile/{n}_tactile/press_depth"]
            per_tip.append(np.array([float(d[i].max()) for i in range(len(d))]))
    return np.mean(per_tip, axis=0)


def main():
    ap = argparse.ArgumentParser()
    for k in ("policy_results", "policy_npz", "expert_hdf5", "ckpt", "demo_joint", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from stflow import checkpoint

    model, cfg = checkpoint.load(args.ckpt, args.device)
    names = list(model.tactile.names)
    joint_std = np.load(args.demo_joint)[:, :8].astype(np.float64).std(0)
    outcomes = load_outcomes(args.policy_results)
    solved = expert_seeds(args.expert_hdf5)
    seeds = sorted(s for s in outcomes if s in solved and (args.policy_npz / f"{s}.npz").exists())

    expert, episodes = {}, {}
    for seed in seeds:
        path = args.expert_hdf5 / f"{seed}.hdf5"
        exp = read_expert(path, cfg.model.image_size, names)
        c = contact_levels(path, names)
        expert[seed] = {"reading": readings(model, exp["frames"], args.device), "joint": exp["joint"],
                        "cz": (c - c.mean()) / max(c.std(), 1e-12)}
        rec = np.load(args.policy_npz / f"{seed}.npz")
        trace = outcomes[seed]["trace"]
        cols, rows = trace["columns"], np.asarray(trace["rows"], dtype=np.float64)
        n = min(len(rows), len(rec["query"]))
        query = rec["query"][:n]
        joint = rows[:n, [cols.index(f"joint_{i}") for i in range(8)]]
        match = open_end_dtw(joint / joint_std, exp["joint"] / joint_std)
        k = np.flatnonzero(query % BLOCK == 0)
        episodes[seed] = {"success": outcomes[seed]["result"] == "success", "steps": k, "jD": match[k],
                          "A": rec["reading"][:n][k].astype(np.float32),
                          "P": rec["predicted"][:n][k].astype(np.float32)}
        episodes[seed]["D"] = expert[seed]["reading"][episodes[seed]["jD"]]
        episodes[seed]["dAD"] = rms(episodes[seed]["A"], episodes[seed]["D"])

    tau = float(np.percentile(np.concatenate([e["dAD"] for e in episodes.values() if e["success"]]), 95))
    for e in episodes.values():
        e["onset"] = None
        if not e["success"]:
            above = e["dAD"] > tau
            hits = [i for i in range(len(above) - 1) if above[i] and above[i + 1]]
            e["onset"] = int(e["steps"][hits[0]]) if hits else None

    def group_steps(e):
        if e["success"]:
            return {"success": np.arange(len(e["steps"]))}
        if e["onset"] is None:
            return {}
        after = e["steps"] >= e["onset"]
        return {"failure, after onset": np.flatnonzero(after), "failure, before onset": np.flatnonzero(~after)}

    def nn_stats(e, seed, vecs, ks):
        bank = expert[seed]["reading"].reshape(len(expert[seed]["reading"]), -1)
        q = vecs[ks].reshape(len(ks), -1)
        jnn = np.argmin(((q[:, None] - bank[None]) ** 2).sum(-1), axis=1)
        cz = expert[seed]["cz"]
        return cz[jnn] - cz[e["jD"][ks]], np.abs(jnn - e["jD"][ks]) / len(bank)

    # (iii) other seeds' expert reading at the same phase
    def other_stats(e, seed, vecs, ks):
        len_e = len(expert[seed]["reading"])
        phi = e["jD"][ks] / (len_e - 1)
        others = [s for s in seeds if s != seed]
        d_same = rms(vecs[ks], e["D"][ks])
        stack = np.stack([expert[s]["reading"][np.round(phi * (len(expert[s]["reading"]) - 1)).astype(int)]
                          for s in others])
        d_other = np.mean([rms(vecs[ks], o) for o in stack], axis=0)
        d_mean = rms(vecs[ks], stack.mean(0))
        return d_other - d_same, d_mean - d_same

    res = {g: {"dcz": [], "phase": [], "S": [], "avg": [], "ep_dcz": [], "ep_S": [], "ep_avg": []} for g in GROUPS}
    res["success (actual A, control)"] = {k: [] for k in res["success"]}
    for seed, e in episodes.items():
        for g, ks in group_steps(e).items():
            if len(ks) == 0:
                continue
            targets = [(g, e["P"])] + ([("success (actual A, control)", e["A"])] if g == "success" else [])
            for label, vecs in targets:
                dcz, phase = nn_stats(e, seed, vecs, ks)
                S, avg = other_stats(e, seed, vecs, ks)
                r = res[label]
                r["dcz"] += dcz.tolist()
                r["phase"] += phase.tolist()
                r["ep_dcz"].append(float(np.median(dcz)))
                r["ep_S"].append(float(S.mean()))
                r["ep_avg"].append(float(avg.mean()))

    lines = ["| group | episodes | steps | (ii) median c_z(NN)-c_z(D) [IQR] | (ii) frac < -1 | (ii) median phase "
             "offset | (iii) S = d_other - d_same [95 % CI] | (iii) d(.,M) - d_same [95 % CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for label, r in res.items():
        if not r["dcz"]:
            continue
        dcz = np.asarray(r["dcz"])
        q1, med, q3 = np.percentile(dcz, [25, 50, 75])
        s = boot_ci(r["ep_S"])
        a = boot_ci(r["ep_avg"])
        lines.append(f"| {label} | {len(r['ep_S'])} | {len(dcz)} | {med:+.2f} [{q1:+.2f}, {q3:+.2f}] | "
                     f"{(dcz < -1).mean():.2f} | {np.median(r['phase']):.3f} | {s[0]:+.4f} [{s[1]:+.4f}, {s[2]:+.4f}] | "
                     f"{a[0]:+.4f} [{a[1]:+.4f}, {a[2]:+.4f}] |")
    note = ("Rows other than the control use the forecast p; the control uses the actual reading A of success "
            "steps. (ii): NN = same-seed expert frame nearest to the vector; c_z = within-seed standardised "
            "contact (press_depth); negative = lighter than the time-matched expert frame. (iii): S > 0 = closer "
            "to its own scene's expert than to other scenes' at the same phase; d(.,M) - d_same < 0 = closer to "
            "the phase average of other scenes than to its own scene.")
    (args.out / "check2_table.md").write_text("\n".join(lines) + "\n\n" + note + f"\n\ntau = {tau:.4f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(-4, 4, 33)
    for i, (label, r) in enumerate(res.items()):
        if r["dcz"]:
            axes[0].hist(r["dcz"], bins=bins, density=True, histtype="step", lw=1.5, color=f"C{i}", label=label)
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set_xlabel("contact of nearest same-seed expert frame minus matched frame (within-seed z)")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)
    labels = [l for l, r in res.items() if r["ep_S"]]
    stats = [boot_ci(res[l]["ep_S"]) for l in labels]
    x = np.arange(len(labels))
    axes[1].bar(x, [s[0] for s in stats], color=[f"C{i}" for i in range(len(labels))], alpha=0.7)
    axes[1].errorbar(x, [s[0] for s in stats], yerr=[[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]],
                     fmt="none", color="k", capsize=3)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([l.replace(", ", ",\n") for l in labels], fontsize=7)
    axes[1].set_ylabel("S = d(other scenes) - d(own scene)")
    fig.tight_layout()
    fig.savefig(args.out / "check2_fig.png", dpi=200)
    print((args.out / "check2_table.md").read_text())


if __name__ == "__main__":
    main()
