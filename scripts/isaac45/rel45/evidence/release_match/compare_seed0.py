"""Seed-0 insert_hole collect on our 4.5 platform vs the authors' released demo 0 (seed 0), frame by frame.

Variants (each one collect run, saved hdf5): A = main 0dafa10 as is; B = main with the gsmini marker code of the
authors' eval commit a9abc0d (run_old_marker.py); C<mm> = main with the adaptive-grasp depth threshold set to <mm>
(run_grasp_thr.py; robot default is 27.5). Every out/insert_hole/seed0_<tag> folder is reported. Each section also
gives the finger qpos at frame 0 and the first saved step (end of pre_move) for demo and variant. Reference row: released demo 2 (seed 2, also rotate pi), i.e. a
different episode from the authors' platform, to show how much two demos differ at all.

Fixed before the runs; descriptive only. Frames are matched by index (save_frequency 2 in both); the step arrays and
frame counts are printed so a phase shift is visible.

Per variant and sensor (left/right):
  trajectory   |ee position - demo| (mm), |joint - demo| (rad, max over joints), |prism position - demo| (mm)
  depth        min depth per frame (mm; gel indentation = far plane - min), mean |depth - demo| over pixels (mm)
  rgb          mean |rgb - demo| (grey levels, JPEG-decoded both sides) -> optical (Taxim) image incl. contact shading
  rgb_marker   mean |rgb_marker - demo|; marker blobs (all channels < 40, connected components >= 5 px): count,
               median area (px), grid extent x/y (px, centroid max - min), mean nearest-centroid shift to demo (px)
Rows: frame 0, 25 %, 50 %, 75 %, last common frame, and the mean over all common frames.
Writes compare_seed0.md (tables) and seed0_frames.png (left sensor rgb_marker and depth at frame 0 and 50 %:
demo, A, B, |A - demo| x4, |B - demo| x4).
"""

import glob
import io
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage

D = Path("/home/yusei/.claude/jobs/7fcd22c8/tmp/tactile_seed0")
DEMO = Path("/home/yusei/Desktop/Robotics/univtac-isaac45/data/isaac45/insert_hole/hdf5")
OUT_MD = D / "compare_seed0.md"
OUT_PNG = D / "seed0_frames.png"


def find_variant(tag):
    hits = sorted(glob.glob(str(D / "out" / "insert_hole" / f"seed0_{tag}" / "**" / "*.hdf5"), recursive=True))
    return Path(hits[0]) if hits else None


def tactile_groups(f):
    names = list(f["tactile"].keys())
    left = [n for n in names if n.startswith("left")]
    right = [n for n in names if n.startswith("right")]
    return {"left": left[0], "right": right[0]}


def jpeg(b):
    return np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"), dtype=np.float32)


class Ep:
    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, "r")
        self.g = tactile_groups(self.f)
        self.n = self.f["step"].shape[0]

    def step(self):
        return self.f["step"][:]

    def ee(self, i):
        return self.f["embodiment/ee"][i]

    def joint(self, i):
        return self.f["embodiment/joint"][i]

    def prism(self, i):
        return self.f["actor/prism"][i] if "actor/prism" in self.f else None

    def depth(self, side, i):
        return self.f[f"tactile/{self.g[side]}/depth"][i].astype(np.float32)

    def img(self, side, key, i):
        return jpeg(self.f[f"tactile/{self.g[side]}/{key}"][i])


def blobs(img):
    mask = (img < 40).all(axis=2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0)
    idx = np.arange(1, n + 1)
    area = ndimage.sum(mask, lab, idx)
    keep = area >= 5
    cen = np.array(ndimage.center_of_mass(mask, lab, idx[keep]))  # (y, x)
    return cen, area[keep]


def blob_stats(img, ref_cen):
    cen, area = blobs(img)
    if len(cen) == 0:
        return dict(count=0, area=np.nan, ext_x=np.nan, ext_y=np.nan, shift=np.nan), cen
    shift = np.nan
    if ref_cen is not None and len(ref_cen):
        d = np.sqrt(((cen[:, None, :] - ref_cen[None, :, :]) ** 2).sum(-1))
        shift = float(d.min(axis=1).mean())
    return dict(count=len(cen), area=float(np.median(area)), ext_x=float(np.ptp(cen[:, 1])),
                ext_y=float(np.ptp(cen[:, 0])), shift=shift), cen


def frame_metrics(demo, ep, i):
    m = {}
    m["ee_mm"] = 1e3 * float(np.linalg.norm(ep.ee(i)[:3] - demo.ee(i)[:3]))
    m["joint_rad"] = float(np.abs(ep.joint(i)[:7] - demo.joint(i)[:7]).max())
    p, q = ep.prism(i), demo.prism(i)
    m["prism_mm"] = 1e3 * float(np.linalg.norm(p[:3] - q[:3])) if p is not None and q is not None else np.nan
    for side in ("left", "right"):
        dd, de = demo.depth(side, i), ep.depth(side, i)
        m[f"{side}_dmin_demo"] = float(dd.min())
        m[f"{side}_dmin"] = float(de.min())
        m[f"{side}_depth_mad"] = float(np.abs(de - dd).mean())
        m[f"{side}_rgb_mad"] = float(np.abs(ep.img(side, "rgb", i) - demo.img(side, "rgb", i)).mean())
        rd, re_ = demo.img(side, "rgb_marker", i), ep.img(side, "rgb_marker", i)
        m[f"{side}_rgbm_mad"] = float(np.abs(re_ - rd).mean())
        sd, cen_d = blob_stats(rd, None)
        se, _ = blob_stats(re_, cen_d)
        for k in ("count", "area", "ext_x", "ext_y"):
            m[f"{side}_{k}_demo"] = sd[k]
            m[f"{side}_{k}"] = se[k]
        m[f"{side}_shift"] = se["shift"]
    return m


COLS = [("ee_mm", "ee mm"), ("joint_rad", "joint rad"), ("prism_mm", "prism mm")]
SIDE_COLS = [("dmin", "depth min mm (demo)"), ("depth_mad", "|depth| mm"), ("rgb_mad", "|rgb|"),
             ("rgbm_mad", "|rgb_marker|"), ("count", "dots (demo)"), ("area", "dot area px (demo)"),
             ("ext_x", "grid x px (demo)"), ("ext_y", "grid y px (demo)"), ("shift", "dot shift px")]


def fmt(m, side, key):
    v = m[f"{side}_{key}"]
    demo_key = f"{side}_{key}_demo" if key != "dmin" else f"{side}_dmin_demo"
    if demo_key in m:
        return f"{v:.2f} ({m[demo_key]:.2f})"
    return f"{v:.2f}"


def report(name, demo, ep, lines):
    n = min(demo.n, ep.n)
    lines.append(f"\n## {name}: {ep.path}\n")
    lines.append(f"finger qpos frame 0 (mm): demo {1e3 * demo.joint(0)[7]:.3f}, this {1e3 * ep.joint(0)[7]:.3f}; "
                 f"first saved step (end of pre_move): demo {demo.step()[0]}, this {ep.step()[0]}")
    lines.append(f"frames: demo {demo.n}, this {ep.n}; steps first/last demo {demo.step()[0]}/{demo.step()[-1]}, "
                 f"this {ep.step()[0]}/{ep.step()[-1]}; step arrays equal on common frames: "
                 f"{bool(np.array_equal(demo.step()[:n], ep.step()[:n]))}")
    picks = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    allm = [frame_metrics(demo, ep, i) for i in range(n)]
    mean = {k: float(np.nanmean([m[k] for m in allm])) for k in allm[0]}
    rows = [(f"frame {i}", allm[i]) for i in picks] + [("mean all", mean)]
    lines.append("| frame | " + " | ".join(c[1] for c in COLS) + " |")
    lines.append("|---" * (len(COLS) + 1) + "|")
    for label, m in rows:
        lines.append(f"| {label} | " + " | ".join(f"{m[k]:.2f}" for k, _ in COLS) + " |")
    for side in ("left", "right"):
        lines.append(f"\n{side} sensor\n")
        lines.append("| frame | " + " | ".join(c[1] for c in SIDE_COLS) + " |")
        lines.append("|---" * (len(SIDE_COLS) + 1) + "|")
        for label, m in rows:
            lines.append(f"| {label} | " + " | ".join(fmt(m, side, k) for k, _ in SIDE_COLS) + " |")
    return allm


def figure(demo, eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min([demo.n] + [e.n for _, e in eps])
    frames = [0, n // 2]
    cols = ["demo"] + [t for t, _ in eps] + [f"|{t}-demo|x4" for t, _ in eps]
    fig, ax = plt.subplots(2 * len(frames), len(cols), figsize=(2.6 * len(cols), 2.2 * 2 * len(frames)))
    for r, i in enumerate(frames):
        dimg, ddep = demo.img("left", "rgb_marker", i), demo.depth("left", i)
        imgs = [dimg] + [e.img("left", "rgb_marker", i) for _, e in eps]
        imgs += [np.clip(4 * np.abs(e.img("left", "rgb_marker", i) - dimg), 0, 255) for _, e in eps]
        deps = [ddep] + [e.depth("left", i) for _, e in eps]
        deps += [np.abs(e.depth("left", i) - ddep) for _, e in eps]
        for c, im in enumerate(imgs):
            ax[2 * r, c].imshow(im.astype(np.uint8))
            ax[2 * r, c].set_title(f"{cols[c]} rgb_marker f{i}", fontsize=7)
        lo, hi = float(ddep.min()), float(ddep.max())
        for c, dp in enumerate(deps):
            if c <= len(eps):
                ax[2 * r + 1, c].imshow(dp, vmin=lo, vmax=hi, cmap="viridis")
            else:
                ax[2 * r + 1, c].imshow(dp, vmin=0, vmax=0.5, cmap="magma")
            ax[2 * r + 1, c].set_title(f"{cols[c]} depth f{i}", fontsize=7)
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


def main():
    demo = Ep(DEMO / "0.hdf5")
    tags = sorted(p.name[len("seed0_"):] for p in (D / "out" / "insert_hole").glob("seed0_*")
                  if "timeout" not in p.name)
    variants = [(t, find_variant(t)) for t in tags]
    if len(sys.argv) > 1 and sys.argv[1] == "--dry":
        variants = [("A", DEMO / "2.hdf5"), ("B", DEMO / "2.hdf5")]
    lines = ["# seed 0 insert_hole: our collect vs released demo 0",
             f"demo 0 tactile groups {demo.g}, frames {demo.n}"]
    report("reference: released demo 2 (seed 2) vs demo 0", demo, Ep(DEMO / "2.hdf5"), lines)
    eps = []
    for tag, p in variants:
        if p is None:
            lines.append(f"\n## {tag}: no hdf5 found")
            continue
        ep = Ep(p)
        lines.append(f"\n{tag} tactile groups {ep.g}")
        report(f"variant {tag}", demo, ep, lines)
        eps.append((tag, ep))
    if eps:
        figure(demo, eps)
        lines.append(f"\nfigure: {OUT_PNG}")
    text = "\n".join(lines)
    OUT_MD.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
