"""Per task: our seed collect (variants under out/<task>/seed0_<V>) vs the released isaac45 demo of the same seed.

Released demo index 0 (seed from metadata; lift_bottle demo 0 = seed 1), reference = released demo index 1 (a
different seed, authors' platform) vs demo 0 to show the scale of episode-to-episode differences.
Fixed before the runs; descriptive only. Per pair:
  episode   frames, first/last saved step, finger qpos (mm) at frame 0, min / median / max over the episode
  frame 0   (first saved frame) per sensor: depth min (mm), mean |depth - demo| (mm), mean |rgb - demo|,
            mean |rgb_marker - demo|, marker mask (rgb_marker / rgb) dots: count, grid centre (y, x), grid extent
            (y, x), dot core darkness (0 = black), dot radius at mask 0.5
  episode   per sensor: depth min over the episode (deepest press), mean |depth - demo| and |rgb - demo| over the
            common frames (index-matched; phase shifts inflate these)
Writes compare_tasks.md.
"""

import io
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage

D = Path("/home/yusei/.claude/jobs/7fcd22c8/tmp/tactile_seed0")
REL = Path("/home/yusei/.claude/jobs/7fcd22c8/tmp/release_sample/isaac45")
IH = Path("/home/yusei/Desktop/Robotics/univtac-isaac45/data/isaac45/insert_hole/hdf5")
TASKS = ["lift_bottle", "pull_out_key", "lift_can", "put_bottle_in_shelf", "insert_hole", "insert_HDMI",
         "insert_tube", "grasp_classify"]


def demo(task, i):
    return (IH if task == "insert_hole" else REL / task / "hdf5") / f"{i}.hdf5"


def jpeg(b):
    return np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"), dtype=np.float32)


class Ep:
    def __init__(self, p):
        self.f = h5py.File(p, "r")
        self.g = {s: [n for n in self.f["tactile"] if n.startswith(s)][0] for s in ("left", "right")}
        self.n = self.f["step"].shape[0]

    def finger(self):
        return 1e3 * self.f["embodiment/joint"][:, 7]

    def depth(self, s, i):
        return self.f[f"tactile/{self.g[s]}/depth"][i].astype(np.float32)

    def img(self, s, k, i):
        return jpeg(self.f[f"tactile/{self.g[s]}/{k}"][i])


def dots(rgb, rgbm):
    m = np.clip(rgbm.sum(2) / np.maximum(rgb.sum(2), 1.0), 0, 1.2)
    lab, n = ndimage.label(m < 0.6)
    if n == 0:
        return dict(n=0)
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(m), lab, idx)
    c = np.array(ndimage.center_of_mass(m < 0.6, lab, idx[area >= 5]))
    yy, xx = np.mgrid[: m.shape[0], : m.shape[1]]
    cores, radii = [], []
    for cy, cx in c:
        r = np.hypot(yy - cy, xx - cx)
        cores.append(m[r <= 2].mean())
        prof = [m[(r >= k) & (r < k + 1)].mean() for k in range(9)]
        rad = np.nan
        for k in range(8):
            if prof[k] < 0.5 <= prof[k + 1]:
                rad = k + 0.5 + (0.5 - prof[k]) / (prof[k + 1] - prof[k])
                break
        radii.append(rad)
    return dict(n=len(c), cy=c[:, 0].mean(), cx=c[:, 1].mean(), ey=np.ptp(c[:, 0]), ex=np.ptp(c[:, 1]),
                core=float(np.median(cores)), rad=float(np.nanmedian(radii)))


def pair(name, a, b, lines):
    fa, fb = a.finger(), b.finger()
    sa, sb = a.f["step"][:], b.f["step"][:]
    lines.append(f"\n### {name}\nframes demo {a.n} / this {b.n}; first-last saved step demo {sa[0]}-{sa[-1]} / "
                 f"this {sb[0]}-{sb[-1]}; finger mm f0 / min / median / max: demo {fa[0]:.2f} / {fa.min():.2f} / "
                 f"{np.median(fa):.2f} / {fa.max():.2f}, this {fb[0]:.2f} / {fb.min():.2f} / {np.median(fb):.2f} / "
                 f"{fb.max():.2f}")
    lines.append("| sensor | f0 depth min (demo) | f0 abs depth diff mm | f0 abs rgb diff | f0 abs rgb_marker diff | "
                 "dots (demo) | grid centre y,x (demo) | grid extent y,x (demo) | dot core (demo) | dot radius (demo) | "
                 "episode deepest depth (demo) | episode mean abs depth / rgb diff |")
    lines.append("|---" * 12 + "|")
    n = min(a.n, b.n)
    for s in ("left", "right"):
        da, db = a.depth(s, 0), b.depth(s, 0)
        ra, rb = a.img(s, "rgb", 0), b.img(s, "rgb", 0)
        ma, mb = a.img(s, "rgb_marker", 0), b.img(s, "rgb_marker", 0)
        xa, xb = dots(ra, ma), dots(rb, mb)
        stride = max(1, n // 40)
        idx = range(0, n, stride)
        dep = np.mean([np.abs(b.depth(s, i) - a.depth(s, i)).mean() for i in idx])
        rgbd = np.mean([np.abs(b.img(s, "rgb", i) - a.img(s, "rgb", i)).mean() for i in idx])
        deep_a = min(a.depth(s, i).min() for i in range(0, a.n, max(1, a.n // 40)))
        deep_b = min(b.depth(s, i).min() for i in range(0, b.n, max(1, b.n // 40)))

        def g(x, k, f="{:.2f}"):
            return f.format(x[k]) if k in x else "-"
        lines.append(
            f"| {s} | {db.min():.2f} ({da.min():.2f}) | {np.abs(db - da).mean():.2f} | {np.abs(rb - ra).mean():.2f} | "
            f"{np.abs(mb - ma).mean():.2f} | {xb['n']} ({xa['n']}) | {g(xb, 'cy', '{:.1f}')},{g(xb, 'cx', '{:.1f}')} "
            f"({g(xa, 'cy', '{:.1f}')},{g(xa, 'cx', '{:.1f}')}) | {g(xb, 'ey', '{:.0f}')},{g(xb, 'ex', '{:.0f}')} "
            f"({g(xa, 'ey', '{:.0f}')},{g(xa, 'ex', '{:.0f}')}) | {g(xb, 'core', '{:.3f}')} ({g(xa, 'core', '{:.3f}')}) | "
            f"{g(xb, 'rad')} ({g(xa, 'rad')}) | {deep_b:.2f} ({deep_a:.2f}) | {dep:.2f} / {rgbd:.2f} |")


def main():
    tasks = sys.argv[1:] or TASKS
    lines = ["# Seed collects vs released isaac45 demo (same seed)"]
    for t in tasks:
        d0 = Ep(demo(t, 0))
        lines.append(f"\n## {t}")
        pair("reference: released demo 1 (other seed) vs demo 0", d0, Ep(demo(t, 1)), lines)
        for v in sorted((D / "out" / t).glob("seed0_*")):
            h = sorted(v.glob("**/*.hdf5"))
            if "timeout" in v.name or not h:
                lines.append(f"\n### {v.name}: no hdf5")
                continue
            pair(v.name, d0, Ep(h[0]), lines)
    text = "\n".join(lines)
    (D / "compare_tasks.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
