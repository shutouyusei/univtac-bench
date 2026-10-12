"""UniVTAC demonstrations -> RDP training cache, and the marker PCA.

    python -m rdp_univtac.prepare pca --tasks <task ...> --out policy/RDP/data/marker_pca.npz
    python -m rdp_univtac.prepare cache --task lift_bottle --out policy/RDP/data/lift_bottle-clean51-50

The cache keeps the raw per-step arrays (joints, normalised marker offsets, 240 x 320 camera frames), so the
PCA can change without re-decoding images. Cameras follow ``policy/task_settings.json`` (head, or head + wrist),
as for the other UniVTAC baselines; frames are resized (aspect not kept) to RDP's 240 x 320 input.

The official PCA is fit once per sensor type on a separate random-contact recording. UniVTAC has no such
recording, so the PCA is fit on the training demonstrations of every task pooled (one sensor type, contact-rich
and contact-free frames of all tasks, both fingertips); it never sees evaluation episodes.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np

from envs.utils.data import HDF5Handler
from . import marker
from .dataset import ArrayBuffer, concat_episodes

IMAGE_HW = (240, 320)
SIDES = ("left", "right")
REPO = Path(__file__).resolve().parents[3]


def resize_frame(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (240, 320, 3) uint8, channel order unchanged."""
    return cv2.resize(img, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_LINEAR)


def demo_paths(hdf5_dir: Path, n: int) -> list[Path]:
    paths = sorted(Path(hdf5_dir).glob("*.hdf5"), key=lambda p: int(p.stem))
    if len(paths) < n:
        raise ValueError(f"{hdf5_dir} has {len(paths)} demonstrations, {n} requested")
    return paths[:n]


def task_cameras(task: str) -> list[str]:
    settings = json.loads((REPO / "policy" / "task_settings.json").read_text())
    kind = settings.get(task, {}).get("camera_type", "head")
    return ["head", "wrist"] if kind == "all" else [kind]


def read_markers(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {s: f[f"tactile/{s}_tactile/marker"][()] for s in SIDES}


def read_demo(path: Path, cameras: list[str]) -> dict:
    with h5py.File(path, "r") as f:
        out = {"joint": f["embodiment/joint"][()].astype(np.float32)}
        for s in SIDES:
            out[f"disp_{s}"] = marker.displacement(f[f"tactile/{s}_tactile/marker"][()])
        for cam in cameras:
            frames = HDF5Handler.stream_to_img(f[f"observation/{cam}/rgb"][()])
            out[f"cam_{cam}"] = np.stack([resize_frame(x) for x in frames])
    return out


def build_cache(paths: list[Path], cameras: list[str], out: Path, workers: int = 8) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        demos = list(ex.map(lambda p: read_demo(p, cameras), paths))
    for key in demos[0]:
        np.save(out / f"{key}.npy", np.concatenate([d[key] for d in demos]))
    meta = {"files": [str(p) for p in paths], "lengths": [len(d["joint"]) for d in demos],
            "cameras": [f"cam_{c}" for c in cameras], "image_hw": list(IMAGE_HW)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load_buffer(cache: Path, pca: marker.MarkerPCA) -> ArrayBuffer:
    cache = Path(cache)
    meta = json.loads((cache / "meta.json").read_text())
    cuts = np.cumsum(meta["lengths"])[:-1]
    split = lambda key: np.split(np.load(cache / f"{key}.npy", mmap_mode="r"), cuts)  # noqa: E731
    tactile = {f"tactile_{s}": [pca.embed(x) for x in split(f"disp_{s}")] for s in SIDES}
    buf = concat_episodes(joint=split("joint"), tactile=tactile, images={})
    # Frames stay memory-mapped (several GB per camera); the cache is already in episode order.
    for c in meta["cameras"]:
        buf.data[c] = np.load(cache / f"{c}.npy", mmap_mode="r")
        assert len(buf.data[c]) == buf.episode_ends[-1], c
    buf.meta = meta
    return buf


def fit_pooled_pca(path_lists: list[list[Path]], n_components: int = 15) -> marker.MarkerPCA:
    x = []
    for paths in path_lists:
        for p in paths:
            m = read_markers(p)
            x += [marker.displacement(m[s]) for s in SIDES]
    return marker.fit_pca(np.concatenate(x), n_components)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pca")
    a.add_argument("--tasks", nargs="+", required=True)
    a.add_argument("--task-config", default="clean51")
    a.add_argument("--episodes", type=int, default=50)
    a.add_argument("--out", required=True)
    b = sub.add_parser("cache")
    b.add_argument("--task", required=True)
    b.add_argument("--task-config", default="clean51")
    b.add_argument("--episodes", type=int, default=50)
    b.add_argument("--out", required=True)
    args = p.parse_args(argv)
    data = REPO / "data"
    if args.cmd == "pca":
        lists = [demo_paths(data / t / args.task_config / "hdf5", args.episodes) for t in args.tasks]
        pca = fit_pooled_pca(lists)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pca.save(Path(args.out))
        print(f"marker PCA over {sum(map(len, lists))} demonstrations of {len(args.tasks)} tasks -> {args.out}")
    else:
        paths = demo_paths(data / args.task / args.task_config / "hdf5", args.episodes)
        build_cache(paths, task_cameras(args.task), Path(args.out))
        print(f"cached {len(paths)} demonstrations of {args.task} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
