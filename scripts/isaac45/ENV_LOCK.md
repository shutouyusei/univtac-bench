# Environment lock: Isaac Sim 4.5 on RTX 5090

Built 2026-10-13 with `bash scripts/isaac45/install.sh` from a fresh Conda environment, then
`--from vcpkg --locked` rebuilt libuipc against the build's own vcpkg binary cache (0 archives restored,
37 stored, so nothing from other vcpkg builds is linked in). The complete `pip freeze` is
`requirements.lock` (263 lines; 249 plain `name==version` lines are what `--locked` pins). The editable
lines carry this machine's paths and the checkout commit at freeze time (the TacEx sources themselves are
upstream 0dafa10, unchanged).

## Machine

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32 GB, sm_120) |
| Driver | 580.178.04 |
| OS / kernel | Ubuntu 24.04.4 LTS / 7.0.0-31-generic |

## Sources

| Component | Revision |
|---|---|
| UniVTAC (this checkout) | upstream `main` 0dafa10 (`fix readme`, 2026-09-07) |
| Isaac Lab | 90b79bb2d44feb8d833f260f2bf37da3487180ba (v2.1.1), `third_party/IsaacLab` |
| cuRobo | 0a50de1ba72db304195d59d9d0b1ed269696047f, `third_party/curobo` |
| TacEx / libuipc | vendored `third_party/TacEx` at upstream 0dafa10 (no change) |
| vcpkg | dd3097e305afa53f7b4312371f62058d2e665320 + `toolchains/vcpkg-overlay-ports/tinygltf` |

## Conda toolchain (`env.yaml`)

python 3.10.22, cuda-toolkit 12.8.2 (nvcc 12.8.93), conda-forge gcc/gxx 12.4.0, cmake 3.26.4, ninja, pkgconfig.

## Key Python packages

| Package | Version |
|---|---|
| isaacsim (+ all `isaacsim-*`, extscache) | 4.5.0.0 |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| numpy | 1.26.4 |
| isaaclab / _assets / _mimic / _rl / _tasks | 0.41.3 / 0.2.2 / 1.0.7 / 0.1.8 / 0.10.36 (editable) |
| nvidia-curobo | editable at 0a50de1 (metadata 0.0.0) |
| warp-lang | 1.0.0 |
| tacex / tacex-assets / tacex-uipc / tacex_tasks | 0.1.0 (editable) |
| pyuipc | 0.9.0 (built here) |
| torch_scatter | 2.1.2+pt27cu128 |
| ipython | 8.39.0 |

## Checks run

- `install.sh --check`: every expected version present (torch_scatter built for pt27cu128), torch CUDA 12.8,
  device RTX 5090.
- `pip check`: no broken requirements.
- `--locked` rebuild of vcpkg / uipc / extras: passes with all 249 lock pins; `pip freeze` afterwards equals
  `requirements.lock` except the checkout commit in the four TacEx editable lines.
- sm_120 smoke without the Isaac Sim app (after the rebuild): warp 1.0.0 compiles and launches a kernel on the
  5090; cuRobo's CUDA extension imports; `libuipc_backend_cuda.so` contains sm_120 code only
  (`cuobjdump --list-elf`).

Not yet run: launching Isaac Sim (first launch asks to accept the NVIDIA Omniverse EULA), an expert episode,
and the released-checkpoint reproduction (see `README.md`).
