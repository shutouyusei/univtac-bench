# Isaac Sim 4.5 environment on RTX 50-series (sm_120)

This directory builds the environment of this upstream `main` checkout (Isaac Sim 4.5) on an RTX 5090.
It is the only change to the upstream tree: simulator code, tasks, policies and configs are untouched, so
results stay comparable with the numbers published on the UniVTAC benchmark (all on Isaac Sim 4.5).

```bash
bash scripts/isaac45/install.sh            # all steps; env UniVTAC45 (UNIVTAC45_CONDA_ENV to rename)
bash scripts/isaac45/install.sh --from uipc # resume from a step: assets env isaacsim isaaclab curobo tacex vcpkg uipc extras
bash scripts/isaac45/install.sh --check    # versions only
bash scripts/isaac45/install.sh --locked   # rebuild with every pip version pinned to requirements.lock
```

The installed versions are recorded in `ENV_LOCK.md` and the full `pip freeze` in `requirements.lock`.
Without `--locked` only torch, torchvision and numpy are pinned (`constraints.txt`) and everything else
resolves to the newest allowed version at install time, as in upstream; with `--locked` the lock's plain
`name==version` lines constrain every pip step (editable installs and direct-URL wheels are pinned by the
steps themselves).

## Deviations from upstream `scripts/install.sh`

Upstream pins CUDA 12.4 and torch 2.5.1+cu124, neither of which has sm_120 support. Everything else
follows upstream; where upstream is silent, the values come from a working Isaac Sim 4.5 + torch 2.7 cu128
setup (the `ENV_LOCK.md` / `setup.sh` of the `anlorla/UniVTAC` fork, RTX 4090) or from the Isaac Sim 5.1
build of the same vendored libuipc (`isaac51` branch of this repository).

| Item | Upstream | Here | Why |
|---|---|---|---|
| Build toolchain (`env.yaml`) | CUDA 12.4, GCC 11.4, cmake 3.26, pkgconfig | CUDA 12.8, conda-forge GCC/G++ 12, cmake 3.26, pkgconfig, + ninja, pip, conda-forge channel | CUDA 12.8 is the first toolkit that targets sm_120; GCC 12 is its host compiler. Same change as the 5.1 build. |
| Installer | `uv pip` | `pip` | resolution may differ in the unpinned packages; use `--locked` to reproduce the recorded set |
| torch / torchvision | 2.5.1+cu124 (re-installed in the cuRobo step) | 2.7.0+cu128 / 0.22.0+cu128 | Isaac Lab v2.1.1's own `isaaclab.sh --install` installs exactly this "for blackwell support"; upstream's later downgrade has no sm_120 kernels. |
| torch_scatter | wheel for torch 2.5.1+cu124 | 2.1.2+pt27cu128 wheel | matches torch |
| numpy | unpinned | 1.26.4 | Isaac Lab requires numpy<2; same as the fork |
| pip constraints | none | `constraints.txt` (torch, torchvision, numpy) for every step after Isaac Sim | without it, `isaaclab.sh --install` resolves the RL extras to the newest torch (2.14, CUDA 13) |
| Isaac Lab | tag v2.1.1 | commit 90b79bb2d (= v2.1.1) | commit instead of tag |
| vcpkg | latest, cloned to `~/Toolchain`, path written into `~/.bashrc` | pinned dd3097e at `~/.cache/univtac-isaac45/vcpkg` (refuses another commit), own binary cache `~/.cache/univtac-isaac45/vcpkg-archives`, no rc edits | reproducible and not mixed with other vcpkg builds (vcpkg's ABI hash sees the compiler wrapper, not the GCC behind it); the pin and the `tinygltf` overlay port (GitHub archive hash) are those of the 5.1 build |
| Host compiler for CMake | system | `toolchains/gcc12-system-ld` / `gxx12-system-ld` (conda GCC 12 with the host linker) | avoids old-linker RELR errors, as in the 5.1 build |
| libuipc / cuRobo CUDA arch | native (setup.py default) / torch default | `CUDAARCHS=120`, `TORCH_CUDA_ARCH_LIST=12.0` (libuipc: sm_120 SASS + compute_120 PTX) | explicit instead of detected; `setup.py` unchanged |
| apt / sudo, demo runs | `sudo apt install ...`, TacEx RL demo, grasp_classify collection | none | install only; system prerequisites (build-essential, git) are assumed |
| TacEx assets | used as checked out | `assets` step: `git lfs fetch https://github.com/univtac/UniVTAC.git` + `git lfs checkout`, then index reset to upstream's tree and the 114 paths marked `skip-worktree` | upstream `main` stores 114 TacEx assets (robot / gel-pad USDs, meshes) as Git LFS pointer *text* in ordinary blobs with no `.gitattributes`, so a plain clone has no robot gel pads (`Could not find prim ... gelpad_left`). Each pointer names the sha256 of its object, so the fetched files are exactly the intended ones; `--check` fails while pointer text remains |
| Extras | transforms3d trimesh tetgen | + `ipython<9` | `policy/ACT` imports IPython at module level; IPython 9 needs psutil>=7, which rl_games (Isaac Lab RL extras) pins below 6 |

cuRobo stays at upstream's 0a50de1 and warp-lang at upstream's 1.0.0 (the fork uses cuRobo d64c4b0 and
warp 1.12.1); if warp 1.0.0 fails on sm_120 at runtime, that is recorded here before it is changed.

## Camera noise on RTX 50-series and the render override

With Isaac Lab's default `balanced` preset (what the upstream eval script gets, since it sets no rendering
mode), the head and wrist cameras render grain-noisy on the RTX 5090 while the tactile images are clean; the
released ACT checkpoints then fail (insert_hole, seeds 1000000-19: vision_only 1/20, univtac 0/20 against the
authors' 6/20 each). The same symptom is reported upstream (univtac/UniVTAC#16, H20; IsaacLab discussion #2869,
RTX 5090 + Isaac Sim 4.5) and is said to be fixed only from Isaac Sim 5.0. `--rendering_mode quality` (#16's fix
on H20) does not remove it here. Both presets enable the stochastic sampled direct lighting and the DL
denoiser (DLSS Ray Reconstruction) that is meant to clean it; switching both off, with the preset otherwise
unchanged, gives clean images close to the authors' demonstrations:

```bash
UNIVTAC45_RENDER_OVERRIDES="/rtx/directLighting/sampledLighting/enabled=false;/rtx-transient/dldenoiser/enabled=false" \
  python scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py insert_hole clean ACT/deploy ...
```

`run_with_render_overrides.py` applies the settings right after Isaac Lab sets its preset (a plain
`--kit_args "--/rtx/..."` is overwritten by the preset) and then runs the upstream script unchanged.
This is a deviation in rendering from the authors' setup (their GPU's denoiser presumably worked); it is
judged by the reproduction check below.

## Reproduction check

The released ACT checkpoints (`data/checkpoints/<task>/{univtac,vision_only}`) ship the authors' Isaac Sim
4.5 evaluation (`metadata.json`: per seed, from 1000000, the initial conditions and the outcome). The first
check of this environment is to evaluate them on the same seeds and compare episode by episode. Their
`log.log` shows `task_config: demo`; `demo.yml` and `clean.yml` differ only in `render_frequency` (GUI window),
so headless evaluation uses `clean`.
