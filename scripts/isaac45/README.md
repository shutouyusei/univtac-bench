# Isaac Sim 4.5 environment on RTX 50-series (sm_120)

This directory builds the environment of this upstream `main` checkout (Isaac Sim 4.5) on an RTX 5090.
It is the only change to the upstream tree: simulator code, tasks, policies and configs are untouched, so
results stay comparable with the numbers published on the UniVTAC benchmark (all on Isaac Sim 4.5).

```bash
bash scripts/isaac45/install.sh            # all steps; env UniVTAC45 (UNIVTAC45_CONDA_ENV to rename)
bash scripts/isaac45/install.sh --from uipc # resume from a step: env isaacsim isaaclab curobo tacex vcpkg uipc extras
bash scripts/isaac45/install.sh --check    # versions only
```

The installed versions are recorded in `ENV_LOCK.md` (with the full `pip freeze`).

## Deviations from upstream `scripts/install.sh`

Upstream pins CUDA 12.4 and torch 2.5.1+cu124, neither of which has sm_120 support. Everything else
follows upstream; where upstream is silent, the values come from a working Isaac Sim 4.5 + torch 2.7 cu128
setup (the `ENV_LOCK.md` / `setup.sh` of the `anlorla/UniVTAC` fork, RTX 4090) or from the Isaac Sim 5.1
build of the same vendored libuipc (`isaac51` branch of this repository).

| Item | Upstream | Here | Why |
|---|---|---|---|
| Build toolchain (`env.yaml`) | CUDA 12.4, GCC 11.4 | CUDA 12.8, conda-forge GCC/G++ 12 | CUDA 12.8 is the first toolkit that targets sm_120; GCC 12 is its host compiler. Same change as the 5.1 build. |
| torch / torchvision | 2.5.1+cu124 (re-installed in the cuRobo step) | 2.7.0+cu128 / 0.22.0+cu128 | Isaac Lab v2.1.1's own `isaaclab.sh --install` installs exactly this "for blackwell support"; upstream's later downgrade has no sm_120 kernels. |
| torch_scatter | wheel for torch 2.5.1+cu124 | 2.1.2+pt27cu128 wheel | matches torch |
| numpy | unpinned | 1.26.4 | Isaac Lab requires numpy<2; same as the fork |
| pip constraints | none | `constraints.txt` (torch, torchvision, numpy) for every step after Isaac Sim | without it, `isaaclab.sh --install` resolves the RL extras to the newest torch (2.14, CUDA 13) |
| Isaac Lab | tag v2.1.1 | commit 90b79bb2d (= v2.1.1) | commit instead of tag |
| vcpkg | latest, cloned to `~/Toolchain`, path written into `~/.bashrc` | pinned dd3097e at `~/.cache/univtac-isaac45/vcpkg`, no rc edits | reproducible; the pin and the `tinygltf` overlay port (GitHub archive hash) are those of the 5.1 build |
| Host compiler for CMake | system | `toolchains/gcc12-system-ld` / `gxx12-system-ld` (conda GCC 12 with the host linker) | avoids old-linker RELR errors, as in the 5.1 build |
| libuipc CUDA arch | native | native (sm_120 with CUDA 12.8) | `setup.py` unchanged |
| apt / sudo, demo runs | `sudo apt install ...`, TacEx RL demo, grasp_classify collection | none | install only; system prerequisites (build-essential, git) are assumed |
| Extras | transforms3d trimesh tetgen | + `ipython<9` | `policy/ACT` imports IPython at module level; IPython 9 needs psutil>=7, which isaacsim-kernel pins lower |

cuRobo stays at upstream's 0a50de1 and warp-lang at upstream's 1.0.0 (the fork uses cuRobo d64c4b0 and
warp 1.12.1); if warp 1.0.0 fails on sm_120 at runtime, that is recorded here before it is changed.

## Reproduction check

The released ACT checkpoints (`data/checkpoints/<task>/{univtac,vision_only}`) ship the authors' Isaac Sim
4.5 evaluation (`metadata.json`: per seed, from 1000000, the initial conditions and the outcome). The first
check of this environment is to evaluate them on the same seeds and compare episode by episode. Their
`log.log` shows `task_config: demo`; `demo.yml` and `clean.yml` differ only in `render_frequency` (GUI window),
so headless evaluation uses `clean`.
