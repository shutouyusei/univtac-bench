# Isaac Sim 4.5 on RTX 50-series

Builds the environment of this upstream `main` checkout on an RTX 5090 (sm_120). Nothing outside
`scripts/isaac45/` changes, so results stay comparable with the numbers published on the benchmark.

```bash
bash scripts/isaac45/install.sh              # all steps into conda env UniVTAC45
bash scripts/isaac45/install.sh --from uipc  # resume at a step
bash scripts/isaac45/install.sh --locked     # every pip version from requirements.lock
bash scripts/isaac45/install.sh --check      # versions and GPU only
```

## Deviations from upstream `scripts/install.sh`

Upstream pins CUDA 12.4 and torch 2.5.1+cu124, which have no sm_120 kernels. Everything else follows
upstream; the remaining values come from the Isaac Sim 5.1 build of the same libuipc (`isaac51` branch).

| Item | Upstream | Here |
|---|---|---|
| toolchain (`env.yaml`) | CUDA 12.4, GCC 11 | CUDA 12.8, conda-forge GCC 12, host linker (`toolchains/`) |
| torch / torchvision / torch_scatter | 2.5.1+cu124 | 2.7.0+cu128 / 0.22.0 / 2.1.2+pt27cu128 |
| pip constraints | none | torch, torchvision, numpy 1.26.4 (`constraints.txt`); `--locked` adds `requirements.lock` |
| vcpkg | latest in `~/Toolchain`, written to `~/.bashrc` | dd3097e in `~/.cache/univtac-isaac45/vcpkg`, own binary cache, `tinygltf` overlay port |
| CUDA arch | detected | `UNIVTAC45_CUDA_ARCH` (120; 89 for RTX 40, its PTX runs on RTX 50) |
| TacEx assets | as checked out | `assets` step fetches the 114 LFS objects upstream commits as pointer text |
| extras | transforms3d trimesh tetgen | + `ipython<9` (policy/ACT imports IPython; 9 conflicts with rl_games) |
| sudo, ~/.bashrc, demo runs | yes | none |

## Camera noise on RTX 50-series

With the default rendering preset the head and wrist cameras render grain-noisy on the RTX 5090 (tactile
images are clean) and the released ACT checkpoints fail (insert_hole seeds 1000000-19: 0-1/20 vs the authors'
6/20). Reported upstream as univtac/UniVTAC#16 and IsaacLab discussion #2869; fixed in Isaac Sim 5.0.
Switching off sampled direct lighting and the DL denoiser gives clean images:

```bash
UNIVTAC45_RENDER_OVERRIDES="/rtx/directLighting/sampledLighting/enabled=false;/rtx-transient/dldenoiser/enabled=false" \
  python scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py insert_hole clean ACT/deploy ...
```

The wrapper applies the settings after Isaac Lab's preset and runs the upstream script unchanged.
RTX 40-series GPUs render cleanly without it.

## Environment built 2026-10-13

RTX 5090, driver 580.178.04, Ubuntu 24.04, upstream 0dafa10. `requirements.lock` is the `pip freeze`
(editable lines carry this machine's paths). Key versions: isaacsim 4.5.0.0, torch 2.7.0+cu128, numpy 1.26.4,
isaaclab 0.41.3 (v2.1.1, 90b79bb2d), cuRobo 0a50de1, warp-lang 1.0.0, tacex 0.1.0, pyuipc 0.9.0.

## Reproduction check

The released checkpoints (`data/checkpoints/<task>/{univtac,vision_only}`) ship the authors' evaluation
(`metadata.json`, seeds from 1000000, task config `demo` = `clean` + GUI). Evaluate them on the same seeds
with `clean` and compare per episode.
