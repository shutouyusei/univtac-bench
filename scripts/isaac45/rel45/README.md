# rel45: the release-matched Isaac Sim 4.5 environment

Everything here is an **as-run copy** of the scripts that produced the Isaac 4.5 results (ACT baselines trained on
the released demos). Absolute paths point at the run machine (checkout `~/Desktop/Robotics/univtac-isaac45`, env
`UniVTAC45`, a job scratch directory for records); they are kept unchanged so the reviewed code is the code that ran.
All of it runs on top of the environment of `scripts/isaac45/` (install, render overrides).

## Why a release-matched environment

The released isaac45 demos (HF `byml/UniVTAC`, `isaac45/<task>/hdf5`) were collected before the public code
reached its current state. Training on those demos and evaluating in the public code mixes two environments:

| | released isaac45 demos | public code (main 0dafa10) |
|---|---|---|
| adaptive-grasp depth threshold (gsmini) | ~28.05 mm (finger e.g. 4.27 mm on insert_HDMI) | 27.5 mm (finger 3.74 mm) |
| marker image | solid black dots | dots + uniform noise 0-80 (`mani_skill_sim.py` draw_markers) |
| marker weights | a9abc0d nearest-face barycentric | Delaunay on recentred points (TacEx 35bc438) |

The released ACT checkpoints were not trained on the released demos (their `dataset_stats.pkl` grip matches the
public code, not the demos), so evaluating *them* in the public code is consistent. Our ACT, trained on the released
demos, is evaluated in `rel`, the environment the demos came from.

## env/ - the evaluation environment

`rel` = public checkout unchanged, plus three import-time patches chained in front of the upstream script:

- `run_marker_noise.py` (MARKER_NOISE=0): verbatim `draw_markers` with the noise level from the environment.
- `run_old_marker.py` (OLD_MARKER_SETUP=0): the a9abc0d `_gen_marker_weight` body; main's `setup()` is kept.
- `run_grasp_thr.py` (GRASP_THR=28.05): overrides the resolved adaptive-grasp threshold. Applied to
  pull_out_key, insert_hole, insert_HDMI, insert_tube, lift_bottle.

`eval_rel45.sh <task> <cfg> <pub|rel> <start_seed> <count>` runs `scripts/eval_policy.py` through these wrappers
and `scripts/isaac45/run_with_render_overrides.py`. `eval_rel45_s2.sh` is the identical command for seeds
1000010-99; it writes a separate record file and allows 200 s per episode.

## train/ - ACT on the released demos, official recipe

Per task (`run_task_rel45_v2.sh <task>`): `fetch_release.py` downloads demos 0-49 and links
`data/<task>/release45`; `train_rel45.sh <task> <cfg>` runs the upstream `process_data.py <task> release45 50`, two
fixes on our processed copy, then the upstream `imitate_episodes.py` (4000 steps, batch 64, seed 0, policy_last).

Fixes needed to train the upstream pipeline at all (released demos untouched):

- `rename_cam_key.py`: the preprocessor saves the head camera as `cam_head`; ACT trains and deploys `cam_high`.
- `trim_state_dim.py`: the preprocessor stores 9 joints (two equal fingers); ACT and deploy use 8.
- `policy/ACT/task_settings.json -> ../task_settings.json`: `process_data.py` reads the file next to itself, deploy
  reads `policy/task_settings.json`; without the link lift_can / insert_tube are processed head-only.
  (`process_data.py` reads the key `downsample_factor`, the file has `downsample`; every value is 1, no effect.)

Configs `policy/ACT/train_config_{,vision_}rel45{,_all}.yml` = upstream `train_config{,_vision}{,_all}.yml` plus
`num_epochs` (detr/main.py needs it to skip argparse; unused, training stops on num_steps) and, for the tactile arm,
`tactile_ckpt` pointing at the released encoder (`data/checkpoints/encoder.pth`; the upstream path does not exist
and is skipped silently). `_all` (head + wrist camera) for lift_can and insert_tube, as `task_settings.json` says.
