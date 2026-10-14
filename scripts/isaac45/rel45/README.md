# rel45: the release-matched Isaac Sim 4.5 environment

The scripts that produced the Isaac 4.5 results (ACT baselines trained on the released demos). `train/` and
`evidence/` are **as-run copies**; `env/` selects the environment with `--env_config` (below), which replaced the
import-time wrappers the runs used. Absolute paths point at the run machine (checkout
`~/Desktop/Robotics/univtac-isaac45`, env `UniVTAC45`, a job scratch directory for records).
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

`rel` = `scripts/eval_policy.py` / `scripts/collect_data.py` with `--env_config rel`
(`task_config/env/rel.yml`); `pub` = the same without it. The three overrides, each defaulting to the public value:

- `marker_noise_level: 0` (`ManiSkillSimulatorCfg`, public 80): solid marker dots; 0 also skips the noise draw.
- `marker_weight_method: nearest_face` (public `delaunay`): the a9abc0d `_gen_marker_weight` body; main's
  `setup()` is kept (no `init_vertices()` re-call).
- `adaptive_grasp_depth_threshold`: 28.05 mm on pull_out_key, insert_hole, insert_HDMI, insert_tube, lift_bottle;
  the other tasks keep the public threshold.

The stage 1 / stage 2 runs applied the same three overrides as import-time wrappers (`run_marker_noise.py`,
`run_old_marker.py` with `OLD_MARKER_SETUP=0`, `run_grasp_thr.py`). The flag code carries their function bodies
unchanged (AST-identical apart from where the noise level is read); the results were not re-run with the flags.
`evidence/release_match/run_task_seed0.sh` still names those wrappers: variant `G28.05` = `--env_config rel`.

`eval_rel45.sh <task> <cfg> <pub|rel> <start_seed> <count>` runs `scripts/eval_policy.py` through
`scripts/isaac45/run_with_render_overrides.py`. `eval_rel45_s2.sh` is the identical command for seeds
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

## evidence/ - why this environment is trusted

`released_ckpts/`: the released ACT checkpoints evaluated in the public 4.5 checkout (no rel patches, see above) on
seeds 1000000-09, against the authors' own shipped evaluation (`data/checkpoints/<task>/<arm>/metadata.json`, same
seeds). Report scripts were fixed before the runs.

| arm | ours, 8 tasks x 10 seeds | authors, same seeds | authors' 100-seed rate inside our interval |
|---|---|---|---|
| vision_only | 29/80 | 30/80 | 8/8 tasks |
| univtac (tactile) | 27/80 | 33/80 | 7/8 tasks (put_bottle_in_shelf: 3/10 vs 8 %) |

`release_match/`: one-seed collects compared with the released demo of the same seed (`compare_*.py`, fixed
before the runs). Variant `G28.05` of `run_task_seed0.sh` is exactly `rel`. Recorded coverage:

- pull_out_key, insert_HDMI, insert_tube, lift_bottle (`compare_tasks.md`): `G28.05` vs the demo, next to the public
  grasp (`G`) and a second demo as reference. E.g. insert_HDMI finger 4.28 vs demo 4.27 mm, frame-0 depth diff 0.01 mm.
- insert_hole (`compare_seed0.md`): variants A-E only; threshold 28.05 gives finger 6.29 vs demo 6.26 mm (public
  5.72). No `G28.05` record of insert_hole is kept, and the collect outputs themselves were deleted.
- lift_can, put_bottle_in_shelf, grasp_classify: no record (these use the public grasp threshold in `rel`).
- `run_task_seed0.sh` variants F, R2-R11 need scripts not included here (marker-axis and render experiments).
