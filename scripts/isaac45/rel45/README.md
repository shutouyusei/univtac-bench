# rel45: the release-matched Isaac Sim 4.5 environment

What it takes to train ACT on the released isaac45 demos and evaluate it in the environment those demos came
from, on top of `scripts/isaac45/` (install, render overrides). `train/` holds the helpers the upstream pipeline
needs. The per-machine drivers that ran the experiments
(tsp jobs, absolute paths, notifications) are not in the repo.

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

## The environment: `--env_config rel`

`rel` = `scripts/eval_policy.py` / `scripts/collect_data.py` with `--env_config rel`
(`task_config/env/rel.yml`); `pub` = the same without it. The three overrides, each defaulting to the public value:

- `marker_noise_level: 0` (`ManiSkillSimulatorCfg`, public 80): solid marker dots; 0 also skips the noise draw.
- `marker_weight_method: nearest_face` (public `delaunay`): the a9abc0d `_gen_marker_weight` body; main's
  `setup()` is kept (no `init_vertices()` re-call).
- `adaptive_grasp_depth_threshold`: 28.05 mm on pull_out_key, insert_hole, insert_HDMI, insert_tube, lift_bottle;
  the other tasks keep the public threshold.

```
python scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py <task> clean ACT/deploy \
    --start_seed 1000000 --max_seed 1000009 --total_num 10 --headless --env_config rel
```

`tacex` is an editable install: every checkout imports TacEx from the checkout it was installed from, so that
checkout must carry the TacEx flags too. With an older TacEx, `BaseTask` raises instead of running with public
markers.

The stage 1 / stage 2 evaluations (10 and 100 seeds) applied the same three overrides as import-time wrappers
chained in front of `eval_policy.py`; the flag code carries their function bodies unchanged (AST-identical apart
from where the noise level is read). Those results were not re-run with the flags.

## train/ - ACT on the released demos, official recipe

Per task, from the checkout root (`policy/ACT` for the upstream steps):

1. `train/fetch_release.py <checkout> <task>`: downloads demos 0-49 and links `data/<task>/release45`.
2. `process_data.py <task> release45 50` (upstream).
3. `train/rename_cam_key.py <checkout> <task>` and `train/trim_state_dim.py <checkout> <task>` on the processed copy.
4. `imitate_episodes.py --task_name sim-<task>-release45-50 --ckpt_dir act_ckpt/act-<task>/clean-50/<cfg>
   --config_path <cfg>.yml --seed 0` (upstream: 4000 steps, batch 64, policy_last). The ckpt dir is where
   `deploy_policy.py` looks for task config `clean` and `TRAIN_CONFIG=<cfg>`.

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

## Why this environment is trusted

Two checks, run before the stage 1 / stage 2 evaluations (reports to be added separately):

- The released ACT checkpoints evaluated in the public 4.5 checkout (`pub`) on seeds 1000000-09, against the
  authors' own shipped evaluation (`data/checkpoints/<task>/<arm>/metadata.json`, same seeds):

  | arm | ours, 8 tasks x 10 seeds | authors, same seeds | authors' 100-seed rate inside our interval |
  |---|---|---|---|
  | vision_only | 29/80 | 30/80 | 8/8 tasks |
  | univtac (tactile) | 27/80 | 33/80 | 7/8 tasks (put_bottle_in_shelf: 3/10 vs 8 %) |

- One-seed collects in `rel` compared with the released demo of the same seed. Recorded coverage:
  - pull_out_key, insert_HDMI, insert_tube, lift_bottle: `rel` vs the demo, next to the public grasp and a second
    demo as reference. E.g. insert_HDMI finger 4.28 vs demo 4.27 mm, frame-0 depth diff 0.01 mm.
  - insert_hole: threshold-only variants; threshold 28.05 gives finger 6.29 vs demo 6.26 mm (public 5.72). No
    record of the full `rel` variant is kept.
  - lift_can, put_bottle_in_shelf, grasp_classify: no record (these use the public grasp threshold in `rel`).
