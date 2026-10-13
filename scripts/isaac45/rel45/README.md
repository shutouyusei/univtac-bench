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
