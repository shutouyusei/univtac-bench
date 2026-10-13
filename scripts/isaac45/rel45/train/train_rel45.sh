#!/bin/bash
# Train ACT on the first 50 released isaac45 demos of one task with the authors' recipe (policy/ACT/train.sh:
# imitate_episodes.py, 4000 steps, batch 64, seed 0), on Isaac 4.5 checkout univtac-isaac45.
#   train_rel45.sh <task> <train config: train_config_rel45 | train_config_vision_rel45 | *_all for lift_can, insert_tube>
# Raw demos: data/<task>/release45 -> ../isaac45/<task> (released files, unchanged). Processed data:
# policy/ACT/data/sim-<task>/release45-50, then two fixes on OUR processed files only (both idempotent):
# rename_cam_key.py (cam_head -> cam_high, the key ACT trains and deploys with) and trim_state_dim.py (qpos/action
# 9 -> 8 dims as deploy feeds joint[:8]). Checkpoint dir act_ckpt/act-<task>/clean-50/<cfg>, i.e. where
# deploy_policy.py looks when evaluating with task config clean and TRAIN_CONFIG=<cfg>; the released checkpoints
# (train_config / train_config_vision symlinks) are not touched. User go 2026-10-13 (IH/GC), stage 1 (6 tasks).
set -euo pipefail
TASK=$1; CFG=$2
D=/home/yusei/.claude/jobs/7fcd22c8/tmp/rel45
A=/home/yusei/Desktop/Robotics/univtac-isaac45/policy/ACT
PY=/home/yusei/miniforge3/envs/UniVTAC45/bin/python
export CUDA_VISIBLE_DEVICES=0
cd "$A"
echo "train $TASK $CFG start $(date)"
if [ ! -d "./data/sim-$TASK/release45-50" ]; then
  echo "processing data $(date)"
  "$PY" process_data.py "$TASK" release45 50
fi
"$PY" -I "$D/rename_cam_key.py" "$TASK"
"$PY" -I "$D/trim_state_dim.py" "$TASK"
CK="./act_ckpt/act-$TASK/clean-50/$CFG"
[ -e "$CK/policy_last.ckpt" ] && { echo "refusing: $CK/policy_last.ckpt exists" >&2; exit 1; }
echo "training $(date)"
"$PY" imitate_episodes.py --task_name "sim-$TASK-release45-50" --ckpt_dir "$CK" --config_path "./$CFG.yml" --seed 0
ls -la "$CK"
[ -f "$CK/policy_last.ckpt" ] && [ -f "$CK/dataset_stats.pkl" ] || { echo "training left no policy_last/stats in $CK" >&2; exit 1; }
echo "train $TASK $CFG done $(date)"
~/.claude/scripts/discord-notify.sh "[rel45] trained $TASK $CFG" || true
