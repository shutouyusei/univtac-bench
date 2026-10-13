#!/bin/bash
# Stage 1, evaluation in the release-matched env (user decision 2026-10-13 22:2x: "残りタスク止めて, rel で評価"):
#   run_task_rel45_v2.sh <task>
# Same stages as run_task_rel45.sh (fetch release demos 0-49, train ACT tactile + vision-only with the official
# recipe, delete our processed copy) but the 10-seed evaluation runs in env `rel` (eval_rel45.sh rel: grasp threshold
# 28.05 on pull_out_key / insert_hole / insert_HDMI / insert_tube / lift_bottle, marker noise 0, a9abc0d marker
# weights), i.e. the environment the released demos were collected in. Stages whose output exists are skipped, so
# tasks already trained (pull_out_key, insert_HDMI) only evaluate.
set -euo pipefail
TASK=$1
D=/home/yusei/.claude/jobs/7fcd22c8/tmp/rel45
ROOT=/home/yusei/Desktop/Robotics/univtac-isaac45
PY=/home/yusei/miniforge3/envs/UniVTAC45/bin/python
case $TASK in
  lift_can|insert_tube) CFGS=(train_config_rel45_all train_config_vision_rel45_all) ;;
  *) CFGS=(train_config_rel45 train_config_vision_rel45) ;;
esac
echo "== stage 1 (rel) $TASK start $(date)"
"$PY" -I "$D/fetch_release.py" "$TASK"
for c in "${CFGS[@]}"; do
  CK=$ROOT/policy/ACT/act_ckpt/act-$TASK/clean-50/$c
  if [ -f "$CK/policy_last.ckpt" ] && [ -f "$CK/dataset_stats.pkl" ]; then echo "have $CK"; continue; fi
  bash "$D/train_rel45.sh" "$TASK" "$c"
done
PROC=$ROOT/policy/ACT/data/sim-$TASK/release45-50
if [ -d "$PROC" ]; then echo "removing processed copy $PROC ($(du -sh "$PROC" | cut -f1))"; rm -rf "$PROC"; fi
for c in "${CFGS[@]}"; do
  if [ -f "$D/rundir_${TASK}_${c}_rel.txt" ]; then echo "have eval $TASK $c rel"; continue; fi
  bash "$D/eval_rel45.sh" "$TASK" "$c" rel 1000000 10 || echo "eval $TASK $c rel exit $?"
done
df -h /home | tail -1
echo "== stage 1 (rel) $TASK done $(date)"
