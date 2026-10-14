#!/bin/bash
# Evaluate our ACT trained on the released isaac45 demos (train_rel45.sh) on Isaac 4.5 (RTX 5090 render overrides):
#   eval_rel45.sh <task> <train config> <env: pub|rel> <start_seed> <count>
#   pub = public code as is; rel = release-matched env, eval_policy.py --env_config rel (task_config/env/rel.yml:
#   marker noise 0, a9abc0d nearest-face marker binding, grasp threshold 28.05 on pull_out_key / insert_hole /
#   insert_HDMI / insert_tube / lift_bottle); verified on one seed per task against the release (evidence/).
# Stage 2 copy of eval_rel45.sh (seeds 1000010-99, user go 2026-10-13 night): identical env and command; the only
# differences are the record file rundir_<task>_<cfg>_<env>_s2.txt (keeps the stage-1 10-seed record) and the
# timeout budget of 200 s per episode (longest stage-1 episode 159 s).
set -uo pipefail
TASK=$1; CFG=$2; ENV=$3; START=$4; COUNT=$5
D=/home/yusei/.claude/jobs/7fcd22c8/tmp/rel45
ROOT=/home/yusei/Desktop/Robotics/univtac-isaac45
PY=/home/yusei/miniforge3/envs/UniVTAC45/bin/python
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 TRAIN_CONFIG=$CFG EP_NUM=50
export UNIVTAC45_RENDER_OVERRIDES="/rtx/directLighting/sampledLighting/enabled=false;/rtx-transient/dldenoiser/enabled=false"
case $ENV in
  pub) ENV_ARGS=() ;;
  rel) ENV_ARGS=(--env_config rel) ;;
  *) echo "env pub or rel" >&2; exit 2 ;;
esac
CK=$ROOT/policy/ACT/act_ckpt/act-$TASK/clean-50/$CFG
[ -f "$CK/policy_last.ckpt" ] && [ -f "$CK/dataset_stats.pkl" ] || { echo "missing checkpoint in $CK" >&2; exit 1; }
cd "$ROOT"
BEFORE=$(ls -d eval_result/ACT/$TASK/deploy/* 2>/dev/null | sort | tail -1)
echo "eval $TASK $CFG env $ENV seeds $START+$COUNT $(date)"
timeout -k 30 $((600 + 200 * COUNT)) "$PY" scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py \
  "$TASK" clean ACT/deploy --start_seed "$START" --max_seed $((START + COUNT - 1)) --total_num "$COUNT" --headless \
  "${ENV_ARGS[@]}"
CODE=$?
RUN=$(ls -d eval_result/ACT/$TASK/deploy/* 2>/dev/null | sort | tail -1)
if [ -n "$RUN" ] && [ "$RUN" != "$BEFORE" ]; then
  echo "$ROOT/$RUN" > "$D/rundir_${TASK}_${CFG}_${ENV}_s2.txt"
  LINE="[rel45 s2] $TASK $CFG $ENV: $(grep 'Final Result' "$RUN/log.log" | tail -1 | sed 's/.*Final/Final/')"
else
  LINE="[rel45 s2] $TASK $CFG $ENV exit $CODE, no result folder"
fi
echo "done $LINE $(date)"
~/.claude/scripts/discord-notify.sh "$LINE" || true
exit $CODE
