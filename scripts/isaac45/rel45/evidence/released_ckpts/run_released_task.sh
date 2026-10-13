#!/bin/bash
# Released ACT checkpoint of any task in the Isaac Sim 4.5 checkout, upstream eval script through the render
# override wrapper (UNIVTAC45_RENDER_OVERRIDES from the caller):
#   run_released_task.sh <task> <arm: univtac|vision_only> <train config name> <start_seed> <count> <tag>
# Links the checkpoint (policy_last.ckpt, dataset_stats.pkl) where policy/ACT/deploy_policy.py looks, evaluates,
# then compares per seed with the authors' 4.5 metadata (compare_released.py) into
# compare_<task>_<arm>_<start>_<count>_<tag>.md. OMNI_KIT_ACCEPT_EULA comes from the caller (user's acceptance).
set -uo pipefail
TASK=$1; ARM=$2; CFG=$3; START=$4; COUNT=$5; TAG=$6
ROOT=/home/yusei/Desktop/Robotics/univtac-isaac45
PY=/home/yusei/miniforge3/envs/UniVTAC45/bin/python
OUT=/home/yusei/.claude/jobs/7fcd22c8/tmp/isaac45
SRC=$ROOT/data/checkpoints/$TASK/$ARM
DST=$ROOT/policy/ACT/act_ckpt/act-$TASK/clean-50/$CFG
mkdir -p "$DST"
for f in policy_last.ckpt dataset_stats.pkl; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f" >&2; exit 1; }
  ln -sfn "$SRC/$f" "$DST/$f"
done
cd "$ROOT"
BEFORE=$(ls -d eval_result/ACT/$TASK/deploy/* 2>/dev/null | sort | tail -1)
echo "released $TASK $ARM ($CFG) seeds $START+$COUNT tag $TAG overrides [${UNIVTAC45_RENDER_OVERRIDES:-}] $(git rev-parse --short HEAD) $(date)"
export CUDA_VISIBLE_DEVICES=0 TRAIN_CONFIG=$CFG EP_NUM=50
"$PY" scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py "$TASK" clean ACT/deploy \
  --start_seed "$START" --max_seed $((START + COUNT - 1)) --total_num "$COUNT" --headless
CODE=$?
RUN=$(ls -d eval_result/ACT/$TASK/deploy/* 2>/dev/null | sort | tail -1)
MD=$OUT/compare_${TASK}_${ARM}_${START}_${COUNT}_${TAG}.md
if [ -z "$RUN" ] || [ "$RUN" = "$BEFORE" ]; then
  echo "no new eval_result folder" >&2; LINE="[isaac45] $TASK $ARM $TAG exit $CODE, no result"
else
  echo "$ROOT/$RUN" > "$OUT/rundir_${TASK}_${ARM}_${TAG}.txt"
  "$PY" -I "$OUT/compare_released.py" "$TASK" "$ARM" "$ROOT/$RUN" | tee "$MD"
  LINE="[isaac45] $TASK $ARM $TAG seeds $START+$COUNT exit $CODE: $(tail -1 "$MD")"
fi
echo "done $LINE $(date)"
~/.claude/scripts/discord-notify.sh "$LINE" || true
exit $CODE
