#!/bin/bash
# UniVTAC-ACT baseline on Isaac 5.1, one task and one stage per call:
#
#   bash policy/ACT/baseline.sh <task> prepare   # link the data, convert 50 demonstrations for ACT
#   bash policy/ACT/baseline.sh <task> train     # 4000 steps x batch 64 (repo recipe); skipped if trained
#   bash policy/ACT/baseline.sh <task> eval      # seeds START_SEED .. START_SEED+SEEDS-1, shared protocol
#
# ACT_ARM=tactile (default: UniVTAC's pretrained tactile encoder, data/checkpoints/encoder.pth, fine-tuned)
# or vision. Tasks whose task_settings camera_type is "all" (lift_can, insert_tube) use the *_all configs
# (head + wrist camera), as the UniVTAC repository does. Environment: EP_NUM (50), SEEDS (100),
# START_SEED (1000000), TASK_CONFIG (clean51), UNIVTAC_DATA (the main checkout's data folder, linked into
# this checkout's data/ when missing), PY (the UniVTAC env's python), DRY_RUN=1 prints the commands only.
# Outputs: policy/ACT/act_ckpt/act-<task>/<config>-<EP_NUM>/<train config>/ (checkpoint,
# act_<train config>.yml, eval_<start>_<count>.json/.log) and eval_result/ACT/<task>/act_<train config>/.
set -euo pipefail

TASKS="grasp_classify insert_HDMI insert_hole insert_tube lift_bottle lift_can pull_out_key put_bottle_in_shelf"
TASK=${1:-}
STAGE=${2:-}
if [ -z "$TASK" ] || [ -z "$STAGE" ]; then
  echo "usage: baseline.sh <task> <prepare|train|eval>" >&2; exit 2
fi
case " $TASKS " in *" $TASK "*) ;; *) echo "unknown task $TASK (one of: $TASKS)" >&2; exit 2 ;; esac

ACT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$ACT_DIR/../.." && pwd)
TASK_CONFIG=${TASK_CONFIG:-clean51}
EP_NUM=${EP_NUM:-50}
SEEDS=${SEEDS:-100}
START_SEED=${START_SEED:-1000000}
ARM=${ACT_ARM:-tactile}
UNIVTAC_DATA=${UNIVTAC_DATA:-$HOME/Desktop/Robotics/univtac-bench/data}
PY=${PY:-$HOME/miniforge3/envs/UniVTAC/bin/python}

CAMERA=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], {}).get('camera_type', 'head'))" \
  "$REPO/policy/task_settings.json" "$TASK")
SUFFIX=""
[ "$CAMERA" = "all" ] && SUFFIX="_all"
case "$ARM" in
  tactile) CFG="train_config$SUFFIX" ;;
  vision) CFG="train_config_vision$SUFFIX" ;;
  *) echo "ACT_ARM must be tactile or vision, got $ARM" >&2; exit 2 ;;
esac
CKPT_REL="./act_ckpt/act-$TASK/$TASK_CONFIG-$EP_NUM/$CFG"
CKPT="$ACT_DIR/${CKPT_REL#./}"

run() {
  echo "+ $*"
  if [ "${DRY_RUN:-0}" != "1" ]; then "$@"; fi
}

link_data() {
  for entry in "$TASK" checkpoints; do
    if [ ! -e "$REPO/data/$entry" ]; then
      run ln -s "$UNIVTAC_DATA/$entry" "$REPO/data/$entry"
    fi
  done
}

case "$STAGE" in
  prepare)
    link_data
    if [ -d "$ACT_DIR/data/sim-$TASK/$TASK_CONFIG-$EP_NUM" ]; then
      echo "ACT data for $TASK already converted"
    else
      cd "$ACT_DIR"
      run "$PY" process_data.py "$TASK" "$TASK_CONFIG" "$EP_NUM"
    fi
    ;;
  train)
    if [ -f "$CKPT/policy_last.ckpt" ]; then
      echo "already trained: $CKPT/policy_last.ckpt"; exit 0
    fi
    cd "$ACT_DIR"
    if [ "${DRY_RUN:-0}" != "1" ]; then mkdir -p "$CKPT"; fi
    run "$PY" imitate_episodes.py --task_name "sim-$TASK-$TASK_CONFIG-$EP_NUM" --ckpt_dir "$CKPT_REL" \
      --config_path "./$CFG.yml" --seed 0
    ;;
  eval)
    if [ "${DRY_RUN:-0}" != "1" ] && [ ! -f "$CKPT/policy_last.ckpt" ]; then
      echo "not trained: $CKPT/policy_last.ckpt" >&2; exit 1
    fi
    DEPLOY="$CKPT/act_$CFG.yml"
    if [ "${DRY_RUN:-0}" != "1" ]; then
      printf 'policy_name: ACT\nseed: 0\nckpt_setting: %s\ninstruction_type: seen\ninstuction_file: null\n' \
        "$EP_NUM" > "$DEPLOY"
    fi
    cd "$REPO"
    run "$PY" scripts/baseline_eval.py --task "$TASK" --task-config "$TASK_CONFIG" --deploy "$DEPLOY" \
      --start "$START_SEED" --count "$SEEDS" --out "$CKPT/eval_${START_SEED}_${SEEDS}.json" \
      --env "TRAIN_CONFIG=$CFG" --env "EP_NUM=$EP_NUM"
    ;;
  *) echo "stage must be prepare, train or eval, got $STAGE" >&2; exit 2 ;;
esac
