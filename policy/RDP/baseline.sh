#!/bin/bash
# Reactive Diffusion Policy baseline on Isaac 5.1, one stage per call:
#
#   bash policy/RDP/baseline.sh all pca          # marker PCA over the 50 training demos of every task (once)
#   bash policy/RDP/baseline.sh <task> prepare   # link the data, cache the task's 50 demonstrations
#   bash policy/RDP/baseline.sh <task> train_at  # asymmetric tokenizer, 601 epochs (resumes, skips when done)
#   bash policy/RDP/baseline.sh <task> train_ldp # latent diffusion policy, 401 epochs (resumes, skips when done)
#   bash policy/RDP/baseline.sh <task> eval      # seeds START_SEED .. START_SEED+SEEDS-1, shared protocol
#
# Environment: EP_NUM (50), SEEDS (100), START_SEED (1000000), TASK_CONFIG (clean51), DEVICE (cuda:0),
# UNIVTAC_DATA (the main checkout's data folder, linked into this checkout's data/ when missing), PY (the
# UniVTAC env's python), RDP_SET (extra --set overrides for training, e.g. for a smoke run), DRY_RUN=1 prints
# the commands only. Outputs: policy/RDP/data/ (cache, marker_pca.npz), policy/RDP/rdp_ckpt/<task>-<config>-<EP>/
# (at_latest.pt, ldp_latest.pt, rdp_policy.pt, train_log.jsonl, rdp.yml, eval_<start>_<count>.json/.log) and
# eval_result/RDP/<task>/rdp/. The eval stage uses scripts/baseline_eval.py.
set -euo pipefail

TASKS="grasp_classify insert_HDMI insert_hole insert_tube lift_bottle lift_can pull_out_key put_bottle_in_shelf"
TASK=${1:-}
STAGE=${2:-}
if [ -z "$TASK" ] || [ -z "$STAGE" ]; then
  echo "usage: baseline.sh <task|all> <pca|prepare|train_at|train_ldp|eval>" >&2; exit 2
fi
if [ "$STAGE" = "pca" ]; then
  [ "$TASK" = "all" ] || { echo "the pca stage runs over all tasks: baseline.sh all pca" >&2; exit 2; }
else
  case " $TASKS " in *" $TASK "*) ;; *) echo "unknown task $TASK (one of: $TASKS)" >&2; exit 2 ;; esac
fi

RDP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$RDP_DIR/../.." && pwd)
TASK_CONFIG=${TASK_CONFIG:-clean51}
EP_NUM=${EP_NUM:-50}
SEEDS=${SEEDS:-100}
START_SEED=${START_SEED:-1000000}
DEVICE=${DEVICE:-cuda:0}
UNIVTAC_DATA=${UNIVTAC_DATA:-$HOME/Desktop/Robotics/univtac-bench/data}
PY=${PY:-$HOME/miniforge3/envs/UniVTAC/bin/python}
export PYTHONPATH="$RDP_DIR:$REPO${PYTHONPATH:+:$PYTHONPATH}"
PCA="$RDP_DIR/data/marker_pca.npz"
CACHE="$RDP_DIR/data/$TASK-$TASK_CONFIG-$EP_NUM"
CKPT="$RDP_DIR/rdp_ckpt/$TASK-$TASK_CONFIG-$EP_NUM"

run() {
  echo "+ $*"
  if [ "${DRY_RUN:-0}" != "1" ]; then "$@"; fi
}

link_data() {
  for t in "$@"; do
    if [ ! -e "$REPO/data/$t" ]; then run ln -s "$UNIVTAC_DATA/$t" "$REPO/data/$t"; fi
  done
}

train() {
  # shellcheck disable=SC2086
  run "$PY" -m rdp_univtac.train "$1" --cache "$CACHE" --pca "$PCA" --out "$CKPT" --device "$DEVICE" \
    ${RDP_SET:+--set $RDP_SET}
}

cd "$RDP_DIR"
case "$STAGE" in
  pca)
    # shellcheck disable=SC2086
    link_data $TASKS
    run "$PY" -m rdp_univtac.prepare pca --tasks $TASKS --task-config "$TASK_CONFIG" --episodes "$EP_NUM" --out "$PCA"
    ;;
  prepare)
    link_data "$TASK"
    if [ -f "$CACHE/meta.json" ]; then
      echo "already cached: $CACHE"
    else
      run "$PY" -m rdp_univtac.prepare cache --task "$TASK" --task-config "$TASK_CONFIG" --episodes "$EP_NUM" --out "$CACHE"
    fi
    ;;
  train_at) train at ;;
  train_ldp) train ldp ;;
  eval)
    if [ "${DRY_RUN:-0}" != "1" ] && [ ! -f "$CKPT/rdp_policy.pt" ]; then
      echo "not trained: $CKPT/rdp_policy.pt" >&2; exit 1
    fi
    DEPLOY="$CKPT/rdp.yml"
    if [ "${DRY_RUN:-0}" != "1" ]; then
      printf 'policy_name: RDP\nseed: 0\ninstruction_type: seen\ninstuction_file: null\nrdp_bundle: %s\nrdp_device: %s\n' \
        "$CKPT/rdp_policy.pt" "$DEVICE" > "$DEPLOY"
    fi
    cd "$REPO"
    run "$PY" scripts/baseline_eval.py --task "$TASK" --task-config "$TASK_CONFIG" --deploy "$DEPLOY" \
      --start "$START_SEED" --count "$SEEDS" --out "$CKPT/eval_${START_SEED}_${SEEDS}.json"
    ;;
  *) echo "stage must be pca, prepare, train_at, train_ldp or eval, got $STAGE" >&2; exit 2 ;;
esac
