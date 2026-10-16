#!/bin/bash
# Evaluate a stflow checkpoint on Isaac Sim 4.5 in the public environment (upstream main, no env patches):
#   eval_stflow_pub45.sh <task> <task_config> <deploy yml> <start_seed> <count>
# Called by streaming-tactile-flow scripts/run_experiment.py --eval-script, which puts stflow on PYTHONPATH and
# sets OMNI_KIT_ACCEPT_EULA. Runs in the 4.5 checkout UNIVTAC45_ROOT (default ~/Desktop/Robotics/univtac-isaac45),
# writing eval_result/stflow/<task>/<deploy yml stem>/<time> there. That checkout's policy/stflow must be a copy of
# this branch's policy/stflow (a symlink would make the adapter resolve the repo root to this branch instead), and
# is checked file by file before every run.
# The render overrides remove the RTX 5090 camera noise; RunPod 4090 jobs run without them, so numbers from the
# 5090 compare only with other 5090 numbers.
set -uo pipefail
TASK=$1; TCFG=$2; YML=$3; START=$4; COUNT=$5
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRANCH_ROOT=$(cd "$HERE/../.." && pwd)
ROOT=${UNIVTAC45_ROOT:-$HOME/Desktop/Robotics/univtac-isaac45}
PY=${UNIVTAC45_PY:-$HOME/miniforge3/envs/UniVTAC45/bin/python}
: "${OMNI_KIT_ACCEPT_EULA:?must be set by the caller, who accepted the Omniverse EULA}"
if ! diff -rq -x __pycache__ "$BRANCH_ROOT/policy/stflow" "$ROOT/policy/stflow"; then
  echo "$ROOT/policy/stflow differs from $BRANCH_ROOT/policy/stflow; copy it again before evaluating" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export UNIVTAC45_RENDER_OVERRIDES="/rtx/directLighting/sampledLighting/enabled=false;/rtx-transient/dldenoiser/enabled=false"
cd "$ROOT"
echo "stflow pub eval $TASK $TCFG $YML seeds $START+$COUNT policy/stflow from $BRANCH_ROOT $(date)"
timeout -k 30 $((600 + 200 * COUNT)) "$PY" scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py \
  "$TASK" "$TCFG" "$YML" --start_seed "$START" --max_seed $((START + COUNT - 1)) --total_num "$COUNT" --headless
