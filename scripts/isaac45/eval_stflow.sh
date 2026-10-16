#!/bin/bash
# Evaluate a stflow checkpoint on Isaac Sim 4.5; run from a UniVTAC checkout by stflow's
# run_experiment.py (stflow on PYTHONPATH, OMNI_KIT_ACCEPT_EULA set):
#   eval_stflow.sh <task> <task_config> <deploy yml> <start_seed> <count>
set -uo pipefail
TASK=$1; TCFG=$2; YML=$3; START=$4; COUNT=$5
PY=${UNIVTAC45_PY:-$HOME/miniforge3/envs/UniVTAC45/bin/python}
: "${OMNI_KIT_ACCEPT_EULA:?must be set by the caller}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export UNIVTAC45_RENDER_OVERRIDES="/rtx/directLighting/sampledLighting/enabled=false;/rtx-transient/dldenoiser/enabled=false"
timeout -k 30 $((600 + 200 * COUNT)) "$PY" scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py \
  "$TASK" "$TCFG" "$YML" --start_seed "$START" --max_seed $((START + COUNT - 1)) --total_num "$COUNT" --headless
