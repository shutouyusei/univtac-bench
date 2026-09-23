#!/bin/bash
# Usage: bash eval_policy.sh <task_name> <task_config> <policy_config> <gpu_id> [extra eval_policy.py args]
# Extra arguments (for example --total_num 2 --max_seed 1000005 --headless) are
# forwarded to scripts/eval_policy.py.
TASK_NAME=${1}
TASK_CONFIG=${2}
POLICY_CONIFG=${3}
GPU=${4}

export CUDA_VISIBLE_DEVICES=$GPU
python scripts/eval_policy.py $TASK_NAME $TASK_CONFIG $POLICY_CONIFG "${@:5}"

# bash eval_policy.sh insert_hole clean51 ACT/deploy 0
