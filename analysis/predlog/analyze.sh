#!/bin/bash
# analyze.sh <expert hdf5 dir> <out dir>: analyze.py on the predlog runs (spec.md).
T=/home/yusei/.claude/jobs/7fcd22c8/tmp/predlog
SIM=/home/yusei/Desktop/Robotics/univtac-bench/.claude/worktrees/vrr-aux-target/.claude/worktrees/stflow-adapter/.claude/worktrees/stflow-adapter-review/.claude/worktrees/stflow-controller/.claude/worktrees/stflow-predlog
cd $T
PYTHONPATH=/home/yusei/Desktop/Robotics/stflow-slowfast/src /home/yusei/miniforge3/envs/UniVTAC/bin/python $T/analyze.py \
  --policy_results $SIM/eval_result/stflow/insert_hole/predlog_k10_tp_st --policy_npz $T/policy_npz \
  --expert_hdf5 "$1" --ckpt /home/yusei/Desktop/Robotics/streaming-tactile-flow/outputs/ep50_sfp_tr5_block_adapter_k10_tp_st/checkpoints/last \
  --demo_joint /home/yusei/Desktop/Robotics/stflow-slowfast/cache/insert_hole-clean51-50/joint.npy --out "$2" ${3:-}
