#!/bin/bash
# run_expert.sh: the scripted expert on insert_hole seeds 1000000-1000099 in collect mode, saved like the training
# demos (hdf5 per successful seed, suc_map.txt) under the stflow-predlog worktree's data/insert_hole/clean51, which
# holds no training data. Used as the demo-like reference of the same scene.
set -uo pipefail
T=/home/yusei/.claude/jobs/7fcd22c8/tmp/predlog
SIM=/home/yusei/Desktop/Robotics/univtac-bench/.claude/worktrees/vrr-aux-target/.claude/worktrees/stflow-adapter/.claude/worktrees/stflow-adapter-review/.claude/worktrees/stflow-controller/.claude/worktrees/stflow-predlog
cd $SIM
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0
echo "expert collect $(git log --oneline -1) $(date)"
conda run -n UniVTAC --no-capture-output python scripts/collect_data.py insert_hole clean51 \
  --start_seed 1000000 --max_seed 1000099 --headless --config-overrides collect_settings.episode_num=100 \
  >> $T/expert_collect.log 2>&1 || true
N=$(ls $SIM/data/insert_hole/clean51/hdf5/*.hdf5 2>/dev/null | wc -l)
echo "expert hdf5 files: $N"
~/.claude/scripts/discord-notify.sh "predlog expert collect (insert_hole 100 eval seeds) done: $N hdf5" || true
echo "done $(date)"
