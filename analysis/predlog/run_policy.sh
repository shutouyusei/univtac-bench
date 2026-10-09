#!/bin/bash
# run_policy.sh: k10_tp_st on insert_hole seeds 1000000-1000099 with stflow_trace + stflow_prediction_log, from the
# stflow-predlog univtac worktree with the stflow-slowfast code (the checkpoint's own). Pass 1 runs the range in one
# process; passes 2-3 re-run each seed that recorded no outcome on its own. Outcomes + traces:
# eval_result/stflow/insert_hole/predlog_k10_tp_st/*/metadata.json; forecasts: $T/policy_npz/<seed>.npz.
# Do not switch the worktree's branch while queued.
set -uo pipefail
T=/home/yusei/.claude/jobs/7fcd22c8/tmp/predlog
SIM=/home/yusei/Desktop/Robotics/univtac-bench/.claude/worktrees/vrr-aux-target/.claude/worktrees/stflow-adapter/.claude/worktrees/stflow-adapter-review/.claude/worktrees/stflow-controller/.claude/worktrees/stflow-predlog
CKPT=/home/yusei/Desktop/Robotics/streaming-tactile-flow/outputs/ep50_sfp_tr5_block_adapter_k10_tp_st/checkpoints/last
PY=/home/yusei/miniforge3/envs/UniVTAC/bin/python
RES=$SIM/eval_result/stflow/insert_hole/predlog_k10_tp_st
YML=$T/predlog_k10_tp_st.yml
cat > $YML <<EOF
policy_name: stflow
seed: 0
instruction_type: seen
stflow_ckpt_dir: $CKPT
stflow_trace: true
stflow_prediction_log: {dir: $T/policy_npz, frames_every: 5}
EOF
cd $SIM
export PYTHONPATH=/home/yusei/Desktop/Robotics/stflow-slowfast/src OMNI_KIT_ACCEPT_EULA=YES
echo "policy run $(git log --oneline -1) $(date)"
START=1000000; LAST=1000099

evaluate() {  # evaluate <first seed> <count>
  conda run -n UniVTAC --no-capture-output bash eval_policy.sh insert_hole clean51 $YML 0 \
    --start_seed $1 --total_num $2 --max_seed $(($1 + $2 - 1)) --headless >> $T/policy_eval.log 2>&1 || true
}

evaluate $START $((LAST - START + 1))
for pass in 2 3; do
  MISSING=$($PY $T/missing.py $RES $START $LAST)
  [ -z "$MISSING" ] && break
  echo "pass $pass: $(echo $MISSING | wc -w) seeds without an outcome"
  for S in $MISSING; do evaluate $S 1; done
done
SUMMARY=$($PY $T/missing.py $RES $START $LAST --summary)
echo "$SUMMARY"
~/.claude/scripts/discord-notify.sh "predlog policy run (k10_tp_st, insert_hole 100 seeds) done: $SUMMARY" || true
echo "done $(date)"
