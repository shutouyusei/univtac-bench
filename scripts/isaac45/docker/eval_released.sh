#!/usr/bin/env bash
# Evaluate one released ACT checkpoint with upstream's scripts/eval_policy.py, inside the evaluation image.
#
# Usage: eval_released.sh <task> <arm: univtac|vision_only> <start_seed> <count> <task_config: clean|demo|...>
#
# The checkpoint is linked where policy/ACT/deploy_policy.py looks for it, under the train config named in the
# checkpoint's own eval log (train_config, train_config_all, ...), so camera setup and config match what the
# authors evaluated. Environment:
#   OMNI_KIT_ACCEPT_EULA         Isaac Sim's EULA acceptance (required by Isaac Sim, passed in by the caller)
#   UNIVTAC45_RENDER_OVERRIDES   optional RTX overrides (scripts/isaac45/run_with_render_overrides.py)
#   EPISODE_TIMEOUT_S            wall-clock budget per episode, default 400 (whole run: 600 + count x this)
#   OUT_ROOT                     default /workspace/out
# Output in OUT_ROOT/<task>_<arm>_<start_seed>_<count>/: eval_result/ (upstream's output for this run; deploy/<time>/
# while it runs), eval.log, manifest.json, and DONE (holding the exit code), written last on every exit, early
# failures included: a collector waits for DONE.
set -uo pipefail

if [[ $# -ne 5 ]]; then
    sed -n '2,16p' "$0" >&2
    exit 2
fi
TASK="$1"
ARM="$2"
START="$3"
COUNT="$4"
TASK_CONFIG="$5"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${CONDA_DIR:-/opt/conda}/envs/UniVTAC45/bin/python"
RESULTS="${ROOT}/scripts/isaac45/runpod/results.py"
CKPT="${ROOT}/data/checkpoints/${TASK}/${ARM}"
OUT="${OUT_ROOT:-/workspace/out}/${TASK}_${ARM}_${START}_${COUNT}"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TRAIN_CONFIG="unknown"

# manifest.json and DONE on every exit, early failures included, so a collector never waits for a dead job.
mkdir -p "${OUT}"
finish() {
    local code=$? start="${START}" count="${COUNT}"
    [[ "${start}" =~ ^[0-9]+$ ]] || start=0
    [[ "${count}" =~ ^[0-9]+$ ]] || count=0
    python3 "${RESULTS}" manifest --out "${OUT}/manifest.json" --task "${TASK}" --arm "${ARM}" \
        --start-seed "${start}" --count "${count}" --task-config "${TASK_CONFIG}" \
        --train-config "${TRAIN_CONFIG}" --exit-code "${code}" --started "${STARTED}" --run-dir "${OUT}/eval_result" \
        --checkpoint-sha256 "${ROOT}/data/checkpoints.sha256" || true
    echo "${code}" > "${OUT}/DONE"
}
trap finish EXIT

fail() { echo "$*" >&2; exit 2; }
[[ "${ARM}" == "univtac" || "${ARM}" == "vision_only" ]] || fail "arm must be univtac or vision_only, not '${ARM}'"
[[ "${START}" =~ ^[0-9]+$ && "${COUNT}" =~ ^[1-9][0-9]*$ ]] || fail "start_seed / count must be integers"
[[ -f "${ROOT}/task_config/${TASK_CONFIG}.yml" ]] || fail "no task_config/${TASK_CONFIG}.yml"
[[ -f "${CKPT}/policy_last.ckpt" && -f "${CKPT}/dataset_stats.pkl" ]] || fail "no released checkpoint at ${CKPT}"
[[ -n "${OMNI_KIT_ACCEPT_EULA:-}" ]] || fail "OMNI_KIT_ACCEPT_EULA is not set"

TRAIN_CONFIG="$(python3 "${RESULTS}" train-config "${CKPT}/log.log")" || { TRAIN_CONFIG="unknown"; fail "cannot read the train config"; }
DST="${ROOT}/policy/ACT/act_ckpt/act-${TASK}/${TASK_CONFIG}-50/${TRAIN_CONFIG}"
mkdir -p "${DST}"
ln -sfn "${CKPT}/policy_last.ckpt" "${DST}/policy_last.ckpt"
ln -sfn "${CKPT}/dataset_stats.pkl" "${DST}/dataset_stats.pkl"

# Upstream writes eval_result/ACT/<task>/deploy/<time>/ as it goes; pointing that folder into OUT (the pod volume)
# keeps finished episodes even if the pod is stopped before the run ends.
DEPLOY="${ROOT}/eval_result/ACT/${TASK}/deploy"
mkdir -p "$(dirname "${DEPLOY}")" "${OUT}/deploy"
rm -rf "${DEPLOY}"
ln -s "${OUT}/deploy" "${DEPLOY}"
LIMIT=$((600 + ${EPISODE_TIMEOUT_S:-400} * COUNT))
echo "[eval] ${TASK} ${ARM} (${TRAIN_CONFIG}) seeds ${START}+${COUNT} task_config ${TASK_CONFIG}" \
     "overrides [${UNIVTAC45_RENDER_OVERRIDES:-}] limit ${LIMIT}s"

cd "${ROOT}"
TRAIN_CONFIG="${TRAIN_CONFIG}" EP_NUM=50 CUDA_VISIBLE_DEVICES=0 timeout -k 60 "${LIMIT}" \
    "${PY}" scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py "${TASK}" "${TASK_CONFIG}" ACT/deploy \
    --start_seed "${START}" --max_seed $((START + COUNT - 1)) --total_num "${COUNT}" --headless \
    > "${OUT}/eval.log" 2>&1
CODE=$?

RUN="$(ls -d "${OUT}/deploy"/* 2>/dev/null | sort | tail -1)"
if [[ -n "${RUN}" ]]; then
    mv "${RUN}" "${OUT}/eval_result"
    rmdir "${OUT}/deploy" 2>/dev/null || true
else
    echo "[eval] upstream wrote no eval_result folder" >&2
fi
exit "${CODE}"
