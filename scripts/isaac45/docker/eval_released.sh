#!/usr/bin/env bash
# Evaluate one released ACT checkpoint with upstream's scripts/eval_policy.py (inside the image).
#
# Usage: eval_released.sh <task> <univtac|vision_only> <start_seed> <count> <task_config>
# The checkpoint is linked under the train config named in its own log.log (the authors' setup).
# Env: OUT_ROOT and OMNI_KIT_ACCEPT_EULA (required), UNIVTAC45_RENDER_OVERRIDES, EPISODE_TIMEOUT_S.
# Writes OUT_ROOT/<task>_<arm>_<start>_<count>/: eval_result/, eval.log, manifest.json, and DONE (exit code)
# last on every exit, so a collector never waits on a dead job. A job whose DONE exists is not run again.
set -uo pipefail

usage() { awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0" >&2; exit 2; }
[[ $# -eq 5 ]] || usage
TASK="$1" ARM="$2" START="$3" COUNT="$4" TASK_CONFIG="$5"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${CONDA_DIR:-/opt/conda}/envs/UniVTAC45/bin/python"
RESULTS="${ROOT}/scripts/isaac45/runpod/results.py"
CKPT="${ROOT}/data/checkpoints/${TASK}/${ARM}"
OUT="${OUT_ROOT:?set OUT_ROOT}/${TASK}_${ARM}_${START}_${COUNT}"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TRAIN_CONFIG="unknown"

if [[ -f "${OUT}/DONE" ]]; then
    echo "[eval] ${OUT##*/}: already done"
    exit 0
fi
mkdir -p "${OUT}"
finish() {
    local code=$? start="${START}" count="${COUNT}"
    [[ "${start}" =~ ^[0-9]+$ ]] || start=0
    [[ "${count}" =~ ^[0-9]+$ ]] || count=0
    python3 "${RESULTS}" manifest --out "${OUT}/manifest.json" --task "${TASK}" --arm "${ARM}" \
        --start-seed "${start}" --count "${count}" --task-config "${TASK_CONFIG}" --train-config "${TRAIN_CONFIG}" \
        --exit-code "${code}" --started "${STARTED}" --run-dir "${OUT}/eval_result" \
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
for f in policy_last.ckpt dataset_stats.pkl; do ln -sfn "${CKPT}/${f}" "${DST}/${f}"; done

# Upstream writes eval_result/ACT/<task>/deploy/<time>/ as it goes; linking it into OUT (the pod volume) keeps
# finished episodes if the pod is stopped mid-run.
DEPLOY="${ROOT}/eval_result/ACT/${TASK}/deploy"
mkdir -p "$(dirname "${DEPLOY}")" "${OUT}/deploy"
rm -rf "${DEPLOY}"
ln -s "${OUT}/deploy" "${DEPLOY}"
LIMIT="$(python3 "${RESULTS}" run-limit "${COUNT}")"
echo "[eval] ${TASK} ${ARM} (${TRAIN_CONFIG}) seeds ${START}+${COUNT} task_config ${TASK_CONFIG}" \
     "overrides [${UNIVTAC45_RENDER_OVERRIDES:-}] limit ${LIMIT}s"

cd "${ROOT}"
TRAIN_CONFIG="${TRAIN_CONFIG}" EP_NUM=50 CUDA_VISIBLE_DEVICES=0 timeout -k 60 "${LIMIT}" \
    "${PY}" scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py "${TASK}" "${TASK_CONFIG}" ACT/deploy \
    --start_seed "${START}" --max_seed $((START + COUNT - 1)) --total_num "${COUNT}" --headless \
    > "${OUT}/eval.log" 2>&1
CODE=$?

RUN="$(ls -d "${OUT}/deploy"/* 2>/dev/null | tail -1)"
if [[ -n "${RUN}" ]]; then
    mv "${RUN}" "${OUT}/eval_result"
    rmdir "${OUT}/deploy" 2>/dev/null || true
else
    echo "[eval] upstream wrote no eval_result folder" >&2
fi
exit "${CODE}"
