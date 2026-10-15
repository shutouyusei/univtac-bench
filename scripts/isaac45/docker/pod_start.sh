#!/usr/bin/env bash
# Entrypoint of the evaluation image on a RunPod pod.
#
# Starts sshd with the account's public keys (RunPod passes them as PUBLIC_KEY) so results can be pulled with
# rsync, runs the evaluation named in EVAL_JOB ("<task> <arm> <start_seed> <count> <task_config>", the
# arguments of eval_released.sh) if one is given, and then stays up until the collector deletes the pod.
# Without EVAL_JOB the pod only idles with sshd, for manual work.
set -uo pipefail

mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
    printf '%s\n' "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A >/dev/null
/usr/sbin/sshd

if [[ -n "${EVAL_JOB:-}" ]]; then
    read -r -a job <<< "${EVAL_JOB}"
    out_root="${OUT_ROOT:-/workspace/out}"
    mkdir -p "${out_root}"
    # A restarted pod (host maintenance) keeps its volume: a job that already finished is not run again.
    if [[ -f "${out_root}/${job[0]:-}_${job[1]:-}_${job[2]:-}_${job[3]:-}/DONE" ]]; then
        echo "[pod] ${EVAL_JOB}: already done" >> "${out_root}/pod_stdout.log"
    else
        bash "$(dirname "${BASH_SOURCE[0]}")/eval_released.sh" "${job[@]}" >> "${out_root}/pod_stdout.log" 2>&1 &
    fi
fi
exec sleep infinity
