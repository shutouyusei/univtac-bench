#!/usr/bin/env bash
# Image entrypoint on a RunPod pod: sshd with PUBLIC_KEY (for result pulls), then eval_released.sh with the
# arguments in EVAL_JOB if set; stays up until the collector deletes the pod.
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
    mkdir -p "${OUT_ROOT}"
    bash "$(dirname "${BASH_SOURCE[0]}")/eval_released.sh" "${job[@]}" >> "${OUT_ROOT}/pod_stdout.log" 2>&1 &
fi
exec sleep infinity
