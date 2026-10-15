#!/usr/bin/env bash
# Tests for scripts/isaac45/cuda_arch.sh: bash scripts/isaac45/tests/test_cuda_arch.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../cuda_arch.sh
source "${HERE}/cuda_arch.sh"

failures=0
expect() {
    local input="$1" wanted="$2" got
    got="$(torch_arch_list "${input}" 2>/dev/null)"
    if [[ "${got}" == "${wanted}" ]]; then
        echo "ok   '${input}' -> '${got}'"
    else
        echo "FAIL '${input}' -> '${got}', wanted '${wanted}'"
        failures=$((failures + 1))
    fi
}
expect_rejected() {
    local input="$1" code
    torch_arch_list "${input}" >/dev/null 2>&1
    code=$?
    if [[ "${code}" -eq 2 ]]; then
        echo "ok   '${input}' rejected"
    else
        echo "FAIL '${input}' exit ${code}, wanted 2"
        failures=$((failures + 1))
    fi
}

expect "120" "12.0"
expect "89;120" "8.9;12.0"
expect "86;89;120" "8.6;8.9;12.0"
expect_rejected "abc"
expect_rejected "8"
expect_rejected "89;"
expect_rejected ""

[[ "${failures}" -eq 0 ]] && echo "all passed" || { echo "${failures} failed"; exit 1; }
