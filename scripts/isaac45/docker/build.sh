#!/usr/bin/env bash
# Build the evaluation image from one commit (a fresh clone: tracked files only), optionally push it.
#
# Usage: scripts/isaac45/docker/build.sh <commit> <image:tag> [--push]
#   --push  log in with REGISTRY_HOST/USER/TOKEN from the env, push, log out, print the digest pods should run.
# Env overrides of build args: UNIVTAC45_BUILD_JOBS UPSTREAM_COMMIT CHECKPOINT_REPO CHECKPOINT_REVISION
# MINIFORGE_VERSION. The SM stays the Dockerfile's 89 (a local UNIVTAC45_CUDA_ARCH=120 would exclude RTX 40s).
# VCPKG_DOWNLOAD_SEED: vcpkg source archives to reuse (default: the local install's cache; "" = download all).
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && "$3" != "--push" ) ]]; then
    awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0" >&2
    exit 2
fi
COMMIT="$1" IMAGE="$2" PUSH="${3:-}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SHA="$(git -C "${REPO}" rev-parse --verify "${COMMIT}^{commit}")"
CONTEXT="$(mktemp -d)"
trap 'rm -rf "${CONTEXT}"' EXIT

echo "[build] ${IMAGE} from ${SHA}"
git clone -q --no-local --no-checkout "${REPO}" "${CONTEXT}/univtac"
git -C "${CONTEXT}/univtac" checkout -q --detach "${SHA}"

build_args=()
for name in UNIVTAC45_BUILD_JOBS UPSTREAM_COMMIT CHECKPOINT_REPO CHECKPOINT_REVISION MINIFORGE_VERSION; do
    if [[ -n "${!name:-}" ]]; then build_args+=(--build-arg "${name}=${!name}"); fi
done
SEED="${VCPKG_DOWNLOAD_SEED-${HOME}/.cache/univtac-isaac45/vcpkg/downloads}"
if [[ -n "${SEED}" && -d "${SEED}" ]]; then
    echo "[build] vcpkg source archives from ${SEED} ($(find "${SEED}" -maxdepth 1 -type f | wc -l) files)"
    build_args+=(--build-context "vcpkg-downloads=${SEED}")
fi

docker build --file "${CONTEXT}/univtac/scripts/isaac45/docker/Dockerfile" \
    --label "org.opencontainers.image.revision=${SHA}" \
    --label "org.opencontainers.image.source=https://github.com/shutouyusei/univtac-bench" \
    "${build_args[@]}" --tag "${IMAGE}" "${CONTEXT}/univtac"
echo "[build] local image id $(docker image inspect --format '{{.Id}}' "${IMAGE}")"

if [[ -n "${PUSH}" ]]; then
    : "${REGISTRY_HOST:?set REGISTRY_HOST (e.g. source .env)}" "${REGISTRY_USER:?set REGISTRY_USER}" \
      "${REGISTRY_TOKEN:?set REGISTRY_TOKEN}"
    trap 'docker logout "${REGISTRY_HOST}" >/dev/null 2>&1 || true; rm -rf "${CONTEXT}"' EXIT
    printf '%s' "${REGISTRY_TOKEN}" | docker login "${REGISTRY_HOST}" -u "${REGISTRY_USER}" --password-stdin
    docker push "${IMAGE}"
    echo "[push] $(docker image inspect --format '{{index .RepoDigests 0}}' "${IMAGE}")"
fi
