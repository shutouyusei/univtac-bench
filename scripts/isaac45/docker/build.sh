#!/usr/bin/env bash
# Build the evaluation image from one commit of this repository, and optionally push it.
#
# Usage: scripts/isaac45/docker/build.sh <commit> <image:tag> [--push]
#   e.g. scripts/isaac45/docker/build.sh HEAD ghcr.io/<user>/univtac45-eval:$(git rev-parse --short HEAD) --push
#
# The build context is a fresh clone of <commit>: tracked files and .git only, so nothing untracked in the
# working checkout (trained checkpoints, local configs, results) reaches the image.
# --push logs in with REGISTRY_HOST / REGISTRY_USER / REGISTRY_TOKEN from the environment
# (e.g. `set -a; source .env; set +a`), pushes, logs out again, and prints the pushed digest, which is
# what a pod should run.
# Build arguments of the Dockerfile can be overridden through the environment of the same name
# (UNIVTAC45_BUILD_JOBS, UPSTREAM_COMMIT, CHECKPOINT_REVISION, MINIFORGE_VERSION). The SM version stays the
# Dockerfile's 89: a UNIVTAC45_CUDA_ARCH=120 exported for local builds would make an image RTX 40s cannot run.
# VCPKG_DOWNLOAD_SEED (default: the local install's ~/.cache/univtac-isaac45/vcpkg/downloads, if present) hands
# libuipc's vcpkg build the source archives already downloaded once, so a GitHub archive outage does not stop the
# build; vcpkg accepts a file only if its SHA512 matches the port. Set it to "" to download everything.
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && "$3" != "--push" ) ]]; then
    sed -n '2,17p' "$0" >&2
    exit 2
fi
COMMIT="$1"
IMAGE="$2"
PUSH="${3:-}"

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SHA="$(git -C "${REPO}" rev-parse --verify "${COMMIT}^{commit}")"
CONTEXT="$(mktemp -d)"
trap 'rm -rf "${CONTEXT}"' EXIT

echo "[build] ${IMAGE} from ${SHA}"
git clone -q --no-local --no-checkout "${REPO}" "${CONTEXT}/univtac"
git -C "${CONTEXT}/univtac" checkout -q --detach "${SHA}"

build_args=()
for name in UNIVTAC45_BUILD_JOBS UPSTREAM_COMMIT CHECKPOINT_REVISION MINIFORGE_VERSION; do
    if [[ -n "${!name:-}" ]]; then build_args+=(--build-arg "${name}=${!name}"); fi
done
SEED="${VCPKG_DOWNLOAD_SEED-${HOME}/.cache/univtac-isaac45/vcpkg/downloads}"
if [[ -n "${SEED}" && -d "${SEED}" ]]; then
    echo "[build] vcpkg source archives from ${SEED} ($(find "${SEED}" -maxdepth 1 -type f | wc -l) files)"
    build_args+=(--build-context "vcpkg-downloads=${SEED}")
fi

docker build \
    --file "${CONTEXT}/univtac/scripts/isaac45/docker/Dockerfile" \
    --label "org.opencontainers.image.revision=${SHA}" \
    --label "org.opencontainers.image.source=https://github.com/shutouyusei/univtac-bench" \
    "${build_args[@]}" \
    --tag "${IMAGE}" \
    "${CONTEXT}/univtac"
echo "[build] local image id $(docker image inspect --format '{{.Id}}' "${IMAGE}")"

if [[ "${PUSH}" == "--push" ]]; then
    : "${REGISTRY_HOST:?set REGISTRY_HOST (e.g. source .env)}"
    : "${REGISTRY_USER:?set REGISTRY_USER}"
    : "${REGISTRY_TOKEN:?set REGISTRY_TOKEN}"
    trap 'docker logout "${REGISTRY_HOST}" >/dev/null 2>&1 || true; rm -rf "${CONTEXT}"' EXIT
    printf '%s' "${REGISTRY_TOKEN}" | docker login "${REGISTRY_HOST}" -u "${REGISTRY_USER}" --password-stdin
    docker push "${IMAGE}"
    echo "[push] $(docker image inspect --format '{{index .RepoDigests 0}}' "${IMAGE}")"
fi
