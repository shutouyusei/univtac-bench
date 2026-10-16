#!/usr/bin/env bash
# Isaac Sim 4.5 environment of this upstream checkout on RTX 50-series GPUs (see README.md).
#
# Usage: bash scripts/isaac45/install.sh [--from STEP | --only STEP] [--locked] [--check]
#   STEP      assets env isaacsim isaaclab curobo tacex vcpkg uipc extras
#   --only    run one step and skip the final check (one Docker layer)
#   --locked  pin every pip step to requirements.lock
#   --check   verify the installed environment and exit
# Env: UNIVTAC45_CONDA_ENV (UniVTAC45), UNIVTAC45_CUDA_ARCH (120; 89 for RTX 40, its PTX runs on RTX 50),
#      UNIVTAC45_BUILD_JOBS (8), UNIVTAC45_VCPKG_ROOT (~/.cache/univtac-isaac45/vcpkg)
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="${ROOT}/scripts/isaac45"
ENV_NAME="${UNIVTAC45_CONDA_ENV:-UniVTAC45}"
CUDA_ARCH="${UNIVTAC45_CUDA_ARCH:-120}"
JOBS="${UNIVTAC45_BUILD_JOBS:-8}"
VCPKG_ROOT="${UNIVTAC45_VCPKG_ROOT:-${HOME}/.cache/univtac-isaac45/vcpkg}"
VCPKG_ARCHIVES="${VCPKG_ROOT}-archives"           # own binary cache: the shared one mixes in other builds
VCPKG_COMMIT=dd3097e305afa53f7b4312371f62058d2e665320
ISAACLAB_DIR="${ROOT}/third_party/IsaacLab"
ISAACLAB_COMMIT=90b79bb2d44feb8d833f260f2bf37da3487180ba    # v2.1.1
CUROBO_DIR="${ROOT}/third_party/curobo"
CUROBO_COMMIT=0a50de1ba72db304195d59d9d0b1ed269696047f      # upstream scripts/install.sh
TACEX="${ROOT}/third_party/TacEx/source"
TORCH_SCATTER_WHEEL=https://data.pyg.org/whl/torch-2.7.0%2Bcu128/torch_scatter-2.1.2%2Bpt27cu128-cp310-cp310-linux_x86_64.whl
LFS_REMOTE=https://github.com/univtac/UniVTAC.git
STEPS=(assets env isaacsim isaaclab curobo tacex vcpkg uipc extras)

FROM=assets ONLY="" LOCKED=0 CHECK=0 LOCK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift ;;
        --only) ONLY="$2"; FROM="$2"; shift ;;
        --locked) LOCKED=1 ;;
        --check) CHECK=1 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
[[ " ${STEPS[*]} " == *" ${FROM} "* ]] || { echo "unknown step '${FROM}' (one of: ${STEPS[*]})" >&2; exit 2; }
[[ "${CUDA_ARCH}" =~ ^[0-9]{2,3}$ ]] || { echo "UNIVTAC45_CUDA_ARCH must be one SM, e.g. 120 or 89" >&2; exit 2; }

index() { local i; for i in "${!STEPS[@]}"; do [[ "${STEPS[i]}" == "$1" ]] && { echo "$i"; return; }; done; }
want() { if [[ -n "${ONLY}" ]]; then [[ "${ONLY}" == "$1" ]]; else (( $(index "$1") >= $(index "${FROM}") )); fi; }

if (( ! CHECK )) && want assets; then
    # Upstream stores 114 TacEx assets as LFS pointer text in plain blobs. Fetch the objects, then keep the
    # index at upstream's tree so the real files stay out of git status.
    echo "[assets] TacEx LFS objects from ${LFS_REMOTE}"
    git -C "${ROOT}" lfs fetch "${LFS_REMOTE}"
    git -C "${ROOT}" lfs checkout
    staged="$(mktemp)"
    git -C "${ROOT}" diff --cached --name-only -- third_party/TacEx > "${staged}"
    if [[ -s "${staged}" ]]; then
        git -C "${ROOT}" reset -q --pathspec-from-file="${staged}"
        git -C "${ROOT}" update-index --skip-worktree --stdin < "${staged}"
    fi
    echo "[assets] $(wc -l < "${staged}") files materialised"
    rm -f "${staged}"
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
activate() { set +u; conda activate "$1"; local s=$?; set -u; return "${s}"; }

if (( ! CHECK )) && want env; then
    echo "[env] conda environment ${ENV_NAME}: Python 3.10 + CUDA 12.8 + GCC 12"
    activate "${ENV_NAME}" >/dev/null 2>&1 \
        || conda create --name "${ENV_NAME}" --yes --override-channels --channel https://repo.anaconda.com/pkgs/main python=3.10 pip
    conda env update --name "${ENV_NAME}" --file "${HERE}/env.yaml"
fi
activate "${ENV_NAME}"

PYTHON="${CONDA_PREFIX}/bin/python"
PIP=("${PYTHON}" -m pip)
"${PYTHON}" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))' || { echo "${ENV_NAME} must use Python 3.10" >&2; exit 1; }
"${CONDA_PREFIX}/bin/nvcc" --version | grep -q "release 12\.8" || { echo "${ENV_NAME} needs CUDA 12.8 (run --only env)" >&2; exit 1; }

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_HOME="${CONDA_PREFIX}" CUDA_PATH="${CONDA_PREFIX}" CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export CUDAARCHS="${CUDA_ARCH}" TORCH_CUDA_ARCH_LIST="${CUDA_ARCH:0:-1}.${CUDA_ARCH: -1}+PTX"
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" MAX_JOBS="${JOBS}"
# conda GCC 12 with the host linker (toolchains/): the conda linker fails on RELR relocations
export UNIVTAC_GCC12="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" UNIVTAC_GXX12="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
export CC="${HERE}/toolchains/gcc12-system-ld" CXX="${HERE}/toolchains/gxx12-system-ld" CUDAHOSTCXX="${HERE}/toolchains/gxx12-system-ld"

check() {
    local pointers
    pointers="$(grep -rlI '^version https://git-lfs.github.com/spec/v1' "${TACEX}" "${ROOT}/assets" 2>/dev/null | wc -l || true)"
    if (( pointers )); then
        echo "[wrong] ${pointers} asset files are still LFS pointer text (run --only assets)"
        return 1
    fi
    echo "[ok] no LFS pointer text in TacEx sources or assets/"
    "${PYTHON}" "${HERE}/check_env.py"
}
if (( CHECK )); then check; exit; fi

if (( LOCKED )); then
    # plain name==version lines only; editable and URL installs are pinned by the steps
    LOCK="$(mktemp --suffix=.txt)"
    trap 'rm -f "${LOCK}"' EXIT
    grep -E '^[A-Za-z0-9_.-]+==[^ ]+$' "${HERE}/requirements.lock" > "${LOCK}"
    export PIP_CONSTRAINT="${LOCK}"
    echo "[locked] $(wc -l < "${LOCK}") pins from requirements.lock"
fi

if want isaacsim; then
    echo "[isaacsim] torch 2.7.0+cu128, Isaac Sim 4.5.0.0"
    "${PIP[@]}" install --upgrade pip "setuptools<82" wheel
    "${PIP[@]}" install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    "${PIP[@]}" install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com
    "${PIP[@]}" install numpy==1.26.4
fi
# later steps must not replace torch / torchvision / numpy
export PIP_CONSTRAINT="${HERE}/constraints.txt${LOCK:+ ${LOCK}}"

if want isaaclab; then
    echo "[isaaclab] Isaac Lab v2.1.1"
    [[ -d "${ISAACLAB_DIR}/.git" ]] || git clone https://github.com/isaac-sim/IsaacLab "${ISAACLAB_DIR}"
    git -C "${ISAACLAB_DIR}" fetch -q origin
    git -C "${ISAACLAB_DIR}" checkout -q --detach "${ISAACLAB_COMMIT}"
    "${PIP[@]}" install flatdict==4.0.1 --no-build-isolation
    (cd "${ISAACLAB_DIR}" && ./isaaclab.sh --install)
fi

if want curobo; then
    echo "[curobo] ${CUROBO_COMMIT:0:7}, warp-lang 1.0.0"
    [[ -d "${CUROBO_DIR}/.git" ]] || git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
    git -C "${CUROBO_DIR}" fetch -q origin
    git -C "${CUROBO_DIR}" checkout -q --detach "${CUROBO_COMMIT}"
    "${PIP[@]}" install warp-lang==1.0.0 --no-build-isolation
    "${PIP[@]}" install -e "${CUROBO_DIR}" --no-build-isolation
fi

if want tacex; then
    echo "[tacex] TacEx core packages"
    "${PIP[@]}" install -e "${TACEX}/tacex" -e "${TACEX}/tacex_assets" -e "${TACEX}/tacex_tasks"
    "${PIP[@]}" uninstall -y torch_scatter || true    # TacEx pulls the wheel of another torch
    "${PIP[@]}" install "${TORCH_SCATTER_WHEEL}"
fi

if want vcpkg; then
    echo "[vcpkg] ${VCPKG_COMMIT:0:7} at ${VCPKG_ROOT}"
    if [[ ! -x "${VCPKG_ROOT}/vcpkg" ]]; then
        [[ -d "${VCPKG_ROOT}/.git" ]] || git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"
        git -C "${VCPKG_ROOT}" fetch -q origin "${VCPKG_COMMIT}"
        git -C "${VCPKG_ROOT}" checkout -q --detach "${VCPKG_COMMIT}"
        "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
    fi
    [[ "$(git -C "${VCPKG_ROOT}" rev-parse HEAD)" == "${VCPKG_COMMIT}" ]] \
        || { echo "${VCPKG_ROOT} is not at ${VCPKG_COMMIT}; move it aside or set UNIVTAC45_VCPKG_ROOT" >&2; exit 1; }
fi

if want uipc; then
    echo "[uipc] libuipc + tacex_uipc for SM ${CUDA_ARCH}"
    export CMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    export VCPKG_OVERLAY_PORTS="${HERE}/toolchains/vcpkg-overlay-ports"
    export VCPKG_BINARY_SOURCES="clear;files,${VCPKG_ARCHIVES},readwrite"
    mkdir -p "${VCPKG_ARCHIVES}"
    rm -rf "${TACEX}/tacex_uipc/build"
    "${PIP[@]}" install -e "${TACEX}/tacex_uipc" -v --no-build-isolation
fi

if want extras; then
    echo "[extras] upstream extras + ipython<9 for policy/ACT"
    "${PIP[@]}" install transforms3d trimesh tetgen "ipython<9"
fi

[[ -n "${ONLY}" ]] || check
