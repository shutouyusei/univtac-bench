#!/usr/bin/env bash
# Isaac Sim 4.5 environment for this upstream checkout on RTX 50-series GPUs (sm_120).
#
# Follows upstream scripts/install.sh (Isaac Sim 4.5.0 + Isaac Lab v2.1.1 + cuRobo 0a50de1 + the vendored
# TacEx/libuipc) with only the changes sm_120 needs, listed in scripts/isaac45/README.md:
#   * CUDA toolkit 12.8 + conda-forge GCC/G++ 12 (scripts/isaac45/env.yaml) instead of CUDA 12.4 + GCC 11.4;
#   * torch 2.7.0+cu128 kept (Isaac Lab v2.1.1's own installer puts it in for Blackwell); upstream's script
#     reinstalls 2.5.1+cu124, which has no sm_120 kernels; torch_scatter is the matching pt27cu128 wheel;
#   * vcpkg pinned to the revision the Isaac Sim 5.1 build of the same libuipc uses, with its tinygltf
#     hash overlay, cloned outside the repository; no sudo, no edits to ~/.bashrc, no demo runs.
# Nothing outside the named Conda environment, this checkout's ignored third_party/ and VCPKG_ROOT is
# touched.
#
# Usage: bash scripts/isaac45/install.sh [--from STEP] [--only STEP] [--check]
#   STEP: env isaacsim isaaclab curobo tacex vcpkg uipc extras
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="${PROJECT_ROOT}/scripts/isaac45"
CONDA_ENV_NAME="${UNIVTAC45_CONDA_ENV:-UniVTAC45}"
CUDA_ARCH="${UNIVTAC45_CUDA_ARCH:-120}"
BUILD_JOBS="${UNIVTAC45_BUILD_JOBS:-8}"
VCPKG_ROOT="${UNIVTAC45_VCPKG_ROOT:-${HOME}/.cache/univtac-isaac45/vcpkg}"
VCPKG_COMMIT="dd3097e305afa53f7b4312371f62058d2e665320"
ISAACLAB_DIR="${PROJECT_ROOT}/third_party/IsaacLab"
ISAACLAB_COMMIT="90b79bb2d44feb8d833f260f2bf37da3487180ba"   # v2.1.1
CUROBO_DIR="${PROJECT_ROOT}/third_party/curobo"
CUROBO_COMMIT="0a50de1ba72db304195d59d9d0b1ed269696047f"     # upstream scripts/install.sh
WARP_VERSION="${UNIVTAC45_WARP_VERSION:-1.0.0}"              # upstream pin for cuRobo
TORCH_SCATTER_WHEEL="https://data.pyg.org/whl/torch-2.7.0%2Bcu128/torch_scatter-2.1.2%2Bpt27cu128-cp310-cp310-linux_x86_64.whl"
UIPC_DIR="${PROJECT_ROOT}/third_party/TacEx/source/tacex_uipc"
STEPS=(env isaacsim isaaclab curobo tacex vcpkg uipc extras)

FROM=""
ONLY=""
CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift ;;
        --only) ONLY="$2"; shift ;;
        --check) CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done

want() {
    local step="$1"
    if [[ -n "${ONLY}" ]]; then [[ "${ONLY}" == "${step}" ]]; return; fi
    if [[ -z "${FROM}" ]]; then return 0; fi
    local s
    for s in "${STEPS[@]}"; do
        [[ "${s}" == "${FROM}" ]] && return 0
        [[ "${s}" == "${step}" ]] && return 1
    done
    return 1
}

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
activate() { set +u; conda activate "$1"; local s=$?; set -u; return "${s}"; }

if [[ "${CHECK_ONLY}" -eq 0 ]] && want env; then
    echo "[env] Conda environment ${CONDA_ENV_NAME} (Python 3.10) + build toolchain"
    if ! activate "${CONDA_ENV_NAME}" >/dev/null 2>&1; then
        conda create --name "${CONDA_ENV_NAME}" --yes --override-channels \
            --channel https://repo.anaconda.com/pkgs/main python=3.10 pip
    fi
    conda env update --name "${CONDA_ENV_NAME}" --file "${HERE}/env.yaml"
fi
activate "${CONDA_ENV_NAME}"

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
PIP=("${PYTHON_BIN}" -m pip)
[[ "$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[:2] == (3, 10))')" == "True" ]] \
    || { echo "${CONDA_ENV_NAME} must use Python 3.10" >&2; exit 1; }
"${CONDA_PREFIX}/bin/nvcc" --version | grep -q "release 12\.8" \
    || { echo "${CONDA_ENV_NAME} must carry the CUDA 12.8 toolkit (run --only env)" >&2; exit 1; }

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_HOME="${CONDA_PREFIX}" CUDA_PATH="${CONDA_PREFIX}" CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export CUDAARCHS="${CUDA_ARCH}" TORCH_CUDA_ARCH_LIST="${CUDA_ARCH:0:${#CUDA_ARCH}-1}.${CUDA_ARCH: -1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" MAX_JOBS="${BUILD_JOBS}"
# Conda GCC 12 with the host linker (avoids old-linker RELR errors), as in the Isaac Sim 5.1 build.
export UNIVTAC_GCC12="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export UNIVTAC_GXX12="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
export CC="${HERE}/toolchains/gcc12-system-ld" CXX="${HERE}/toolchains/gxx12-system-ld"
export CUDAHOSTCXX="${CXX}"

check() {
    "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

expected = {"isaacsim": "4.5.0.0", "torch": "2.7.0", "torchvision": "0.22.0", "numpy": "1.26.4",
            "tacex": "0.1.0", "tacex-assets": "0.1.0", "tacex-uipc": "0.1.0"}
failed = False
for package in [*expected, "isaaclab", "nvidia-curobo", "warp-lang", "torch-scatter", "pyuipc"]:
    try:
        installed = version(package)
    except PackageNotFoundError:
        print(f"[missing] {package}")
        failed = True
        continue
    wanted = expected.get(package)
    ok = wanted is None or installed.split("+")[0] == wanted
    print(f"[{'ok' if ok else 'wrong'}] {package}=={installed}")
    failed |= not ok
import torch
print(f"[{'ok' if torch.version.cuda == '12.8' else 'wrong'}] torch CUDA {torch.version.cuda}; "
      f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
raise SystemExit(1 if failed else 0)
PY
}

if [[ "${CHECK_ONLY}" -eq 1 ]]; then check; exit $?; fi

if want isaacsim; then
    echo "[isaacsim] torch 2.7.0+cu128, Isaac Sim 4.5.0.0"
    "${PIP[@]}" install --upgrade pip
    "${PIP[@]}" install "setuptools<82" wheel
    "${PIP[@]}" install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    "${PIP[@]}" install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com
    "${PIP[@]}" install numpy==1.26.4
fi
# Every later pip step keeps torch / torchvision / numpy (scripts/isaac45/constraints.txt).
export PIP_CONSTRAINT="${HERE}/constraints.txt"

if want isaaclab; then
    echo "[isaaclab] Isaac Lab v2.1.1 (${ISAACLAB_COMMIT})"
    [[ -d "${ISAACLAB_DIR}/.git" ]] || git clone https://github.com/isaac-sim/IsaacLab "${ISAACLAB_DIR}"
    git -C "${ISAACLAB_DIR}" fetch -q origin
    git -C "${ISAACLAB_DIR}" checkout -q --detach "${ISAACLAB_COMMIT}"
    "${PIP[@]}" install flatdict==4.0.1 --no-build-isolation
    (cd "${ISAACLAB_DIR}" && ./isaaclab.sh --install)
fi

if want curobo; then
    echo "[curobo] ${CUROBO_COMMIT}, warp-lang ${WARP_VERSION}"
    [[ -d "${CUROBO_DIR}/.git" ]] || git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
    git -C "${CUROBO_DIR}" fetch -q origin
    git -C "${CUROBO_DIR}" checkout -q --detach "${CUROBO_COMMIT}"
    "${PIP[@]}" install "warp-lang==${WARP_VERSION}" --no-build-isolation
    "${PIP[@]}" install -e "${CUROBO_DIR}" --no-build-isolation
fi

if want tacex; then
    echo "[tacex] vendored TacEx core packages (tacex.sh -i without the demo run)"
    "${PIP[@]}" install -e "${PROJECT_ROOT}/third_party/TacEx/source/tacex"
    "${PIP[@]}" install -e "${PROJECT_ROOT}/third_party/TacEx/source/tacex_assets"
    "${PIP[@]}" install -e "${PROJECT_ROOT}/third_party/TacEx/source/tacex_tasks"
    # upstream replaces torch_scatter with the wheel of its torch; ours is 2.7.0+cu128
    "${PIP[@]}" uninstall -y torch_scatter || true
    "${PIP[@]}" install "${TORCH_SCATTER_WHEEL}"
fi

if want vcpkg; then
    echo "[vcpkg] ${VCPKG_COMMIT} at ${VCPKG_ROOT}"
    if [[ ! -x "${VCPKG_ROOT}/vcpkg" ]]; then
        [[ -d "${VCPKG_ROOT}/.git" ]] || { mkdir -p "$(dirname "${VCPKG_ROOT}")"; git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}"; }
        git -C "${VCPKG_ROOT}" fetch -q origin "${VCPKG_COMMIT}"
        git -C "${VCPKG_ROOT}" checkout -q --detach "${VCPKG_COMMIT}"
        "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
    fi
fi

if want uipc; then
    echo "[uipc] libuipc + tacex_uipc for sm_${CUDA_ARCH}"
    # vcpkg only for this build: exported globally it would leak into unrelated CMake builds (egl_probe).
    export CMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    export VCPKG_OVERLAY_PORTS="${HERE}/toolchains/vcpkg-overlay-ports"
    rm -rf "${UIPC_DIR}/build"
    "${PIP[@]}" install -e "${UIPC_DIR}" -v --no-build-isolation
fi

if want extras; then
    echo "[extras] upstream extras + IPython for policy/ACT"
    "${PIP[@]}" install transforms3d trimesh tetgen
    # policy/ACT imports IPython at module level; IPython 9 needs psutil>=7 (isaacsim-kernel pins lower)
    "${PIP[@]}" install "ipython<9"
    "${PIP[@]}" install numpy==1.26.4
fi

check
