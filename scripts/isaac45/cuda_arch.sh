# The CUDA builds target one SM version (UNIVTAC45_CUDA_ARCH, e.g. 120 for RTX 50, 89 for RTX 40): libuipc passes
# the CMake value unquoted to set_target_properties, so a list ("89;120") breaks its configure step.
# CMake's plain "89" compiles sm_89 code plus compute_89 PTX, which newer GPUs (sm_120) compile at load time; the
# torch extensions (cuRobo) get the same pair through "+PTX", e.g. "8.9+PTX".
torch_arch_list() {
    local sm="$1"
    if [[ ! "${sm}" =~ ^[0-9]{2,3}$ ]]; then
        echo "bad SM version '${sm}' (one value, e.g. 120 or 89)" >&2
        return 2
    fi
    echo "${sm:0:${#sm}-1}.${sm: -1}+PTX"
}
