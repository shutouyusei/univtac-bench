# torch_arch_list SM: TORCH_CUDA_ARCH_LIST for one SM, e.g. 89 -> "8.9+PTX" (the PTX, like CMake's default, also
# runs on newer GPUs). One SM only: libuipc passes the CMake value unquoted, so a list ("89;120") breaks it.
torch_arch_list() {
    local sm="$1"
    if [[ ! "${sm}" =~ ^[0-9]{2,3}$ ]]; then
        echo "bad SM version '${sm}' (one value, e.g. 120 or 89)" >&2
        return 2
    fi
    echo "${sm:0:${#sm}-1}.${sm: -1}+PTX"
}
