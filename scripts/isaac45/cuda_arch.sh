# SM versions for the CUDA builds come as a ';'-separated list ("120", or "89;120" for RTX 40 + 50 series).
# CMake takes that list as is (CUDAARCHS); torch extensions (cuRobo) want major.minor, e.g. "8.9;12.0".
torch_arch_list() {
    local list="$1" archs a out=()
    if [[ ! "${list}" =~ ^[0-9]{2,3}(;[0-9]{2,3})*$ ]]; then
        echo "bad SM version list '${list}' (expected e.g. 120 or 89;120)" >&2
        return 2
    fi
    IFS=';' read -r -a archs <<< "${list}"
    for a in "${archs[@]}"; do
        out+=("${a:0:${#a}-1}.${a: -1}")
    done
    (IFS=';'; echo "${out[*]}")
}
