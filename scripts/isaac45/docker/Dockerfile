# Isaac Sim 4.5 + this repository's upstream code, for evaluating the released ACT checkpoints on cloud GPUs.
#
# Build with scripts/isaac45/docker/build.sh: its context is a fresh clone of one commit (tracked files and
# .git only; the assets step needs .git to fetch TacEx's LFS objects from upstream).
#
# The environment is the one scripts/isaac45/install.sh --locked builds locally, run one step per layer,
# for SM 8.9 (RTX 40 native; its PTX runs on RTX 50), as no GPU is visible at build time. The released checkpoints
# come from the release's HuggingFace copy at a fixed commit. The build fails if any file outside scripts/isaac45
# differs from the upstream commit, so the image evaluates upstream's code.

# Source archives libuipc's vcpkg build may take from a local vcpkg downloads folder (build.sh passes one as the build
# context "vcpkg-downloads") instead of fetching them; vcpkg checks each file's SHA512 against its port, so only the
# same sources are used. Empty by default: everything is downloaded.
FROM scratch AS vcpkg-downloads

FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ARG MINIFORGE_VERSION=26.3.2-3
ARG UNIVTAC45_CUDA_ARCH=89
ARG UNIVTAC45_BUILD_JOBS=8
ARG UPSTREAM_COMMIT=0dafa10262e22f486f160a55d6f11aeab12d8e7b
ARG CHECKPOINT_REPO=byml/UniVTAC
ARG CHECKPOINT_REVISION=e1aee7b0c95543b535e0146b2de3ee1bc6ddaabd

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CONDA_DIR=/opt/conda \
    UNIVTAC_ROOT=/workspace/univtac
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Build tools as on the local machine (the compiler wrappers use the system linker, libuipc builds with
# make, vcpkg needs zip/unzip/tar), the libraries Isaac Sim's renderer loads, ffmpeg for the eval videos,
# and sshd + rsync so results can be pulled from a pod.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config ca-certificates curl git git-lfs zip unzip tar python3 \
        ffmpeg openssh-server rsync \
        libatomic1 libegl1 libgl1 libglu1-mesa libgomp1 libvulkan1 vulkan-tools \
        libsm6 libice6 libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxt6 libxcursor1 libxinerama1 \
        libglib2.0-0t64 libfontconfig1 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Vulkan and EGL find the NVIDIA driver libraries (mounted by the container runtime) through these.
RUN mkdir -p /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.242"}}\n' \
        > /etc/vulkan/icd.d/nvidia_icd.json \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

WORKDIR ${UNIVTAC_ROOT}

# Released checkpoints first: they do not depend on the code, so code changes keep this layer.
# The release has two official copies (README: HuggingFace and ModelScope) with the same policy weights. The
# ModelScope one, which upstream's data/download.sh fetches, has no dataset_stats.pkl for insert_hole and lift_bottle,
# which evaluation needs, and other eval logs for put_bottle_in_shelf; the HuggingFace one is complete. The downloader
# (huggingface_hub) lives in its own environment, outside the locked UniVTAC45 one.
RUN conda create -y -q -n download python=3.11 pip \
    && conda run -n download python -m pip install -q --no-cache-dir huggingface_hub \
    && conda run -n download python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${CHECKPOINT_REPO}', repo_type='dataset', revision='${CHECKPOINT_REVISION}', allow_patterns='checkpoints/*', local_dir='data')" \
    && rm -rf data/.cache \
    && (cd data && find checkpoints -type f -print0 | sort -z | xargs -0 sha256sum > checkpoints.sha256) \
    && { echo "repo ${CHECKPOINT_REPO} (HuggingFace dataset) revision ${CHECKPOINT_REVISION}"; \
         conda run -n download python -m pip show huggingface_hub | grep -E '^(Name|Version)'; } > data/checkpoints.source \
    && conda clean -afy

# Steps that only need scripts/isaac45 (the toolchain, Isaac Sim, Isaac Lab, cuRobo clone into third_party/).
COPY scripts/isaac45 scripts/isaac45
ENV UNIVTAC45_CUDA_ARCH=${UNIVTAC45_CUDA_ARCH} UNIVTAC45_BUILD_JOBS=${UNIVTAC45_BUILD_JOBS}
RUN bash scripts/isaac45/install.sh --locked --only env && conda clean -afy
RUN --mount=type=cache,target=/root/.cache/pip bash scripts/isaac45/install.sh --locked --only isaacsim
RUN --mount=type=cache,target=/root/.cache/pip bash scripts/isaac45/install.sh --locked --only isaaclab
RUN --mount=type=cache,target=/root/.cache/pip bash scripts/isaac45/install.sh --locked --only curobo

# The whole commit; stop here if it is not upstream's code plus scripts/isaac45.
COPY . .
RUN git config --global --add safe.directory "${UNIVTAC_ROOT}" \
    && if ! git diff --quiet "${UPSTREAM_COMMIT}" HEAD -- . ':(exclude)scripts/isaac45'; then \
           echo "files outside scripts/isaac45 differ from upstream ${UPSTREAM_COMMIT}:" >&2; \
           git diff --stat "${UPSTREAM_COMMIT}" HEAD -- . ':(exclude)scripts/isaac45' >&2; \
           exit 1; \
       fi \
    && test -z "$(git status --porcelain --untracked-files=no)"

RUN bash scripts/isaac45/install.sh --locked --only assets
RUN --mount=type=cache,target=/root/.cache/pip bash scripts/isaac45/install.sh --locked --only tacex
RUN bash scripts/isaac45/install.sh --locked --only vcpkg
RUN --mount=type=bind,from=vcpkg-downloads,target=/vcpkg-seed \
    --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /root/.cache/univtac-isaac45/vcpkg/downloads \
    && find /vcpkg-seed -maxdepth 1 -type f -exec cp -n {} /root/.cache/univtac-isaac45/vcpkg/downloads/ \; \
    && echo "[vcpkg] $(find /vcpkg-seed -maxdepth 1 -type f | wc -l) source archives seeded" \
    && bash scripts/isaac45/install.sh --locked --only uipc
RUN --mount=type=cache,target=/root/.cache/pip bash scripts/isaac45/install.sh --locked --only extras
RUN bash scripts/isaac45/install.sh --check

# What this image is, for the manifest each evaluation writes.
RUN printf '{"fork_commit": "%s", "upstream_commit": "%s", "cuda_arch": "%s", "miniforge": "%s", "checkpoints": "%s", "built_utc": "%s"}\n' \
        "$(git rev-parse HEAD)" "${UPSTREAM_COMMIT}" "${UNIVTAC45_CUDA_ARCH}" "${MINIFORGE_VERSION}" \
        "${CHECKPOINT_REPO}@${CHECKPOINT_REVISION}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /workspace/build_info.json

ENTRYPOINT ["bash", "/workspace/univtac/scripts/isaac45/docker/pod_start.sh"]
