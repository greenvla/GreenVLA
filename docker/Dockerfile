# syntax=docker/dockerfile:1.10
# ============================================================================
# Green-VLA — whole-body policy inference server
#
# Build:
#   docker build -f docker/Dockerfile -t green-vla:latest .
#
# Run offline with the checkpoint and Qwen backbone included in the image:
#   docker run --gpus all --rm -p 8999:8999 \
#     green-vla:latest --device cuda:0
#
# The build includes the published checkpoint; a /ckpt mount overrides it.
# A checkpoint is a directory with `pretrained_model/` (config.json + model.safetensors) and
# `norm_stats/`. The pinned Qwen snapshot is downloaded to /opt/hf-cache during
# the build. Runtime HF access is disabled; do not hide this cache with an empty mount.
# ============================================================================

# The devel image, not runtime: causal-conv1d is compiled from source and needs
# nvcc. CUDA 12.8 supports Blackwell (sm_120).
ARG CUDA_VERSION=12.8.1
ARG OS_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}

ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda \
    # HF_HUB_CACHE points at the directory that holds models--<org>--<name>, which is what a
    # warm cache copied from elsewhere looks like; HF_HOME keeps the rest of the HF state.
    HF_HUB_CACHE=/opt/hf-cache \
    HF_HOME=/opt/hf-home \
    CHECKPOINT_PATH=/ckpt \
    TOKENIZERS_PARALLELISM=false

# ffmpeg is the one system dependency the server itself needs: torchcodec links
# the libav* libraries and the import chain of serve_policy reaches it.
# build-essential and git are here for the kernels compiled further down.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# uv comes straight from its published image; Ubuntu 22.04 ships Python 3.10,
# so the 3.11 interpreter comes from uv rather than apt.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv
RUN export UV_PYTHON_INSTALL_DIR=/opt/python \
    && uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Install the inference dependencies, including the simulator wire boundary.
COPY pyproject.toml /app/pyproject.toml
WORKDIR /app
RUN uv pip install --no-cache --python /opt/venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match \
        -r /app/pyproject.toml

RUN hf download SberRoboticsCenter/GreenChallengeModel \
    --local-dir /ckpt

# Build causal-conv1d against the installed PyTorch/CUDA runtime. It adds sm_120
# with nvcc >=12.8; its setup.py supplies explicit -gencode flags. The pinned
# official FlashAttention wheel already contains sm_80/90/100/120 kernels.
ARG TORCH_CUDA_ARCH_LIST="8.9;12.0"
ARG MAX_JOBS=8
RUN TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} \
       uv pip install --no-cache --python /opt/venv/bin/python --no-deps --no-build-isolation \
       --group /app/pyproject.toml:linear-attention \
    && TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} \
       CAUSAL_CONV1D_FORCE_BUILD=TRUE \
       uv pip install --no-cache --python /opt/venv/bin/python --no-deps --no-build-isolation \
       --group /app/pyproject.toml:causal-conv1d \
    && uv pip install --no-cache --python /opt/venv/bin/python --no-deps \
       --group /app/pyproject.toml:flash-attention

ARG USER=greenvla
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} ${USER} \
    && useradd --create-home --shell /bin/bash --uid ${UID} --gid ${GID} ${USER} \
    && mkdir -p /opt/hf-cache /opt/hf-home /ckpt \
    && chown -R ${UID}:${GID} /opt/hf-cache /opt/hf-home /ckpt /app

# Keep the model layer cached when only inference code changes. This uses the
# same pinned revision and required files as the launcher profiles, including
# weights, tokenizer, chat template and image/video processors. The helper binds
# refs/main to that revision because runtime resolves Qwen by its HF repo ID.
COPY --chown=${UID}:${GID} tools/repro017799/profile.json tools/repro017799/prepare_assets.py /app/tools/repro017799/
USER ${USER}
RUN python /app/tools/repro017799/prepare_assets.py --cache /opt < /dev/null

COPY --chown=${UID}:${GID} . /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

EXPOSE 8999

# The server answers a plain HTTP GET /healthz before the WebSocket upgrade.
# start-period covers loading the checkpoint onto the GPU.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request,os; p=os.environ.get('POLICY_PORT','8999'); sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+p+'/healthz', timeout=4).status == 200 else 1)"]

# Same infrastructure interface; raw VLA + wire boundary (learned corrections opt-in).
# CHECKPOINT_PATH / POLICY_PORT remain runtime settings. See docs/INFRA.md.
ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD []
