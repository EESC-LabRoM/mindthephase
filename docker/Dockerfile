FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git curl cmake build-essential libglib2.0-0 libglu1-mesa \
    libvulkan1 vulkan-tools pkg-config python3-pip \
    libgl1 \
    libx11-6 \
    libvulkan1 \
    libxt6 libxrender1 libxext6 libsm6 libice6 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENV UV_CACHE_DIR="/root/.cache/uv"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /workspace/
COPY uv.lock /workspace/uv.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra-index-url https://pypi.nvidia.com

ENTRYPOINT ["/bin/bash"]
