FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

# Default workdir used by test scripts (pyxis/enroot requires it to exist)
RUN mkdir -p /workspace

# System deps: Python 3.12, ffmpeg, build tools for transformer-engine-torch
RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        git \
        python3 python3-dev python3-venv \
        ffmpeg \
        gcc g++ ninja-build \
        libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
