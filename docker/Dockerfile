# Test environment matching nvcr.io/nvidian/qiwu:fast-infer-v9
#
# Build:
#   docker build -t gitlab-master.nvidia.com:5005/sil/flashsim:base-v0.1dev0 -f docker/Dockerfile .
#
# Run tests (GPU required):
#   docker/run_tests.sh
#
FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /workspace/flashsim

# Copy only the package manifest so dep installation is decoupled from source
# changes (source is bind-mounted at runtime via run_tests.sh).
COPY pyproject.toml README.md ./
# Stub the package so setuptools can register an editable install without the
# full source tree; the real source replaces this stub at runtime.
RUN mkdir -p flashsim && touch flashsim/__init__.py

# Install all project dependencies from pyproject.toml (including dev/streaming
# extras). torch and transformer-engine are pre-installed by the base image with
# NVIDIA custom builds; the >=2.11.0a0 / >=2.12 lower bounds accept them.
RUN pip install --no-cache-dir -e ".[dev,streaming,streaming_viser]"

# Source code is bind-mounted at runtime (see docker/run_tests.sh).
# PYTHONPATH ensures projects/ and flashsim/ are importable via their source paths.
ENV PYTHONPATH=/workspace/flashsim
