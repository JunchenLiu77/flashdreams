#!/usr/bin/env bash
# Run the flashdreams test suite inside a fresh NVIDIA PyTorch docker container.
#
# Use this on a local machine that has docker and at least one GPU.
# The script installs flashdreams + integration packages on the fly, then
# invokes pytest. Caches for uv / huggingface / flashdreams are bind-mounted
# from the host so subsequent runs are fast.
#
# Usage:
#   ./tests/run_tests_docker.sh [TEST_TARGET...]
#
# Environment overrides:
#   FLASHDREAMS_TEST_IMAGE         (default: gitlab-master.nvidia.com:5005/sil/flashdreams:base-v0.3-20260424-55bd566)
#   FLASHDREAMS_UV_CACHE_DIR       (default: ${HOME}/.cache/uv)
#   FLASHDREAMS_HF_CACHE_DIR       (default: ${HOME}/.cache/huggingface)
#   FLASHDREAMS_CACHE_DIR          (default: ${HOME}/.cache/flashdreams)
#   FLASHDREAMS_TRITON_CACHE_DIR   (default: ${HOME}/.cache/triton)
#
# Examples:
#   # Run all tests
#   ./tests/run_tests_docker.sh
#
#   # Run a specific test file
#   ./tests/run_tests_docker.sh flashdreams/tests/test_attention.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FLASHDREAMS_TEST_IMAGE:-gitlab-master.nvidia.com:5005/sil/flashdreams:base-v0.3-20260424-55bd566}"

UV_CACHE_HOST="${FLASHDREAMS_UV_CACHE_DIR:-${HOME}/.cache/uv}"
HF_CACHE_HOST="${FLASHDREAMS_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
FLASHDREAMS_CACHE_HOST="${FLASHDREAMS_CACHE_DIR:-${HOME}/.cache/flashdreams}"
TRITON_CACHE_HOST="${FLASHDREAMS_TRITON_CACHE_DIR:-${HOME}/.cache/triton}"

mkdir -p "${UV_CACHE_HOST}" "${HF_CACHE_HOST}" "${FLASHDREAMS_CACHE_HOST}" "${TRITON_CACHE_HOST}"

docker run --rm -i \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${REPO_ROOT}:/workspace/flashdreams" \
    -v "${UV_CACHE_HOST}:/root/.cache/uv" \
    -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
    -v "${FLASHDREAMS_CACHE_HOST}:/root/.cache/flashdreams" \
    -v "${TRITON_CACHE_HOST}:/root/.cache/triton" \
    -v "${HOME}/.netrc:/root/.netrc:ro" \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRITON_CACHE_DIR=/root/.cache/triton \
    -e UV_LINK_MODE=copy \
    -e UV_PROJECT_ENVIRONMENT=/tmp/flashdreams-venv \
    -w /workspace/flashdreams \
    "${IMAGE}" \
    bash -s -- "$@" <<'EOF'
set -euo pipefail

# UV_PROJECT_ENVIRONMENT is set via docker -e so the venv lives outside the
# bind-mounted workspace, avoiding root-owned .venv on the host.
uv venv --clear
uv sync --frozen --extra dev

exec bash /workspace/flashdreams/tests/run_tests_local.sh "$@"
EOF
