#!/usr/bin/env bash
# Run the flashsim test suite inside the base NVIDIA PyTorch container.
#
# Usage:
#   docker/run_tests.sh [TEST_TARGET...]
#
# Examples:
#   # Run all tests:
#   docker/run_tests.sh
#
#   # Run a specific test file:
#   docker/run_tests.sh integrations/streaming_ws/tests/test_streaming_ws_protocol.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FLASHSIM_TEST_IMAGE:-nvcr.io/nvidia/pytorch:26.02-py3}"
TEST_TARGETS=("$@")

UV_CACHE_HOST="${FLASHSIM_UV_CACHE_DIR:-${HOME}/.cache/uv}"
HF_CACHE_HOST="${FLASHSIM_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
FLASHSIM_CACHE_HOST="${FLASHSIM_CACHE_DIR:-${HOME}/.cache/flashsim}"

mkdir -p "${UV_CACHE_HOST}" "${HF_CACHE_HOST}" "${FLASHSIM_CACHE_HOST}"

docker run --rm -i \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${REPO_ROOT}:/workspace/flashsim" \
    -v "${UV_CACHE_HOST}:/root/.cache/uv" \
    -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
    -v "${FLASHSIM_CACHE_HOST}:/root/.cache/flashsim" \
    -e HF_HOME=/root/.cache/huggingface \
    -e UV_LINK_MODE=copy \
    -w /workspace/flashsim \
    "${IMAGE}" \
    bash -s -- "${TEST_TARGETS[@]}" <<'EOF'
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --break-system-packages --no-cache-dir uv
fi

INSTALL_TARGETS=("flashsim[dev]")
for integration_dir in integrations/*; do
    if [[ -f "${integration_dir}/pyproject.toml" ]]; then
        INSTALL_TARGETS+=("${integration_dir}[dev]")
    fi
done

INSTALL_ARGS=()
for target in "${INSTALL_TARGETS[@]}"; do
    INSTALL_ARGS+=("-e" "${target}")
done

uv pip install --system --break-system-packages --no-build-isolation "${INSTALL_ARGS[@]}"

if [[ $# -eq 0 ]]; then
    TEST_TARGETS=()
    shopt -s nullglob
    for test_file in flashsim/tests/test_*.py integrations/*/tests/test_*.py tests/test_*.py; do
        TEST_TARGETS+=("${test_file}")
    done
    shopt -u nullglob
else
    TEST_TARGETS=("$@")
fi

if [[ ${#TEST_TARGETS[@]} -eq 0 ]]; then
    echo "No test targets found (expected flashsim/tests, integrations/*/tests, or tests)." >&2
    exit 1
fi

echo "=== Running pytest for ${#TEST_TARGETS[@]} target(s) ==="
python -m pytest -m "not manual" "${TEST_TARGETS[@]}"
EOF
