#!/usr/bin/env bash
# Run the flashsim test suite inside the Docker test container.
#
# Usage:
#   docker/run_tests.sh [OPTIONS] [PYTEST_ARGS...]
#
# Options:
#   --build    Rebuild the Docker image before running.
#
# Examples:
#   # Run all tests:
#   docker/run_tests.sh
#
#   # Run a specific test file:
#   docker/run_tests.sh tests/test_streaming_ws_protocol.py
#
#   # Rebuild image then run:
#   docker/run_tests.sh --build
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FLASHSIM_TEST_IMAGE:-gitlab-master.nvidia.com:5005/sil/flashsim:base-v0.1dev1}"

BUILD=0
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=1; shift ;;
        *)       PYTEST_ARGS+=("$1"); shift ;;
    esac
done

if [[ $BUILD -eq 1 ]]; then
    docker build -t "${IMAGE}" -f "${REPO_ROOT}/docker/Dockerfile" "${REPO_ROOT}"
fi

if [[ ${#PYTEST_ARGS[@]} -eq 0 ]]; then
    PYTEST_ARGS=(tests/ projects/)
fi

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${REPO_ROOT}:/workspace/flashsim" \
    "${IMAGE}" \
    python -m pytest -m "not manual" "${PYTEST_ARGS[@]}"
