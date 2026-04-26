#!/usr/bin/env bash
# Script to request an interactive Slurm node.
#
# Usage:
#   bash ./scripts/slurm_interactive.sh [NUM_GPUS] [--account NAME] [--partition NAME]
#
# Examples:
#   # 4 GPUs on the defaults with default account=nvr_torontoai_videogen, partition=batch
#   bash ./scripts/slurm_interactive.sh 4
#
#   # Override account / partition for this run
#   bash ./scripts/slurm_interactive.sh 4 --account <MY_ACCOUNT> --partition <MY_PARTITION>

set -euo pipefail

NUM_GPUS=4
SLURM_ACCOUNT="nvr_torontoai_videogen"
SLURM_PARTITION="batch"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)     SLURM_ACCOUNT="$2";        shift 2 ;;
        --account=*)   SLURM_ACCOUNT="${1#*=}";   shift   ;;
        --partition)   SLURM_PARTITION="$2";      shift 2 ;;
        --partition=*) SLURM_PARTITION="${1#*=}"; shift   ;;
        -h|--help)     sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --*)           echo "Unknown option: $1" >&2; exit 2 ;;
        *)             NUM_GPUS="$1";             shift   ;;
    esac
done

REPO_ROOT="${FLASHSIM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${FLASHSIM_TEST_IMAGE:-gitlab-master.nvidia.com/sil/flashsim:base-v0.3-20260424-55bd566}"

UV_CACHE_HOST="${FLASHSIM_UV_CACHE_DIR:-${HOME}/.cache/uv}"
HF_CACHE_HOST="${FLASHSIM_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
FLASHSIM_CACHE_HOST="${FLASHSIM_CACHE_DIR:-${HOME}/.cache/flashsim}"
TRITON_CACHE_HOST="${FLASHSIM_TRITON_CACHE_DIR:-${HOME}/.cache/triton}"

mkdir -p "${UV_CACHE_HOST}" "${HF_CACHE_HOST}" "${FLASHSIM_CACHE_HOST}" "${TRITON_CACHE_HOST}"

srun -A "${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --qos=interactive \
    --nodes=1 \
    --gpus-per-node="${NUM_GPUS}" \
    --cpus-per-gpu=36 \
    --exclusive \
    --time=4:00:00 \
    --pty \
    --container-image="${IMAGE}" \
    --container-mounts="${REPO_ROOT}:/workspace/flashsim,${UV_CACHE_HOST}:/root/.cache/uv,${HF_CACHE_HOST}:/root/.cache/huggingface,${FLASHSIM_CACHE_HOST}:/root/.cache/flashsim,${TRITON_CACHE_HOST}:/root/.cache/triton,/lustre:/lustre" \
    --container-workdir=/workspace/flashsim \
    --container-writable \
    --container-mount-home \
    --container-remap-root \
    --export=ALL,HF_HOME=/root/.cache/huggingface,UV_LINK_MODE=copy,TRITON_CACHE_DIR=/root/.cache/triton \
    /bin/bash
