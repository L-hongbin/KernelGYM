#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

NVCC="/usr/local/cuda-12.9/bin/nvcc"
PROXY="${KERNELGYM_PROXY:-${KERNELGYM_FALLBACK_PROXY:-http://192.168.28.186:7897}}"
RECREATE=0

usage() {
    echo "Usage: ./create_venv.sh [--recreate]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 1 || ( "${1:-}" != "" && "${1:-}" != "--recreate" ) ]]; then
    usage >&2
    exit 2
fi

if [[ "${1:-}" == "--recreate" ]]; then
    RECREATE=1
fi

PROFILE="internal"
if [[ -L /ms || ! -e /ms ]]; then
    PROFILE="external"
fi

network() {
    if "$@"; then
        return 0
    fi
    if [[ "${PROFILE}" != "external" || -z "${PROXY}" ]]; then
        return 1
    fi
    echo "Retrying with proxy ${PROXY}"
    env HTTP_PROXY="${PROXY}" HTTPS_PROXY="${PROXY}" ALL_PROXY="${PROXY}" \
        http_proxy="${PROXY}" https_proxy="${PROXY}" all_proxy="${PROXY}" "$@"
}

check_cuda129() {
    if [[ ! -x "${NVCC}" ]]; then
        echo "CUDA 12.9 nvcc not found at ${NVCC}" >&2
        exit 1
    fi
    local version
    version="$("${NVCC}" --version)"
    echo "${version}" | tail -n 1
    if [[ "${version}" != *"release 12.9"* && "${version}" != *"V12.9"* ]]; then
        echo "Expected CUDA 12.9 nvcc at ${NVCC}" >&2
        exit 1
    fi
}

cd "${ROOT_DIR}"
echo "profile=${PROFILE}"
echo "nvcc=${NVCC}"
check_cuda129

if ! command -v uv >/dev/null 2>&1; then
    network pip install uv
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is still not available on PATH after pip install uv" >&2
    exit 1
fi

if [[ "${RECREATE}" == "1" && -e .venv ]]; then
    rm -rf .venv
fi

if [[ ! -e .venv ]]; then
    uv venv -p 3.12
fi

source .venv/bin/activate
network uv pip install -e ".[dev]"
network uv pip install -r requirements-cuda129.txt

python scripts/validate_cuda129.py
