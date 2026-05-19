#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

NVCC="/usr/local/cuda-12.9/bin/nvcc"
PROXY="${KERNELGYM_PROXY:-${KERNELGYM_FALLBACK_PROXY:-http://192.168.28.186:7897}}"
RECREATE=0

usage() {
    echo "Usage: ./ensure_venv.sh [--recreate]"
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

run_apt_get() {
    if [[ "$(id -u)" == "0" ]]; then
        network env DEBIAN_FRONTEND=noninteractive apt-get "$@"
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        echo "redis-server is missing and sudo is not available for apt-get install" >&2
        exit 1
    fi
    network sudo -E env DEBIAN_FRONTEND=noninteractive apt-get "$@"
}

ensure_redis_server() {
    if ! command -v redis-server >/dev/null 2>&1; then
        if ! command -v apt-get >/dev/null 2>&1; then
            echo "redis-server is missing and apt-get is not available" >&2
            exit 1
        fi
        echo "Installing redis-server"
        run_apt_get update
        run_apt_get install -y --no-install-recommends redis-server
    fi
    if ! command -v redis-server >/dev/null 2>&1; then
        echo "redis-server is still not available after installation" >&2
        exit 1
    fi
    redis-server --version
}

ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        network pip install uv
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is still not available on PATH after pip install uv" >&2
        exit 1
    fi
}

ensure_python_env() {
    local install_deps=0
    if [[ "${RECREATE}" == "1" && -e .venv ]]; then
        rm -rf .venv
    fi
    if [[ ! -e .venv ]]; then
        uv venv -p 3.12
        install_deps=1
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    if ! python - <<'PY' >/dev/null 2>&1
import redis
import torch
import tvm_ffi
PY
    then
        install_deps=1
    fi
    if [[ "${install_deps}" == "1" ]]; then
        network uv pip install -e ".[dev,tvm-ffi]"
        network uv pip install -r requirements-cuda129.txt
    fi
}

cd "${ROOT_DIR}"
echo "profile=${PROFILE}"
echo "nvcc=${NVCC}"
check_cuda129
ensure_redis_server
ensure_uv
ensure_python_env
python scripts/validate_cuda129.py
