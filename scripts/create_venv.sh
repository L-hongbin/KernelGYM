#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV="${KERNELGYM_VENV:-${ROOT_DIR}/.venv}"
PYTHON="${KERNELGYM_PYTHON:-python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
FALLBACK_PROXY="${KERNELGYM_FALLBACK_PROXY:-http://192.168.28.186:7897}"
RECREATE=0

case "${1:-}" in
    "")
        ;;
    --recreate)
        RECREATE=1
        ;;
    -h|--help)
        echo "Usage: scripts/create_venv.sh [--recreate]"
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        echo "Usage: scripts/create_venv.sh [--recreate]" >&2
        exit 2
        ;;
esac

if [[ $# -gt 1 ]]; then
    echo "Too many arguments." >&2
    exit 2
fi

if [[ -L /ms || ! -e /ms ]]; then
    PROFILE="external"
else
    PROFILE="internal"
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

run() {
    echo "+ $*"
    "$@"
}

run_with_proxy() {
    local proxy="$1"
    shift
    echo "+ HTTP_PROXY=${proxy} $*"
    env HTTP_PROXY="${proxy}" HTTPS_PROXY="${proxy}" ALL_PROXY="${proxy}" \
        http_proxy="${proxy}" https_proxy="${proxy}" all_proxy="${proxy}" "$@"
}

network() {
    if [[ -n "${KERNELGYM_PROXY:-}" ]]; then
        run_with_proxy "${KERNELGYM_PROXY}" "$@"
    elif run "$@"; then
        return 0
    elif [[ "${PROFILE}" == "external" && -n "${FALLBACK_PROXY}" ]]; then
        echo "Retrying with proxy ${FALLBACK_PROXY}"
        run_with_proxy "${FALLBACK_PROXY}" "$@"
    else
        return 1
    fi
}

cd "${ROOT_DIR}"
echo "profile=${PROFILE}"
echo "venv=${VENV}"
echo "CUDA_HOME=${CUDA_HOME}"

if command -v uv >/dev/null 2>&1; then
    UV=(uv)
else
    network "${PYTHON}" -m pip install uv
    UV=("${PYTHON}" -m uv)
fi

if [[ "${RECREATE}" == "1" && -e "${VENV}" ]]; then
    run rm -rf "${VENV}"
fi

if [[ ! -e "${VENV}" ]]; then
    run "${UV[@]}" venv --python "${PYTHON}" "${VENV}"
fi

network "${UV[@]}" pip install --python "${VENV}/bin/python" -e ".[dev]"
network "${UV[@]}" pip install --python "${VENV}/bin/python" -r requirements-cuda129.txt

run "${VENV}/bin/python" scripts/validate_cuda129.py
