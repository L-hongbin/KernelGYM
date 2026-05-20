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
    # Prefer the canonical CUDA 12.9 install; fall back to whatever nvcc is on
    # PATH so machines that symlink /usr/local/cuda or only expose nvcc through
    # the environment still pass. The resolved binary must report release 12.9
    # exactly — torch is pinned to the +cu129 wheel and the deployed driver
    # line is sized for 12.9, so any other release breaks runtime linkage.
    local candidate=""
    if [[ -x "${NVCC}" ]]; then
        candidate="${NVCC}"
    elif command -v nvcc >/dev/null 2>&1; then
        candidate="$(command -v nvcc)"
    else
        echo "nvcc not found (tried ${NVCC} and \$PATH)" >&2
        exit 1
    fi
    local version major minor
    version="$("${candidate}" --version)"
    echo "${version}" | tail -n 1
    # Extract "release X.Y" from nvcc --version output.
    if [[ ! "${version}" =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
        echo "Could not parse nvcc release from output of ${candidate}" >&2
        exit 1
    fi
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    if (( major != 12 || minor != 9 )); then
        echo "Expected nvcc release 12.9, got ${major}.${minor} at ${candidate}" >&2
        exit 1
    fi
    NVCC="${candidate}"
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
    # Drop UV_PROJECT_ENVIRONMENT so a host-wide override (e.g. /opt/venv on
    # some shared images) cannot redirect either `uv venv` or `uv pip install`
    # away from the repo-local .venv we want.
    unset UV_PROJECT_ENVIRONMENT
    local install_deps=0
    if [[ "${RECREATE}" == "1" && -e .venv ]]; then
        rm -rf .venv
    fi
    if [[ ! -e .venv ]]; then
        # Pass the path explicitly so uv writes to ./.venv even on hosts that
        # tried to coerce a different location through env or config.
        uv venv .venv -p 3.12
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
        # Prefer the locally-staged +cu129 wheels (./wheels/*.whl) over the
        # configured index when they exist. uv resolves --find-links before
        # the index, so an intranet that only has the standard PyPI mirror
        # can still satisfy the cu129 pins. If ./wheels/ is empty or absent,
        # uv just falls through to the configured index.
        local find_links_arg=()
        if compgen -G "${ROOT_DIR}/wheels/*.whl" >/dev/null; then
            echo "Installing torch/torchvision from local wheels under ${ROOT_DIR}/wheels"
            find_links_arg=(--find-links "${ROOT_DIR}/wheels")
        fi
        network uv pip install "${find_links_arg[@]}" -r requirements-cuda129.txt
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
