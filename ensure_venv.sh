#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/runtime_paths.sh"

NVCC="/usr/local/cuda-12.9/bin/nvcc"
PYTHON_TARGET="${KERNELGYM_PYTHON_TARGET:-/usr/bin/python3.12}"
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

check_paths() {
    kernelgym_require_local_venv_path
    if [[ ! -x "${PYTHON_TARGET}" ]]; then
        echo "python target not executable: ${PYTHON_TARGET}" >&2
        exit 1
    fi
    if [[ ! -d "${KERNELGYM_OFFLINE_WHEEL_DIR}" ]]; then
        echo "offline wheel directory not found: ${KERNELGYM_OFFLINE_WHEEL_DIR}" >&2
        exit 1
    fi
    if ! compgen -G "${KERNELGYM_OFFLINE_WHEEL_DIR}/*.whl" >/dev/null; then
        echo "offline wheel directory is empty: ${KERNELGYM_OFFLINE_WHEEL_DIR}" >&2
        exit 1
    fi
}

check_cuda129() {
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

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return
    fi

    # Bootstrap uv itself from the same offline wheelhouse. This touches only
    # root's node-local user site and never consults a package index.
    "${PYTHON_TARGET}" -m pip install --user --no-index \
        --find-links "${KERNELGYM_OFFLINE_WHEEL_DIR}" uv
    UV_BIN="$("${PYTHON_TARGET}" -m site --user-base)/bin/uv"
    if [[ ! -x "${UV_BIN}" ]]; then
        echo "uv is unavailable after offline bootstrap from ${KERNELGYM_OFFLINE_WHEEL_DIR}" >&2
        exit 1
    fi
}

ensure_python_env() {
    unset UV_PROJECT_ENVIRONMENT
    if [[ "${RECREATE}" == "1" && -e "${KERNELGYM_LOCAL_VENV_DIR}" ]]; then
        rm -rf -- "${KERNELGYM_LOCAL_VENV_DIR}"
    fi
    if [[ ! -e "${KERNELGYM_LOCAL_VENV_DIR}" ]]; then
        mkdir -p "$(dirname "${KERNELGYM_LOCAL_VENV_DIR}")"
        "${UV_BIN}" venv "${KERNELGYM_LOCAL_VENV_DIR}" --python "${PYTHON_TARGET}"
    fi

    # shellcheck disable=SC1091
    source "${KERNELGYM_LOCAL_VENV_DIR}/bin/activate"
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/scripts/scrub_venv_env.sh"

    echo
    echo "=== Install exclusively from offline wheelhouse ==="
    "${UV_BIN}" pip install --offline --no-cache --no-index \
        --find-links "${KERNELGYM_OFFLINE_WHEEL_DIR}" \
        "setuptools==81.0.0" "wheel==0.48.0"
    "${UV_BIN}" pip install --offline --no-cache --no-index \
        --find-links "${KERNELGYM_OFFLINE_WHEEL_DIR}" \
        -r "${ROOT_DIR}/requirements-offline.txt"
    "${UV_BIN}" pip install --offline --no-cache --no-index \
        --find-links "${KERNELGYM_OFFLINE_WHEEL_DIR}" \
        --no-deps -e "${ROOT_DIR}" \
        --no-build-isolation
}

cd "${ROOT_DIR}"
check_paths

echo "=== Environment ==="
echo "root=${ROOT_DIR}"
echo "local_venv=${KERNELGYM_LOCAL_VENV_DIR}"
echo "wheel_path=${WHELL_PATH}"
echo "offline_wheels=${KERNELGYM_OFFLINE_WHEEL_DIR}"
echo "offline_redis=${KERNELGYM_OFFLINE_REDIS_DIR}"
if [[ -e "${ROOT_DIR}/.venv" ]]; then
    echo "deprecated_shared_venv=${ROOT_DIR}/.venv (ignored)" >&2
fi

echo
echo "=== CUDA toolchain ==="
echo "nvcc=${NVCC}"
check_cuda129

echo
echo "=== redis-server ==="
bash "${ROOT_DIR}/scripts/ensure_redis.sh"

echo
echo "=== uv ==="
ensure_uv
echo "uv=${UV_BIN}"

echo
echo "=== Python venv ==="
ensure_python_env

echo
echo "=== Validate runtime (CUDA + torch + redis) ==="
python "${ROOT_DIR}/scripts/validate_runtime.py"
