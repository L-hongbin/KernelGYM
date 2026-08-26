#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/runtime_paths.sh"

PYTHON_TARGET="${KERNELGYM_PYTHON_TARGET:-/usr/bin/python3.12}"

if [[ ! -x "${PYTHON_TARGET}" ]]; then
    echo "python target not executable: ${PYTHON_TARGET}" >&2
    exit 1
fi

kernelgym_require_local_venv_path

if [[ ! -d "${KERNELGYM_OFFLINE_WHEEL_DIR}" ]]; then
    echo "offline wheel directory not found: ${KERNELGYM_OFFLINE_WHEEL_DIR}" >&2
    exit 1
fi

if ! compgen -G "${KERNELGYM_OFFLINE_WHEEL_DIR}/*.whl" >/dev/null; then
    echo "offline wheel directory is empty: ${KERNELGYM_OFFLINE_WHEEL_DIR}" >&2
    exit 1
fi

if [[ ! -d "${KERNELGYM_OFFLINE_REDIS_DIR}" ]]; then
    echo "offline Redis directory not found: ${KERNELGYM_OFFLINE_REDIS_DIR}" >&2
    exit 1
fi

mkdir -p "$(dirname "${KERNELGYM_LOCAL_VENV_DIR}")"

if [[ -e "${ROOT_DIR}/.venv" ]]; then
    echo "deprecated_shared_venv=${ROOT_DIR}/.venv (ignored)" >&2
fi

echo "python_target=${PYTHON_TARGET}"
echo "local_venv=${KERNELGYM_LOCAL_VENV_DIR}"
echo "offline_wheels=${KERNELGYM_OFFLINE_WHEEL_DIR}"
echo "offline_redis=${KERNELGYM_OFFLINE_REDIS_DIR}"

if [[ ! -x "${KERNELGYM_LOCAL_VENV_DIR}/bin/python" ]]; then
    echo "local venv is not bootstrapped; run: bash ${ROOT_DIR}/ensure_venv.sh" >&2
fi
