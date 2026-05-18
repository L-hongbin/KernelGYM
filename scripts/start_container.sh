#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

NAME="${KERNELGYM_CONTAINER_NAME:-kernelgym-reward}"
IMAGE="${KERNELGYM_CONTAINER_IMAGE:-192.168.14.129:80/library/slime:nightly-dev-20260430b}"
CUDA_PATH="${KERNELGYM_CUDA_PATH:-/usr/local/cuda-12.9}"
SHM_SIZE="${KERNELGYM_SHM_SIZE:-256g}"
REPLACE="${KERNELGYM_CONTAINER_REPLACE:-1}"

usage() {
    echo "Usage: scripts/start_container.sh"
    echo "Env: KERNELGYM_CONTAINER_NAME KERNELGYM_CONTAINER_IMAGE KERNELGYM_CUDA_PATH KERNELGYM_SHM_SIZE"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 ]]; then
    usage >&2
    exit 2
fi

if [[ ! -d "${CUDA_PATH}" ]]; then
    echo "CUDA path not found: ${CUDA_PATH}" >&2
    exit 1
fi

DOCKER=(docker)
if ! docker info >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
    DOCKER=(sudo docker)
fi

cd "${ROOT_DIR}"

if [[ "${REPLACE}" == "1" ]]; then
    "${DOCKER[@]}" rm -f "${NAME}" >/dev/null 2>&1 || true
fi

"${DOCKER[@]}" run -d \
    --name "${NAME}" \
    --gpus all \
    --network host \
    --privileged \
    --tmpfs "/dev/shm:rw,nosuid,nodev,exec,size=${SHM_SIZE}" \
    -v /nfs:/nfs \
    -v "${CUDA_PATH}:${CUDA_PATH}:ro" \
    -w "${ROOT_DIR}" \
    "${IMAGE}" \
    sleep infinity

echo "container=${NAME}"
echo "image=${IMAGE}"
echo "enter: ${DOCKER[*]} exec -it ${NAME} bash"
