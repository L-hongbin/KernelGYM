#!/usr/bin/env bash
set -euo pipefail

GPU_CLOCK="${KERNELGYM_GPU_CLOCK:-2700}"
POWER_LIMIT="${KERNELGYM_POWER_LIMIT:-400}"
NVIDIA_SMI="${KERNELGYM_NVIDIA_SMI:-nvidia-smi}"
SUDO=()
DRY_RUN=0
PERSISTENCE=1

usage() {
    echo "Usage: scripts/lock_gpu_clocks.sh [--sudo] [--dry-run] [--no-persistence] [--gpu-clock N] [--power-limit N]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sudo)
            SUDO=(sudo)
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-persistence)
            PERSISTENCE=0
            shift
            ;;
        --gpu-clock)
            GPU_CLOCK="$2"
            shift 2
            ;;
        --power-limit)
            POWER_LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" == "0" ]]; then
        "$@"
    fi
}

if [[ "${PERSISTENCE}" == "1" ]]; then
    run "${SUDO[@]}" "${NVIDIA_SMI}" -pm 1
fi
run "${SUDO[@]}" "${NVIDIA_SMI}" -lgc "${GPU_CLOCK},${GPU_CLOCK}"
run "${SUDO[@]}" "${NVIDIA_SMI}" -pl "${POWER_LIMIT}"
