#!/usr/bin/env bash

# Runtime Python packages live on each node's local system disk. The wheelhouse
# and repository remain shared so every node installs the same artifacts and
# continues to write logs to the shared checkout.
export KERNELGYM_LOCAL_VENV_DIR="${KERNELGYM_LOCAL_VENV_DIR:-/root/kernelgym-reward-only/.venv}"
export KERNELGYM_OFFLINE_WHEEL_DIR="${KERNELGYM_OFFLINE_WHEEL_DIR:-/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only/wheels}"
export KERNELGYM_OFFLINE_REDIS_DIR="${KERNELGYM_OFFLINE_REDIS_DIR:-/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only/wheels/redis/ubuntu-24.04-amd64}"

case "${KERNELGYM_LOCAL_VENV_DIR}" in
    /*) ;;
    *)
        echo "KERNELGYM_LOCAL_VENV_DIR must be an absolute path: ${KERNELGYM_LOCAL_VENV_DIR}" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

case "${KERNELGYM_OFFLINE_WHEEL_DIR}" in
    /*) ;;
    *)
        echo "KERNELGYM_OFFLINE_WHEEL_DIR must be an absolute path: ${KERNELGYM_OFFLINE_WHEEL_DIR}" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

case "${KERNELGYM_OFFLINE_REDIS_DIR}" in
    /*) ;;
    *)
        echo "KERNELGYM_OFFLINE_REDIS_DIR must be an absolute path: ${KERNELGYM_OFFLINE_REDIS_DIR}" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

kernelgym_require_local_venv_path() {
    case "${KERNELGYM_LOCAL_VENV_DIR}" in
        /|/root|/nfs|/nfs/*|/ms|/ms/*)
            echo "refusing non-local or unsafe venv path: ${KERNELGYM_LOCAL_VENV_DIR}" >&2
            return 1
            ;;
    esac

    local probe_path="${KERNELGYM_LOCAL_VENV_DIR}"
    while [[ ! -e "${probe_path}" && "${probe_path}" != "/" ]]; do
        probe_path="$(dirname "${probe_path}")"
    done
    if ! command -v findmnt >/dev/null 2>&1; then
        echo "findmnt is required to prove the venv path is node-local" >&2
        return 1
    fi
    local fs_type
    if ! fs_type="$(findmnt -n -o FSTYPE -T "${probe_path}" 2>/dev/null)" || [[ -z "${fs_type}" ]]; then
        echo "could not determine filesystem type for local venv path: ${KERNELGYM_LOCAL_VENV_DIR}" >&2
        return 1
    fi
    case "${fs_type}" in
        nfs|nfs4|cifs|smb3|lustre|ceph|gpfs|9p|virtiofs|fuse.*)
            echo "venv path resolves to shared filesystem ${fs_type}: ${KERNELGYM_LOCAL_VENV_DIR}" >&2
            return 1
            ;;
    esac
}
