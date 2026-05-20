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
    local venv_created=0
    if [[ "${RECREATE}" == "1" && -e .venv ]]; then
        rm -rf .venv
    fi
    if [[ ! -e .venv ]]; then
        # Pass the path explicitly so uv writes to ./.venv even on hosts that
        # tried to coerce a different location through env or config.
        uv venv .venv -p 3.12
        venv_created=1
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/scripts/scrub_venv_env.sh"

    # Split the install into two independent steps so a broken third-party
    # import doesn't force a re-install of the editable package, and a missing
    # editable install doesn't force a reinstall of the heavy CUDA wheels.
    local needs_editable=${venv_created}
    local needs_thirdparty=${venv_created}
    if [[ "${needs_editable}" == "0" ]]; then
        if ! python -c "import kernelgym" >/dev/null 2>&1; then
            needs_editable=1
        fi
    fi
    if [[ "${needs_thirdparty}" == "0" ]]; then
        if ! python - <<'PY' >/dev/null 2>&1
import redis
import torch
import tvm_ffi
PY
        then
            needs_thirdparty=1
        fi
    fi

    if [[ "${needs_editable}" == "1" ]]; then
        echo
        echo "=== Install editable kernelgym + dev/tvm-ffi extras ==="
        network uv pip install -e ".[dev,tvm-ffi]"
    else
        echo "kernelgym already installed (editable), skipping reinstall"
    fi

    if [[ "${needs_thirdparty}" == "1" ]]; then
        echo
        echo "=== Install CUDA 12.9 runtime deps (torch, torchvision, apache-tvm-ffi) ==="
        # When ./wheels/ is staged, reconcile against it first using
        # --no-deps. uv with explicit wheel paths is idempotent: packages
        # already at the wheel's version are no-ops, only mismatched ones
        # (e.g. an old nvidia-nvjitlink-cu12 dragged in earlier from the
        # intranet) get replaced. This avoids redundantly reinstalling a
        # correct torch wheel just to fix a broken transitive dep.
        local find_links_arg=()
        if compgen -G "${ROOT_DIR}/wheels/*.whl" >/dev/null; then
            echo "Reconciling local wheels under ${ROOT_DIR}/wheels"
            network uv pip install --no-deps "${ROOT_DIR}"/wheels/*.whl
            find_links_arg=(--find-links "${ROOT_DIR}/wheels")
        fi
        # Run the requirements file too so apache-tvm-ffi (not in ./wheels/)
        # is fetched from the configured index. Anything that was reconciled
        # from local wheels above already satisfies the pins, so uv only
        # touches what's missing.
        network uv pip install "${find_links_arg[@]}" -r requirements-cuda129.txt
    else
        echo "torch/redis/tvm_ffi already importable, skipping requirements-cuda129.txt install"
    fi
}

cd "${ROOT_DIR}"

echo "=== Environment ==="
echo "profile=${PROFILE}"
echo "root=${ROOT_DIR}"

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

echo
echo "=== Python venv ==="
ensure_python_env

echo
echo "=== Validate runtime (CUDA + torch + redis) ==="
python scripts/validate_runtime.py
