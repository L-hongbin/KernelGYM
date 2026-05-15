#!/bin/bash
# Stop KernelGym API server, workers, and monitor, then optionally clear Redis keys.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTNAME_VALUE="${HOSTNAME:-$(hostname 2>/dev/null || echo local)}"
DEFAULT_ENV_FILE="${ROOT_DIR}/.env.${HOSTNAME_VALUE}"
LEGACY_ENV_FILE="${ROOT_DIR}/.env"
if [ -n "${ENV_FILE:-}" ]; then
    ENV_FILE="${ENV_FILE}"
elif [ -f "${DEFAULT_ENV_FILE}" ]; then
    ENV_FILE="${DEFAULT_ENV_FILE}"
else
    ENV_FILE="${LEGACY_ENV_FILE}"
fi

REDIS_HOST="localhost"
REDIS_PORT=""
REDIS_PASSWORD=""
REDIS_KEY_PREFIX="kernelgym"
API_PORT=""
KERNELGYM_TMP_ROOT="${KERNELGYM_TMP_ROOT:-}"
KERNELGYM_TMPDIR="${KERNELGYM_TMPDIR:-}"

read_env_value() {
    local name="$1"
    if [ ! -f "${ENV_FILE}" ]; then
        return 0
    fi
    grep "^${name}=" "${ENV_FILE}" 2>/dev/null \
        | head -n 1 \
        | cut -d'=' -f2- \
        | tr -d '"' \
        | tr -d ' ' \
        || true
}

if [ -f "${ENV_FILE}" ]; then
    API_PORT="$(read_env_value API_PORT)"
    REDIS_HOST="$(read_env_value REDIS_HOST)"
    REDIS_PORT="$(read_env_value REDIS_PORT)"
    REDIS_PASSWORD="$(read_env_value REDIS_PASSWORD)"
    REDIS_KEY_PREFIX="$(read_env_value REDIS_KEY_PREFIX)"
    KERNELGYM_TMP_ROOT="$(read_env_value KERNELGYM_TMP_ROOT)"
    KERNELGYM_TMPDIR="$(read_env_value KERNELGYM_TMPDIR)"
fi

if [ -z "${REDIS_HOST}" ]; then
    REDIS_HOST="localhost"
fi
if [ -z "${REDIS_KEY_PREFIX}" ]; then
    REDIS_KEY_PREFIX="kernelgym"
fi
TMP_ROOT="${KERNELGYM_TMPDIR:-${TMPDIR:-/tmp}}"
if [ -z "${KERNELGYM_TMPDIR}" ] && [ -n "${KERNELGYM_TMP_ROOT}" ]; then
    TMP_ROOT="${KERNELGYM_TMP_ROOT}/kernelgym_${HOSTNAME_VALUE}"
fi
kill_processes() {
    local pattern="$1"
    local description="$2"

    echo "Stopping ${description}..."
    local pids=""
    if command -v pgrep >/dev/null 2>&1; then
        pids="$(pgrep -f "${pattern}" || true)"
    else
        pids="$(ps aux | grep "${pattern}" | grep -v grep | awk '{print $2}' || true)"
    fi

    if [ -z "${pids}" ]; then
        echo "No ${description} processes found."
        return
    fi

    echo "${pids}" | xargs -r kill -TERM || true
    sleep 2

    local remaining=""
    if command -v pgrep >/dev/null 2>&1; then
        remaining="$(pgrep -f "${pattern}" || true)"
    else
        remaining="$(ps aux | grep "${pattern}" | grep -v grep | awk '{print $2}' || true)"
    fi

    if [ -n "${remaining}" ]; then
        echo "Force killing ${description}..."
        echo "${remaining}" | xargs -r kill -KILL || true
    fi
}

maybe_clear_kernel_caches() {
    prompt_clear_cache_root() {
        local root="$1"
        local title="$2"
        local empty_message="$3"
        local clear_message="$4"
        local skip_message="$5"
        local cache_paths=()
        local existing_paths=()
        local kernelgym_cuda_build_group=()
        local cuda_agent_group=()
        local path=""
        local answer=""
        local total_bytes=""

        shopt -s nullglob
        cache_paths+=("${root}/kernelgym_compile_cache")
        cache_paths+=("${root}/kernelgym_cuda_cache")
        cache_paths+=("${root}/kernelgym_cuda_agent_cache")
        cache_paths+=("${root}/kernelgym_manual_ninja_object_cache")
        for path in "${root}"/kernelgym_cuda_build_*; do
            cache_paths+=("${path}")
            kernelgym_cuda_build_group+=("${path}")
        done
        for path in "${root}"/cuda_agent_*; do
            cache_paths+=("${path}")
            cuda_agent_group+=("${path}")
        done
        shopt -u nullglob

        for path in "${cache_paths[@]}"; do
            if [ -e "${path}" ]; then
                existing_paths+=("${path}")
            fi
        done

        if [ "${#existing_paths[@]}" -eq 0 ]; then
            echo "${empty_message}"
            return
        fi

        total_bytes="$(du -sb "${existing_paths[@]}" 2>/dev/null | tail -n 1 | awk '{print $1}' || true)"
        if [ -z "${total_bytes}" ] || [ "${total_bytes}" -le 0 ] 2>/dev/null; then
            echo "${empty_message}"
            return
        fi

        echo "${title}"
        for path in \
            "${root}/kernelgym_compile_cache" \
            "${root}/kernelgym_cuda_cache" \
            "${root}/kernelgym_cuda_agent_cache" \
            "${root}/kernelgym_manual_ninja_object_cache"; do
            if [ -e "${path}" ]; then
                du -sh "${path}" 2>/dev/null || true
            fi
        done
        if [ "${#kernelgym_cuda_build_group[@]}" -gt 0 ]; then
            local kernelgym_cuda_build_size
            kernelgym_cuda_build_size="$(du -sch "${kernelgym_cuda_build_group[@]}" 2>/dev/null | tail -n 1 | awk '{print $1}')"
            if [ -n "${kernelgym_cuda_build_size}" ]; then
                echo "${kernelgym_cuda_build_size}	${root}/kernelgym_cuda_build_*"
            fi
        fi
        if [ "${#cuda_agent_group[@]}" -gt 0 ]; then
            local cuda_agent_size
            cuda_agent_size="$(du -sch "${cuda_agent_group[@]}" 2>/dev/null | tail -n 1 | awk '{print $1}')"
            if [ -n "${cuda_agent_size}" ]; then
                echo "${cuda_agent_size}	${root}/cuda_agent_*"
            fi
        fi
        echo "Total:"
        du -sch "${existing_paths[@]}" 2>/dev/null | tail -n 1 || true
        echo "${clear_message} [y/N] (default: N in 5s)"
        if ! read -r -t 5 answer; then
            answer=""
        fi

        case "${answer}" in
            y|Y)
                echo "${clear_message}..."
                rm -rf "${existing_paths[@]}"
                ;;
            *)
                echo "${skip_message}"
                ;;
        esac
    }

    prompt_clear_cache_root \
        "${TMP_ROOT}" \
        "Local kernel compile/cache directories:" \
        "No local kernel compile/cache directories found." \
        "Clear these local kernel compile/cache directories?" \
        "Skipping local kernel compile/cache cleanup."

    if [ "${TMP_ROOT}" != "/tmp" ]; then
        prompt_clear_cache_root \
            "/tmp" \
            "Legacy /tmp kernel compile/cache directories:" \
            "No legacy /tmp kernel compile/cache directories found." \
            "Clear these legacy /tmp kernel compile/cache directories?" \
            "Skipping legacy /tmp kernel compile/cache cleanup."
    fi
}

kill_compile_processes() {
    echo "Stopping KernelGym compile subprocesses..."
    kill_processes "ninja.*(${TMP_ROOT}/kernelgym_|${TMP_ROOT}/cuda_agent_)" "KernelGym ninja compile processes"
    kill_processes "nvcc.*(${TMP_ROOT}/kernelgym_|${TMP_ROOT}/cuda_agent_)" "KernelGym nvcc compile processes"
    kill_processes "ptxas.*(${TMP_ROOT}/kernelgym_|${TMP_ROOT}/cuda_agent_|tmpxft_)" "KernelGym ptxas compile processes"
    kill_processes "cicc.*(${TMP_ROOT}/kernelgym_|${TMP_ROOT}/cuda_agent_|tmpxft_)" "KernelGym cicc compile processes"
    kill_processes "fatbinary.*(${TMP_ROOT}/kernelgym_|${TMP_ROOT}/cuda_agent_|tmpxft_)" "KernelGym fatbinary compile processes"
    kill_processes "cc1plus.*(${TMP_ROOT}/kernelgym_|${TMP_ROOT}/cuda_agent_)" "KernelGym cc1plus compile processes"
    kill_processes "c\\+\\+.*(${TMP_ROOT}/kernelgym_|${TMP_ROOT}/cuda_agent_)" "KernelGym c++ compile processes"
    kill_processes "gcc.*(${TMP_ROOT}/kernelgym_|${TMP_ROOT}/cuda_agent_)" "KernelGym gcc compile processes"
}

stop_local_redis() {
    if [ -z "${REDIS_PORT}" ]; then
        echo "REDIS_PORT not set; skipping Redis shutdown."
        return
    fi
    if [ "${REDIS_HOST}" != "localhost" ] && [ "${REDIS_HOST}" != "127.0.0.1" ]; then
        echo "Redis host is ${REDIS_HOST}; skipping Redis shutdown for non-local Redis."
        return
    fi
    if ! command -v redis-cli >/dev/null 2>&1; then
        echo "redis-cli not found; using process fallback for Redis shutdown."
    else
        REDIS_AUTH_ARGS=()
        if [ -n "${REDIS_PASSWORD}" ]; then
            REDIS_AUTH_ARGS=(-a "${REDIS_PASSWORD}" --no-auth-warning)
        fi
        echo "Stopping local Redis on ${REDIS_HOST}:${REDIS_PORT}..."
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" \
            SHUTDOWN NOSAVE >/dev/null 2>&1 || true
        sleep 1
    fi

    local redis_pids=""
    if command -v pgrep >/dev/null 2>&1; then
        redis_pids="$(pgrep -f "redis-server.*(:|\\s)${REDIS_PORT}(\\s|$)" || true)"
    else
        redis_pids="$(ps aux | grep -E "redis-server.*(:|[[:space:]])${REDIS_PORT}([[:space:]]|$)" | grep -v grep | awk '{print $2}' || true)"
    fi
    if [ -n "${redis_pids}" ]; then
        echo "Force killing local Redis on port ${REDIS_PORT}..."
        echo "${redis_pids}" | xargs -r kill -KILL || true
    fi
}

echo "Stopping KernelGym processes..."

kill_processes "kernelgym.server.api.server" "KernelGym API server"
kill_processes "kernelgym.worker.worker_monitor" "KernelGym worker monitor"
kill_processes "kernelgym.worker.single_worker" "KernelGym GPU workers"
kill_processes "kernelgym.worker.gpu_worker" "KernelGym GPU worker core"
kill_processes "uvicorn.*kernelgym" "Uvicorn server"

echo "Stopping multiprocessing worker processes..."
kill_processes "multiprocessing.spawn" "multiprocessing spawn workers"
kill_processes "multiprocessing.resource_tracker" "multiprocessing resource tracker"
kill_compile_processes || true

if command -v redis-cli >/dev/null 2>&1; then
    if [ -n "${REDIS_PORT}" ]; then
        REDIS_AUTH_ARGS=()
        if [ -n "${REDIS_PASSWORD}" ]; then
            REDIS_AUTH_ARGS=(-a "${REDIS_PASSWORD}" --no-auth-warning)
        fi
        echo "Clearing Redis keys with prefix '${REDIS_KEY_PREFIX}:' on ${REDIS_HOST}:${REDIS_PORT}..."
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" \
            --scan --pattern "${REDIS_KEY_PREFIX}:*" \
            | xargs -r redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" DEL >/dev/null 2>&1 || true
    else
        echo "REDIS_PORT not set; skipping Redis cleanup."
    fi
else
    echo "redis-cli not found; skipping Redis cleanup."
fi

stop_local_redis || true

maybe_clear_kernel_caches || true

echo "KernelGym stopped."
exit 0
