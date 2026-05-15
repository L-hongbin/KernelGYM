#!/bin/bash

# Start KernelGym API server, worker monitor, and GPU workers.
# Assumes this script is run from any location; it will resolve the repo root.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_CONFIGURE="${ROOT_DIR}/scripts/auto_configure.sh"
HOSTNAME_VALUE="${HOSTNAME:-$(hostname 2>/dev/null || echo local)}"
DEFAULT_ENV_FILE="${ROOT_DIR}/.env.${HOSTNAME_VALUE}"
LEGACY_ENV_FILE="${ROOT_DIR}/.env"
if [ -n "${ENV_FILE:-}" ]; then
    ENV_FILE="${ENV_FILE}"
elif [ -f "${DEFAULT_ENV_FILE}" ]; then
    ENV_FILE="${DEFAULT_ENV_FILE}"
elif [ -f "${LEGACY_ENV_FILE}" ]; then
    ENV_FILE="${LEGACY_ENV_FILE}"
else
    ENV_FILE="${DEFAULT_ENV_FILE}"
fi
USING_LEGACY_ENV=0
if [ "${ENV_FILE}" = "${LEGACY_ENV_FILE}" ]; then
    USING_LEGACY_ENV=1
fi

LOG_DIR=""
LOG_DIR_OVERRIDE=""
EVAL_RESULTS_PATH_OVERRIDE=""
PY_LOG_DIR=""
PY_LOG_DIR_OVERRIDE=""
AUTO_CONFIGURE_ARGS=()

cd "${ROOT_DIR}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)
            LOG_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --log-dir=*)
            LOG_DIR_OVERRIDE="${1#*=}"
            shift 1
            ;;
        --eval-results-path)
            EVAL_RESULTS_PATH_OVERRIDE="$2"
            shift 2
            ;;
        --eval-results-path=*)
            EVAL_RESULTS_PATH_OVERRIDE="${1#*=}"
            shift 1
            ;;
        --py-log-dir)
            PY_LOG_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --py-log-dir=*)
            PY_LOG_DIR_OVERRIDE="${1#*=}"
            shift 1
            ;;
        --use-indexed-ports)
            AUTO_CONFIGURE_ARGS+=("--use-indexed-ports")
            shift 1
            ;;
        --force-config)
            AUTO_CONFIGURE_ARGS+=("--force")
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            shift 1
            ;;
    esac
done

if ! ENV_FILE="${ENV_FILE}" bash "${ROOT_DIR}/stop_all.sh"; then
    echo "Warning: stop_all.sh returned non-zero; continuing startup after best-effort cleanup."
fi

if [ ! -f "${ENV_FILE}" ]; then
    echo "No env file found at ${ENV_FILE}. Running auto configuration..."
    if [ ! -f "${AUTO_CONFIGURE}" ]; then
        echo "Auto configuration script not found at ${AUTO_CONFIGURE}"
        exit 1
    fi
    chmod +x "${AUTO_CONFIGURE}"
    ENV_FILE="${ENV_FILE}" "${AUTO_CONFIGURE}" "${AUTO_CONFIGURE_ARGS[@]}"
elif [ ${#AUTO_CONFIGURE_ARGS[@]} -gt 0 ]; then
    echo "Re-running auto configuration with explicit flags..."
    chmod +x "${AUTO_CONFIGURE}"
    ENV_FILE="${ENV_FILE}" "${AUTO_CONFIGURE}" "${AUTO_CONFIGURE_ARGS[@]}"
fi

set -o allexport
source "${ENV_FILE}"
set +o allexport

if [ "${USING_LEGACY_ENV}" = "1" ]; then
    echo "Warning: using legacy env file ${ENV_FILE}. Consider migrating to ${DEFAULT_ENV_FILE} for per-host config isolation."
fi

if [ -z "${LOG_DIR}" ]; then
    LOG_DIR="${LOG_DIR:-logs/${HOSTNAME_VALUE}}"
fi
if [ -n "${LOG_DIR_OVERRIDE}" ]; then
    LOG_DIR="${LOG_DIR_OVERRIDE}"
fi
if [ -n "${EVAL_RESULTS_PATH_OVERRIDE}" ]; then
    EVAL_RESULTS_PATH="${EVAL_RESULTS_PATH_OVERRIDE}"
fi
if [ -z "${PY_LOG_DIR}" ]; then
    PY_LOG_DIR="${PY_LOG_DIR:-py_logs/${HOSTNAME_VALUE}}"
fi
if [ -n "${PY_LOG_DIR_OVERRIDE}" ]; then
    PY_LOG_DIR="${PY_LOG_DIR_OVERRIDE}"
fi
KERNELGYM_TMP_ROOT="${KERNELGYM_TMP_ROOT:-/tmp}"
if [ -z "${KERNELGYM_TMPDIR:-}" ]; then
    KERNELGYM_TMPDIR="${KERNELGYM_TMP_ROOT}/kernelgym_${HOSTNAME_VALUE}"
fi

mkdir -p "${ROOT_DIR}/${LOG_DIR}"
PROCESS_OUTPUT_DIR="${ROOT_DIR}/${LOG_DIR}"
mkdir -p "${KERNELGYM_TMPDIR}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
REDIS_KEY_PREFIX="${REDIS_KEY_PREFIX:-kernelgym}"

PYTHONPATH="${ROOT_DIR}"
export PYTHONPATH
export LOG_DIR
export EVAL_RESULTS_PATH
export PY_LOG_DIR
export KERNELGYM_TMP_ROOT
export KERNELGYM_TMPDIR
export TMPDIR="${KERNELGYM_TMPDIR}"
export TMP="${KERNELGYM_TMPDIR}"
export TEMP="${KERNELGYM_TMPDIR}"

launch_background_process() {
    local log_file="$1"
    shift
    if command -v setsid >/dev/null 2>&1; then
        setsid nohup "$@" > "${log_file}" 2>&1 < /dev/null &
    else
        nohup "$@" > "${log_file}" 2>&1 < /dev/null &
    fi
    echo $!
}

port_is_open() {
    local host="$1"
    local port="$2"
    python - "$host" "$port" <<PY
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1):
        pass
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

echo "Checking Redis..."
if ! port_is_open "${REDIS_HOST}" "${REDIS_PORT}"; then
    if [ "${REDIS_HOST}" != "localhost" ] && [ "${REDIS_HOST}" != "127.0.0.1" ]; then
        echo "Redis is not reachable at ${REDIS_HOST}:${REDIS_PORT}. Please start it first."
        exit 1
    fi
    echo "Starting Redis on ${REDIS_HOST}:${REDIS_PORT}..."
    if [ -n "${REDIS_PASSWORD}" ]; then
        redis-server --port "${REDIS_PORT}" --requirepass "${REDIS_PASSWORD}" --daemonize yes
    else
        redis-server --port "${REDIS_PORT}" --daemonize yes
    fi
    sleep 2
fi
echo "Log Path: ${ROOT_DIR}/${LOG_DIR}"
echo "Python Log Path: ${ROOT_DIR}/kernelgym/${PY_LOG_DIR}"
echo "Process Output Path: ${PROCESS_OUTPUT_DIR}"
echo "KernelGym Tmp Path: ${KERNELGYM_TMPDIR}"
echo "TMPDIR=${TMPDIR}"

echo "Starting API server..."
API_PID="$(launch_background_process "${PROCESS_OUTPUT_DIR}/api_server.log" python -m kernelgym.server.api.server)"
echo "API server PID: ${API_PID}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-10907}"
if [ "${API_HOST}" = "0.0.0.0" ] || [ "${API_HOST}" = "::" ]; then
    echo "API server URLs:"
    echo "  http://127.0.0.1:${API_PORT}"
    echo "  http://localhost:${API_PORT}"
    HOSTNAME_VALUE="$(hostname)"
    if [ -n "${HOSTNAME_VALUE}" ] && [ "${HOSTNAME_VALUE}" != "localhost" ]; then
        echo "  http://${HOSTNAME_VALUE}:${API_PORT}"
    fi
else
    echo "API server URL: http://${API_HOST}:${API_PORT}"
fi

echo "Starting worker monitor..."
MONITOR_PID="$(launch_background_process "${PROCESS_OUTPUT_DIR}/worker_monitor.log" python -m kernelgym.worker.worker_monitor --persistent)"
echo "Worker monitor PID: ${MONITOR_PID}"

sleep 2

GPU_LIST="$(python - <<'PY'
import os, json
raw = os.environ.get("GPU_DEVICES", "")
if not raw:
    print("0")
    raise SystemExit(0)
try:
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        print(" ".join(str(x) for x in parsed))
    else:
        print(str(parsed))
except Exception:
    print(" ".join([x.strip() for x in raw.split(",") if x.strip()]))
PY
)"

if ! command -v redis-cli >/dev/null 2>&1; then
    echo "Warning: redis-cli not found; worker monitor persistent metadata will be skipped."
fi

REDIS_AUTH_ARGS=()
if [ -n "${REDIS_PASSWORD}" ]; then
    REDIS_AUTH_ARGS=(-a "${REDIS_PASSWORD}")
fi

if command -v redis-cli >/dev/null 2>&1; then
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
        DEL "${REDIS_KEY_PREFIX}:expected_workers" > /dev/null 2>&1 || true
fi

echo "Starting GPU workers..."
for gpu in ${GPU_LIST}; do
    WORKER_ID="worker_gpu_${gpu}"
    echo "Launching ${WORKER_ID} on cuda:${gpu}"
    WORKER_PID="$(launch_background_process "${PROCESS_OUTPUT_DIR}/worker_gpu_${gpu}.log" python -m kernelgym.worker.single_worker \
        --worker-id "${WORKER_ID}" \
        --device "cuda:${gpu}" \
        --persistent)"

    if command -v redis-cli >/dev/null 2>&1; then
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
            SADD "${REDIS_KEY_PREFIX}:expected_workers" "${WORKER_ID}" > /dev/null 2>&1 || true
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
            HSET "${REDIS_KEY_PREFIX}:expected_worker:${WORKER_ID}" \
            device "cuda:${gpu}" hostname "$(hostname)" node_id "${NODE_ID:-}" > /dev/null 2>&1 || true
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
            HSET "${REDIS_KEY_PREFIX}:worker_process:${WORKER_ID}" \
            pid "${WORKER_PID}" start_time "$(date -Iseconds)" device "cuda:${gpu}" > /dev/null 2>&1 || true
    fi

    echo "Worker PID: ${WORKER_PID}"
    sleep 0.3
done

CPU_COMPILE_WORKERS="${CPU_COMPILE_WORKERS:-0}"
WORKER_USE_BLOCKING_TASK_POLL="${WORKER_USE_BLOCKING_TASK_POLL:-true}"
export WORKER_USE_BLOCKING_TASK_POLL
WORKER_TASK_POLL_BLOCK_TIMEOUT_SEC="${WORKER_TASK_POLL_BLOCK_TIMEOUT_SEC:-1}"
export WORKER_TASK_POLL_BLOCK_TIMEOUT_SEC
GPU_WORKER_CLASS="${GPU_WORKER_CLASS:-${WORKER_SPLIT_COMPILE_MODE:-legacy}}"
export GPU_WORKER_CLASS
WORKER_SPLIT_COMPILE_MODE="${WORKER_SPLIT_COMPILE_MODE:-background_compile}"
export WORKER_SPLIT_COMPILE_MODE
WORKER_BACKGROUND_COMPILE_LIMIT="${WORKER_BACKGROUND_COMPILE_LIMIT:-2}"
export WORKER_BACKGROUND_COMPILE_LIMIT
WORKER_COMPILE_POOL_SIZE="${WORKER_COMPILE_POOL_SIZE:-2}"
export WORKER_COMPILE_POOL_SIZE
WORKER_COMPILE_MAX_TASKS_PER_WORKER="${WORKER_COMPILE_MAX_TASKS_PER_WORKER:-10}"
export WORKER_COMPILE_MAX_TASKS_PER_WORKER
GPU_WORKERS_POLL_CPU_TASKS="${GPU_WORKERS_POLL_CPU_TASKS:-true}"
export GPU_WORKERS_POLL_CPU_TASKS
WORKER_POOL_SIZE="${WORKER_POOL_SIZE:-1}"
export WORKER_POOL_SIZE
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-1}"
export MAX_TASKS_PER_WORKER
echo "CPU_COMPILE_WORKERS=${CPU_COMPILE_WORKERS}"
echo "WORKER_USE_BLOCKING_TASK_POLL=${WORKER_USE_BLOCKING_TASK_POLL}"
echo "WORKER_TASK_POLL_BLOCK_TIMEOUT_SEC=${WORKER_TASK_POLL_BLOCK_TIMEOUT_SEC}"
echo "GPU_WORKER_CLASS=${GPU_WORKER_CLASS}"
echo "WORKER_SPLIT_COMPILE_MODE=${WORKER_SPLIT_COMPILE_MODE}"
echo "WORKER_BACKGROUND_COMPILE_LIMIT=${WORKER_BACKGROUND_COMPILE_LIMIT}"
echo "WORKER_COMPILE_POOL_SIZE=${WORKER_COMPILE_POOL_SIZE}"
echo "WORKER_COMPILE_MAX_TASKS_PER_WORKER=${WORKER_COMPILE_MAX_TASKS_PER_WORKER}"
echo "GPU_WORKERS_POLL_CPU_TASKS=${GPU_WORKERS_POLL_CPU_TASKS}"
echo "WORKER_POOL_SIZE=${WORKER_POOL_SIZE}"
echo "MAX_TASKS_PER_WORKER=${MAX_TASKS_PER_WORKER}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-auto}"
if [ "${CPU_COMPILE_WORKERS}" -gt 0 ] 2>/dev/null; then
    echo "Starting CPU compile workers..."
    for ((cpu_worker=0; cpu_worker<CPU_COMPILE_WORKERS; cpu_worker++)); do
        WORKER_ID="worker_cpu_compile_${cpu_worker}"
        DEVICE_ID="cpu:${cpu_worker}"
        echo "Launching ${WORKER_ID} on ${DEVICE_ID}"
        WORKER_PID="$(launch_background_process "${PROCESS_OUTPUT_DIR}/${WORKER_ID}.log" python -m kernelgym.worker.single_worker \
            --worker-id "${WORKER_ID}" \
            --device "${DEVICE_ID}" \
            --persistent)"

        if command -v redis-cli >/dev/null 2>&1; then
            redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
                SADD "${REDIS_KEY_PREFIX}:expected_workers" "${WORKER_ID}" > /dev/null 2>&1 || true
            redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
                HSET "${REDIS_KEY_PREFIX}:expected_worker:${WORKER_ID}" \
                device "${DEVICE_ID}" hostname "$(hostname)" node_id "${NODE_ID:-}" worker_type "cpu" > /dev/null 2>&1 || true
            redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
                HSET "${REDIS_KEY_PREFIX}:worker_process:${WORKER_ID}" \
                pid "${WORKER_PID}" start_time "$(date -Iseconds)" device "${DEVICE_ID}" > /dev/null 2>&1 || true
        fi

        echo "Worker PID: ${WORKER_PID}"
        sleep 0.1
    done
fi

echo "KernelGym started."
echo "Logs: ${ROOT_DIR}/${LOG_DIR}"
