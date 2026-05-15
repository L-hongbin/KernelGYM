#!/bin/bash

# KernelGym auto-configuration script.
# Generates a .env file with detected IP and available ports (ARNOLD-aware).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTNAME_VALUE="${HOSTNAME:-$(hostname 2>/dev/null || echo local)}"
DEFAULT_ENV_FILE="${ROOT_DIR}/.env.${HOSTNAME_VALUE}"
ENV_FILE="${ENV_FILE:-${DEFAULT_ENV_FILE}}"

FORCE=false
USE_INDEXED_PORTS=false
ENABLE_SAVE_RESULTS=false
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --force)
            FORCE=true
            shift 1
            ;;
        --use-indexed-ports)
            USE_INDEXED_PORTS=true
            shift 1
            ;;
        --save-eval-results)
            ENABLE_SAVE_RESULTS=true
            shift 1
            ;;
        *)
            shift 1
            ;;
    esac
done

if [ -f "${ENV_FILE}" ] && [ "${FORCE}" = false ]; then
    echo "Found existing .env at ${ENV_FILE}. Use --force to overwrite."
    exit 0
fi

get_available_ports() {
    local role="${ARNOLD_ROLE:-}"
    local worker_id="${ARNOLD_ID:-}"

    local varname="ARNOLD_${role^^}_${worker_id}_PORT"
    local allports="${!varname:-}"

    local need_probe=false
    if [ "${USE_INDEXED_PORTS}" = true ]; then
        need_probe=true
        allports=""
    elif [ -z "${allports}" ] || [[ "${allports}" != *,* ]]; then
        need_probe=true
    fi
    if [ "${need_probe}" = true ]; then
        local ports_list=()
        local idx=0
        while true; do
            local indexed_varname="PORT${idx}"
            local pv="${!indexed_varname:-}"
            if [ -z "${pv}" ]; then
                break
            fi
            ports_list+=("${pv}")
            idx=$((idx+1))
        done
        if [ ${#ports_list[@]} -gt 0 ]; then
            local joined=""
            for p in "${ports_list[@]}"; do
                if [ -n "${joined}" ]; then
                    joined="${joined},${p}"
                else
                    joined="${p}"
                fi
            done
            allports="${joined}"
            echo "Using indexed environment ports: PORT0..$((idx-1))" >&2
        fi
    fi

    if [ -z "${allports}" ]; then
        echo "No ARNOLD ports found, using fallback ports" >&2
        allports="8000,8001,8002,8003,8004,8005,8006,8007,8008,8009"
    fi

    echo "${allports}"
}

get_ip_address() {
    local role="${ARNOLD_ROLE:-}"
    local worker_id="${ARNOLD_ID:-}"
    local ipvarname="ARNOLD_${role^^}_${worker_id}_HOST"
    local ipaddress="${!ipvarname:-}"

    if [ -z "${ipaddress}" ]; then
        ipaddress="$(hostname -I | awk '{print $1}')"
        if [ -z "${ipaddress}" ]; then
            ipaddress="127.0.0.1"
        fi
    fi

    echo "${ipaddress}"
}

is_port_available() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ! ss -tlnp | grep -Fq ":${port} "
    else
        ! lsof -iTCP -sTCP:LISTEN -P 2>/dev/null | grep -Fq ":${port}"
    fi
}

select_ports() {
    local available_ports="$1"
    local needed_ports=("redis" "api" "metrics")

    IFS=',' read -ra PORT_ARRAY <<< "${available_ports}"

    local selected_ports=()
    local port_index=0

    for service in "${needed_ports[@]}"; do
        local found_port=""

        for ((i=port_index; i<${#PORT_ARRAY[@]}; i++)); do
            local port="${PORT_ARRAY[i]}"
            port="${port//[[:space:]]/}"
            if is_port_available "${port}"; then
                found_port="${port}"
                port_index=$((i+1))
                break
            fi
        done

        if [ -z "${found_port}" ]; then
            echo "Could not find available port for ${service}"
            exit 1
        fi

        selected_ports+=("${found_port}")
    done

    echo "${selected_ports[@]}"
}

detect_gpus_json() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [ "${count}" -gt 0 ]; then
            local devices="["
            for ((i=0; i<"${count}"; i++)); do
                if [ "${i}" -gt 0 ]; then
                    devices="${devices},"
                fi
                devices="${devices}${i}"
            done
            devices="${devices}]"
            echo "${devices}"
            return
        fi
    fi
    echo "[0]"
}

detect_torch_cuda_arch_list() {
    if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        echo "${TORCH_CUDA_ARCH_LIST}"
        return
    fi

    detect_local_torch_cuda_arch_list
}

detect_local_torch_cuda_arch_list() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi

    local caps=""
    caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
    if [ -z "${caps}" ]; then
        caps="$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader 2>/dev/null || true)"
    fi
    if [ -z "${caps}" ]; then
        echo ""
        return
    fi

    local arch_list=""
    local seen_arches=";"
    local cap
    while IFS= read -r cap; do
        cap="${cap//[[:space:]]/}"
        cap="${cap//,/}"
        if [[ ! "${cap}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            continue
        fi
        if [[ "${seen_arches}" == *";${cap};"* ]]; then
            continue
        fi
        seen_arches="${seen_arches}${cap};"
        if [ -n "${arch_list}" ]; then
            arch_list="${arch_list};${cap}"
        else
            arch_list="${cap}"
        fi
    done <<< "${caps}"

    echo "${arch_list}"
}

torch_cuda_arch_list_has_multiple_values() {
    local arch_list="$1"
    arch_list="${arch_list//\"/}"
    arch_list="${arch_list//\'/}"
    arch_list="${arch_list//;/ }"
    # shellcheck disable=SC2206
    local arches=( ${arch_list} )
    [ "${#arches[@]}" -gt 1 ]
}

detect_cpu_threads() {
    if command -v nproc >/dev/null 2>&1; then
        nproc --all
        return
    fi
    getconf _NPROCESSORS_ONLN 2>/dev/null || echo "1"
}

detect_cpu_model() {
    if command -v lscpu >/dev/null 2>&1; then
        lscpu | awk -F: '/Model name:/ {sub(/^[ \t]+/, "", $2); print $2; exit}'
        return
    fi
    awk -F: '/model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || echo "unknown"
}

detect_lscpu_field() {
    local field="$1"
    if command -v lscpu >/dev/null 2>&1; then
        lscpu | awk -F: -v key="${field}" '$1 == key {sub(/^[ \t]+/, "", $2); print $2; exit}'
    fi
}

detect_cpu_physical_cores() {
    local sockets
    local cores_per_socket
    sockets="$(detect_lscpu_field "Socket(s)")"
    cores_per_socket="$(detect_lscpu_field "Core(s) per socket")"
    if [[ "${sockets}" =~ ^[0-9]+$ && "${cores_per_socket}" =~ ^[0-9]+$ ]]; then
        echo $((sockets * cores_per_socket))
        return
    fi
    detect_cpu_threads
}

detect_memory_total_mb() {
    awk '/MemTotal:/ {print int($2 / 1024); exit}' /proc/meminfo 2>/dev/null || echo "0"
}

detect_path_size_human() {
    local path="$1"
    if [ ! -e "${path}" ]; then
        echo "unavailable"
        return
    fi
    if command -v df >/dev/null 2>&1; then
        df -h "${path}" 2>/dev/null | awk 'NR == 2 {print $2 " total, " $4 " available"}'
        return
    fi
    echo "unknown"
}

env_quote() {
    local value="${1:-}"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '"%s"' "${value}"
}

validate_non_negative_integer() {
    local value="${1:-}"
    [[ "${value}" =~ ^[0-9]+$ ]]
}

resolve_cpu_compile_workers() {
    if [ -n "${CPU_COMPILE_WORKERS:-}" ]; then
        if ! validate_non_negative_integer "${CPU_COMPILE_WORKERS}"; then
            echo "Invalid CPU_COMPILE_WORKERS value: ${CPU_COMPILE_WORKERS}. Expected a non-negative integer." >&2
            exit 1
        fi
        echo "${CPU_COMPILE_WORKERS}"
        return
    fi

    local prompt="CPU_COMPILE_WORKERS not set. Enter CPU compile worker count [default: 0]: "
    local input=""
    if [ -r /dev/tty ]; then
        printf "%s" "${prompt}" > /dev/tty
        if read -r -t 5 input < /dev/tty; then
            input="${input//[[:space:]]/}"
            if [ -z "${input}" ]; then
                echo "0"
                return
            fi
            if ! validate_non_negative_integer "${input}"; then
                echo "Invalid CPU compile worker count: ${input}. Expected a non-negative integer." >&2
                exit 1
            fi
            echo "${input}"
            return
        fi
        printf "\n" > /dev/tty
    else
        echo "CPU_COMPILE_WORKERS not set and no interactive tty available. Defaulting to 0." >&2
    fi

    echo "0"
}

API_HOST="${API_HOST:-$(get_ip_address)}"

AVAILABLE_PORTS="$(get_available_ports)"
SELECTED_PORTS="$(select_ports "${AVAILABLE_PORTS}")"

REDIS_PORT="$(echo "${SELECTED_PORTS}" | awk '{print $1}')"
API_PORT="$(echo "${SELECTED_PORTS}" | awk '{print $2}')"
METRICS_PORT="$(echo "${SELECTED_PORTS}" | awk '{print $3}')"
GPU_DEVICES="${GPU_DEVICES:-$(detect_gpus_json)}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(detect_torch_cuda_arch_list)}"
RECOMMENDED_TORCH_CUDA_ARCH_LIST=""
if [ -n "${TORCH_CUDA_ARCH_LIST}" ] && torch_cuda_arch_list_has_multiple_values "${TORCH_CUDA_ARCH_LIST}"; then
    RECOMMENDED_TORCH_CUDA_ARCH_LIST="$(detect_local_torch_cuda_arch_list)"
fi
if [ -n "${TORCH_CUDA_ARCH_LIST}" ]; then
    TORCH_CUDA_ARCH_LIST_CONFIG="TORCH_CUDA_ARCH_LIST=$(env_quote "${TORCH_CUDA_ARCH_LIST}")"
else
    TORCH_CUDA_ARCH_LIST_CONFIG="# TORCH_CUDA_ARCH_LIST=  # Set manually for compile-only nodes without visible target GPUs."
fi

CPU_ARCH="${CPU_ARCH:-$(uname -m 2>/dev/null || echo unknown)}"
CPU_MODEL="${CPU_MODEL:-$(detect_cpu_model)}"
CPU_THREADS="${CPU_THREADS:-$(detect_cpu_threads)}"
CPU_PHYSICAL_CORES="${CPU_PHYSICAL_CORES:-$(detect_cpu_physical_cores)}"
CPU_SOCKETS="${CPU_SOCKETS:-$(detect_lscpu_field "Socket(s)")}"
CPU_SOCKETS="${CPU_SOCKETS:-1}"
CPU_NUMA_NODES="${CPU_NUMA_NODES:-$(detect_lscpu_field "NUMA node(s)")}"
CPU_NUMA_NODES="${CPU_NUMA_NODES:-1}"
MEMORY_TOTAL_MB="${MEMORY_TOTAL_MB:-$(detect_memory_total_mb)}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_DB="${REDIS_DB:-0}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
REDIS_KEY_PREFIX="${REDIS_KEY_PREFIX:-kernelgym}"

API_WORKERS="${API_WORKERS:-4}"
API_RELOAD="${API_RELOAD:-false}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_DIR="${LOG_DIR:-logs/${HOSTNAME_VALUE}}"
PY_LOG_DIR="${PY_LOG_DIR:-py_logs/${HOSTNAME_VALUE}}"
KERNELGYM_TMP_ROOT="${KERNELGYM_TMP_ROOT:-/tmp}"
KERNELGYM_TMPDIR="${KERNELGYM_TMPDIR:-${KERNELGYM_TMP_ROOT}/kernelgym_${HOSTNAME_VALUE}}"
ENABLE_METRICS="${ENABLE_METRICS:-true}"
ENABLE_PROFILING="${ENABLE_PROFILING:-true}"
VERBOSE_ERROR_TRACEBACK="${VERBOSE_ERROR_TRACEBACK:-true}"
SAVE_EVAL_RESULTS="${SAVE_EVAL_RESULTS:-false}"
EVAL_RESULTS_PATH="${EVAL_RESULTS_PATH:-${LOG_DIR}/eval_results.jsonl}"

if [ "${ENABLE_SAVE_RESULTS}" = true ]; then
    SAVE_EVAL_RESULTS=true
fi

DEFAULT_TOOLKIT="${DEFAULT_TOOLKIT:-kernelbench}"
DEFAULT_BACKEND_ADAPTER="${DEFAULT_BACKEND_ADAPTER:-kernelbench}"
DEFAULT_BACKEND="${DEFAULT_BACKEND:-triton}"
CUDA_BUILD_BACKEND="${CUDA_BUILD_BACKEND:-manual_ninja}"
DETAILED_COMPILE_TIMING="${DETAILED_COMPILE_TIMING:-false}"
CACHE_INDEX="${CACHE_INDEX:-redis}"
COMPILE_ARTIFACT_CACHE_INDEX="${COMPILE_ARTIFACT_CACHE_INDEX:-${CACHE_INDEX}}"
MANUAL_NINJA_OBJECT_CACHE="${MANUAL_NINJA_OBJECT_CACHE:-true}"
MANUAL_NINJA_OBJECT_CACHE_INDEX="${MANUAL_NINJA_OBJECT_CACHE_INDEX:-${CACHE_INDEX}}"

NODE_ID="${NODE_ID:-$(hostname 2>/dev/null || echo "")}"
WORKER_POOL_SIZE="${WORKER_POOL_SIZE:-1}"
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-1}"
CPU_COMPILE_WORKERS="$(resolve_cpu_compile_workers)"
GPU_WORKERS_POLL_CPU_TASKS="${GPU_WORKERS_POLL_CPU_TASKS:-true}"

cat > "${ENV_FILE}" <<EOF
# KernelGym Auto-Generated Configuration
# Generated on: $(date)

# Network
API_HOST=${API_HOST}
API_PORT=${API_PORT}
API_WORKERS=${API_WORKERS}
API_RELOAD=${API_RELOAD}

# GPU
GPU_DEVICES=${GPU_DEVICES}
GPU_MEMORY_LIMIT=16GB
${TORCH_CUDA_ARCH_LIST_CONFIG}
NODE_ID=${NODE_ID}

# CPU
CPU_ARCH=${CPU_ARCH}
CPU_MODEL=$(env_quote "${CPU_MODEL}")
CPU_THREADS=${CPU_THREADS}
CPU_PHYSICAL_CORES=${CPU_PHYSICAL_CORES}
CPU_SOCKETS=${CPU_SOCKETS}
CPU_NUMA_NODES=${CPU_NUMA_NODES}
MEMORY_TOTAL_MB=${MEMORY_TOTAL_MB}
CPU_COMPILE_WORKERS=${CPU_COMPILE_WORKERS}
GPU_WORKERS_POLL_CPU_TASKS=${GPU_WORKERS_POLL_CPU_TASKS}

# Redis
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT}
REDIS_DB=${REDIS_DB}
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_KEY_PREFIX=${REDIS_KEY_PREFIX}

# Worker pool
WORKER_POOL_SIZE=${WORKER_POOL_SIZE}
MAX_TASKS_PER_WORKER=${MAX_TASKS_PER_WORKER}

# Defaults
DEFAULT_TOOLKIT=${DEFAULT_TOOLKIT}
DEFAULT_BACKEND_ADAPTER=${DEFAULT_BACKEND_ADAPTER}
DEFAULT_BACKEND=${DEFAULT_BACKEND}

# CUDA build
CUDA_BUILD_BACKEND=${CUDA_BUILD_BACKEND}
DETAILED_COMPILE_TIMING=${DETAILED_COMPILE_TIMING}
CACHE_INDEX=${CACHE_INDEX}
COMPILE_ARTIFACT_CACHE_INDEX=${COMPILE_ARTIFACT_CACHE_INDEX}
MANUAL_NINJA_OBJECT_CACHE=${MANUAL_NINJA_OBJECT_CACHE}
MANUAL_NINJA_OBJECT_CACHE_INDEX=${MANUAL_NINJA_OBJECT_CACHE_INDEX}

# Logging
LOG_LEVEL=${LOG_LEVEL}
LOG_DIR=${LOG_DIR}
PY_LOG_DIR=${PY_LOG_DIR}
KERNELGYM_TMP_ROOT=${KERNELGYM_TMP_ROOT}
KERNELGYM_TMPDIR=${KERNELGYM_TMPDIR}

# Metrics
ENABLE_METRICS=${ENABLE_METRICS}
METRICS_PORT=${METRICS_PORT}

# Profiling
ENABLE_PROFILING=${ENABLE_PROFILING}

# Errors
VERBOSE_ERROR_TRACEBACK=${VERBOSE_ERROR_TRACEBACK}

# Result persistence
SAVE_EVAL_RESULTS=${SAVE_EVAL_RESULTS}
EVAL_RESULTS_PATH=${EVAL_RESULTS_PATH}
EOF

COLOR_BLUE=""
COLOR_RESET=""
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    COLOR_BLUE=$'\033[34m'
    COLOR_RESET=$'\033[0m'
fi

echo "Wrote configuration to ${ENV_FILE}"
echo "${COLOR_BLUE}KernelGym optimization options:${COLOR_RESET}"
echo "  CUDA_BUILD_BACKEND=${CUDA_BUILD_BACKEND}  # cpp_extension_load or manual_ninja"
echo "  MANUAL_NINJA_OBJECT_CACHE=${MANUAL_NINJA_OBJECT_CACHE}  # true enables manual_ninja .o reuse"
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-unset}"
echo "  CPU_COMPILE_WORKERS=${CPU_COMPILE_WORKERS}"
echo "  KERNELGYM_TMP_ROOT=${KERNELGYM_TMP_ROOT}  # root for KERNELGYM_TMPDIR; default: /tmp"
echo "  KERNELGYM_TMPDIR=${KERNELGYM_TMPDIR}"
if [ -d "/dev/shm" ]; then
    DEV_SHM_SIZE="$(detect_path_size_human /dev/shm)"
    echo "  Optional: KERNELGYM_TMP_ROOT=/dev/shm may be faster for temporary compile files (/dev/shm: ${DEV_SHM_SIZE})"
fi
echo "KernelGym cache index options:"
echo "  CACHE_INDEX=${CACHE_INDEX}  # optional: memory or redis"
echo "  COMPILE_ARTIFACT_CACHE_INDEX=${COMPILE_ARTIFACT_CACHE_INDEX}  # optional: memory or redis"
echo "  MANUAL_NINJA_OBJECT_CACHE_INDEX=${MANUAL_NINJA_OBJECT_CACHE_INDEX}  # optional: memory/fs or redis"
echo "  REDIS=${REDIS_HOST}:${REDIS_PORT} prefix=${REDIS_KEY_PREFIX}"
echo "KernelGym analysis/debug options:"
echo "  DETAILED_COMPILE_TIMING=${DETAILED_COMPILE_TIMING}  # true reads .ninja_log for nvcc/cpp/link details"
if [ -n "${RECOMMENDED_TORCH_CUDA_ARCH_LIST}" ] && [ "${RECOMMENDED_TORCH_CUDA_ARCH_LIST}" != "${TORCH_CUDA_ARCH_LIST}" ]; then
    echo "Detected multiple TORCH_CUDA_ARCH_LIST values: ${TORCH_CUDA_ARCH_LIST}"
    echo "Recommended for this node's visible GPUs: TORCH_CUDA_ARCH_LIST=${RECOMMENDED_TORCH_CUDA_ARCH_LIST}"
    echo "Keep the multi-arch value only for compile-only nodes that target other GPU architectures."
fi
