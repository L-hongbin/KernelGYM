#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SAMPLES_PATH="${SAMPLES_PATH:-${ROOT_DIR}/logs/evaluate_split_compile_samples_100.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/logs/evaluate_split_compile_bench_repro}"
SERVER_URL="${SERVER_URL:-http://10.1.17.13:8001}"
SCENARIOS="${SCENARIOS:-gpu_only_false,gpu_only_true,gpu_cpu_false,gpu_cpu_true}"
GPU_WORKERS="${GPU_WORKERS:-8}"
CPU_WORKERS="${CPU_WORKERS:-8}"
WAIT_READY_TIMEOUT="${WAIT_READY_TIMEOUT:-240}"
BLOCKING_TASK_POLL="${WORKER_USE_BLOCKING_TASK_POLL:-true}"
MANAGE_ENV="${MANAGE_ENV:-true}"
ENABLE_COMPILE_ARTIFACT_CACHE="${ENABLE_COMPILE_ARTIFACT_CACHE:-false}"

case "${BLOCKING_TASK_POLL,,}" in
  true|1|yes|y)
    export WORKER_USE_BLOCKING_TASK_POLL=true
    ;;
  false|0|no|n)
    export WORKER_USE_BLOCKING_TASK_POLL=false
    ;;
  *)
    echo "Invalid WORKER_USE_BLOCKING_TASK_POLL: ${BLOCKING_TASK_POLL}" >&2
    exit 1
    ;;
esac

cd "${ROOT_DIR}"

if [[ ! -f "${SAMPLES_PATH}" ]]; then
  echo "Samples file not found: ${SAMPLES_PATH}" >&2
  echo "Generate it once with test_script/test_evaluate_split_compile_dataset.py --extract-only" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/test_evaluate_split_compile_dataset.py"
  --reuse-samples
  --samples-path "${SAMPLES_PATH}"
  --results-dir "${RESULTS_DIR}"
  --server-url "${SERVER_URL}"
  --scenarios "${SCENARIOS}"
  --gpu-workers "${GPU_WORKERS}"
  --cpu-workers "${CPU_WORKERS}"
  --wait-ready-timeout "${WAIT_READY_TIMEOUT}"
)

case "${ENABLE_COMPILE_ARTIFACT_CACHE,,}" in
  true|1|yes|y)
    CMD+=(--enable-compile-artifact-cache)
    ;;
  false|0|no|n)
    ;;
  *)
    echo "Invalid ENABLE_COMPILE_ARTIFACT_CACHE: ${ENABLE_COMPILE_ARTIFACT_CACHE}" >&2
    exit 1
    ;;
esac

case "${MANAGE_ENV,,}" in
  true|1|yes|y)
    CMD+=(--manage-env)
    ;;
  false|0|no|n)
    ;;
  *)
    echo "Invalid MANAGE_ENV: ${MANAGE_ENV}" >&2
    exit 1
    ;;
esac

exec "${CMD[@]}" "$@"
