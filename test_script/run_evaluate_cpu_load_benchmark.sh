#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SERVER_URL="${SERVER_URL:-http://10.1.17.13:8001}"
SAMPLES_PATH="${SAMPLES_PATH:-${ROOT_DIR}/logs/evaluate_split_compile_samples_100.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/logs/evaluate_cpu_load_bench}"
SCENARIO="${SCENARIO:-gpu_cpu_true}"
CONCURRENCY="${CONCURRENCY:-32}"
GPU_WORKERS="${GPU_WORKERS:-8}"
CPU_WORKERS="${CPU_WORKERS:-8}"
WAIT_READY_TIMEOUT="${WAIT_READY_TIMEOUT:-240}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-2}"
MANAGE_ENV="${MANAGE_ENV:-true}"
WORKER_USE_BLOCKING_TASK_POLL="${WORKER_USE_BLOCKING_TASK_POLL:-true}"

case "${WORKER_USE_BLOCKING_TASK_POLL,,}" in
  true|1|yes|y)
    export WORKER_USE_BLOCKING_TASK_POLL=true
    ;;
  false|0|no|n)
    export WORKER_USE_BLOCKING_TASK_POLL=false
    ;;
  *)
    echo "Invalid WORKER_USE_BLOCKING_TASK_POLL: ${WORKER_USE_BLOCKING_TASK_POLL}" >&2
    exit 1
    ;;
esac

case "${MANAGE_ENV,,}" in
  true|1|yes|y)
    MANAGE_ENV_FLAG=1
    ;;
  false|0|no|n)
    MANAGE_ENV_FLAG=0
    ;;
  *)
    echo "Invalid MANAGE_ENV: ${MANAGE_ENV}" >&2
    exit 1
    ;;
esac

cd "${ROOT_DIR}"
mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${SAMPLES_PATH}" ]]; then
  echo "Samples file not found: ${SAMPLES_PATH}" >&2
  echo "Generate it once with: test_script/test_evaluate_split_compile_dataset.py --extract-only" >&2
  exit 1
fi

HEALTH_SAMPLES_FILE="${RESULTS_DIR}/health_samples.jsonl"
HEALTH_SUMMARY_FILE="${RESULTS_DIR}/health_summary.json"
WORKERS_BEFORE_FILE="${RESULTS_DIR}/workers_before.json"
WORKERS_AFTER_FILE="${RESULTS_DIR}/workers_after.json"
BENCH_RESULTS_DIR="${RESULTS_DIR}/benchmark"

python3 - "${SERVER_URL}" "${WORKERS_BEFORE_FILE}" <<'PY'
import json
import sys
import urllib.request

server_url = sys.argv[1].rstrip("/")
output_path = sys.argv[2]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

payload = {}
try:
    with opener.open(f"{server_url}/resources/status", timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    payload = {"error": str(exc)}

workers = sorted((payload.get("workers") or {}).keys()) if isinstance(payload, dict) else []
summary = {
    "workers": workers,
    "gpu_workers": [w for w in workers if w.startswith("worker_gpu_")],
    "cpu_workers": [w for w in workers if w.startswith("worker_cpu_compile_")],
    "raw": payload,
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
PY

python3 - "${SERVER_URL}" "${HEALTH_INTERVAL}" "${HEALTH_SAMPLES_FILE}" <<'PY' &
import json
import signal
import sys
import time
import urllib.error
import urllib.request

server_url = sys.argv[1].rstrip("/")
interval = max(0.5, float(sys.argv[2]))
output_path = sys.argv[3]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
stop = False

def _stop(*_args):
    global stop
    stop = True

signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

with open(output_path, "a", encoding="utf-8") as handle:
    while not stop:
        started = time.time()
        sample = {
            "timestamp": started,
            "ok": False,
        }
        try:
            with opener.open(f"{server_url}/health", timeout=max(5, int(interval + 5))) as response:
                data = json.loads(response.read().decode("utf-8"))
            memory_usage = data.get("memory_usage") or {}
            gpu_status = data.get("gpu_status") or {}
            gpu_memory_values = []
            for info in gpu_status.values():
                if not isinstance(info, dict):
                    continue
                value = str(info.get("memory_used_percent") or "").rstrip("%")
                if not value:
                    continue
                try:
                    gpu_memory_values.append(float(value))
                except ValueError:
                    continue
            sample.update({
                "ok": True,
                "cpu_percent": memory_usage.get("cpu_percent"),
                "memory_percent": memory_usage.get("memory_percent"),
                "gpu_memory_used_percent_avg": round(sum(gpu_memory_values) / len(gpu_memory_values), 3) if gpu_memory_values else None,
                "gpu_memory_used_percent_max": round(max(gpu_memory_values), 3) if gpu_memory_values else None,
                "gpu_count": len(gpu_status) if isinstance(gpu_status, dict) else 0,
            })
        except Exception as exc:
            sample["error"] = str(exc)
        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        handle.flush()
        remaining = interval - (time.time() - started)
        if remaining > 0:
            time.sleep(remaining)
PY
HEALTH_MONITOR_PID=$!

cleanup() {
  if [[ -n "${HEALTH_MONITOR_PID:-}" ]]; then
    kill "${HEALTH_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${HEALTH_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

BENCH_CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/test_evaluate_split_compile_dataset.py"
  --reuse-samples
  --samples-path "${SAMPLES_PATH}"
  --results-dir "${BENCH_RESULTS_DIR}"
  --server-url "${SERVER_URL}"
  --scenarios "${SCENARIO}"
  --concurrency "${CONCURRENCY}"
  --gpu-workers "${GPU_WORKERS}"
  --cpu-workers "${CPU_WORKERS}"
  --wait-ready-timeout "${WAIT_READY_TIMEOUT}"
)

if [[ "${MANAGE_ENV_FLAG}" -eq 1 ]]; then
  BENCH_CMD+=(--manage-env)
fi

"${BENCH_CMD[@]}" "$@"

python3 - "${SERVER_URL}" "${WORKERS_AFTER_FILE}" <<'PY'
import json
import sys
import urllib.request

server_url = sys.argv[1].rstrip("/")
output_path = sys.argv[2]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

payload = {}
try:
    with opener.open(f"{server_url}/resources/status", timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    payload = {"error": str(exc)}

workers = sorted((payload.get("workers") or {}).keys()) if isinstance(payload, dict) else []
summary = {
    "workers": workers,
    "gpu_workers": [w for w in workers if w.startswith("worker_gpu_")],
    "cpu_workers": [w for w in workers if w.startswith("worker_cpu_compile_")],
    "raw": payload,
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
PY

kill "${HEALTH_MONITOR_PID}" >/dev/null 2>&1 || true
wait "${HEALTH_MONITOR_PID}" >/dev/null 2>&1 || true
HEALTH_MONITOR_PID=""

python3 - "${HEALTH_SAMPLES_FILE}" "${HEALTH_SUMMARY_FILE}" "${WORKERS_BEFORE_FILE}" "${WORKERS_AFTER_FILE}" "${BENCH_RESULTS_DIR}/summary_all.json" <<'PY'
import json
import sys
from pathlib import Path
from statistics import mean

samples_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
workers_before_path = Path(sys.argv[3])
workers_after_path = Path(sys.argv[4])
benchmark_summary_path = Path(sys.argv[5])

def parse_number(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None

samples = []
if samples_path.exists():
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                continue

cpu_values = [parse_number(item.get("cpu_percent")) for item in samples if parse_number(item.get("cpu_percent")) is not None]
memory_values = [parse_number(item.get("memory_percent")) for item in samples if parse_number(item.get("memory_percent")) is not None]
gpu_avg_values = [parse_number(item.get("gpu_memory_used_percent_avg")) for item in samples if parse_number(item.get("gpu_memory_used_percent_avg")) is not None]
gpu_max_values = [parse_number(item.get("gpu_memory_used_percent_max")) for item in samples if parse_number(item.get("gpu_memory_used_percent_max")) is not None]

def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return round(values[idx], 3)

health_summary = {
    "samples": len(samples),
    "ok_samples": sum(1 for item in samples if item.get("ok")),
    "error_samples": sum(1 for item in samples if not item.get("ok")),
    "cpu_percent": {
        "min": round(min(cpu_values), 3) if cpu_values else None,
        "avg": round(mean(cpu_values), 3) if cpu_values else None,
        "p50": pct(cpu_values, 0.50),
        "p95": pct(cpu_values, 0.95),
        "max": round(max(cpu_values), 3) if cpu_values else None,
    },
    "memory_percent": {
        "min": round(min(memory_values), 3) if memory_values else None,
        "avg": round(mean(memory_values), 3) if memory_values else None,
        "p50": pct(memory_values, 0.50),
        "p95": pct(memory_values, 0.95),
        "max": round(max(memory_values), 3) if memory_values else None,
    },
    "gpu_memory_used_percent_avg": {
        "min": round(min(gpu_avg_values), 3) if gpu_avg_values else None,
        "avg": round(mean(gpu_avg_values), 3) if gpu_avg_values else None,
        "p50": pct(gpu_avg_values, 0.50),
        "p95": pct(gpu_avg_values, 0.95),
        "max": round(max(gpu_avg_values), 3) if gpu_avg_values else None,
    },
    "gpu_memory_used_percent_max": {
        "min": round(min(gpu_max_values), 3) if gpu_max_values else None,
        "avg": round(mean(gpu_max_values), 3) if gpu_max_values else None,
        "p50": pct(gpu_max_values, 0.50),
        "p95": pct(gpu_max_values, 0.95),
        "max": round(max(gpu_max_values), 3) if gpu_max_values else None,
    },
    "workers_before": json.loads(workers_before_path.read_text(encoding="utf-8")) if workers_before_path.exists() else {},
    "workers_after": json.loads(workers_after_path.read_text(encoding="utf-8")) if workers_after_path.exists() else {},
    "benchmark_summary_path": str(benchmark_summary_path),
}

summary_path.write_text(json.dumps(health_summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(health_summary, indent=2, ensure_ascii=False))
PY
