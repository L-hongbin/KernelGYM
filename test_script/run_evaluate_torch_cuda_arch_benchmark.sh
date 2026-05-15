#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_BENCHMARK="${SCRIPT_DIR}/run_evaluate_split_compile_benchmark.sh"
ENV_PATH="${ENV_PATH:-${ROOT_DIR}/.env}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST_BENCH:-}"
BASELINE_ARCH_LIST="${BASELINE_TORCH_CUDA_ARCH_LIST_BENCH:-}"
BASELINE_LABEL="${BASELINE_LABEL:-no_arch}"
ARCH_LABEL="${ARCH_LABEL:-with_arch}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/logs/evaluate_torch_cuda_arch_bench}"
SCENARIOS="${SCENARIOS:-gpu_cpu_true}"
MANAGE_ENV="${MANAGE_ENV:-true}"
ENABLE_COMPILE_ARTIFACT_CACHE=false
CLEAN_COMPILE_OUTPUTS="${CLEAN_COMPILE_OUTPUTS:-true}"
CLEAN_EXTRA_PATHS="${CLEAN_EXTRA_PATHS:-}"

detect_arch_list() {
  if [[ -n "${ARCH_LIST}" ]]; then
    printf '%s\n' "${ARCH_LIST}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  local caps
  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  if [[ -z "${caps}" ]]; then
    caps="$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader 2>/dev/null || true)"
  fi
  printf '%s\n' "${caps}" | awk '
    {
      gsub(/[[:space:],]/, "", $0)
      if ($0 ~ /^[0-9]+(\.[0-9]+)?$/ && !seen[$0]++) {
        if (out != "") out = out ";" $0; else out = $0
      }
    }
    END { print out }
  '
}

set_env_value() {
  local key="$1"
  local value="$2"
  if [[ ! -f "${ENV_PATH}" ]]; then
    touch "${ENV_PATH}"
  fi
  if grep -q "^${key}=" "${ENV_PATH}"; then
    sed -i.bak "s#^${key}=.*#${key}=\"${value}\"#" "${ENV_PATH}"
    rm -f "${ENV_PATH}.bak"
  else
    printf '%s="%s"\n' "${key}" "${value}" >> "${ENV_PATH}"
  fi
}

delete_env_value() {
  local key="$1"
  if [[ -f "${ENV_PATH}" ]]; then
    sed -i.bak "/^${key}=/d" "${ENV_PATH}"
    rm -f "${ENV_PATH}.bak"
  fi
}

clean_compile_outputs() {
  case "${CLEAN_COMPILE_OUTPUTS,,}" in
    true|1|yes|y)
      ;;
    false|0|no|n)
      echo "Skipping compile-output cleanup because CLEAN_COMPILE_OUTPUTS=${CLEAN_COMPILE_OUTPUTS}"
      return
      ;;
    *)
      echo "Invalid CLEAN_COMPILE_OUTPUTS: ${CLEAN_COMPILE_OUTPUTS}" >&2
      exit 1
      ;;
  esac

  local paths=(
    "/tmp/kernelgym_compile_cache"
    "/tmp/kernelgym_cuda_cache"
    "/tmp/kernelgym_cuda_agent_cache"
    "/tmp/kernelgym_cuda_build_*"
    "/tmp/cuda_agent_*"
    "/tmp/torch_extensions"
    "${HOME:-}/.cache/torch_extensions"
  )
  if [[ -n "${TORCH_EXTENSIONS_DIR:-}" ]]; then
    paths+=("${TORCH_EXTENSIONS_DIR}")
  fi
  if [[ -n "${KERNELGYM_TMPDIR:-}" ]]; then
    paths+=("${KERNELGYM_TMPDIR}")
  fi
  if [[ -n "${CLEAN_EXTRA_PATHS}" ]]; then
    IFS=':' read -ra EXTRA <<< "${CLEAN_EXTRA_PATHS}"
    paths+=("${EXTRA[@]}")
  fi

  echo "Cleaning compile outputs..."
  local path
  for path in "${paths[@]}"; do
    if [[ -z "${path}" || "${path}" == "/" ]]; then
      continue
    fi
    shopt -s nullglob
    local matches=( ${path} )
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
      continue
    fi
    printf '  rm -rf'
    printf ' %q' "${matches[@]}"
    printf '\n'
    rm -rf -- "${matches[@]}"
  done
}

run_case() {
  local label="$1"
  local arch_value="$2"
  shift 2
  local case_results="${RESULTS_ROOT}/${label}"

  if [[ -n "${arch_value}" ]]; then
    echo "Running ${label}: TORCH_CUDA_ARCH_LIST=${arch_value}"
    set_env_value "TORCH_CUDA_ARCH_LIST" "${arch_value}"
    export TORCH_CUDA_ARCH_LIST="${arch_value}"
  else
    echo "Running ${label}: TORCH_CUDA_ARCH_LIST unset"
    delete_env_value "TORCH_CUDA_ARCH_LIST"
    unset TORCH_CUDA_ARCH_LIST
  fi

  clean_compile_outputs

  RESULTS_DIR="${case_results}" \
  SCENARIOS="${SCENARIOS}" \
  MANAGE_ENV="${MANAGE_ENV}" \
  ENABLE_COMPILE_ARTIFACT_CACHE="${ENABLE_COMPILE_ARTIFACT_CACHE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  "${BASE_BENCHMARK}" "$@"
}

summarize_cases() {
  "${PYTHON_BIN}" - "${RESULTS_ROOT}" "${BASELINE_LABEL}" "${ARCH_LABEL}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = sys.argv[2:]
summary = {}
for label in labels:
    path = root / label / "summary_all.json"
    if not path.exists():
        summary[label] = {"error": f"missing {path}"}
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    scenarios = {k: v for k, v in data.items() if isinstance(v, dict) and "wall_elapsed_sec" in v}
    summary[label] = {
        name: {
            "count": item.get("count"),
            "wall_elapsed_sec": item.get("wall_elapsed_sec"),
            "avg_request_sec": (item.get("per_request_sec") or {}).get("avg"),
            "p50_request_sec": (item.get("per_request_sec") or {}).get("p50"),
            "p95_request_sec": (item.get("per_request_sec") or {}).get("p95"),
            "compile_log_duration_sec": item.get("compile_log_duration_sec"),
            "compile_timing": item.get("compile_timing"),
            "manual_ninja_object_cache": item.get("manual_ninja_object_cache"),
            "compiled_counts": item.get("compiled_counts"),
            "build_backend_counts": item.get("build_backend_counts"),
            "status_counts": item.get("status_counts"),
        }
        for name, item in scenarios.items()
    }

base, tuned = labels
comparison = {}
for scenario, base_item in summary.get(base, {}).items():
    tuned_item = summary.get(tuned, {}).get(scenario)
    if not isinstance(base_item, dict) or not isinstance(tuned_item, dict):
        continue
    base_wall = base_item.get("wall_elapsed_sec")
    tuned_wall = tuned_item.get("wall_elapsed_sec")
    if base_wall and tuned_wall:
        scenario_comparison = {
            "baseline_wall_sec": base_wall,
            "arch_wall_sec": tuned_wall,
            "speedup": round(base_wall / tuned_wall, 4),
            "wall_delta_sec": round(base_wall - tuned_wall, 3),
        }
        base_compile = (base_item.get("compile_log_duration_sec") or {}).get("avg")
        tuned_compile = (tuned_item.get("compile_log_duration_sec") or {}).get("avg")
        if base_compile and tuned_compile:
            scenario_comparison["baseline_compile_log_avg_sec"] = base_compile
            scenario_comparison["arch_compile_log_avg_sec"] = tuned_compile
            scenario_comparison["compile_log_speedup"] = round(base_compile / tuned_compile, 4)
            scenario_comparison["compile_log_delta_sec"] = round(base_compile - tuned_compile, 3)
        compile_timing = {}
        for metric in (
            "cpp_extension_load_wall_sec",
            "manual_ninja_build_wall_sec",
            "manual_ninja_import_wall_sec",
            "ninja_wall_sec",
            "cuda_compile_sec",
            "cpp_compile_sec",
            "link_sec",
            "total_wall_sec",
        ):
            base_metric = (
                (base_item.get("compile_timing") or {})
                .get(metric, {})
                .get("avg")
            )
            tuned_metric = (
                (tuned_item.get("compile_timing") or {})
                .get(metric, {})
                .get("avg")
            )
            if base_metric is None or tuned_metric is None:
                continue
            compile_timing[metric] = {
                "baseline_avg_sec": base_metric,
                "arch_avg_sec": tuned_metric,
                "speedup": round(base_metric / tuned_metric, 4) if tuned_metric else None,
                "delta_sec": round(base_metric - tuned_metric, 3),
            }
        if compile_timing:
            scenario_comparison["compile_timing"] = compile_timing
        comparison[scenario] = scenario_comparison
summary["comparison"] = comparison
out = root / "torch_cuda_arch_comparison.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"Wrote {out}")
PY
}

cd "${ROOT_DIR}"
if [[ ! -x "${BASE_BENCHMARK}" ]]; then
  chmod +x "${BASE_BENCHMARK}"
fi

HOSTNAME_VALUE="${HOSTNAME:-$(hostname 2>/dev/null || echo local)}"
HOST_ENV_PATH="${ROOT_DIR}/.env.${HOSTNAME_VALUE}"
if [[ -n "${ENV_FILE:-}" ]]; then
  ENV_PATH="${ENV_FILE}"
elif [[ -f "${ENV_PATH}" ]]; then
  ENV_PATH="${ENV_PATH}"
elif [[ -f "${HOST_ENV_PATH}" ]]; then
  ENV_PATH="${HOST_ENV_PATH}"
else
  echo "Env file not found. Set ENV_FILE or ENV_PATH, or create ${HOST_ENV_PATH}." >&2
  exit 1
fi
export ENV_FILE="${ENV_PATH}"

ENV_BACKUP="$(mktemp)"
cp "${ENV_PATH}" "${ENV_BACKUP}"
restore_env() {
  cp "${ENV_BACKUP}" "${ENV_PATH}"
  rm -f "${ENV_BACKUP}"
}
trap restore_env EXIT

set -o allexport
source "${ENV_PATH}"
set +o allexport
export ENV_FILE="${ENV_PATH}"

ARCH_LIST="$(detect_arch_list)"
if [[ -z "${ARCH_LIST}" ]]; then
  echo "Unable to detect TORCH_CUDA_ARCH_LIST. Set TORCH_CUDA_ARCH_LIST_BENCH, for example 8.0 or 9.0." >&2
  exit 1
fi

mkdir -p "${RESULTS_ROOT}"

run_case "${BASELINE_LABEL}" "${BASELINE_ARCH_LIST}" "$@"
run_case "${ARCH_LABEL}" "${ARCH_LIST}" "$@"
summarize_cases
