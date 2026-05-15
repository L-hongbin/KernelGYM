#!/usr/bin/env python3
"""Benchmark /evaluate with sampled cuda-agent dataset examples.

The script can:
1. extract 100 deterministic examples from the SFT parquet with a 6:4
   compiled/uncompiled ratio based on the first feedback,
2. run /evaluate with split_compile_and_execute true/false,
3. optionally switch Env between GPU-only and GPU+CPU worker layouts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = "/data/ssd1/lhb/SFT/prompt_v4/parallel_drkernel_minimax_results_sft.parquet"
DEFAULT_SERVER_URL = os.getenv("KERNELGYM_SERVER_URL", "http://10.1.17.13:8001")
DEFAULT_SAMPLES = REPO_ROOT / "logs" / "evaluate_split_compile_samples_100.jsonl"
DEFAULT_RESULTS_DIR = REPO_ROOT / "logs" / "evaluate_split_compile_bench"
DIRECT_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def resolve_env_path() -> Path:
    explicit = os.getenv("ENV_FILE") or os.getenv("ENV_PATH")
    if explicit:
        return Path(explicit)
    hostname = os.getenv("HOSTNAME") or os.uname().nodename
    host_env = REPO_ROOT / f".env.{hostname}"
    if host_env.exists():
        return host_env
    return REPO_ROOT / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--samples-path", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument(
        "--enable-compile-artifact-cache",
        action="store_true",
        help="Set enable_compile_artifact_cache=true in /evaluate payloads.",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--compiled-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N samples after extraction/loading.")
    parser.add_argument("--task-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--num-perf-trials", type=int, default=1)
    parser.add_argument("--gpu-workers", type=int, default=8)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Submit compile-only /evaluate requests and skip execution/correctness/performance.",
    )
    parser.add_argument(
        "--worker-log-dir",
        default=None,
        help="Directory containing worker logs. Defaults to LOG_DIR from ENV_FILE when available.",
    )
    parser.add_argument(
        "--scenarios",
        default="gpu_only_false,gpu_only_true,gpu_cpu_false,gpu_cpu_true",
        help=(
            "Comma separated scenarios. Choices: gpu_only_false,gpu_only_true,"
            "gpu_cpu_false,gpu_cpu_true,current_false,current_true"
        ),
    )
    parser.add_argument(
        "--manage-env",
        action="store_true",
        help="Edit .env CPU_COMPILE_WORKERS and restart Env before gpu_only/gpu_cpu scenarios.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only write the sampled JSONL, do not submit /evaluate requests.",
    )
    parser.add_argument(
        "--reuse-samples",
        action="store_true",
        help="Use existing --samples-path instead of re-extracting from parquet.",
    )
    parser.add_argument("--wait-ready-timeout", type=int, default=180)
    return parser.parse_args()


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 10):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with DIRECT_HTTP_OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        values[key.strip()] = value
    return values


COMPLETED_TASK_RE = re.compile(
    r"Worker\s+(?P<worker_id>\S+)\s+completed task\s+"
    r"(?P<task_id>\S+_compile)\s+in\s+(?P<duration_sec>[0-9]+(?:\.[0-9]+)?)s"
)


def worker_log_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.worker_log_dir:
        candidates = [Path(args.worker_log_dir)]
    else:
        env_values = read_env_values(resolve_env_path())
        log_dir = env_values.get("LOG_DIR")
        node_id = env_values.get("NODE_ID")
        candidates = []
        if log_dir:
            log_path = Path(log_dir)
            candidates.append(log_path if log_path.is_absolute() else REPO_ROOT / log_path)
            candidates.append(REPO_ROOT / "kernelgym" / log_path)
        if node_id:
            candidates.append(REPO_ROOT / "logs" / node_id)
            candidates.append(REPO_ROOT / "kernelgym" / "logs" / node_id)
    seen: set[Path] = set()
    for directory in candidates:
        if directory in seen or not directory.exists():
            continue
        seen.add(directory)
        paths.extend(sorted(directory.glob("worker*.log")))
        workers_log = directory / "workers.log"
        if workers_log.exists():
            paths.append(workers_log)
    unique: list[Path] = []
    seen_files: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen_files:
            unique.append(path)
            seen_files.add(resolved)
    return unique


def parse_compile_task_log_durations(
    task_ids: set[str], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    if not task_ids:
        return {}
    matches: dict[str, dict[str, Any]] = {}
    for path in worker_log_paths(args):
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for line_no, line in enumerate(handle, start=1):
                match = COMPLETED_TASK_RE.search(line)
                if not match:
                    continue
                task_id = match.group("task_id")
                if task_id not in task_ids:
                    continue
                matches[task_id] = {
                    "task_id": task_id,
                    "worker_id": match.group("worker_id"),
                    "duration_sec": round(float(match.group("duration_sec")), 3),
                    "log_path": str(path),
                    "line_no": line_no,
                    "line": line.strip(),
                }
    return matches


def messages_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [x for x in value if isinstance(x, dict)]


def first_assistant_and_feedback(messages: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    assistant_idx = None
    for i, message in enumerate(messages):
        if message.get("role") == "assistant":
            assistant_idx = i
            break
    if assistant_idx is None:
        return None, None
    code = str(messages[assistant_idx].get("content") or "")
    feedback = None
    for message in messages[assistant_idx + 1 :]:
        text = str(message.get("content") or "")
        if "Server feedback" in text and '"compiled"' in text:
            feedback = text
            break
    return code, feedback


def feedback_compiled(feedback: str | None) -> bool | None:
    if not feedback:
        return None
    match = re.search(r'"compiled"\s*:\s*(true|false)', feedback, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"


def reference_code_from_row(row: Any) -> str:
    for key in ("reward", "ground_truth", "original_python_code"):
        if key in row and row[key] is not None:
            text = str(row[key])
            if text.strip():
                return text
    return ""


def extract_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    df = pd.read_parquet(args.parquet)
    records: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        messages = messages_list(row["messages"])
        kernel_code, feedback = first_assistant_and_feedback(messages)
        compiled = feedback_compiled(feedback)
        reference_code = reference_code_from_row(row)
        if compiled is None or not kernel_code or not reference_code:
            continue
        records.append(
            {
                "source_row": int(row_idx),
                "uuid": str(row.get("uuid", "")),
                "entry_point": str(row.get("entry_point", "Model") or "Model"),
                "reference_code": reference_code,
                "kernel_code": kernel_code,
                "feedback_compiled": bool(compiled),
                "feedback_excerpt": (feedback or "")[:2000],
            }
        )

    compiled_records = [item for item in records if item["feedback_compiled"]]
    uncompiled_records = [item for item in records if not item["feedback_compiled"]]
    uncompiled_count = args.sample_size - args.compiled_count
    if len(compiled_records) < args.compiled_count or len(uncompiled_records) < uncompiled_count:
        raise RuntimeError(
            f"Not enough records for requested ratio: compiled={len(compiled_records)}, "
            f"uncompiled={len(uncompiled_records)}"
        )

    compiled_sample = pd.DataFrame(compiled_records).sample(
        n=args.compiled_count, random_state=args.seed
    ).to_dict("records")
    uncompiled_sample = pd.DataFrame(uncompiled_records).sample(
        n=uncompiled_count, random_state=args.seed + 1
    ).to_dict("records")
    sample = pd.DataFrame(compiled_sample + uncompiled_sample).sample(
        frac=1.0, random_state=args.seed + 2
    ).to_dict("records")

    path = Path(args.samples_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in sample:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sample)} samples to {path}")
    print(json.dumps(Counter(item["feedback_compiled"] for item in sample), indent=2))
    return sample


def load_samples(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_env_value(key: str, value: str) -> None:
    env_path = resolve_env_path()
    if not env_path.exists():
        raise FileNotFoundError(
            f"Env file not found: {env_path}. Set ENV_FILE or ENV_PATH to the KernelGYM env file."
        )
    lines = env_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def restart_env(label: str, cpu_workers: int) -> None:
    set_env_value("CPU_COMPILE_WORKERS", str(cpu_workers))
    env_path = resolve_env_path()
    log_path = REPO_ROOT / "logs" / f"start_all_{label}.log"
    cmd = ["bash", "start_all_with_monitor.sh"]
    env = os.environ.copy()
    env["ENV_FILE"] = str(env_path)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def resources_status(server_url: str, timeout: int = 10) -> dict[str, Any]:
    return request_json("GET", f"{server_url}/resources/status", timeout=timeout)


def wait_ready(
    server_url: str,
    expected_gpu_workers: int | None,
    expected_cpu_workers: int | None,
    timeout_sec: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last: dict[str, Any] = {}
    while time.time() < deadline:
        try:
            status = resources_status(server_url)
            workers = sorted((status.get("workers") or {}).keys())
            gpu_workers = [w for w in workers if w.startswith("worker_gpu_")]
            cpu_workers = [w for w in workers if w.startswith("worker_cpu_compile_")]
            if (
                (expected_gpu_workers is None or len(gpu_workers) >= expected_gpu_workers)
                and (
                    expected_cpu_workers is None or len(cpu_workers) >= expected_cpu_workers
                )
            ):
                return status
            last = {"gpu_workers": gpu_workers, "cpu_workers": cpu_workers}
        except Exception as exc:
            last = {"error": str(exc)}
        time.sleep(2)
    raise TimeoutError(f"Env not ready after {timeout_sec}s: {last}")


def build_payload(sample: dict[str, Any], task_id: str, split: bool, args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "task_id": task_id,
        "reference_code": sample["reference_code"],
        "kernel_code": sample["kernel_code"],
        "workflow": "kernelbench",
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": "cuda_agent",
        "entry_point": sample.get("entry_point") or "Model",
        "timeout": args.task_timeout,
        "priority": "normal",
        "num_correct_trials": args.num_correct_trials,
        "num_perf_trials": args.num_perf_trials,
        "enable_profiling": False,
        "verbose_errors": True,
        "split_compile_and_execute": split,
        "enable_compile_artifact_cache": args.enable_compile_artifact_cache,
    }
    if args.compile_only:
        payload.update(
            {
                "task_stage": "compile",
                "pure_compile_task": True,
                "split_compile_and_execute": False,
                "run_correctness": False,
                "run_performance": False,
                "measure_performance": False,
                "enable_profiling": False,
            }
        )
    return payload


def submit_one(
    sample_idx: int,
    sample: dict[str, Any],
    scenario: str,
    split: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = f"{scenario}_{sample_idx:03d}_{int(time.time() * 1000)}"
    payload = build_payload(sample, task_id, split, args)
    start = time.perf_counter()
    result: dict[str, Any]
    try:
        result = request_json(
            "POST",
            f"{args.server_url}/evaluate",
            payload=payload,
            timeout=args.request_timeout,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        result = {"status": "http_error", "error_message": body, "http_status": exc.code}
    except Exception as exc:
        result = {"status": "client_error", "error_message": str(exc)}
    elapsed = time.perf_counter() - start
    metadata = result.get("metadata") or {}
    compile_artifact = metadata.get("compile_artifact") or {}
    compile_timing = (
        compile_artifact.get("compile_timing")
        if isinstance(compile_artifact, dict)
        else None
    )
    error_message = result.get("error_message")
    infra_error = is_infra_error(result)
    return {
        "task_id": task_id,
        "scenario": scenario,
        "split_requested": split,
        "sample_idx": sample_idx,
        "source_row": sample.get("source_row"),
        "feedback_compiled": sample.get("feedback_compiled"),
        "elapsed_sec": round(elapsed, 3),
        "status": result.get("status"),
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "error_code": result.get("error_code"),
        "error_message": error_message,
        "infra_error": infra_error,
        "metadata_split_compile_and_execute": metadata.get("split_compile_and_execute"),
        "metadata_precompiled_artifact_used": metadata.get("precompiled_artifact_used"),
        "metadata_inline_gpu_execute_completed": metadata.get("inline_gpu_execute_completed"),
        "metadata_compile_artifact_cache_hit": metadata.get("compile_artifact_cache_hit"),
        "metadata_compile_artifact_cache_enabled": metadata.get(
            "compile_artifact_cache_enabled"
        ),
        "metadata_compile_only": metadata.get("compile_only"),
        "metadata_task_stage": metadata.get("task_stage"),
        "build_backend": (
            compile_artifact.get("build_backend")
            if isinstance(compile_artifact, dict)
            else None
        )
        or (
            compile_timing.get("build_backend")
            if isinstance(compile_timing, dict)
            else None
        ),
        "compile_timing": compile_timing if isinstance(compile_timing, dict) else None,
        "worker_id": metadata.get("worker_id") or metadata.get("inline_compile_worker_id"),
        "worker_device": metadata.get("worker_device") or metadata.get("inline_compile_worker_device"),
        "raw_result": result,
    }


def is_infra_error(result: dict[str, Any]) -> bool:
    error_message = str(result.get("error_message") or "")
    if result.get("status") in {"client_error", "http_error"}:
        return True
    return "Worker shutdown" in error_message


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def metric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "avg": round(mean(values), 3) if values else None,
        "p50": round(median(values), 3) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(item["elapsed_sec"]) for item in results]
    compile_log_durations = [
        float(item["compile_log_duration_sec"])
        for item in results
        if item.get("compile_log_duration_sec") is not None
    ]
    compile_timing_keys = [
        "cpp_extension_load_wall_sec",
        "manual_ninja_build_wall_sec",
        "manual_ninja_import_wall_sec",
        "total_wall_sec",
        "ninja_wall_sec",
        "cuda_compile_sec",
        "cpp_compile_sec",
        "link_sec",
        "other_sec",
        "copy_so_sec",
    ]
    compile_timing_summary = {}
    for key in compile_timing_keys:
        values = [
            float((item.get("compile_timing") or {}).get(key))
            for item in results
            if (item.get("compile_timing") or {}).get(key) is not None
        ]
        compile_timing_summary[key] = metric_summary(values)
    object_cache_items = [
        (item.get("compile_timing") or {}).get("manual_ninja_object_cache")
        for item in results
        if isinstance((item.get("compile_timing") or {}).get("manual_ninja_object_cache"), dict)
    ]
    object_cache_object_statuses = Counter()
    object_cache_object_name_statuses: dict[str, Counter[str]] = {}
    object_cache_index_statuses = Counter()
    object_cache_index_statuses_by_object: dict[str, Counter[str]] = {}
    object_cache_skipped = Counter()
    object_cache_skipped_by_object: dict[str, Counter[str]] = {}
    for cache_item in object_cache_items:
        for obj in cache_item.get("objects") or []:
            object_name = str(obj.get("object"))
            cache_status = str(obj.get("cache_status"))
            index_status = str(obj.get("index_status"))
            object_cache_object_statuses[cache_status] += 1
            object_cache_object_name_statuses.setdefault(object_name, Counter())[cache_status] += 1
            object_cache_index_statuses[index_status] += 1
            object_cache_index_statuses_by_object.setdefault(object_name, Counter())[index_status] += 1
        for skipped in cache_item.get("skipped") or []:
            object_name = str(skipped.get("object"))
            reason = str(skipped.get("reason"))
            object_cache_skipped[reason] += 1
            object_cache_skipped_by_object.setdefault(object_name, Counter())[reason] += 1
    object_cache_summary = {
        "enabled_count": len(object_cache_items),
        "hits": sum(int(item.get("hits") or 0) for item in object_cache_items),
        "misses": sum(int(item.get("misses") or 0) for item in object_cache_items),
        "lookup_wall_sec": metric_summary(
            [
                float(item.get("lookup_wall_sec"))
                for item in object_cache_items
                if item.get("lookup_wall_sec") is not None
            ]
        ),
        "store_wall_sec": metric_summary(
            [
                float(item.get("store_wall_sec"))
                for item in object_cache_items
                if item.get("store_wall_sec") is not None
            ]
        ),
        "object_status_counts": dict(object_cache_object_statuses),
        "object_name_status_counts": {
            name: dict(counter)
            for name, counter in sorted(object_cache_object_name_statuses.items())
        },
        "index_status_counts": dict(object_cache_index_statuses),
        "index_status_counts_by_object": {
            name: dict(counter)
            for name, counter in sorted(object_cache_index_statuses_by_object.items())
        },
        "skipped_reason_counts": dict(object_cache_skipped),
        "skipped_reason_counts_by_object": {
            name: dict(counter)
            for name, counter in sorted(object_cache_skipped_by_object.items())
        },
    }
    non_infra_results = [item for item in results if not item.get("infra_error")]
    groups = Counter(
        (item.get("feedback_compiled"), item.get("compiled"), item.get("status"))
        for item in results
    )
    consistency = sum(
        1 for item in results if bool(item.get("feedback_compiled")) == bool(item.get("compiled"))
    )
    non_infra_consistency = sum(
        1
        for item in non_infra_results
        if bool(item.get("feedback_compiled")) == bool(item.get("compiled"))
    )
    return {
        "count": len(results),
        "total_elapsed_sec": round(sum(elapsed), 3),
        "wall_elapsed_sec": None,
        "per_request_sec": {
            "min": min(elapsed) if elapsed else None,
            "avg": round(mean(elapsed), 3) if elapsed else None,
            "p50": round(median(elapsed), 3) if elapsed else None,
            "p95": percentile(elapsed, 0.95),
            "max": max(elapsed) if elapsed else None,
        },
        "status_counts": dict(Counter(item.get("status") for item in results)),
        "compiled_counts": dict(Counter(str(item.get("compiled")) for item in results)),
        "build_backend_counts": dict(Counter(str(item.get("build_backend")) for item in results)),
        "infra_error_count": sum(1 for item in results if item.get("infra_error")),
        "feedback_vs_result_counts": {str(k): v for k, v in groups.items()},
        "feedback_compiled_match_count": consistency,
        "feedback_compiled_match_rate": round(consistency / len(results), 4) if results else None,
        "feedback_compiled_match_count_excluding_infra": non_infra_consistency,
        "feedback_compiled_match_rate_excluding_infra": (
            round(non_infra_consistency / len(non_infra_results), 4) if non_infra_results else None
        ),
        "metadata_split_counts": dict(
            Counter(str(item.get("metadata_split_compile_and_execute")) for item in results)
        ),
        "metadata_precompiled_counts": dict(
            Counter(str(item.get("metadata_precompiled_artifact_used")) for item in results)
        ),
        "inline_gpu_execute_counts": dict(
            Counter(str(item.get("metadata_inline_gpu_execute_completed")) for item in results)
        ),
        "compile_artifact_cache_hit_counts": dict(
            Counter(str(item.get("metadata_compile_artifact_cache_hit")) for item in results)
        ),
        "compile_artifact_cache_enabled_counts": dict(
            Counter(str(item.get("metadata_compile_artifact_cache_enabled")) for item in results)
        ),
        "compile_log_duration_sec": metric_summary(compile_log_durations),
        "compile_timing": compile_timing_summary,
        "manual_ninja_object_cache": object_cache_summary,
    }


def compile_task_id_for_result(item: dict[str, Any]) -> str | None:
    value = item.get("task_id")
    if isinstance(value, str) and value:
        return value if value.endswith("_compile") else f"{value}_compile"
    return None


def attach_compile_log_durations(
    results: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    task_ids = {
        task_id
        for task_id in (compile_task_id_for_result(item) for item in results)
        if task_id
    }
    log_matches = parse_compile_task_log_durations(task_ids, args)
    enriched = []
    for item in results:
        item = dict(item)
        task_id = compile_task_id_for_result(item)
        item["compile_log_task_id"] = task_id
        match = log_matches.get(task_id or "")
        item["compile_log_found"] = bool(match)
        item["compile_log_duration_sec"] = (
            match.get("duration_sec") if match else None
        )
        item["compile_log_worker_id"] = match.get("worker_id") if match else None
        item["compile_log_path"] = match.get("log_path") if match else None
        item["compile_log_line_no"] = match.get("line_no") if match else None
        enriched.append(item)
    return enriched


def compare_cross_scenario(all_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_scenario = {
        scenario: {item["sample_idx"]: item for item in results}
        for scenario, results in all_results.items()
    }
    scenarios = sorted(by_scenario)
    if not scenarios:
        return {"scenarios": [], "matched_samples": 0, "mismatch_count": 0, "mismatch_examples": []}
    common = set(by_scenario[scenarios[0]])
    for scenario in scenarios[1:]:
        common &= set(by_scenario[scenario])

    mismatch_examples = []
    matched = 0
    for sample_idx in sorted(common):
        signatures = {}
        for scenario in scenarios:
            item = by_scenario[scenario][sample_idx]
            signatures[scenario] = {
                "status": item.get("status"),
                "compiled": item.get("compiled"),
                "correctness": item.get("correctness"),
                "error_code": item.get("error_code"),
                "infra_error": item.get("infra_error"),
            }
        first = next(iter(signatures.values()))
        if all(value == first for value in signatures.values()):
            matched += 1
        elif len(mismatch_examples) < 20:
            mismatch_examples.append({"sample_idx": sample_idx, "signatures": signatures})

    return {
        "scenarios": scenarios,
        "matched_samples": matched,
        "common_samples": len(common),
        "match_rate": round(matched / len(common), 4) if common else None,
        "mismatch_count": len(common) - matched,
        "mismatch_examples": mismatch_examples,
    }


def run_scenario(
    scenario: str,
    samples: list[dict[str, Any]],
    split: bool,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(submit_one, idx, sample, scenario, split, args)
            for idx, sample in enumerate(samples)
        ]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            results.append(item)
            done = len(results)
            if done % 10 == 0 or done == len(samples):
                print(f"{scenario}: completed {done}/{len(samples)}")
    summary = summarize(results)
    summary["wall_elapsed_sec"] = round(time.perf_counter() - started, 3)
    return results, summary


def scenario_config(name: str, cpu_workers: int) -> tuple[int | None, bool]:
    table = {
        "gpu_only_false": (0, False),
        "gpu_only_true": (0, True),
        "gpu_cpu_false": (cpu_workers, False),
        "gpu_cpu_true": (cpu_workers, True),
        "current_false": (None, False),
        "current_true": (None, True),
    }
    if name not in table:
        raise ValueError(f"Unknown scenario: {name}")
    return table[name]


def main() -> None:
    args = parse_args()
    if args.reuse_samples and Path(args.samples_path).exists():
        samples = load_samples(args.samples_path)
        print(f"Loaded {len(samples)} samples from {args.samples_path}")
    else:
        samples = extract_samples(args)
    if args.extract_only:
        return
    if args.limit > 0:
        samples = samples[: args.limit]
        print(f"Running limited sample set: {len(samples)} samples")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, Any] = {}
    all_results: dict[str, list[dict[str, Any]]] = {}
    for scenario in [x.strip() for x in args.scenarios.split(",") if x.strip()]:
        cpu_workers, split = scenario_config(scenario, args.cpu_workers)
        if cpu_workers is not None:
            if not args.manage_env:
                raise RuntimeError(
                    f"Scenario {scenario} needs --manage-env to switch CPU_COMPILE_WORKERS={cpu_workers}"
                )
            print(f"Restarting Env for {scenario}: CPU_COMPILE_WORKERS={cpu_workers}")
            restart_env(scenario, cpu_workers)
        status = wait_ready(args.server_url, args.gpu_workers, cpu_workers, args.wait_ready_timeout)
        worker_names = sorted((status.get("workers") or {}).keys())
        print(f"{scenario}: workers={worker_names}")

        results, summary = run_scenario(scenario, samples, split, args)
        if args.compile_only:
            wall_elapsed_sec = summary["wall_elapsed_sec"]
            results = attach_compile_log_durations(results, args)
            summary = summarize(results)
            summary["wall_elapsed_sec"] = wall_elapsed_sec
        all_results[scenario] = results
        summary["workers_before"] = worker_names
        all_summaries[scenario] = summary

        result_path = results_dir / f"{scenario}.jsonl"
        with result_path.open("w", encoding="utf-8") as handle:
            for item in sorted(results, key=lambda x: x["sample_idx"]):
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        summary_path = results_dir / f"{scenario}.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({scenario: summary}, indent=2, ensure_ascii=False))

    combined = results_dir / "summary_all.json"
    all_summaries["cross_scenario_consistency"] = compare_cross_scenario(all_results)
    combined.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote combined summary to {combined}")


if __name__ == "__main__":
    main()
