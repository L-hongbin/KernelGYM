#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_URL = "http://10.1.17.13:8001"
DEFAULT_SAMPLES = REPO_ROOT / "logs" / "evaluate_split_compile_samples_100.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "full_evaluate_timing_20.jsonl"
WORKERS_LOG = REPO_ROOT / "kernelgym" / "logs" / "workers.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark full /evaluate latency on known-good compiled+correct samples and "
            "extract Env-side processing times from worker logs."
        )
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--samples-path", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--task-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--num-perf-trials", type=int, default=1)
    parser.add_argument("--enable-profiling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--split-compile-and-execute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-compile-artifact-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source-rows", default=None, help="Comma-separated source_row overrides.")
    return parser.parse_args()


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return json.loads(resp.read().decode(charset))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                yield idx, json.loads(line)


def collect_good_source_rows(limit: int) -> list[int]:
    best_elapsed: dict[int, float] = {}
    for path in sorted((REPO_ROOT / "logs").glob("**/gpu_cpu_true.jsonl")):
        for _, item in iter_jsonl(path):
            if item.get("compiled") is True and item.get("correctness") is True:
                row = item.get("source_row")
                if row is None:
                    continue
                row = int(row)
                elapsed = item.get("elapsed_sec")
                if elapsed is None:
                    continue
                elapsed = float(elapsed)
                prev = best_elapsed.get(row)
                if prev is None or elapsed < prev:
                    best_elapsed[row] = elapsed
    ordered = sorted(best_elapsed.items(), key=lambda kv: kv[1])
    return [row for row, _ in ordered[:limit]]


def load_samples(samples_path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for _, item in iter_jsonl(samples_path):
        row = item.get("source_row")
        if row is not None:
            result[int(row)] = item
    return result


def build_payload(sample: dict[str, Any], task_id: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
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
        "enable_profiling": args.enable_profiling,
        "verbose_errors": True,
        "split_compile_and_execute": args.split_compile_and_execute,
        "enable_compile_artifact_cache": args.enable_compile_artifact_cache,
    }


def resources_status(server_url: str) -> dict[str, Any]:
    return request_json("GET", f"{server_url}/resources/status", timeout=20)


def read_workers_log_text() -> str:
    if not WORKERS_LOG.exists():
        return ""
    return WORKERS_LOG.read_text(encoding="utf-8", errors="replace")


def extract_env_timings(log_text: str, task_ids: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for task_id in task_ids:
        pattern = re.compile(rf"completed task {re.escape(task_id)} in ([0-9]+(?:\.[0-9]+)?)s")
        matches = pattern.findall(log_text)
        if matches:
            result[task_id] = float(matches[-1])
    return result


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
    }


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status = resources_status(args.server_url)
    workers = sorted((status.get("workers") or {}).keys())
    print(json.dumps({
        "server_url": args.server_url,
        "worker_count": len(workers),
        "gpu_workers": len([w for w in workers if w.startswith("worker_gpu_")]),
        "cpu_workers": len([w for w in workers if w.startswith("worker_cpu_compile_")]),
    }, ensure_ascii=False))

    if args.source_rows:
        source_rows = [int(x.strip()) for x in args.source_rows.split(",") if x.strip()]
    else:
        source_rows = collect_good_source_rows(args.sample_count)
    if len(source_rows) < args.sample_count:
        raise RuntimeError(f"Only found {len(source_rows)} good source rows, need {args.sample_count}")

    sample_map = load_samples(samples_path)
    selected: list[tuple[int, dict[str, Any]]] = []
    for row in source_rows[: args.sample_count]:
        sample = sample_map.get(row)
        if not sample:
            raise KeyError(f"Missing source_row={row} in {samples_path}")
        selected.append((row, sample))

    output_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []

    def run_one(item: tuple[int, tuple[int, dict[str, Any]]]) -> dict[str, Any]:
        idx, (row, sample) = item
        task_id = f"full_eval_timing_{row}_{idx}_{int(time.time() * 1000)}"
        payload = build_payload(sample, task_id, args)
        log_before = read_workers_log_text()
        start = time.perf_counter()
        try:
            result = request_json("POST", f"{args.server_url}/evaluate", payload=payload, timeout=args.request_timeout)
            request_error = None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            result = {"status": "http_error", "http_status": exc.code, "error_message": body}
            request_error = f"HTTP {exc.code}"
        except Exception as exc:
            result = {"status": "client_error", "error_message": str(exc)}
            request_error = str(exc)
        wall = time.perf_counter() - start
        log_after = read_workers_log_text()
        delta_log = log_after[len(log_before):] if log_after.startswith(log_before) else log_after

        metadata = result.get("metadata") or {}
        related_ids = [task_id, f"{task_id}_compile", f"{task_id}_kernel", f"{task_id}_ref"]
        related_ids = [x for x in related_ids if x]
        env_timings = extract_env_timings(delta_log, related_ids)

        record = {
            "task_id": task_id,
            "sample_order": idx,
            "source_row": row,
            "entry_point": sample.get("entry_point"),
            "request_to_response_sec": round(wall, 3),
            "status": result.get("status"),
            "compiled": result.get("compiled"),
            "correctness": result.get("correctness"),
            "error_code": result.get("error_code"),
            "request_error": request_error,
            "processing_time": result.get("processing_time"),
            "submitted_at": result.get("submitted_at"),
            "completed_at": result.get("completed_at"),
            "reference_runtime": result.get("reference_runtime"),
            "kernel_runtime": result.get("kernel_runtime"),
            "worker_id": metadata.get("worker_id") or metadata.get("inline_compile_worker_id"),
            "worker_device": metadata.get("worker_device") or metadata.get("inline_compile_worker_device"),
            "split_compile_and_execute": metadata.get("split_compile_and_execute"),
            "precompiled_artifact_used": metadata.get("precompiled_artifact_used"),
            "inline_gpu_execute_completed": metadata.get("inline_gpu_execute_completed"),
            "related_task_ids": related_ids,
            "env_log_timings_sec": env_timings,
        }
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, item) for item in enumerate(selected)]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))

    good = [r for r in records if r.get("status") == "completed"]
    wall_values = [r["request_to_response_sec"] for r in good]
    env_values = []
    compile_values = []
    execute_values = []
    ref_values = []
    for r in good:
        timings = r.get("env_log_timings_sec") or {}
        if r["task_id"] in timings:
            env_values.append(timings[r["task_id"]])
        for k, v in timings.items():
            if str(k).endswith("_compile"):
                compile_values.append(v)
            elif str(k).endswith("_ref"):
                ref_values.append(v)
            elif k != r["task_id"]:
                execute_values.append(v)

    summary = {
        "requested_samples": args.sample_count,
        "completed_samples": len(good),
        "request_to_response_sec": summarize(wall_values),
        "env_processing_sec_from_logs": summarize(env_values),
        "compile_stage_sec_from_logs": summarize(compile_values),
        "execute_stage_sec_from_logs": summarize(execute_values),
        "reference_stage_sec_from_logs": summarize(ref_values),
        "output_path": str(output_path),
    }
    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
