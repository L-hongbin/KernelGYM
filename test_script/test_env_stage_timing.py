#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_URL = "http://10.1.17.13:8001"
DEFAULT_SAMPLES = REPO_ROOT / "logs" / "evaluate_split_compile_samples_100.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark KernelGym env stage timings on one known-good compiled+correct sample "
            "under a GPU+CPU worker environment."
        )
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--samples-path", default=str(DEFAULT_SAMPLES))
    parser.add_argument(
        "--benchmark-result",
        default=None,
        help=(
            "Optional gpu_cpu_true benchmark result JSONL used to pick a compiled+correct sample. "
            "If omitted, the script searches logs/**/gpu_cpu_true.jsonl."
        ),
    )
    parser.add_argument("--source-row", type=int, default=None, help="Force a specific source_row from samples JSONL.")
    parser.add_argument("--sample-idx", type=int, default=None, help="Force a specific sample_idx from samples JSONL line order.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--task-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--num-perf-trials", type=int, default=1)
    parser.add_argument(
        "--enable-compile-artifact-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--split-compile-and-execute",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--print-raw-result",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
            if not line:
                continue
            yield idx, json.loads(line)


def find_gpu_cpu_true_result_file(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark result file not found: {path}")
        return path
    candidates = sorted(REPO_ROOT.glob("logs/**/gpu_cpu_true.jsonl"))
    for path in candidates:
        for _, item in iter_jsonl(path):
            if item.get("compiled") is True and item.get("correctness") is True:
                return path
    raise FileNotFoundError("No gpu_cpu_true benchmark result JSONL with compiled+correct samples found under logs/.")


def pick_source_row_from_result(path: Path) -> tuple[int, dict[str, Any]]:
    for _, item in iter_jsonl(path):
        if item.get("compiled") is True and item.get("correctness") is True:
            source_row = item.get("source_row")
            if source_row is None:
                continue
            return int(source_row), item
    raise ValueError(f"No compiled+correct record found in {path}")


def load_sample(samples_path: Path, source_row: int | None, sample_idx: int | None) -> tuple[int, dict[str, Any]]:
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples JSONL not found: {samples_path}")
    for idx, item in iter_jsonl(samples_path):
        if source_row is not None and int(item.get("source_row", -1)) == source_row:
            return idx, item
        if sample_idx is not None and idx == sample_idx:
            return idx, item
    needle = f"source_row={source_row}" if source_row is not None else f"sample_idx={sample_idx}"
    raise ValueError(f"Sample not found in {samples_path} for {needle}")


def build_payload(
    sample: dict[str, Any],
    task_id: str,
    args: argparse.Namespace,
    *,
    pure_compile_task: bool,
    run_correctness: bool,
    run_performance: bool,
) -> dict[str, Any]:
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
        "enable_profiling": False,
        "verbose_errors": True,
        "split_compile_and_execute": args.split_compile_and_execute,
        "enable_compile_artifact_cache": args.enable_compile_artifact_cache,
        "pure_compile_task": pure_compile_task,
        "run_correctness": run_correctness,
        "run_performance": run_performance,
        "measure_performance": run_performance,
    }


def resources_status(server_url: str) -> dict[str, Any] | None:
    try:
        return request_json("GET", f"{server_url}/resources/status", timeout=30)
    except Exception:
        return None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples_path)

    picked_result = None
    source_row = args.source_row
    if source_row is None and args.sample_idx is None:
        result_file = find_gpu_cpu_true_result_file(args.benchmark_result)
        source_row, picked_result = pick_source_row_from_result(result_file)
        print(f"Picked sample from benchmark result: {result_file}")
        print(
            json.dumps(
                {
                    "source_row": source_row,
                    "sample_idx": picked_result.get("sample_idx"),
                    "elapsed_sec": picked_result.get("elapsed_sec"),
                    "worker_id": picked_result.get("worker_id"),
                    "worker_device": picked_result.get("worker_device"),
                },
                ensure_ascii=False,
            )
        )

    sample_line_idx, sample = load_sample(samples_path, source_row=source_row, sample_idx=args.sample_idx)
    print(
        json.dumps(
            {
                "sample_line_idx": sample_line_idx,
                "source_row": sample.get("source_row"),
                "entry_point": sample.get("entry_point"),
                "feedback_compiled": sample.get("feedback_compiled"),
                "kernel_code_len": len(sample.get("kernel_code", "")),
                "reference_code_len": len(sample.get("reference_code", "")),
            },
            ensure_ascii=False,
        )
    )

    status = resources_status(args.server_url) or {}
    workers = sorted((status.get("workers") or {}).keys())
    gpu_workers = [w for w in workers if w.startswith("worker_gpu_")]
    cpu_workers = [w for w in workers if w.startswith("worker_cpu_compile_")]
    print(
        json.dumps(
            {
                "server_url": args.server_url,
                "gpu_workers": len(gpu_workers),
                "cpu_compile_workers": len(cpu_workers),
                "worker_names_sample": workers[:8],
            },
            ensure_ascii=False,
        )
    )

    modes = [
        ("compile_only", dict(pure_compile_task=True, run_correctness=False, run_performance=False)),
        ("compile_plus_correctness", dict(pure_compile_task=False, run_correctness=True, run_performance=False)),
        ("compile_plus_performance", dict(pure_compile_task=False, run_correctness=False, run_performance=True)),
        ("full", dict(pure_compile_task=False, run_correctness=True, run_performance=True)),
    ]

    all_results: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in modes}

    for name, flags in modes:
        print(f"\n=== {name} ===")
        for repeat_idx in range(args.repeats):
            task_id = f"stage_timing_{name}_{sample.get('source_row', sample_line_idx)}_{repeat_idx}_{int(time.time() * 1000)}"
            payload = build_payload(sample, task_id, args, **flags)
            started = time.perf_counter()
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
            except Exception as exc:  # pragma: no cover - operational path
                result = {"status": "client_error", "error_message": str(exc)}
            wall = time.perf_counter() - started
            record = {
                "wall_time_sec": wall,
                "processing_time": result.get("processing_time"),
                "status": result.get("status"),
                "compiled": result.get("compiled"),
                "correctness": result.get("correctness"),
                "error_code": result.get("error_code"),
                "error_message": result.get("error_message"),
                "metadata": result.get("metadata") or {},
            }
            all_results[name].append(record)
            print(
                json.dumps(
                    {
                        "repeat": repeat_idx,
                        "wall_time_sec": round(wall, 3),
                        "processing_time": result.get("processing_time"),
                        "status": result.get("status"),
                        "compiled": result.get("compiled"),
                        "correctness": result.get("correctness"),
                        "error_code": result.get("error_code"),
                    },
                    ensure_ascii=False,
                )
            )
            if args.print_raw_result:
                print(json.dumps(result, ensure_ascii=False, indent=2))

    summary: dict[str, dict[str, Any]] = {}
    for name in all_results:
        walls = [item["wall_time_sec"] for item in all_results[name]]
        procs = [item["processing_time"] for item in all_results[name] if item["processing_time"] is not None]
        summary[name] = {
            "median_wall": median(walls),
            "mean_wall": mean(walls),
            "median_processing": median(procs),
            "statuses": [item["status"] for item in all_results[name]],
        }

    compile_med = summary["compile_only"]["median_wall"]
    correctness_med = summary["compile_plus_correctness"]["median_wall"]
    performance_med = summary["compile_plus_performance"]["median_wall"]
    full_med = summary["full"]["median_wall"]

    estimated_correctness = None
    estimated_performance = None
    if full_med is not None and performance_med is not None:
        estimated_correctness = full_med - performance_med
    if full_med is not None and correctness_med is not None:
        estimated_performance = full_med - correctness_med

    stage_candidates = {
        "compile_only_total": compile_med,
        "estimated_correctness_increment": estimated_correctness,
        "estimated_performance_increment": estimated_performance,
    }
    ranked = sorted(
        [(k, v) for k, v in stage_candidates.items() if v is not None],
        key=lambda item: item[1],
        reverse=True,
    )

    print("\n=== Summary ===")
    for name, info in summary.items():
        print(
            json.dumps(
                {
                    "mode": name,
                    "median_wall": round(info["median_wall"], 3) if info["median_wall"] is not None else None,
                    "mean_wall": round(info["mean_wall"], 3) if info["mean_wall"] is not None else None,
                    "median_processing": info["median_processing"],
                    "statuses": info["statuses"],
                },
                ensure_ascii=False,
            )
        )

    print("\n=== Estimated Stage Cost ===")
    print(
        json.dumps(
            {
                "compile_only_total": compile_med,
                "estimated_correctness_increment": estimated_correctness,
                "estimated_performance_increment": estimated_performance,
                "heaviest_stage_estimate": ranked[0][0] if ranked else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if ranked:
        print(f"\nLikely heaviest stage under current GPU_CPU_TRUE env: {ranked[0][0]} ({fmt(ranked[0][1])})")


if __name__ == "__main__":
    main()
