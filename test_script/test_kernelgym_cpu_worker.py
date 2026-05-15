#!/usr/bin/env python
"""Submit a CPU-routed KernelGym task and report which worker handled it."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_URL = os.getenv("KERNELGYM_SERVER_URL", "http://10.1.17.13:8001")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test KernelGym CPU worker routing by submitting a task with "
            "required_resource=cpu/task_stage=compile."
        )
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="KernelGym API server URL.")
    parser.add_argument(
        "--mode",
        choices=("route", "cuda-agent-compile", "stress"),
        default="route",
        help=(
            "route checks CPU queue routing; cuda-agent-compile runs real compile-only tasks; "
            "stress submits unassigned concurrent compile tasks through normal Env routing."
        ),
    )
    parser.add_argument("--task-id", default=None, help="Optional task id. Defaults to a timestamped id.")
    parser.add_argument("--timeout", type=int, default=20, help="Task timeout field sent to KernelGym.")
    parser.add_argument("--request-timeout", type=int, default=40, help="HTTP request timeout in seconds.")
    parser.add_argument(
        "--assigned-worker",
        default=None,
        help="Optional worker id, e.g. worker_cpu_compile_0, to force a worker-specific queue.",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(REPO_ROOT / "logs"),
        help="KernelGym launcher logs directory used to find the handling worker.",
    )
    parser.add_argument(
        "--internal-logs-dir",
        default=str(REPO_ROOT / "kernelgym" / "logs"),
        help="KernelGym internal RotatingFileHandler logs directory.",
    )
    parser.add_argument("--log-match-timeout", type=float, default=3.0, help="Seconds to wait for worker logs.")
    parser.add_argument("--print-full-response", action="store_true", help="Print full JSON responses.")
    parser.add_argument("--stress-requests", type=int, default=32, help="Number of concurrent stress requests.")
    parser.add_argument(
        "--monitor-resources",
        action="store_true",
        help="Sample /health while stress mode is running and summarize CPU/GPU load.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=2.0,
        help="Seconds between /health samples when --monitor-resources is enabled.",
    )
    return parser.parse_args()


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 10):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def short_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def build_kernel_simple_payload(task_id: str, timeout: int, assigned_worker: str | None):
    kernel_code = r'''
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        return x + 1


def get_inputs():
    return [torch.randn(2, 3)]
'''
    payload = {
        "task_id": task_id,
        "kernel_code": kernel_code,
        "workflow": "kernel_simple",
        "toolkit": "kernel_simple",
        "backend_adapter": "kernelbench",
        "backend": "cuda",
        "entry_point": "ModelNew",
        "device": "cpu:0",
        "priority": "normal",
        "timeout": timeout,
        "required_resource": "cpu",
        "task_stage": "compile",
        "run_correctness": False,
        "run_performance": False,
        "enable_profiling": False,
    }
    if assigned_worker:
        payload["assigned_worker"] = assigned_worker
    return payload


def build_cuda_agent_compile_payload(task_id: str, timeout: int, assigned_worker: str | None):
    reference_code = r'''
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        return x + 1


def get_inputs():
    return [torch.randn(32)]


def get_init_inputs():
    return []
'''
    kernel_code = r'''
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void add_one_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + 1.0f;
    }
}

extern "C" void add_one_launcher(
    float* output,
    const float* input,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_one_kernel<<<blocks, threads, 0, stream>>>(output, input, n);
}
```

### APPLY_BINDINGS
```cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void add_one_launcher(
    float* output,
    const float* input,
    int n,
    cudaStream_t stream
);

torch::Tensor add_one_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_one_launcher(output.data_ptr<float>(), input.data_ptr<float>(), input.numel(), stream);
    return output;
}

void register_add_one(pybind11::module& m) {
    m.def("add_one_forward", &add_one_forward, "Add one forward");
}

REGISTER_BINDING(add_one, register_add_one);
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def forward(self, x):
        return cuda_extension.add_one_forward(x)
```
'''
    payload = {
        "task_id": task_id,
        "reference_code": reference_code,
        "kernel_code": kernel_code,
        "workflow": "kernelbench",
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": "cuda_agent",
        "entry_point": "Model",
        "device": "cpu:0",
        "priority": "normal",
        "timeout": timeout,
        "required_resource": "cpu",
        "task_stage": "compile",
        "run_correctness": False,
        "run_performance": False,
        "measure_performance": False,
        "enable_profiling": False,
    }
    if assigned_worker:
        payload["assigned_worker"] = assigned_worker
    return payload


def build_cuda_agent_compile_error_payload(task_id: str, timeout: int, assigned_worker: str | None):
    payload = build_cuda_agent_compile_payload(task_id, timeout, assigned_worker)
    payload["kernel_code"] = r'''
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void broken_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x
    if (idx < n) {
        output[idx] = input[idx] + missing_symbol;
    }
}

extern "C" void broken_launcher(
    float* output,
    const float* input,
    int n,
    cudaStream_t stream
) {
    broken_kernel<<<1, 256, 0, stream>>>(output, input, n);
}
```

### APPLY_BINDINGS
```cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void broken_launcher(
    float* output,
    const float* input,
    int n,
    cudaStream_t stream
);

torch::Tensor broken_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    broken_launcher(output.data_ptr<float>(), input.data_ptr<float>(), input.numel(), stream);
    return output;
}

void register_broken(pybind11::module& m) {
    m.def("broken_forward", &broken_forward, "Broken forward");
}

REGISTER_BINDING(broken, register_broken);
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def forward(self, x):
        return cuda_extension.broken_forward(x)
```
'''
    return payload


def summarize_resources(resources: dict[str, Any]):
    cpu_workers, gpu_workers = split_workers(resources)
    print(f"CPU workers online: {sorted(cpu_workers)}")
    print(f"GPU workers online: {sorted(gpu_workers)}")
    cpu = resources.get("cpu") or {}
    if cpu:
        print(
            "CPU info: "
            f"{cpu.get('cpu_threads')} threads, "
            f"{cpu.get('cpu_physical_cores')} physical cores, "
            f"CPU_COMPILE_WORKERS={cpu.get('cpu_compile_workers')}"
        )


def split_workers(resources: dict[str, Any]):
    workers = resources.get("workers") or {}
    cpu_workers = {
        worker_id: info
        for worker_id, info in workers.items()
        if str(info.get("worker_type") or "").lower() == "cpu" or str(info.get("device") or "").startswith("cpu:")
    }
    gpu_workers = {
        worker_id: info
        for worker_id, info in workers.items()
        if str(info.get("device") or "").startswith("cuda:")
    }
    return cpu_workers, gpu_workers


def find_worker_logs(logs_dirs: list[Path], task_id: str):
    matches = []
    for logs_dir in logs_dirs:
        if not logs_dir.exists():
            continue
        patterns = (
            "worker_cpu_compile_*.log",
            "worker_gpu_*.log",
            "workers.log",
            "api.log",
            "kernelgym.log",
        )
        for pattern in patterns:
            for path in sorted(logs_dir.glob(pattern)):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if task_id not in text:
                    continue
                worker_id = path.stem
                for line in text.splitlines():
                    if task_id in line and "Worker " in line:
                        marker = line.split("Worker ", 1)[1].split(" ", 1)[0]
                        if marker:
                            worker_id = marker
                            break
                matches.append((worker_id, str(path)))
    return matches


def wait_for_worker_logs(logs_dirs: list[Path], task_id: str, timeout: float):
    deadline = time.time() + max(0.0, timeout)
    while True:
        matches = find_worker_logs(logs_dirs, task_id)
        if matches or time.time() >= deadline:
            return matches
        time.sleep(0.2)


def unique_matches(matches: list[tuple[str, str]]):
    seen = set()
    result = []
    for item in matches:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def submit_workflow(server_url: str, task_id: str, payload: dict[str, Any], request_timeout: int):
    request_payload = {
        "workflow": payload["workflow"],
        "task_id": task_id,
        "force_refresh": True,
        "payload": payload,
    }
    return request_json(
        "POST",
        f"{server_url}/workflow/submit",
        payload=request_payload,
        timeout=request_timeout,
    )


def print_workflow_response(response: dict[str, Any], print_full_response: bool):
    if print_full_response:
        print(short_json(response))
        return

    result = response.get("result") or {}
    print(short_json({
        "task_id": response.get("task_id"),
        "status": response.get("status"),
        "error_message": response.get("error_message") or result.get("error_message"),
        "error_code": response.get("error_code") or result.get("error_code"),
        "result_status": result.get("status"),
        "compiled": result.get("compiled"),
        "metadata": result.get("metadata"),
    }))


def parse_percent(value: Any):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip().rstrip("%")
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_numbers(values: list[float]):
    values = [value for value in values if value is not None]
    if not values:
        return {"min": None, "avg": None, "max": None}
    return {
        "min": round(min(values), 3),
        "avg": round(sum(values) / len(values), 3),
        "max": round(max(values), 3),
    }


class HealthMonitor:
    def __init__(self, server_url: str, interval: float):
        self.server_url = server_url.rstrip("/")
        self.interval = max(0.5, float(interval))
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 3.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                sample = request_json("GET", f"{self.server_url}/health", timeout=max(5, int(self.interval + 3)))
                self.samples.append(sample)
            except Exception as exc:
                self.errors.append(str(exc))
            self._stop.wait(self.interval)

    def summary(self):
        cpu_values: list[float] = []
        memory_values: list[float] = []
        gpu_memory_values: dict[str, list[float]] = {}

        for sample in self.samples:
            memory_usage = sample.get("memory_usage") or {}
            cpu = parse_percent(memory_usage.get("cpu_percent"))
            memory = parse_percent(memory_usage.get("memory_percent"))
            if cpu is not None:
                cpu_values.append(cpu)
            if memory is not None:
                memory_values.append(memory)

            gpu_status = sample.get("gpu_status") or {}
            for device, info in gpu_status.items():
                if not isinstance(info, dict):
                    continue
                used = parse_percent(info.get("memory_used_percent"))
                if used is not None:
                    gpu_memory_values.setdefault(str(device), []).append(used)

        per_gpu_memory = {
            device: summarize_numbers(values)
            for device, values in sorted(gpu_memory_values.items())
        }
        all_gpu_memory = [
            value
            for values in gpu_memory_values.values()
            for value in values
        ]
        return {
            "samples": len(self.samples),
            "errors": len(self.errors),
            "cpu_percent": summarize_numbers(cpu_values),
            "memory_percent": summarize_numbers(memory_values),
            "gpu_memory_used_percent": {
                "all_devices": summarize_numbers(all_gpu_memory),
                "per_device": per_gpu_memory,
            },
            "note": "/health reports GPU memory allocated/reserved percent from torch, not SM utilization.",
        }


def stress_submit_one(server_url: str, task_id: str, timeout: int, request_timeout: int):
    payload = build_cuda_agent_compile_payload(task_id, timeout, assigned_worker=None)
    started = time.perf_counter()
    try:
        response = submit_workflow(server_url, task_id, payload, request_timeout)
        elapsed = time.perf_counter() - started
        result = response.get("result") or {}
        metadata = result.get("metadata") or {}
        return {
            "task_id": task_id,
            "assigned_worker": None,
            "elapsed_sec": elapsed,
            "status": response.get("status"),
            "compiled": result.get("compiled"),
            "error_code": response.get("error_code") or result.get("error_code"),
            "error_message": response.get("error_message") or result.get("error_message"),
            "worker_id": metadata.get("worker_id"),
            "worker_device": metadata.get("worker_device"),
        }
    except Exception as exc:
        return {
            "task_id": task_id,
            "assigned_worker": None,
            "elapsed_sec": time.perf_counter() - started,
            "status": "request_failed",
            "compiled": False,
            "error_code": type(exc).__name__,
            "error_message": str(exc),
            "worker_id": None,
            "worker_device": None,
        }


def percentile(values: list[float], fraction: float):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return values[index]


def summarize_stress(label: str, results: list[dict[str, Any]], total_elapsed: float):
    durations = [float(item["elapsed_sec"]) for item in results]
    compiled_count = sum(1 for item in results if item.get("compiled") is True)
    failed_items = [
        item
        for item in results
        if item.get("compiled") is not True or item.get("status") != "completed"
    ]
    summary = {
        "label": label,
        "requests": len(results),
        "total_elapsed_sec": round(total_elapsed, 3),
        "compiled_true": compiled_count,
        "failed_or_uncompiled": len(failed_items),
        "per_request_sec": {
            "min": round(min(durations), 3) if durations else None,
            "p50": round(percentile(durations, 0.50), 3) if durations else None,
            "p95": round(percentile(durations, 0.95), 3) if durations else None,
            "max": round(max(durations), 3) if durations else None,
        },
        "status_counts": dict(Counter(str(item.get("status")) for item in results)),
        "worker_counts": dict(Counter(str(item.get("worker_id")) for item in results)),
        "error_codes": dict(Counter(str(item.get("error_code")) for item in failed_items)),
    }
    print(short_json(summary))
    if failed_items:
        print("failed_or_uncompiled_examples:")
        for item in failed_items[:3]:
            print(short_json({
                "task_id": item.get("task_id"),
                "assigned_worker": item.get("assigned_worker"),
                "status": item.get("status"),
                "compiled": item.get("compiled"),
                "error_code": item.get("error_code"),
                "error_message": str(item.get("error_message"))[:500],
            }))


def run_stress_phase(
    args,
    server_url: str,
    phase_name: str,
    task_prefix: str,
    request_count: int,
    timeout: int,
    request_timeout: int,
):
    print(f"\n=== STRESS PHASE: {phase_name} ===")
    print(f"requests: {request_count}")
    print("assigned_worker: <not set>")

    monitor = HealthMonitor(server_url, args.monitor_interval) if args.monitor_resources else None
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        if monitor:
            monitor.start()
        with concurrent.futures.ThreadPoolExecutor(max_workers=request_count) as executor:
            futures = []
            for index in range(request_count):
                phase_slug = phase_name.lower().replace("+", "plus").replace(" ", "_")
                phase_task_id = f"{task_prefix}_{phase_slug}_{index:03d}"
                futures.append(executor.submit(
                    stress_submit_one,
                    server_url,
                    phase_task_id,
                    timeout,
                    request_timeout,
                ))
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    finally:
        total_elapsed = time.perf_counter() - started
        if monitor:
            monitor.stop()
    summarize_stress(phase_name, results, total_elapsed)
    if monitor:
        print("\n=== RESOURCE MONITOR SUMMARY ===")
        print(short_json(monitor.summary()))
    return results, total_elapsed


def run_stress_mode(args, server_url: str, task_id: str, resources_before: dict[str, Any] | None):
    if resources_before is None:
        resources_before = request_json("GET", f"{server_url}/resources/status", timeout=args.request_timeout)

    cpu_workers, gpu_workers = split_workers(resources_before)
    gpu_worker_ids = sorted(gpu_workers)
    cpu_worker_ids = sorted(cpu_workers)

    print("\n=== STRESS CONFIG ===")
    print(short_json({
        "stress_requests": args.stress_requests,
        "gpu_workers": gpu_worker_ids,
        "cpu_workers": cpu_worker_ids,
        "timeout": args.timeout,
        "request_timeout": args.request_timeout,
        "note": "Stress requests do not set assigned_worker; routing is controlled by the currently running Env workers.",
    }))

    if not gpu_worker_ids:
        raise RuntimeError("No GPU workers online; cannot run stress mode.")

    warmup_task_id = f"{task_id}_gpu_warmup"
    print("\n=== GPU WORKER COMPILE WARMUP ===")
    print(f"task_id: {warmup_task_id}")
    warmup = stress_submit_one(server_url, warmup_task_id, args.timeout, args.request_timeout)
    print(short_json(warmup))
    if warmup.get("compiled") is not True:
        print("GPU warmup did not compile successfully; stress phases will still run for diagnostics.", file=sys.stderr)

    stress_results, stress_total_elapsed = run_stress_phase(
        args,
        server_url,
        "normal_routing",
        task_id,
        args.stress_requests,
        args.timeout,
        args.request_timeout,
    )

    print("\n=== STRESS COMPARISON ===")
    print(short_json({
        "normal_routing_total_elapsed_sec": round(stress_total_elapsed, 3),
        "normal_routing_compiled": sum(1 for item in stress_results if item.get("compiled") is True),
        "normal_routing_workers_used": dict(Counter(str(item.get("worker_id")) for item in stress_results)),
    }))


def main():
    args = parse_args()
    server_url = args.server_url.rstrip("/")
    task_id = args.task_id or f"cpu_route_test_{int(time.time())}"

    print("=== KERNELGYM CPU WORKER TEST ===")
    print(f"server_url: {server_url}")
    print(f"task_id: {task_id}")
    print(f"mode: {args.mode}")
    if args.assigned_worker:
        print(f"assigned_worker: {args.assigned_worker}")

    resources_before = None
    try:
        resources_before = request_json("GET", f"{server_url}/resources/status", timeout=args.request_timeout)
        print("\n=== RESOURCES BEFORE ===")
        summarize_resources(resources_before)
    except Exception as exc:
        print(f"Failed to read /resources/status: {exc}", file=sys.stderr)

    if args.mode == "stress":
        run_stress_mode(args, server_url, task_id, resources_before)
        return

    if args.mode == "cuda-agent-compile":
        payload = build_cuda_agent_compile_payload(task_id, args.timeout, args.assigned_worker)
    else:
        payload = build_kernel_simple_payload(task_id, args.timeout, args.assigned_worker)
    log_task_ids = [task_id]

    print("\n=== SUBMIT CPU-ROUTED TASK ===")
    print("payload routing fields:")
    print(short_json({k: payload[k] for k in ("required_resource", "task_stage", "device", "timeout")}))

    response = None
    try:
        response = submit_workflow(server_url, task_id, payload, args.request_timeout)
        print("\n=== WORKFLOW RESPONSE ===")
        print_workflow_response(response, args.print_full_response)
    except urllib.error.URLError as exc:
        print(f"Submit request failed or timed out: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"Submit request failed: {exc}", file=sys.stderr)

    if args.mode == "cuda-agent-compile":
        error_task_id = f"{task_id}_bad_cuda"
        log_task_ids.append(error_task_id)
        error_payload = build_cuda_agent_compile_error_payload(error_task_id, args.timeout, args.assigned_worker)

        print("\n=== SUBMIT BAD CUDA CPU-ROUTED TASK ===")
        print(f"bad_cuda_task_id: {error_task_id}")
        print("payload routing fields:")
        print(short_json({k: error_payload[k] for k in ("required_resource", "task_stage", "device", "timeout")}))

        try:
            error_response = submit_workflow(server_url, error_task_id, error_payload, args.request_timeout)
            print("\n=== BAD CUDA WORKFLOW RESPONSE ===")
            print_workflow_response(error_response, args.print_full_response)
        except urllib.error.URLError as exc:
            print(f"Bad CUDA submit request failed or timed out: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"Bad CUDA submit request failed: {exc}", file=sys.stderr)

    try:
        queue_status = request_json("GET", f"{server_url}/queue/status", timeout=args.request_timeout)
        print("\n=== QUEUE STATUS ===")
        print(short_json(queue_status))
    except Exception as exc:
        print(f"Failed to read /queue/status: {exc}", file=sys.stderr)

    print("\n=== WORKER LOG MATCHES ===")
    logs_dirs = [Path(args.logs_dir), Path(args.internal_logs_dir)]
    for log_task_id in log_task_ids:
        print(f"task_id: {log_task_id}")
        matches = unique_matches(wait_for_worker_logs(logs_dirs, log_task_id, args.log_match_timeout))
        if matches:
            for worker_id, path in matches:
                print(f"  {worker_id}: {path}")
        else:
            print(f"  No task_id match found under: {', '.join(str(path) for path in logs_dirs)}")

    print("\nExpected outcome:")
    print("- If a worker_cpu_compile_* log contains the task_id, CPU queue routing worked.")
    print("- route mode may fail with 'CUDA is required for kernel_simple'; that is OK for routing.")
    print("- cuda-agent-compile mode should return real compilation success/failure from cuda_agent.")
    print("- In cuda-agent-compile mode, the *_bad_cuda task shows the response shape for broken CUDA.")
    print("- If no CPU worker is online, resource_pending.cpu should increase or the submit may time out.")


if __name__ == "__main__":
    main()
