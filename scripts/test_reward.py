#!/usr/bin/env python3
"""Smoke-test the reward API end-to-end.

Probes ``/health`` then submits a tiny KernelBench evaluation (vector add
reference vs a hand-written CUDA add kernel built through the CUDA-Agent
backend) to ``/evaluate``. Prints the compile / correctness / speedup
outcome. Exits 0 when the API is healthy and the request returned a
non-failed status.

Uses stdlib only (urllib + json) so it can run from any Python without
activating the venv. Bypasses ``http_proxy`` so LAN probes don't get
routed through a corporate proxy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid


REFERENCE_CODE = '''
import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference: element-wise add of two 1-D tensors."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    return [torch.randn(4096, device="cuda"), torch.randn(4096, device="cuda")]


def get_init_inputs():
    return []
'''

# CUDA-Agent submission: three labelled sections in one string that the
# kernelbench cuda_agent backend parses into (.cu + binding .cpp + ModelNew).
KERNEL_CODE = '''
### CUDA_KERNELS
```cpp
#include <torch/extension.h>

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void launch_add_kernel(const float* a, const float* b, float* out, int n) {
    constexpr int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block>>>(a, b, out, n);
}
```

### APPLY_BINDINGS
```cpp
#include <torch/extension.h>

void launch_add_kernel(const float* a, const float* b, float* out, int n);

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "input shapes must match");
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = torch::empty_like(a_c);
    launch_add_kernel(a_c.data_ptr<float>(),
                      b_c.data_ptr<float>(),
                      out.data_ptr<float>(),
                      static_cast<int>(a_c.numel()));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "elementwise add (CUDA)");
}
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    """CUDA-backed element-wise add."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return cuda_extension.add(a, b)
```
'''


def _http_get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Server returned a non-2xx; still try to surface the JSON body.
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"error": str(exc)}
        return exc.code, payload


def _build_request(task_id: str, timeout: int, run_performance: bool) -> dict:
    return {
        "task_id": task_id,
        "reference_code": REFERENCE_CODE,
        "kernel_code": KERNEL_CODE,
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": "auto",
        "num_correct_trials": 3,
        "num_perf_trials": 20,
        "num_warmup": 3,
        "perf_trim_count": 0,
        "timeout": timeout,
        "priority": "normal",
        "entry_point": "Model",
        "force_refresh": True,
        "run_performance": run_performance,
    }


def _disable_proxy_for_host(host: str) -> None:
    """Make sure urllib doesn't route LAN probes through http_proxy."""
    # Append the host to no_proxy if not already exempt. Easier than building
    # a custom opener for one request.
    existing = os.environ.get("no_proxy", "")
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if "*" not in parts and host not in parts:
        parts.append(host)
    os.environ["no_proxy"] = ",".join(parts)
    os.environ["NO_PROXY"] = os.environ["no_proxy"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=os.environ.get("KERNELGYM_REWARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("KERNELGYM_REWARD_PORT", "20111")))
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request HTTP timeout AND server-side per-task timeout (seconds)",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Override the generated task_id (default: random uuid4 hex)",
    )
    parser.add_argument("--no-perf", action="store_true", help="Skip the performance-timing phase")
    parser.add_argument("--verbose", "-v", action="store_true", help="Dump the full JSON response")
    args = parser.parse_args()

    _disable_proxy_for_host(args.host)
    base = f"http://{args.host}:{args.port}"

    print("=== /health probe ===")
    print(f"url: {base}/health")
    try:
        health = _http_get_json(f"{base}/health", timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"status: DOWN ({type(exc).__name__}: {exc})")
        return 1
    api_status = health.get("status", "?")
    gpus = health.get("gpu_status", {}) or {}
    gpus_ok = sum(1 for v in gpus.values() if isinstance(v, dict) and v.get("available"))
    print(f"status: {api_status}")
    print(f"gpus_available: {gpus_ok}/{len(gpus)}")
    if api_status != "healthy":
        return 1

    task_id = args.task_id or f"reward_smoke_{uuid.uuid4().hex[:12]}"
    payload = _build_request(task_id, timeout=args.timeout, run_performance=not args.no_perf)

    print()
    print("=== POST /evaluate ===")
    print(f"task_id: {task_id}")
    print(f"timeout: {args.timeout}s, run_performance: {not args.no_perf}")
    sent_at = time.time()
    http_status, body = _http_post_json(
        f"{base}/evaluate",
        payload,
        # Allow some headroom over the server-side task timeout for queueing.
        timeout=args.timeout + 60,
    )
    elapsed = time.time() - sent_at
    print(f"http_status: {http_status}")
    print(f"elapsed_s: {elapsed:.2f}")

    if http_status >= 400:
        print(json.dumps(body, indent=2)[:1500])
        return 1

    status = body.get("status", "?")
    compiled = body.get("compiled")
    correctness = body.get("correctness")
    speedup = body.get("speedup")
    reference_runtime = body.get("reference_runtime")
    kernel_runtime = body.get("kernel_runtime")
    error_msg = body.get("error_message")

    print(f"task_status: {status}")
    print(f"compiled: {compiled}")
    print(f"correctness: {correctness}")
    if speedup is not None:
        print(f"speedup: {speedup}")
    if reference_runtime is not None:
        print(f"reference_runtime_ms: {reference_runtime}")
    if kernel_runtime is not None:
        print(f"kernel_runtime_ms: {kernel_runtime}")
    if error_msg:
        print(f"error_message: {error_msg[:500]}")

    if args.verbose:
        print()
        print("=== full response ===")
        print(json.dumps(body, indent=2))

    return 0 if status not in (None, "failed", "?") else 1


if __name__ == "__main__":
    raise SystemExit(main())
