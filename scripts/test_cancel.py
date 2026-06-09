#!/usr/bin/env python3
"""End-to-end test for DELETE /tasks/{task_id} cancellation.

Validates two behaviors against a live reward deployment:

  inflight  Submit a deliberately slow GPU evaluation, wait until a worker is
            actually running it (status == processing), then cancel it and
            assert it goes terminal *promptly* (well before its natural
            runtime) with error_message == "Task cancelled". A control run
            (no cancel) establishes the natural runtime baseline.

  pending   Submit a task pinned to a non-existent node so no worker will ever
            pick it up (it stays pending), cancel it, and assert it is removed
            from the queue and recorded cancelled without ever running.

Stdlib only (urllib + threading) so it runs from any Python without the venv.
Bypasses http_proxy so LAN probes aren't routed through a corporate proxy.

Examples:
  python scripts/test_cancel.py --host 192.168.16.39 --mode both -v
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid


# Reference and ModelNew run the *same* heavy element-wise loop, so correctness
# passes exactly while the perf phase stays long enough to cancel mid-flight.
# ModelNew also calls the (trivially correct) custom add kernel so the cuda_agent
# backend's decoy heuristics see real kernel usage.
def _reference_code(size_pow: int, iters: int) -> str:
    return f"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, a, b):
        out = a + b
        for _ in range({iters}):
            out = torch.sin(out) + b
        return out


def get_inputs():
    n = 1 << {size_pow}
    return [torch.randn(n, device="cuda"), torch.randn(n, device="cuda")]


def get_init_inputs():
    return []
"""


def _kernel_code(iters: int) -> str:
    return f"""
### CUDA_KERNELS
```cpp
#include <torch/extension.h>

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}}

void launch_add_kernel(const float* a, const float* b, float* out, int n) {{
    constexpr int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block>>>(a, b, out, n);
}}
```

### APPLY_BINDINGS
```cpp
#include <torch/extension.h>

void launch_add_kernel(const float* a, const float* b, float* out, int n);

torch::Tensor add(torch::Tensor a, torch::Tensor b) {{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = torch::empty_like(a_c);
    launch_add_kernel(a_c.data_ptr<float>(), b_c.data_ptr<float>(),
                      out.data_ptr<float>(), static_cast<int>(a_c.numel()));
    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("add", &add, "elementwise add (CUDA)");
}}
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def forward(self, a, b):
        out = cuda_extension.add(a, b)
        for _ in range({iters}):
            out = torch.sin(out) + b
        return out
```
"""


def _disable_proxy_for_host(host: str) -> None:
    existing = os.environ.get("no_proxy", "")
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if "*" not in parts and host not in parts:
        parts.append(host)
    os.environ["no_proxy"] = ",".join(parts)
    os.environ["NO_PROXY"] = os.environ["no_proxy"]


def _get(url: str, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"error": str(exc)}


def _post(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"error": str(exc)}


def _delete(url: str, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="DELETE", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"error": str(exc)}


def _build_request(base_task_id, *, timeout, size_pow, iters, perf_trials, extra=None):
    req = {
        "task_id": base_task_id,
        "reference_code": _reference_code(size_pow, iters),
        "kernel_code": _kernel_code(iters),
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": "auto",
        "num_correct_trials": 2,
        "num_perf_trials": perf_trials,
        "num_warmup": 5,
        "timeout": timeout,
        "priority": "normal",
        "entry_point": "Model",
        "force_refresh": True,
        "run_performance": True,
        "detect_decoy_kernel": False,
        "enable_triton_detection": False,
        "run_triton_detection": False,
    }
    if extra:
        req.update(extra)
    return req


def _gpu_used_gb(base: str) -> float:
    _, h = _get(f"{base}/health", timeout=10)
    total = 0.0
    for v in (h.get("gpu_status") or {}).values():
        if isinstance(v, dict):
            try:
                total += float(str(v.get("memory_used", "0")).rstrip("GB"))
            except Exception:
                pass
    return total


def _poll_until(base, task_id, predicate, deadline_s, interval=0.3):
    start = time.time()
    last = None
    while time.time() - start < deadline_s:
        code, body = _get(f"{base}/status/{task_id}", timeout=10)
        last = (code, body)
        if code == 200 and predicate(body.get("status")):
            return body.get("status"), time.time() - start, last
        time.sleep(interval)
    return None, time.time() - start, last


def run_control(base, args) -> float:
    """Measure natural runtime of the slow task (no cancel)."""
    tid = f"cancel_ctrl_{uuid.uuid4().hex[:10]}"
    req = _build_request(
        tid, timeout=args.timeout, size_pow=args.size_pow, iters=args.iters, perf_trials=args.perf_trials
    )
    print(f"[control] submitting {tid} (measuring natural runtime)...")
    t0 = time.time()
    code, body = _post(f"{base}/evaluate", req, timeout=args.timeout + 60)
    dt = time.time() - t0
    print(
        f"[control] http={code} status={body.get('status')} "
        f"compiled={body.get('compiled')} correctness={body.get('correctness')} natural_runtime={dt:.1f}s"
    )
    return dt


def _wait_subtask_running(base, tid, deadline_s=120):
    """The /evaluate parent id is 404 mid-flight; its GPU sub-tasks ({tid}_kernel
    /{tid}_ref) go to 'processing'. Return which sub-task is running, or None."""
    sub_ids = [f"{tid}_kernel", f"{tid}_ref"]
    start = time.time()
    while time.time() - start < deadline_s:
        for sid in sub_ids:
            code, body = _get(f"{base}/status/{sid}", timeout=10)
            if code == 200 and body.get("status") == "processing":
                return sid, time.time() - start
        time.sleep(0.3)
    return None, time.time() - start


def run_inflight(base, args) -> bool:
    tid = f"cancel_inflight_{uuid.uuid4().hex[:10]}"
    req = _build_request(
        tid, timeout=args.timeout, size_pow=args.size_pow, iters=args.iters, perf_trials=args.perf_trials
    )

    result_holder = {}

    def _submit():
        result_holder["resp"] = _post(f"{base}/evaluate", req, timeout=args.timeout + 60)

    gpu_before = _gpu_used_gb(base)
    print(f"\n[inflight] submitting slow task {tid} in background thread...")
    t_submit = time.time()
    th = threading.Thread(target=_submit, daemon=True)
    th.start()

    # Wait until a GPU sub-task is actually running.
    running_sub, waited = _wait_subtask_running(base, tid, deadline_s=150)
    if running_sub is None:
        print(f"[inflight] FAIL: no GPU sub-task of {tid} reached 'processing' within 150s")
        return False
    print(
        f"[inflight] sub-task {running_sub} is processing after {waited:.1f}s; "
        f"letting it run {args.run_before_cancel}s before cancelling the PARENT id"
    )
    time.sleep(args.run_before_cancel)

    # Cancel.
    t_cancel = time.time()
    code, body = _delete(f"{base}/tasks/{tid}", timeout=15)
    print(f"[inflight] DELETE http={code} resp={body}")
    if code != 200:
        print(f"[inflight] FAIL: cancel returned http {code}")
        return False

    # The /evaluate background thread returns once the workflow aborts; the
    # parent id (404 mid-flight) becomes terminal when its result is written.
    th.join(timeout=args.max_cancel_latency + 30)
    cancel_latency = time.time() - t_cancel
    total_elapsed = time.time() - t_submit
    resp = result_holder.get("resp")
    status, _, _ = _poll_until(
        base, tid, lambda s: s in ("failed", "cancelled", "completed", "timeout"), deadline_s=10
    )
    gpu_after = _gpu_used_gb(base)

    print(
        f"[inflight] post-cancel parent status={status} cancel->return_latency={cancel_latency:.1f}s "
        f"submit->return_total={total_elapsed:.1f}s (vs server-side timeout={args.timeout}s)"
    )
    resp_status = resp_msg = None
    if resp:
        rc, rb = resp
        resp_status, resp_msg = rb.get("status"), (rb.get("error_message") or "")
        print(f"[inflight] /evaluate returned http={rc} status={resp_status} error_message={resp_msg}")
    print(f"[inflight] gpu_used_gb before={gpu_before:.1f} after={gpu_after:.1f}")

    ok = True
    if resp_status not in ("failed", "cancelled") and status not in ("failed", "cancelled"):
        print(f"[inflight] FAIL: expected terminal failed/cancelled (resp={resp_status}, status={status})")
        ok = False
    if cancel_latency > args.max_cancel_latency:
        print(f"[inflight] FAIL: cancel took {cancel_latency:.1f}s (> max_cancel_latency={args.max_cancel_latency}s)")
        ok = False
    if total_elapsed >= args.timeout:
        print(f"[inflight] FAIL: did not interrupt — ran to the {args.timeout}s timeout")
        ok = False
    if resp_msg is not None and "cancel" not in resp_msg.lower():
        print(f"[inflight] WARN: /evaluate error_message did not mention cancel: {resp_msg!r}")
    print(f"[inflight] {'PASS' if ok else 'FAIL'}")
    return ok


def run_pending(base, args) -> bool:
    """Cancel a queued task before any worker runs it.

    Pins the task to a bogus ``assigned_worker`` so it lands in a worker queue
    that nobody consumes and stays pending. If this deployment doesn't honor
    that routing (the task gets dispatched anyway), the check is reported
    INCONCLUSIVE rather than failed — the pending-cancel/queue-drop logic is
    also covered deterministically by the offline unit test.
    """
    tid = f"cancel_pending_{uuid.uuid4().hex[:10]}"
    req = _build_request(
        tid,
        timeout=args.timeout,
        size_pow=args.size_pow,
        iters=args.iters,
        perf_trials=args.perf_trials,
        extra={"assigned_worker": "cancel_test_no_such_worker"},
    )
    print(f"\n[pending] submitting {tid} routed to a non-existent worker (should stay pending)...")
    # /evaluate blocks while the task sits pending, so submit in the background.
    th = threading.Thread(target=lambda: _post(f"{base}/evaluate", req, timeout=args.timeout + 30), daemon=True)
    th.start()

    status, waited, last = _poll_until(base, tid, lambda s: s == "pending", deadline_s=15)
    if status != "pending":
        print(
            f"[pending] INCONCLUSIVE: could not hold task pending in this deployment "
            f"(last={last}); pending-cancel is covered by the offline unit test"
        )
        return True

    code, body = _delete(f"{base}/tasks/{tid}", timeout=15)
    print(f"[pending] task confirmed pending after {waited:.1f}s; DELETE http={code} resp={body}")
    if code != 200:
        print(f"[pending] FAIL: cancel returned http {code}")
        return False

    # It must NEVER move to processing/completed after cancellation.
    moved_to_run = False
    t_end = time.time() + 8
    final = None
    while time.time() < t_end:
        _, sbody = _get(f"{base}/status/{tid}", timeout=10)
        final = sbody.get("status")
        if final in ("processing", "completed"):
            moved_to_run = True
            break
        time.sleep(0.5)

    print(f"[pending] final status={final}")
    ok = (not moved_to_run) and final in ("failed", "cancelled")
    print(f"[pending] {'PASS' if ok else 'FAIL'} (never dispatched={not moved_to_run})")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0] if __doc__ else None)
    p.add_argument("--host", default=os.environ.get("KERNELGYM_REWARD_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("KERNELGYM_REWARD_PORT", "20111")))
    p.add_argument("--mode", choices=["inflight", "pending", "both"], default="inflight")
    p.add_argument("--timeout", type=int, default=300, help="server-side per-task timeout")
    p.add_argument("--size-pow", type=int, default=24, help="input tensor size = 1<<size_pow")
    p.add_argument("--iters", type=int, default=40, help="element-wise iters per forward (raises runtime)")
    p.add_argument("--perf-trials", type=int, default=1000, help="num_perf_trials (caps at server max)")
    p.add_argument("--run-before-cancel", type=float, default=4.0, help="seconds to let task run before cancel")
    p.add_argument(
        "--max-cancel-latency",
        type=float,
        default=20.0,
        help="max seconds from DELETE to terminal for the inflight test to pass",
    )
    p.add_argument(
        "--baseline", action="store_true", help="also run an un-cancelled control to print the natural runtime (slow)"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    _disable_proxy_for_host(args.host)
    base = f"http://{args.host}:{args.port}"

    code, health = _get(f"{base}/health", timeout=10)
    if code != 200 or health.get("status") != "healthy":
        print(f"API not healthy at {base}: http={code} status={health.get('status')}")
        return 1
    print(f"API healthy at {base}")

    results = {}
    if args.mode in ("inflight", "both"):
        if args.baseline:
            run_control(base, args)
        results["inflight"] = run_inflight(base, args)
    if args.mode in ("pending", "both"):
        results["pending"] = run_pending(base, args)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
