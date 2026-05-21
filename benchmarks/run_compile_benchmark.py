#!/usr/bin/env python3
"""Compile-speed benchmark runner for the reward service.

Sends a sequence of /evaluate requests against a running service, captures
all server-side timing fields, and prints one JSON record per run on stdout.
Supports both cuda_agent and tvm_ffi backends via parallel fixtures under
``benchmarks/kernels/``.

Typical use:

    # Default scenario: cold -> warm -> 2 novels -> warm again
    python benchmarks/run_compile_benchmark.py --backend cuda_agent
    python benchmarks/run_compile_benchmark.py --backend tvm_ffi

    # Persist outputs for later comparison
    python benchmarks/run_compile_benchmark.py --backend tvm_ffi \\
        --scenario sequence \\
        --out benchmarks/results/$(date +%Y%m%d_%H%M%S)_tvm_ffi.jsonl

The runner intentionally sends ``force_refresh: True`` so the per-task result
cache is always bypassed; what you measure is real compile + load + profile
behaviour, plus whatever lower-level caches (object cache / compile artifact
cache) the service has enabled.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCENARIOS: dict[str, list[tuple[str, str | None]]] = {
    # (label, mutate_tag)  — mutate_tag=None means "use unmutated fixture"
    "sequence": [
        ("identical-1", None),
        ("identical-2", None),
        ("mutant-1", "unique-A"),
        ("mutant-2", "unique-B"),
        ("identical-3", None),
    ],
    "warm-only": [
        ("warm-1", None),
        ("warm-2", None),
        ("warm-3", None),
    ],
    "novel": [
        ("novel-1", "fresh-1"),
        ("novel-2", "fresh-2"),
        ("novel-3", "fresh-3"),
    ],
}


def _http_get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, body: dict, timeout: float) -> tuple[int, dict]:
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
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            payload = {"error": str(exc)}
        return exc.code, payload


def _disable_proxy(host: str) -> None:
    existing = os.environ.get("no_proxy", "")
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if "*" not in parts and host not in parts:
        parts.append(host)
    os.environ["no_proxy"] = ",".join(parts)
    os.environ["NO_PROXY"] = os.environ["no_proxy"]


def _load_fixture(backend: str) -> tuple[str, str, str]:
    mod_name = {
        "cuda_agent": "kernels.cuda_agent_vector_add",
        "tvm_ffi": "kernels.tvm_ffi_vector_add",
    }[backend]
    mod = importlib.import_module(mod_name)
    return mod.REFERENCE_CODE, mod.KERNEL_CODE, mod.BACKEND


def _mutate(kernel_code: str, mutate_tag: str) -> str:
    """Insert a unique comment in the CUDA source so the compile-cache key
    changes. Triggers a cold compile while keeping the kernel semantically
    identical to the original (the comment is non-functional)."""
    marker = "__global__"
    return kernel_code.replace(marker, f"// bench-mutant-{mutate_tag}\n{marker}", 1)


def _build_request(
    backend: str,
    task_id: str,
    timeout: int,
    reference_code: str,
    kernel_code: str,
) -> dict:
    return {
        "task_id": task_id,
        "reference_code": reference_code,
        "kernel_code": kernel_code,
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": backend,
        "num_correct_trials": 3,
        "num_perf_trials": 20,
        "num_warmup": 3,
        "perf_trim_count": 0,
        "timeout": timeout,
        "priority": "normal",
        "entry_point": "Model",
        "force_refresh": True,
        "run_performance": True,
    }


def _summarize(body: dict | None, sent_at: float) -> dict:
    """Extract the canonical timing fields exposed by the reward service."""
    if not isinstance(body, dict):
        body = {}
    md = body.get("metadata") or {}
    ct = md.get("compile_timing") or {}
    oc = ct.get("manual_ninja_object_cache") or {}
    return {
        "elapsed_s": round(time.time() - sent_at, 3),
        "status": body.get("status"),
        "compiled": body.get("compiled"),
        "correctness": body.get("correctness"),
        "speedup": body.get("speedup"),
        "reference_runtime_ms": body.get("reference_runtime"),
        "kernel_runtime_ms": body.get("kernel_runtime"),
        # task / pool
        "kg_kernel_total_s": md.get("kg_kernel_total_s"),
        "kg_kernel_backend_compile_s": md.get("kg_kernel_backend_compile_s"),
        "kg_kernel_backend_load_s": md.get("kg_kernel_backend_load_s"),
        "kg_kernel_performance_step_s": md.get("kg_kernel_performance_step_s"),
        "kg_kernel_correctness_s": md.get("kg_kernel_correctness_s"),
        "kg_reference_total_s": md.get("kg_reference_total_s"),
        "wg_pool_total_s": md.get("wg_pool_total_s"),
        # caches
        "compile_artifact_cache_enabled": md.get("compile_artifact_cache_enabled"),
        "compile_artifact_cache_hit": md.get("compile_artifact_cache_hit"),
        "object_cache_hits": oc.get("hits"),
        "object_cache_misses": oc.get("misses"),
        "object_cache_skipped": (len(oc.get("skipped") or []) if oc else None),
        # ninja internals (cuda_agent manual ninja only)
        "manual_ninja_build_wall_sec": ct.get("manual_ninja_build_wall_sec"),
        "manual_ninja_import_wall_sec": ct.get("manual_ninja_import_wall_sec"),
        "build_backend": md.get("build_backend"),
        # diag
        "error_message": body.get("error_message"),
    }


def run_once(
    *,
    host: str,
    port: int,
    timeout: int,
    backend: str,
    label: str,
    mutate_tag: str | None,
) -> dict:
    reference, kernel, real_backend = _load_fixture(backend)
    if mutate_tag is not None:
        kernel = _mutate(kernel, mutate_tag)
    task_id = f"bench_{backend}_{uuid.uuid4().hex[:12]}"
    payload = _build_request(real_backend, task_id, timeout, reference, kernel)
    sent_at = time.time()
    http_code, body = _http_post_json(f"http://{host}:{port}/evaluate", payload, timeout + 60)
    summary = _summarize(body, sent_at)
    return {
        "label": label,
        "backend": real_backend,
        "mutate_tag": mutate_tag,
        "task_id": task_id,
        "host": host,
        "port": port,
        "http_status": http_code,
        "timestamp_unix": time.time(),
        **summary,
    }


def _print_pretty(rec: dict) -> None:
    cache = []
    if rec.get("compile_artifact_cache_hit"):
        cache.append("artifact-HIT")
    if rec.get("object_cache_hits") or rec.get("object_cache_misses"):
        cache.append(
            f"oc {rec.get('object_cache_hits')}/{rec.get('object_cache_misses')}/{rec.get('object_cache_skipped')}"
        )
    cache_str = ("[" + " ".join(cache) + "]") if cache else ""
    print(
        f"[{rec['label']:<11s}] elapsed={rec.get('elapsed_s'):>6}s  "
        f"compile={rec.get('kg_kernel_backend_compile_s')}s  "
        f"perf={rec.get('kg_kernel_performance_step_s')}s  "
        f"status={rec.get('status')}  {cache_str}",
        file=sys.stderr,
    )


def health_probe(host: str, port: int) -> None:
    try:
        d = _http_get_json(f"http://{host}:{port}/health", timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"# health: DOWN ({type(exc).__name__}: {exc})", file=sys.stderr)
        sys.exit(2)
    status = d.get("status", "?")
    gpus = d.get("gpu_status", {}) or {}
    ok = sum(1 for v in gpus.values() if isinstance(v, dict) and v.get("available"))
    print(f"# health: {status} gpus={ok}/{len(gpus)}", file=sys.stderr)
    if status != "healthy":
        sys.exit(2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default=os.environ.get("KERNELGYM_REWARD_HOST", "192.168.16.40"))
    p.add_argument("--port", type=int, default=int(os.environ.get("KERNELGYM_REWARD_PORT", "20111")))
    p.add_argument("--timeout", type=int, default=240, help="server-side per-task timeout (seconds)")
    p.add_argument("--backend", required=True, choices=["cuda_agent", "tvm_ffi"])
    p.add_argument("--scenario", default="sequence", choices=list(SCENARIOS.keys()))
    p.add_argument("--out", default=None, help="append JSONL output to this file (still printed on stdout)")
    p.add_argument("--no-health", action="store_true", help="skip /health probe")
    args = p.parse_args()

    _disable_proxy(args.host)
    if not args.no_health:
        health_probe(args.host, args.port)

    runs = SCENARIOS[args.scenario]
    out_fh = open(args.out, "a") if args.out else None
    print(
        f"# scenario={args.scenario} backend={args.backend} host={args.host}:{args.port} timeout={args.timeout}",
        file=sys.stderr,
    )

    try:
        for label, mutate_tag in runs:
            rec = run_once(
                host=args.host,
                port=args.port,
                timeout=args.timeout,
                backend=args.backend,
                label=label,
                mutate_tag=mutate_tag,
            )
            line = json.dumps(rec, separators=(",", ":"))
            print(line)
            if out_fh is not None:
                out_fh.write(line + "\n")
                out_fh.flush()
            _print_pretty(rec)
    finally:
        if out_fh is not None:
            out_fh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
