#!/usr/bin/env python3
"""Drive the 100-sample-per-binding reward-time breakdown against .40.

For each binding (``cuda_agent`` / ``pybind11`` / ``tvm_ffi``):

  * read ``benchmarks/data_27b/samples_<binding>.jsonl`` (100 rows)
  * for each row, POST ``/evaluate`` with the exact kernel_code the 27B
    model emitted and the matching reference_code from
    ``eval_outputs/.../reference.py``
  * capture the server's full ``compile_timing`` + ``kg_*`` breakdown
  * append one JSON record per sample to a JSONL file under
    ``benchmarks/results/``

Single shot per (sample, binding) — no warmup, no retry. The runner
sends ``force_refresh: True`` so the per-request result cache is
bypassed and we measure real compile + correctness + perf work.

Usage:

    python benchmarks/run_27b_breakdown.py            # all 3 bindings
    python benchmarks/run_27b_breakdown.py --binding cuda_agent
    python benchmarks/run_27b_breakdown.py --limit 5  # smoke test

By default the runner writes to
``benchmarks/results/<timestamp>_27b_breakdown_<binding>.jsonl``.
Each line is a self-contained record so the run is resumable / a
crash mid-way still yields partial data.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_27b"
RESULTS_DIR = ROOT / "results"

ALL_BINDINGS = ("cuda_agent", "pybind11", "tvm_ffi")


def _disable_proxy(host: str) -> None:
    existing = os.environ.get("no_proxy", "")
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if "*" not in parts and host not in parts:
        parts.append(host)
    os.environ["no_proxy"] = ",".join(parts)
    os.environ["NO_PROXY"] = os.environ["no_proxy"]


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
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


def _health_probe(host: str, port: int) -> None:
    try:
        d = _http_get_json(f"http://{host}:{port}/health", timeout=10)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"# health: DOWN ({type(exc).__name__}: {exc})")
    status = d.get("status", "?")
    gpus = d.get("gpu_status", {}) or {}
    ok = sum(1 for v in gpus.values() if isinstance(v, dict) and v.get("available"))
    print(f"# health: {status} gpus={ok}/{len(gpus)}", file=sys.stderr, flush=True)
    if status != "healthy":
        sys.exit(1)


def _build_request(sample: dict, *, timeout: int) -> dict:
    return {
        "task_id": f"bench27b_{sample['binding']}_{sample['uid']}",
        "reference_code": sample["reference_code"],
        "kernel_code": sample["kernel_code"],
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": sample["backend"],
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
    """Extract the canonical timing + outcome fields the reward service exposes."""
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
        # task-level breakdown
        "kg_kernel_total_s": md.get("kg_kernel_total_s"),
        "kg_kernel_backend_compile_s": md.get("kg_kernel_backend_compile_s"),
        "kg_kernel_backend_load_s": md.get("kg_kernel_backend_load_s"),
        "kg_kernel_performance_step_s": md.get("kg_kernel_performance_step_s"),
        "kg_kernel_correctness_s": md.get("kg_kernel_correctness_s"),
        "kg_reference_total_s": md.get("kg_reference_total_s"),
        "wg_pool_total_s": md.get("wg_pool_total_s"),
        # cache state
        "compile_artifact_cache_enabled": md.get("compile_artifact_cache_enabled"),
        "compile_artifact_cache_hit": md.get("compile_artifact_cache_hit"),
        "object_cache_hits": oc.get("hits"),
        "object_cache_misses": oc.get("misses"),
        "object_cache_skipped": (len(oc.get("skipped") or []) if oc else None),
        # ninja internals
        "manual_ninja_build_wall_sec": ct.get("manual_ninja_build_wall_sec"),
        "manual_ninja_import_wall_sec": ct.get("manual_ninja_import_wall_sec"),
        "build_backend": md.get("build_backend"),
        # diag
        "error_message": body.get("error_message"),
        "decoy_kernel": body.get("decoy_kernel"),
    }


def _run_one(sample: dict, *, host: str, port: int, timeout: int) -> dict:
    payload = _build_request(sample, timeout=timeout)
    sent_at = time.time()
    http_code, body = _http_post_json(f"http://{host}:{port}/evaluate", payload, timeout + 60)
    summary = _summarize(body, sent_at)
    return {
        "uid": sample["uid"],
        "problem_id": sample["problem_id"],
        "sample_id": sample["sample_id"],
        "binding": sample["binding"],
        "backend": sample["backend"],
        "original_score": sample.get("score"),
        "http_status": http_code,
        "timestamp_unix": time.time(),
        **summary,
    }


def _load_samples(binding: str) -> list[dict]:
    """Read 100 samples for ``binding``. Prefers the xz-compressed file
    (the canonical tracked artifact); falls back to plain .jsonl if a
    caller has it materialized side-by-side (e.g. mid-run debugging)."""
    xz_path = DATA_DIR / f"samples_{binding}.jsonl.xz"
    plain_path = DATA_DIR / f"samples_{binding}.jsonl"
    if xz_path.is_file():
        opener = lambda: lzma.open(xz_path, "rt", encoding="utf-8")  # noqa: E731
    elif plain_path.is_file():
        opener = lambda: plain_path.open(encoding="utf-8")  # noqa: E731
    else:
        sys.exit(f"missing samples file: {xz_path} (and no .jsonl fallback)")
    rows: list[dict] = []
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _print_one(rec: dict, idx: int, total: int) -> None:
    elapsed = rec.get("elapsed_s")
    status = rec.get("status")
    compiled = rec.get("compiled")
    correct = rec.get("correctness")
    speedup = rec.get("speedup")
    compile_s = rec.get("kg_kernel_backend_compile_s")
    perf_s = rec.get("kg_kernel_performance_step_s")
    flags = []
    if compiled:
        flags.append("compiled")
    if correct:
        flags.append("correct")
    if rec.get("decoy_kernel"):
        flags.append("decoy")
    flag_str = "/".join(flags) or "fail"
    speedup_str = f"speedup={speedup:.2f}" if isinstance(speedup, (int, float)) else ""
    print(
        f"[{rec['binding']:10s}] [{idx:>3d}/{total}] uid={rec['uid'][:14]} "
        f"elapsed={elapsed}s status={status} {flag_str} "
        f"compile={compile_s}s perf={perf_s}s {speedup_str}",
        file=sys.stderr,
        flush=True,
    )


def run_binding(
    binding: str,
    *,
    host: str,
    port: int,
    timeout: int,
    limit: int | None,
    out_path: Path,
) -> int:
    samples = _load_samples(binding)
    if limit is not None:
        samples = samples[:limit]
    total = len(samples)
    print(f"# binding={binding} samples={total} -> {out_path}", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_count = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, sample in enumerate(samples, start=1):
            try:
                rec = _run_one(sample, host=host, port=port, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "uid": sample["uid"],
                    "problem_id": sample["problem_id"],
                    "sample_id": sample["sample_id"],
                    "binding": sample["binding"],
                    "backend": sample["backend"],
                    "http_status": 0,
                    "timestamp_unix": time.time(),
                    "elapsed_s": None,
                    "status": "runner-exception",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
                fail_count += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            _print_one(rec, i, total)
    return fail_count


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default=os.environ.get("KERNELGYM_REWARD_HOST", "192.168.16.40"))
    p.add_argument("--port", type=int, default=int(os.environ.get("KERNELGYM_REWARD_PORT", "20111")))
    p.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="per-task server-side timeout (seconds); HTTP timeout is +60",
    )
    p.add_argument(
        "--binding",
        choices=list(ALL_BINDINGS),
        action="append",
        default=None,
        help="restrict to one or more bindings (default: all three)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of samples per binding (smoke test)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"where to write per-binding JSONL (default: {RESULTS_DIR})",
    )
    p.add_argument("--no-health", action="store_true")
    p.add_argument(
        "--tag",
        default=None,
        help="suffix appended to the per-binding output filename (default: timestamp)",
    )
    args = p.parse_args()

    _disable_proxy(args.host)
    if not args.no_health:
        _health_probe(args.host, args.port)

    bindings = args.binding or list(ALL_BINDINGS)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    total_fail = 0
    for binding in bindings:
        out = args.results_dir / f"{tag}_27b_breakdown_{binding}.jsonl"
        total_fail += run_binding(
            binding,
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            limit=args.limit,
            out_path=out,
        )
    print(f"# done. runner exceptions: {total_fail}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
