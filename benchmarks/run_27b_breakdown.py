#!/usr/bin/env python3
"""Drive the 74-problem paired 3-binding reward-time breakdown.

For every problem_id covered by all three bindings (extracted by
``benchmarks/data_27b/extract.py``, see the manifest there), submit
its (cuda_agent, pybind11, tvm_ffi) rollouts to the .40 reward
service back-to-back. The three submissions for the same problem
happen within seconds of each other, so OS / FS / page-cache state
is comparable across the bindings — removing the "first binding
pays the cold-cache penalty" confound that an all-of-A-then-all-of-B
order would introduce.

Single shot per (sample, binding). force_refresh=True so the
per-request result cache is bypassed and we measure real compile +
correctness + perf work each time.

Resume support: if a result file already contains a (binding, uid)
row, that (binding, sample) pair is skipped. To start fresh, delete
the result files or use a new ``--tag``.

Optional ``--seed`` shuffles the problem-index order with that seed
before iterating, so any residual page-cache effects from earlier
samples in the run are randomized rather than systematic.

Usage:

    python benchmarks/run_27b_breakdown.py                  # 74 problems × 3
    python benchmarks/run_27b_breakdown.py --limit 3        # smoke test
    python benchmarks/run_27b_breakdown.py --binding cuda_agent  # one binding only
    python benchmarks/run_27b_breakdown.py --seed 2026      # shuffled order
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import lzma
import os
import random
import sys
import threading
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
        # ninja internals (cuda_agent / pybind11 only — tvm_ffi uses tvm_ffi.cpp.build)
        "manual_ninja_build_wall_sec": ct.get("manual_ninja_build_wall_sec"),
        "manual_ninja_import_wall_sec": ct.get("manual_ninja_import_wall_sec"),
        "build_backend": md.get("build_backend"),
        # Trial counts — recorded so the same JSONL can prove every
        # binding ran the same correctness + perf machinery (codex
        # audit caveat — previously these were only retrievable from
        # the .40 Redis metadata, not from the JSONL itself).
        "num_correct_trials": md.get("num_correct_trials"),
        "correctness_trials_run": md.get("correctness_trials_run"),
        "kg_kernel_perf_num_trials": md.get("kg_kernel_perf_num_trials"),
        "kg_kernel_perf_num_warmup": md.get("kg_kernel_perf_num_warmup"),
        "kg_kernel_perf_num_profile_trials": md.get("kg_kernel_perf_num_profile_trials"),
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
    """Read 74 samples for ``binding``. Prefers xz; falls back to .jsonl."""
    xz_path = DATA_DIR / f"samples_{binding}.jsonl.xz"
    plain_path = DATA_DIR / f"samples_{binding}.jsonl"
    if xz_path.is_file():
        opener = lambda: lzma.open(xz_path, "rt", encoding="utf-8")  # noqa: E731
    elif plain_path.is_file():
        opener = lambda: plain_path.open(encoding="utf-8")  # noqa: E731
    else:
        sys.exit(f"missing samples file: {xz_path}")
    rows: list[dict] = []
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_done_uids(path: Path) -> set[str]:
    """Set of uids already present in the per-binding output file."""
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            uid = rec.get("uid")
            if isinstance(uid, str):
                done.add(uid)
    return done


def _print_one(rec: dict, idx: int, total: int) -> None:
    elapsed = rec.get("elapsed_s")
    status = rec.get("status")
    compiled = rec.get("compiled")
    correct = rec.get("correctness")
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
    print(
        f"[{rec['binding']:10s}] [{idx:>3d}/{total}] pid={rec['problem_id']:3d} "
        f"elapsed={elapsed}s status={status} {flag_str} "
        f"compile={compile_s}s perf={perf_s}s",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default=os.environ.get("KERNELGYM_REWARD_HOST", "192.168.16.40"))
    p.add_argument("--port", type=int, default=int(os.environ.get("KERNELGYM_REWARD_PORT", "20111")))
    p.add_argument("--timeout", type=int, default=240, help="per-task server-side timeout (s)")
    p.add_argument(
        "--binding",
        choices=list(ALL_BINDINGS),
        action="append",
        default=None,
        help="restrict to one or more bindings (default: all three interleaved)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of problems processed (smoke test)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="shuffle the problem-index order with this seed (default: keep extractor order)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"where to write per-binding JSONL (default: {RESULTS_DIR})",
    )
    p.add_argument("--no-health", action="store_true")
    p.add_argument("--tag", default=None, help="output filename suffix (default: timestamp)")
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help=(
            "number of in-flight /evaluate requests at once (default: 8 — "
            "matches the v1 GPU-worker count, so each warm-pool subprocess "
            "sees ≤1 task in flight). Different (binding, problem) requests "
            "do not share cache entries and run on different GPUs, so "
            "parallel dispatch does not bias the timing comparison."
        ),
    )
    args = p.parse_args()

    _disable_proxy(args.host)
    if not args.no_health:
        _health_probe(args.host, args.port)

    bindings = args.binding or list(ALL_BINDINGS)
    per_binding_samples = {b: _load_samples(b) for b in bindings}

    # Align by problem_id: every binding's samples list is already in
    # extractor-emitted order (sorted by problem_id), so this is a
    # paranoia check, not a transform.
    lengths = {b: len(per_binding_samples[b]) for b in bindings}
    if len(set(lengths.values())) != 1:
        sys.exit(f"# binding sample counts differ — extractor output drifted? {lengths}")
    if set(bindings) == set(ALL_BINDINGS):
        pids = {b: [s["problem_id"] for s in per_binding_samples[b]] for b in bindings}
        ref = pids[bindings[0]]
        for b in bindings[1:]:
            if pids[b] != ref:
                sys.exit(f"# problem_id alignment mismatch between bindings ({b} vs {bindings[0]})")

    n_problems = lengths[bindings[0]]
    order = list(range(n_problems))
    if args.seed is not None:
        random.Random(args.seed).shuffle(order)
    if args.limit is not None:
        order = order[: args.limit]

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {b: args.results_dir / f"{tag}_27b_breakdown_{b}.jsonl" for b in bindings}
    done = {b: _load_done_uids(out_paths[b]) for b in bindings}
    handles = {b: out_paths[b].open("a", encoding="utf-8") for b in bindings}

    # Build the full submission list in (problem, binding) interleave
    # order — same uid order as before, just flattened so the thread
    # pool can dispatch from a single queue while preserving the
    # original "same problem near in time across bindings" intent.
    submissions: list[dict] = []
    for prob_index in order:
        for binding in bindings:
            sample = per_binding_samples[binding][prob_index]
            if sample["uid"] in done[binding]:
                # Will be skipped at dispatch time; keep in queue so
                # the per-pair index counter stays predictable.
                pass
            submissions.append(sample)

    total_pairs = len(submissions)
    print(
        f"# interleaved order: {len(order)} problems × {len(bindings)} bindings "
        f"= {total_pairs} submissions; concurrency={args.concurrency} "
        f"seed={args.seed} tag={tag}",
        file=sys.stderr,
        flush=True,
    )

    write_lock = threading.Lock()

    def _process(idx_sample: tuple[int, dict]) -> dict:
        idx, sample = idx_sample
        binding = sample["binding"]
        if sample["uid"] in done[binding]:
            return {
                "_skipped": True,
                "_idx": idx,
                "binding": binding,
                "problem_id": sample["problem_id"],
                "uid": sample["uid"],
            }
        try:
            rec = _run_one(sample, host=args.host, port=args.port, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001
            rec = {
                "uid": sample["uid"],
                "problem_id": sample["problem_id"],
                "sample_id": sample["sample_id"],
                "binding": binding,
                "backend": sample["backend"],
                "http_status": 0,
                "timestamp_unix": time.time(),
                "elapsed_s": None,
                "status": "runner-exception",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        rec["_idx"] = idx
        return rec

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(_process, (i, s)) for i, s in enumerate(submissions, start=1)]
            for fut in concurrent.futures.as_completed(futs):
                rec = fut.result()
                idx = rec.pop("_idx")
                binding = rec["binding"]
                if rec.get("_skipped"):
                    print(
                        f"[{binding:10s}] [{idx:>3d}/{total_pairs}] pid={rec['problem_id']:3d} "
                        f"(skipped: already in {out_paths[binding].name})",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                with write_lock:
                    handles[binding].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    handles[binding].flush()
                    done[binding].add(rec["uid"])
                _print_one(rec, idx, total_pairs)
    finally:
        for h in handles.values():
            h.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
