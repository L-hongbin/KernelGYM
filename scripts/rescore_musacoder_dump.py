#!/usr/bin/env python3
"""Re-score MusaCoder generations from a slime dump via KernelGym's load_inline backend.

Extracts (reference, response) pairs from a slime eval dump and scores each in an
isolated subprocess (``score_one_sample.py``) so load_inline JIT builds don't
collide across samples. Aggregates compile / correct / fast@p rates at both the
sample level and the problem level (a problem counts correct if any of its scored
samples is correct), and writes reviewable per-sample evidence.

Designed for incremental debugging: start with ``--num-problems 10``, inspect the
review dir, fix, then 50, then 100. Example (on a GPU host, KernelGym venv active,
PYTHONPATH=<worktree>):

    PYTHONPATH=$WT python scripts/rescore_musacoder_dump.py \
        --num-problems 10 --out /tmp/rescore_10.json --review-dir /tmp/rescore_review
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

DEFAULT_DUMP = (
    "/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/experiments/"
    "EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/"
    "MusaCoder-27B.kb_l1_musa_coder.load_inline.cuda_agent/dumps/rollout_data/eval_0.pt"
)
SAMPLES_PER_PROMPT = 8  # n_samples_per_eval_prompt used when the dump was produced


def _ref_of(sample: dict) -> str | None:
    label = sample.get("label")
    if isinstance(label, dict) and label.get("ground_truth"):
        return label["ground_truth"]
    return (sample.get("metadata") or {}).get("ground_truth")


def _meta(sample: dict):
    meta = sample.get("metadata") or {}
    ei = meta.get("extra_info") or {}
    return meta.get("problem_id") or ei.get("problem_id"), meta.get("name") or ei.get("name")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DEFAULT_DUMP)
    ap.add_argument("--num-problems", type=int, default=10)
    ap.add_argument("--samples-per-problem", type=int, default=1)
    ap.add_argument("--indices-file", default=None, help="JSON list of explicit sample indices (overrides num-problems).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-correct", type=int, default=5)
    ap.add_argument("--num-perf", type=int, default=20)
    ap.add_argument("--backend", default="load_inline", help="KernelGym backend (load_inline | cuda_agent).")
    ap.add_argument("--per-sample-timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=1, help="Concurrent samples (each pinned to one GPU).")
    ap.add_argument("--num-gpus", type=int, default=8, help="GPUs to round-robin pin workers to.")
    ap.add_argument("--out", default="/tmp/rescore_results.json")
    ap.add_argument("--review-dir", default="/tmp/rescore_review")
    ap.add_argument("--worktree", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    obj = torch.load(args.dump, map_location="cpu", weights_only=False)
    samples = obj["samples"] if isinstance(obj, dict) else obj
    worker = str(Path(args.worktree) / "scripts" / "score_one_sample.py")
    review = Path(args.review_dir)
    review.mkdir(parents=True, exist_ok=True)

    if args.indices_file:
        indices = [i for i in json.loads(Path(args.indices_file).read_text()) if 0 <= i < len(samples)]
    else:
        indices = []
        for p in range(1, args.num_problems + 1):
            for k in range(args.samples_per_problem):
                idx = (p - 1) * SAMPLES_PER_PROMPT + k
                if idx < len(samples):
                    indices.append(idx)

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = args.worktree
    total = len(indices)
    done = [0]
    print_lock = threading.Lock()
    num_gpus = max(1, args.num_gpus)
    # GPU pool with one slot per worker, GPUs assigned round-robin. With
    # --workers == num_gpus each sample owns a GPU (cleanest numbers). With
    # --workers > num_gpus, up to ceil(workers/num_gpus) samples share a GPU
    # (e.g. --workers 16 on 8 GPUs = 2 per GPU): each sample's nvcc compile is
    # CPU-bound and leaves its GPU idle, so oversubscribing overlaps compiles
    # with execs and roughly doubles throughput. Trade-off: samples sharing a GPU
    # can OOM the very largest problems (a false failure) and make perf/speedup
    # noisier; prefer --workers == num_gpus for an authoritative run.
    gpu_slots = max(args.workers, num_gpus)
    gpu_pool: "queue.Queue[int]" = queue.Queue()
    for i in range(gpu_slots):
        gpu_pool.put(i % num_gpus)

    def run_one(pos_idx) -> dict:
        pos, idx = pos_idx
        s = samples[idx]
        pid, name = _meta(s)
        reference = _ref_of(s)
        response = s.get("response", "") or ""
        rec = {"idx": idx, "problem_id": pid, "name": name}
        if not reference:
            rec.update(compiled=False, correctness=False, decoy=False, error="no reference in dump")
        else:
            case_dir = review / f"idx_{idx:04d}_p{pid}"
            case_dir.mkdir(parents=True, exist_ok=True)
            ref_file = case_dir / "reference.py"
            resp_file = case_dir / "response.txt"
            ref_file.write_text(reference, encoding="utf-8")
            resp_file.write_text(response, encoding="utf-8")
            # Pin each concurrent sample to one physical GPU (seen as cuda:0 inside
            # the subprocess). Subprocess + per-task build dir already isolate the
            # load_inline build; the GPU pin isolates execution memory.
            gpu = gpu_pool.get()
            try:
                env = dict(base_env)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                cmd = [
                    sys.executable, worker,
                    "--reference-file", str(ref_file),
                    "--response-file", str(resp_file),
                    "--entry-point", "Model",
                    "--device", "cuda:0",
                    "--num-correct", str(args.num_correct),
                    "--num-perf", str(args.num_perf),
                    "--backend", args.backend,
                ]
                try:
                    proc = subprocess.run(
                        cmd, env=env, capture_output=True, text=True, timeout=args.per_sample_timeout
                    )
                    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
                    res = json.loads(line) if line.startswith("{") else {
                        "compiled": False, "correctness": False, "decoy": False,
                        "error": f"no JSON; rc={proc.returncode}; stderr={proc.stderr[-300:]}",
                    }
                except subprocess.TimeoutExpired:
                    res = {"compiled": False, "correctness": False, "decoy": False, "error": "subprocess timeout"}
            finally:
                gpu_pool.put(gpu)
            rec.update(res)
            (case_dir / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        with print_lock:
            done[0] += 1
            print(
                f"[{done[0]}/{total}] idx={idx} pid={pid} {name} "
                f"detected={rec.get('detected_backend')} compiled={rec.get('compiled')} "
                f"correct={rec.get('correctness')} decoy={rec.get('decoy')} "
                f"speedup={rec.get('speedup')} err={(rec.get('error') or '')[:80]}",
                flush=True,
            )
        return rec

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        results = list(ex.map(run_one, list(enumerate(indices))))
    results.sort(key=lambda r: r["idx"])

    # sample-level
    n_s = len(results)
    comp_s = sum(bool(r.get("compiled")) for r in results)
    corr_s = sum(bool(r.get("correctness")) and not bool(r.get("decoy")) for r in results)
    decoy_s = sum(bool(r.get("decoy")) for r in results)
    fast10 = sum(bool(r.get("correctness")) and not bool(r.get("decoy")) and (r.get("speedup") or 0) >= 1.0 for r in results)
    fast12 = sum(bool(r.get("correctness")) and not bool(r.get("decoy")) and (r.get("speedup") or 0) >= 1.2 for r in results)
    # problem-level (any sample correct)
    by_pid: dict = {}
    for r in results:
        by_pid.setdefault(r["problem_id"], []).append(r)
    n_p = len(by_pid)
    comp_p = sum(any(bool(x.get("compiled")) for x in v) for v in by_pid.values())
    corr_p = sum(any(bool(x.get("correctness")) and not bool(x.get("decoy")) for x in v) for v in by_pid.values())

    summary = {
        "dump": args.dump,
        "num_problems": n_p,
        "samples_scored": n_s,
        "sample_level": {
            "compiled": comp_s, "correct": corr_s, "decoy": decoy_s,
            "fast@1.0": fast10, "fast@1.2": fast12,
            "compile_rate": round(comp_s / n_s, 4) if n_s else 0,
            "correct_rate": round(corr_s / n_s, 4) if n_s else 0,
        },
        "problem_level": {
            "compiled_any": comp_p, "correct_any": corr_p,
            "compile_rate": round(comp_p / n_p, 4) if n_p else 0,
            "correct_rate": round(corr_p / n_p, 4) if n_p else 0,
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 80)
    print(f"PROBLEMS={n_p} SAMPLES={n_s}")
    print(f"  sample-level: compiled={comp_s}/{n_s} correct={corr_s}/{n_s} decoy={decoy_s} fast@1.0={fast10} fast@1.2={fast12}")
    print(f"  problem-level(any): compiled={comp_p}/{n_p} correct={corr_p}/{n_p}")
    print(f"wrote {args.out}; per-sample evidence under {args.review_dir}")


if __name__ == "__main__":
    main()
