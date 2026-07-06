#!/usr/bin/env python3
"""Subdivide output-mismatch failures into near-miss vs gross by re-measuring maxdiff.

The 800-run only stored error="Output mismatch" with no magnitude. This re-runs
each mismatch sample (one seeded forward of reference vs ModelNew, from the saved
review-dir ref/response), computes max abs diff and relative diff, and classifies:
  near_miss : passes torch.allclose(atol=rtol=1e-2)  -> almost right, just over 1e-4
  gross     : fails 1e-2                              -> fundamentally wrong
Subprocess-per-sample (load_inline isolation) + GPU pool, like the rescorer.

Worker mode: --one <reference_file> <response_file>  (prints one JSON line).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_one(ref_file: str, resp_file: str) -> dict:
    import torch
    from kernelgym.toolkit.kernelbench.binding_detection import extract_model_code
    from kernelgym.toolkit.kernelbench.exec_types import set_seed
    from kernelgym.toolkit.kernelbench.loading import load_original_model_and_inputs, load_custom_model

    ref = open(ref_file, encoding="utf-8").read()
    resp = open(resp_file, encoding="utf-8").read()
    clean = extract_model_code(resp)
    ctxr: dict = {}
    Model, gi, ginp = load_original_model_and_inputs(ref, ctxr, "Model")
    set_seed(42)
    init = gi()
    init = [x.cuda() if torch.is_tensor(x) else x for x in init]
    if len(init) > 1 and hasattr(init[0], "__len__") and not isinstance(init[0], (str, torch.Tensor)) and len(init[0]) == 0:
        init = init[1]
    set_seed(42)
    m = (Model(*init) if isinstance(init, list) else Model(**init)).cuda()
    bd = tempfile.mkdtemp(prefix="subdiv_")
    ctxn: dict = {}
    MN = load_custom_model(clean, ctxn, bd)
    set_seed(42)
    mn = (MN(*init) if isinstance(init, list) else MN(**init)).cuda()
    set_seed(42)
    x = ginp()
    x = [t.cuda() if torch.is_tensor(t) else t for t in x]
    with torch.no_grad():
        o = m(*x)
        on = mn(*x)
    if getattr(o, "shape", None) != getattr(on, "shape", None):
        return {"class": "shape_mismatch", "maxdiff": None, "rel": None}
    d = (o.float() - on.float()).abs()
    maxdiff = d.max().item()
    rel = (d.max() / o.float().abs().max().clamp(min=1e-9)).item()
    w12 = bool(torch.allclose(o, on, atol=1e-2, rtol=1e-2))
    w13 = bool(torch.allclose(o, on, atol=1e-3, rtol=1e-3))
    return {
        "class": "near_miss<=1e-2" if w12 else "gross>1e-2",
        "within_1e3": w13,
        "maxdiff": maxdiff,
        "rel": rel,
    }


def _cat(name: str) -> str:
    n = (name or "").lower()
    return "conv" if "conv" in n else "non-conv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=2)
    ap.add_argument("--results", default="/tmp/rescore_800.json")
    ap.add_argument("--review-dir", default="/tmp/rescore_review_800")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="/tmp/mismatch_subdiv.json")
    ap.add_argument("--worktree", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    if args.one:
        try:
            print(json.dumps(run_one(args.one[0], args.one[1])))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"class": "error", "err": f"{type(exc).__name__}: {exc}"[:200]}))
        return

    res = json.load(open(args.results, encoding="utf-8"))["results"]
    mism = [
        x for x in res
        if x.get("compiled") and not x.get("correctness") and not x.get("decoy")
        and "mismatch" in (x.get("error") or "").lower()
    ]
    review = Path(args.review_dir)
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = args.worktree
    num_gpus = max(1, args.num_gpus)
    gpu_pool: "queue.Queue[int]" = queue.Queue()
    for i in range(max(args.workers, num_gpus)):
        gpu_pool.put(i % num_gpus)

    def work(x: dict) -> dict:
        idx, pid = x["idx"], x["problem_id"]
        cd = review / f"idx_{idx:04d}_p{pid}"
        rec = {"idx": idx, "problem_id": pid, "name": x.get("name"), "cat": _cat(x.get("name"))}
        gpu = gpu_pool.get()
        try:
            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [sys.executable, __file__, "--one", str(cd / "reference.py"), str(cd / "response.txt")]
            try:
                p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
                line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
                rec.update(json.loads(line) if line.startswith("{") else {"class": "norun", "err": p.stderr[-200:]})
            except subprocess.TimeoutExpired:
                rec.update({"class": "timeout"})
        finally:
            gpu_pool.put(gpu)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(work, mism))

    from collections import Counter
    klass = Counter(r["class"] for r in out)
    klass_conv = Counter(r["class"] for r in out if r["cat"] == "conv")
    klass_non = Counter(r["class"] for r in out if r["cat"] == "non-conv")
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    n = len(out)
    print(f"=== subdivided {n} output_mismatch samples ===")
    print("overall:", dict(klass))
    print("conv:", dict(klass_conv), "| non-conv:", dict(klass_non))
    nm = sum(1 for r in out if r["class"] == "near_miss<=1e-2")
    gr = sum(1 for r in out if r["class"] == "gross>1e-2")
    print(f"near_miss(<=1e-2)={nm} ({100*nm/n:.1f}%)  gross(>1e-2)={gr} ({100*gr/n:.1f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
