#!/usr/bin/env python3
"""Extract paired 74-problem 3-binding rollouts from one 27B run.

Source: the ``cuda-qwen36-27b-l1fullset-mixedauto-t180opt-...`` 27B
offline-eval run in the upstream ``KernelGYM-vllm018-cuda-agent`` repo.
KernelBench Level 1 has 100 problems and the run produced 8 rollouts
per problem (800 turn-1 responses total). The mixedauto backend lets
the model pick a submission shape per rollout, so each problem ends
up with some mix of cuda_agent / pybind11 / tvm_ffi style outputs.

Of the 100 problems:

  * 74 have at least one rollout in EACH of the three styles
  * 25 have rollouts in only two of the three
  *  1 has rollouts in only one

To make the cross-binding timing comparison **paired by problem**
(every reward-time difference between bindings is on the same kernel
problem, removing problem-difficulty as a confound), we pick exactly
those 74 covered problems and emit one rollout per (problem, binding).

Outputs three xz-compressed jsonl files, ROW-ALIGNED so row ``i`` in
all three files refers to the same problem_id:

  benchmarks/data_27b/samples_cuda_agent.jsonl.xz   (74 rows)
  benchmarks/data_27b/samples_pybind11.jsonl.xz     (74 rows)
  benchmarks/data_27b/samples_tvm_ffi.jsonl.xz      (74 rows)

Plus a ``manifest.json`` next to them with:

  * absolute upstream run path
  * SHA-256 of upstream ``graded_results_conversations.jsonl``
  * SHA-256 of each emitted sample file
  * classification rule + counts
  * total-rollout breakdown by binding

Per-row schema:

  {
    "uid":            "test_example_...",
    "problem_id":     58,
    "sample_id":      7,
    "binding":        "cuda_agent" | "pybind11" | "tvm_ffi",
    "backend":        "cuda_agent" | "tvm_ffi",
    "score":          0.63,
    "reference_code": "<contents of reference.py>",
    "kernel_code":    "<turn[0].response — three-section LLM output>"
  }

Classification rule (used by ``classify_binding``):

  * has TVM_FFI_DLL_EXPORT_TYPED_FUNC → tvm_ffi
  * has PYBIND11_MODULE (with or without REGISTER_BINDING) → cuda_agent
  * has REGISTER_BINDING only → pybind11
  * neither → skipped

The (PYBIND11_MODULE + REGISTER_BINDING) "both" case is classified as
cuda_agent because at runtime the cuda_agent backend skips writing
its own scaffold whenever the user's APPLY_BINDINGS already contains
a PYBIND11_MODULE block — so the effective compile path is the
explicit / cuda_agent one regardless of any leftover REGISTER_BINDING
boilerplate.

Determinism: for each covered problem we pick the rollout with the
LOWEST ``sample_id`` for each binding. Re-running this extractor on
the same upstream produces byte-identical samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

UPSTREAM_REPO = Path("/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent")
DEFAULT_RUN = (
    UPSTREAM_REPO
    / "drkernel/logs/cuda-qwen36-27b-l1fullset-mixedauto-t180opt-n8-tp2-seqs16-32k-24-r1.run.20260509-180712"
)

OUT_DIR = Path(__file__).resolve().parent

ALL_BINDINGS: tuple[str, ...] = ("cuda_agent", "pybind11", "tvm_ffi")
BACKEND_FOR = {"cuda_agent": "cuda_agent", "pybind11": "cuda_agent", "tvm_ffi": "tvm_ffi"}

_PYBIND11_MODULE_RE = re.compile(r"\bPYBIND11_MODULE\s*\(")


def classify_binding(response: str) -> str | None:
    has_pyb_module = bool(_PYBIND11_MODULE_RE.search(response))
    has_register = "REGISTER_BINDING" in response
    has_tvm_ffi = "TVM_FFI_DLL_EXPORT_TYPED_FUNC" in response
    if has_tvm_ffi:
        return "tvm_ffi"
    if has_pyb_module:
        # Even if REGISTER_BINDING is also present, the runtime backend
        # skips its scaffold when PYBIND11_MODULE is present, so the
        # effective compile shape is the explicit / cuda_agent one.
        return "cuda_agent"
    if has_register:
        return "pybind11"
    return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_uid_to_problem(run_root: Path) -> dict[str, tuple[int, int, Path]]:
    """Build {uid -> (problem_id, sample_id, problem_dir)} from
    eval_outputs/problem_X_sample_Y/summary.json.
    """
    eval_outputs = run_root / "eval_results/step_0/eval_outputs"
    if not eval_outputs.is_dir():
        raise FileNotFoundError(f"missing eval_outputs at {eval_outputs}")
    mapping: dict[str, tuple[int, int, Path]] = {}
    for problem_dir in sorted(eval_outputs.iterdir()):
        if not problem_dir.is_dir():
            continue
        m = re.match(r"problem_(\d+)_sample_(\d+)$", problem_dir.name)
        if not m:
            continue
        problem_id, sample_id = int(m.group(1)), int(m.group(2))
        summary_path = problem_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        uid = summary.get("uid")
        if isinstance(uid, str):
            mapping[uid] = (problem_id, sample_id, problem_dir)
    return mapping


def iter_turn1(graded_jsonl: Path) -> Iterator[tuple[str, float, str]]:
    """Yield (uid, total_score, turn1_response)."""
    with graded_jsonl.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            uid = row.get("uid")
            turns = row.get("turns") or []
            if not isinstance(uid, str) or not turns:
                continue
            t0 = turns[0]
            resp = t0.get("response")
            if not isinstance(resp, str):
                continue
            score = float(row.get("total_score") or 0.0)
            yield uid, score, resp


def extract(run_root: Path, out_dir: Path) -> dict:
    """Pick one rollout per (binding, problem) for problems covered by
    all three bindings. Write 3 row-aligned xz jsonl files + manifest.
    """
    graded_jsonl = run_root / "eval_results/step_0/graded_results_conversations.jsonl"
    if not graded_jsonl.is_file():
        raise FileNotFoundError(f"missing graded_results at {graded_jsonl}")

    uid_to_problem = build_uid_to_problem(run_root)

    # per_problem[problem_id][binding] = list of (sample_id, uid, score, resp, problem_dir)
    per_problem: dict[int, dict[str, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    total_by_binding: dict[str, int] = defaultdict(int)
    classified = 0
    unclassified = 0

    for uid, score, response in iter_turn1(graded_jsonl):
        binding = classify_binding(response)
        if binding is None:
            unclassified += 1
            continue
        problem_meta = uid_to_problem.get(uid)
        if problem_meta is None:
            continue
        problem_id, sample_id, problem_dir = problem_meta
        per_problem[problem_id][binding].append((sample_id, uid, score, response, problem_dir))
        total_by_binding[binding] += 1
        classified += 1

    # Problems covered by all three bindings.
    covered = sorted(pid for pid, b in per_problem.items() if set(b.keys()) >= set(ALL_BINDINGS))

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_paths: dict[str, Path] = {}
    rows_per_binding: dict[str, list[dict]] = {b: [] for b in ALL_BINDINGS}

    for problem_id in covered:
        problem_buckets = per_problem[problem_id]
        for binding in ALL_BINDINGS:
            # Deterministic pick: lowest sample_id.
            sample_id, uid, score, response, problem_dir = min(problem_buckets[binding])
            ref_path = problem_dir / "reference.py"
            try:
                reference_code = ref_path.read_text(encoding="utf-8")
            except OSError:
                # Should never happen because we already saw summary.json,
                # but guard anyway.
                raise
            rows_per_binding[binding].append(
                {
                    "uid": uid,
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "binding": binding,
                    "backend": BACKEND_FOR[binding],
                    "score": score,
                    "reference_code": reference_code,
                    "kernel_code": response,
                }
            )

    for binding in ALL_BINDINGS:
        out_path = out_dir / f"samples_{binding}.jsonl.xz"
        with lzma.open(out_path, "wt", encoding="utf-8", preset=9 | lzma.PRESET_EXTREME) as f:
            for r in rows_per_binding[binding]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sample_paths[binding] = out_path

    manifest = {
        "upstream_run": str(run_root),
        "graded_results_sha256": sha256_of(graded_jsonl),
        "classification_rule": (
            "tvm_ffi if TVM_FFI_DLL_EXPORT_TYPED_FUNC present; "
            "else cuda_agent if PYBIND11_MODULE present (with or without REGISTER_BINDING); "
            "else pybind11 if REGISTER_BINDING present; else skipped."
        ),
        "totals": {
            "rollouts_classified": classified,
            "rollouts_unclassified": unclassified,
            "per_binding_rollouts": dict(total_by_binding),
            "problems_total": len(per_problem),
            "problems_covered_by_all_3": len(covered),
            "samples_per_binding": len(covered),
        },
        "covered_problem_ids": covered,
        "samples": {
            b: {
                "path": str(sample_paths[b].relative_to(out_dir)),
                "sha256": sha256_of(sample_paths[b]),
                "rows": len(rows_per_binding[b]),
            }
            for b in ALL_BINDINGS
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    manifest = extract(args.run, args.out_dir)
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
