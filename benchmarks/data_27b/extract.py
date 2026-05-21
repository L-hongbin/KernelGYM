#!/usr/bin/env python3
"""Extract 100 real 27B rollouts per binding style.

Source: the ``cuda-qwen36-27b-l1fullset-mixedauto-t180opt-...`` 27B
offline-eval run in the upstream ``KernelGYM-vllm018-cuda-agent`` repo.
That single run uses the ``mixedauto`` backend, meaning the model is
free to pick any of three submission shapes per problem. In practice
the 800 turn-1 responses break down roughly:

  * ~136 explicit ``PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){...}`` only
    (the "cuda_agent" submission shape — the user's own binding TU is
    the only thing compiled)
  * ~403 ``REGISTER_BINDING(name, register_fn)`` only (the "pybind11"
    framework-scaffold shape — the framework writes an extra
    binding.cpp + binding_registry.h, two host C++ TUs get compiled)
  * ~182 ``TVM_FFI_DLL_EXPORT_TYPED_FUNC(...)`` (the "tvm_ffi" shape —
    routed through the tvm_ffi backend)

For each shape we pull 100 distinct turn-1 responses, AS-EMITTED by
the model (no mechanical rewriting between shapes), pair them with
the ``reference.py`` problem source from ``eval_outputs/``, and write
three JSONL files:

  benchmarks/data_27b/samples_cuda_agent.jsonl.xz
  benchmarks/data_27b/samples_pybind11.jsonl.xz
  benchmarks/data_27b/samples_tvm_ffi.jsonl.xz

Files are xz-compressed (~600-720 KB each, well under git pre-commit
size limits) and decoded by the runner via stdlib ``lzma``.

Each row has the minimum fields ``run_27b_breakdown.py`` needs:

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

Idempotent. Re-running overwrites the JSONL files but the picks are
deterministic given a fixed run path + a sorted iteration order.
"""

from __future__ import annotations

import argparse
import json
import lzma
import re
from pathlib import Path
from typing import Iterator

UPSTREAM_REPO = Path("/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent")
DEFAULT_RUN = (
    UPSTREAM_REPO
    / "drkernel/logs/cuda-qwen36-27b-l1fullset-mixedauto-t180opt-n8-tp2-seqs16-32k-24-r1.run.20260509-180712"
)

OUT_DIR = Path(__file__).resolve().parent

PER_BINDING_TARGET = 100

# Regex for detecting binding style on a turn-1 response. The mixedauto
# model occasionally emits BOTH PYBIND11_MODULE and REGISTER_BINDING in
# the same response — we count that as ambiguous and skip it.
_PYBIND11_MODULE_RE = re.compile(r"\bPYBIND11_MODULE\s*\(")


def classify_binding(response: str) -> str | None:
    has_pyb_module = bool(_PYBIND11_MODULE_RE.search(response))
    has_register = "REGISTER_BINDING" in response
    has_tvm_ffi = "TVM_FFI_DLL_EXPORT_TYPED_FUNC" in response
    if has_tvm_ffi:
        return "tvm_ffi"
    if has_pyb_module and not has_register:
        return "cuda_agent"
    if has_register and not has_pyb_module:
        return "pybind11"
    # Either ambiguous (both pyb_module + register_binding) or neither
    # — not a clean exemplar, skip.
    return None


def build_uid_to_outputs_dir(run_root: Path) -> dict[str, Path]:
    """Build {uid -> eval_outputs/problem_X_sample_Y/} index.

    Each summary.json in those dirs carries the uid we need to join
    with graded_results_conversations.jsonl.
    """
    eval_outputs = run_root / "eval_results/step_0/eval_outputs"
    if not eval_outputs.is_dir():
        raise FileNotFoundError(f"missing eval_outputs at {eval_outputs}")
    mapping: dict[str, Path] = {}
    for problem_dir in sorted(eval_outputs.iterdir()):
        if not problem_dir.is_dir():
            continue
        summary_path = problem_dir / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        uid = summary.get("uid")
        if isinstance(uid, str):
            mapping[uid] = problem_dir
    return mapping


def parse_problem_sample_id(folder_name: str) -> tuple[int, int] | None:
    m = re.match(r"problem_(\d+)_sample_(\d+)$", folder_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def iter_turn1(
    graded_jsonl: Path,
) -> Iterator[tuple[str, float, str]]:
    """Yield (uid, total_score, turn1_response) per row in graded_results."""
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


def extract(run_root: Path, out_dir: Path, target: int) -> dict[str, int]:
    graded_jsonl = run_root / "eval_results/step_0/graded_results_conversations.jsonl"
    if not graded_jsonl.is_file():
        raise FileNotFoundError(f"missing graded_results at {graded_jsonl}")

    uid_to_dir = build_uid_to_outputs_dir(run_root)

    picked: dict[str, list[dict]] = {"cuda_agent": [], "pybind11": [], "tvm_ffi": []}
    backend_for = {"cuda_agent": "cuda_agent", "pybind11": "cuda_agent", "tvm_ffi": "tvm_ffi"}
    skipped_no_match = 0
    skipped_no_reference = 0

    for uid, score, response in iter_turn1(graded_jsonl):
        binding = classify_binding(response)
        if binding is None:
            continue
        bucket = picked[binding]
        if len(bucket) >= target:
            # Already full — keep iterating in case other buckets need
            # more, but skip this sample.
            if all(len(picked[b]) >= target for b in picked):
                break
            continue

        problem_dir = uid_to_dir.get(uid)
        if problem_dir is None:
            skipped_no_match += 1
            continue
        ps = parse_problem_sample_id(problem_dir.name)
        if ps is None:
            skipped_no_match += 1
            continue
        problem_id, sample_id = ps

        ref_path = problem_dir / "reference.py"
        if not ref_path.is_file():
            skipped_no_reference += 1
            continue
        try:
            reference_code = ref_path.read_text(encoding="utf-8")
        except OSError:
            skipped_no_reference += 1
            continue

        bucket.append(
            {
                "uid": uid,
                "problem_id": problem_id,
                "sample_id": sample_id,
                "binding": binding,
                "backend": backend_for[binding],
                "score": score,
                "reference_code": reference_code,
                "kernel_code": response,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for binding, rows in picked.items():
        out_path = out_dir / f"samples_{binding}.jsonl.xz"
        with lzma.open(out_path, "wt", encoding="utf-8", preset=9 | lzma.PRESET_EXTREME) as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[binding] = len(rows)

    counts["skipped_no_match"] = skipped_no_match
    counts["skipped_no_reference"] = skipped_no_reference
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--per-binding", type=int, default=PER_BINDING_TARGET)
    args = p.parse_args()

    counts = extract(args.run, args.out_dir, args.per_binding)
    print(json.dumps(counts, indent=2, sort_keys=True))
    targets_ok = all(counts[b] >= args.per_binding for b in ("cuda_agent", "pybind11", "tvm_ffi"))
    return 0 if targets_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
