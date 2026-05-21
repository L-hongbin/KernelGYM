#!/usr/bin/env python3
"""Aggregate the 27B-breakdown JSONL into a per-binding comparison.

Reads ``benchmarks/results/<tag>_27b_breakdown_<binding>.jsonl`` for
each binding and emits:

  * Outcome distribution table (counts per status, plus
    compiled/correctness/decoy/positive-speedup rates).
  * Two percentile tables per timing field:
      - "completed-only" — over samples with status==completed
      - "all-attempts (censored at server timeout)" — every sample is
        counted; timeouts use the timeout value, runner exceptions
        and pure failures use their elapsed_s, all clamped at
        ``--censor-at`` seconds. Headlining "mean elapsed per
        attempt" should always reference THIS table.
  * Per-binding residual:
      kernel_residual_s = kg_kernel_total_s
                          - (backend_compile + backend_load
                             + correctness + performance)
    A non-trivial residual means the breakdown is missing a phase.
  * Backend-specific diagnostics section (manual_ninja_*) — these
    only apply to cuda_agent / pybind11 (manual ninja path), not
    tvm_ffi (uses ``tvm_ffi.cpp.build``). Reported separately so they
    are not mistaken for a cross-binding comparison metric.

Dedupes rows by (binding, uid) keeping the last occurrence — so a
crashed-and-resumed run with the same tag does not double-count.

Usage:

    python benchmarks/aggregate_27b.py --tag fullrun
    python benchmarks/aggregate_27b.py --tag fullrun --markdown > table.md
    python benchmarks/aggregate_27b.py --tag fullrun --json > table.json
    python benchmarks/aggregate_27b.py --tag fullrun --censor-at 240
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

ALL_BINDINGS = ("cuda_agent", "pybind11", "tvm_ffi")

# Cross-binding-safe metrics: present on every backend.
CROSS_BINDING_FIELDS = (
    "elapsed_s",
    "kg_kernel_total_s",
    "kg_kernel_backend_compile_s",
    "kg_kernel_backend_load_s",
    "kg_kernel_performance_step_s",
    "kg_kernel_correctness_s",
    "kg_reference_total_s",
    "wg_pool_total_s",
)

# Backend-specific (manual ninja only — absent for tvm_ffi).
BACKEND_SPECIFIC_FIELDS = (
    "manual_ninja_build_wall_sec",
    "manual_ninja_import_wall_sec",
)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "mean": None}
    return {
        "n": len(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "mean": statistics.fmean(values),
    }


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    by_uid: dict[str, dict] = {}
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
            if not isinstance(uid, str):
                continue
            # Dedupe by uid keeping the LAST seen row (so a resumed
            # run with the same tag overrides earlier partial rows).
            by_uid[uid] = rec
    return list(by_uid.values())


def _kernel_residual(rec: dict) -> float | None:
    total = rec.get("kg_kernel_total_s")
    parts = [
        rec.get("kg_kernel_backend_compile_s"),
        rec.get("kg_kernel_backend_load_s"),
        rec.get("kg_kernel_correctness_s"),
        rec.get("kg_kernel_performance_step_s"),
    ]
    if not isinstance(total, (int, float)):
        return None
    if not all(isinstance(p, (int, float)) for p in parts):
        return None
    return float(total) - sum(parts)  # type: ignore[arg-type]


def _summarize(rows: list[dict], *, censor_at: float) -> dict:
    total = len(rows)
    status_counts: dict[str, int] = {}
    compiled = 0
    correctness = 0
    decoy = 0
    speedup_pos = 0
    speedups_compiled: list[float] = []

    completed_rows: list[dict] = []

    for r in rows:
        st = r.get("status") or "unknown"
        status_counts[st] = status_counts.get(st, 0) + 1
        if r.get("compiled"):
            compiled += 1
        if r.get("correctness"):
            correctness += 1
        if r.get("decoy_kernel"):
            decoy += 1
        sp = r.get("speedup")
        if isinstance(sp, (int, float)):
            if sp > 0:
                speedup_pos += 1
            if r.get("compiled"):
                speedups_compiled.append(float(sp))
        if st == "completed":
            completed_rows.append(r)

    timing_completed: dict[str, dict] = {}
    timing_censored: dict[str, dict] = {}
    for fld in CROSS_BINDING_FIELDS:
        # Completed-only: values from rows where status == completed.
        completed_vals = [r[fld] for r in completed_rows if isinstance(r.get(fld), (int, float))]
        timing_completed[fld] = _stats(completed_vals)

        # All-attempts censored: every row contributes. Missing field
        # in a non-completed row is imputed from elapsed_s (since for
        # failures the request still consumed elapsed_s wall time at
        # the reward layer; per-phase fields are not meaningful, but
        # we fall back to elapsed_s for elapsed_s itself and skip for
        # phase fields).
        censored_vals: list[float] = []
        for r in rows:
            v = r.get(fld)
            if isinstance(v, (int, float)):
                censored_vals.append(min(float(v), censor_at))
            elif fld == "elapsed_s":
                # Use elapsed_s when present even on failures; final
                # ceiling at censor_at.
                ev = r.get("elapsed_s")
                if isinstance(ev, (int, float)):
                    censored_vals.append(min(float(ev), censor_at))
        timing_censored[fld] = _stats(censored_vals)

    timing_backend_specific: dict[str, dict] = {}
    for fld in BACKEND_SPECIFIC_FIELDS:
        vals = [r[fld] for r in completed_rows if isinstance(r.get(fld), (int, float))]
        timing_backend_specific[fld] = _stats(vals)

    residuals = [r for r in (_kernel_residual(rec) for rec in completed_rows) if r is not None]

    return {
        "total": total,
        "completed": len(completed_rows),
        "status_counts": status_counts,
        "compiled": compiled,
        "correctness": correctness,
        "decoy_kernel": decoy,
        "speedup_positive": speedup_pos,
        "speedup_compiled_mean": (statistics.fmean(speedups_compiled) if speedups_compiled else None),
        "speedup_compiled_p50": _percentile(speedups_compiled, 0.50) if speedups_compiled else None,
        "timing_completed": timing_completed,
        "timing_censored": timing_censored,
        "timing_backend_specific": timing_backend_specific,
        "kernel_residual_stats": _stats(residuals),
    }


def _fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}s"


def _render_markdown(summaries: dict[str, dict], *, tag: str, censor_at: float) -> str:
    lines: list[str] = []
    lines.append(f"# 27B 3-binding breakdown — tag `{tag}`")
    lines.append("")
    lines.append(
        f"Per binding: paired by problem_id (same KernelBench Level 1 "
        f"problems across all three). Censored timing percentiles "
        f"clamp at {censor_at:.0f}s (the server-side per-task timeout)."
    )
    lines.append("")

    lines.append("## Outcome distribution")
    lines.append("")
    lines.append(
        "| Binding | Total | Completed | Failed | Timeout | Runner-exc | "
        "Compiled | Correct | Decoy | Speedup>1 | Mean speedup (compiled) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in ALL_BINDINGS:
        s = summaries.get(b)
        if not s:
            lines.append(f"| `{b}` | (no data) | | | | | | | | | |")
            continue
        sc = s["status_counts"]
        ms = s["speedup_compiled_mean"]
        lines.append(
            f"| `{b}` | {s['total']} | {s['completed']} | {sc.get('failed', 0)} "
            f"| {sc.get('timeout', 0)} | {sc.get('runner-exception', 0)} "
            f"| {s['compiled']} | {s['correctness']} | {s['decoy_kernel']} "
            f"| {s['speedup_positive']} "
            f"| {'%.3f' % ms if ms is not None else '—'} |"
        )
    lines.append("")

    for header, key in (
        ("All-attempts (censored at server timeout)", "timing_censored"),
        ("Completed-only", "timing_completed"),
    ):
        lines.append(f"## {header}")
        lines.append("")
        for fld in CROSS_BINDING_FIELDS:
            lines.append(f"### `{fld}`")
            lines.append("")
            lines.append("| Binding | n | p50 | mean | p90 | p99 |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for b in ALL_BINDINGS:
                s = summaries.get(b)
                if not s:
                    continue
                t = s[key][fld]
                lines.append(
                    f"| `{b}` | {t['n']} | {_fmt(t['p50'])} | {_fmt(t['mean'])} "
                    f"| {_fmt(t['p90'])} | {_fmt(t['p99'])} |"
                )
            lines.append("")

    lines.append("## Kernel-phase residual")
    lines.append("")
    lines.append(
        "`kg_kernel_total_s − (backend_compile + backend_load + correctness "
        "+ performance)`. A non-zero residual means kg_kernel_total includes "
        "work not surfaced by the per-phase fields."
    )
    lines.append("")
    lines.append("| Binding | n | p50 | mean | p90 | p99 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for b in ALL_BINDINGS:
        s = summaries.get(b)
        if not s:
            continue
        t = s["kernel_residual_stats"]
        lines.append(
            f"| `{b}` | {t['n']} | {_fmt(t['p50'])} | {_fmt(t['mean'])} | {_fmt(t['p90'])} | {_fmt(t['p99'])} |"
        )
    lines.append("")

    lines.append("## Backend-specific diagnostics (manual ninja path only)")
    lines.append("")
    lines.append(
        "These fields exist for the cuda_agent + pybind11 manual-ninja "
        "compile path; tvm_ffi compiles via `tvm_ffi.cpp.build` and "
        "exposes neither. Do NOT use these to compare against tvm_ffi."
    )
    lines.append("")
    for fld in BACKEND_SPECIFIC_FIELDS:
        lines.append(f"### `{fld}`")
        lines.append("")
        lines.append("| Binding | n | p50 | mean | p90 | p99 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for b in ALL_BINDINGS:
            s = summaries.get(b)
            if not s:
                continue
            t = s["timing_backend_specific"][fld]
            if t["n"] == 0:
                lines.append(f"| `{b}` | 0 | — | — | — | — |")
            else:
                lines.append(
                    f"| `{b}` | {t['n']} | {_fmt(t['p50'])} | {_fmt(t['mean'])} "
                    f"| {_fmt(t['p90'])} | {_fmt(t['p99'])} |"
                )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument(
        "--censor-at",
        type=float,
        default=240.0,
        help="server-side timeout used by the runner; censors the all-attempts table",
    )
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    summaries: dict[str, dict] = {}
    for b in ALL_BINDINGS:
        path = args.results_dir / f"{args.tag}_27b_breakdown_{b}.jsonl"
        rows = _load(path)
        if rows:
            summaries[b] = _summarize(rows, censor_at=args.censor_at)

    if not summaries:
        print(f"# no input found for tag={args.tag} in {args.results_dir}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True, default=str))
        return 0

    print(_render_markdown(summaries, tag=args.tag, censor_at=args.censor_at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
