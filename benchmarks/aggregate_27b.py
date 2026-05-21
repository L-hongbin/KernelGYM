#!/usr/bin/env python3
"""Aggregate the 27B-breakdown JSONL into a per-binding table.

Reads ``benchmarks/results/<tag>_27b_breakdown_<binding>.jsonl`` files,
emits a comparison table covering:

  * status distribution (completed / failed / timeout / runner-exception)
  * outcome rates (compiled, correctness, positive-speedup)
  * timing percentiles (p50/p90/p99) and mean for each of:
        elapsed_s
        kg_kernel_total_s
        kg_kernel_backend_compile_s
        kg_kernel_performance_step_s
        kg_kernel_correctness_s
        kg_reference_total_s
        wg_pool_total_s
        manual_ninja_build_wall_sec

The aggregation runs on COMPLETED samples only when computing timing
percentiles (a fail/timeout/runner-exception sample has no meaningful
``kg_*`` breakdown). Status counts cover every row.

Usage:

    python benchmarks/aggregate_27b.py --tag fullrun
    python benchmarks/aggregate_27b.py --tag fullrun --markdown > table.md
    python benchmarks/aggregate_27b.py --tag fullrun --json > table.json
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

TIMING_FIELDS = (
    "elapsed_s",
    "kg_kernel_total_s",
    "kg_kernel_backend_compile_s",
    "kg_kernel_performance_step_s",
    "kg_kernel_correctness_s",
    "kg_reference_total_s",
    "wg_pool_total_s",
    "manual_ninja_build_wall_sec",
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
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _summarize_binding(rows: list[dict]) -> dict:
    total = len(rows)
    status_counts: dict[str, int] = {}
    compiled = 0
    correctness = 0
    speedup_pos = 0
    speedups_compiled: list[float] = []

    for r in rows:
        st = r.get("status") or "unknown"
        status_counts[st] = status_counts.get(st, 0) + 1
        if r.get("compiled"):
            compiled += 1
        if r.get("correctness"):
            correctness += 1
        sp = r.get("speedup")
        if isinstance(sp, (int, float)):
            if sp > 0:
                speedup_pos += 1
            if r.get("compiled"):
                speedups_compiled.append(float(sp))

    completed_rows = [r for r in rows if r.get("status") == "completed"]
    n_completed = len(completed_rows)

    timing: dict[str, dict] = {}
    for fld in TIMING_FIELDS:
        vals = [r[fld] for r in completed_rows if isinstance(r.get(fld), (int, float))]
        if not vals:
            timing[fld] = {"n": 0, "p50": None, "p90": None, "p99": None, "mean": None}
            continue
        timing[fld] = {
            "n": len(vals),
            "p50": _percentile(vals, 0.50),
            "p90": _percentile(vals, 0.90),
            "p99": _percentile(vals, 0.99),
            "mean": statistics.fmean(vals),
        }

    return {
        "total": total,
        "completed": n_completed,
        "status_counts": status_counts,
        "compiled": compiled,
        "correctness": correctness,
        "speedup_positive": speedup_pos,
        "speedup_compiled_mean": (statistics.fmean(speedups_compiled) if speedups_compiled else None),
        "speedup_compiled_p50": _percentile(speedups_compiled, 0.50) if speedups_compiled else None,
        "timing": timing,
    }


def _fmt_secs(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}s"


def _render_markdown(summaries: dict[str, dict], tag: str) -> str:
    lines: list[str] = []
    lines.append(f"# 27B breakdown — tag `{tag}`")
    lines.append("")
    lines.append("## Outcome distribution")
    lines.append("")
    lines.append(
        "| Binding | Total | Completed | Failed | Timeout | Runner-exc | Compiled | Correct | Speedup>1 | Mean speedup (compiled) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in ALL_BINDINGS:
        s = summaries.get(b)
        if not s:
            lines.append(f"| `{b}` | (no data) | | | | | | | | |")
            continue
        sc = s["status_counts"]
        ms = s["speedup_compiled_mean"]
        lines.append(
            f"| `{b}` | {s['total']} | {s['completed']} | {sc.get('failed', 0)} "
            f"| {sc.get('timeout', 0)} | {sc.get('runner-exception', 0)} "
            f"| {s['compiled']} | {s['correctness']} | {s['speedup_positive']} "
            f"| {'%.3f' % ms if ms is not None else '—'} |"
        )
    lines.append("")
    lines.append("## Reward-time breakdown (completed samples only, p50 / mean / p90 / p99)")
    lines.append("")
    for fld in TIMING_FIELDS:
        lines.append(f"### `{fld}`")
        lines.append("")
        lines.append("| Binding | n | p50 | mean | p90 | p99 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for b in ALL_BINDINGS:
            s = summaries.get(b)
            if not s:
                continue
            t = s["timing"][fld]
            lines.append(
                f"| `{b}` | {t['n']} | {_fmt_secs(t['p50'])} | {_fmt_secs(t['mean'])} "
                f"| {_fmt_secs(t['p90'])} | {_fmt_secs(t['p99'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True, help="filename prefix used by run_27b_breakdown.py --tag")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--markdown", action="store_true", help="emit a markdown table")
    p.add_argument("--json", action="store_true", help="emit the raw aggregated JSON")
    args = p.parse_args()

    summaries: dict[str, dict] = {}
    for b in ALL_BINDINGS:
        path = args.results_dir / f"{args.tag}_27b_breakdown_{b}.jsonl"
        rows = _load(path)
        if rows:
            summaries[b] = _summarize_binding(rows)

    if not summaries:
        print(f"# no input found for tag={args.tag} in {args.results_dir}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True, default=str))
        return 0

    if args.markdown or not (args.json or args.markdown):
        # default to markdown
        print(_render_markdown(summaries, tag=args.tag))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
