#!/usr/bin/env python3
"""Format a reward-node check report from /health and /workers/status JSON.

The companion bash script `check_node.sh` does the HTTP probing and pipes a
combined ``{"health": ..., "workers": ...}`` JSON document to this program on
stdin. We keep the rendering here so the bash side stays a thin wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys


def render_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print rows as a `+--+--+` ASCII table with left-aligned cells."""
    str_rows = [list(map(str, r)) for r in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print()
    print(f"=== {title} ===")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in str_rows:
        print(fmt.format(*row))
    print(sep)


def _device_key(name: str) -> tuple[int, str]:
    """Sort key for `cuda:N` device strings and `worker_gpu_N` worker ids."""
    tail = name.rsplit(":", 1)[-1] if ":" in name else name.rsplit("_", 1)[-1]
    return (int(tail) if tail.isdigit() else 10**9, name)


def render_summary(base: str, health: dict, workers: dict) -> int:
    """Print vertical key:value summary; return the process exit code."""
    status = health.get("status", "?")
    gpus = health.get("gpu_status", {}) or {}
    gpus_ok = sum(1 for v in gpus.values() if v.get("available"))
    queue = health.get("queue_status", {}) or {}
    online = sum(1 for w in workers.values() if isinstance(w, dict) and w.get("status") == "online")
    top = "UP" if status == "healthy" else "WARN"
    fields: list[tuple[str, object]] = [
        ("status", top),
        ("url", base),
        ("api_status", status),
        ("gpus_available", f"{gpus_ok}/{len(gpus)}"),
        ("workers_online", f"{online}/{len(workers)}"),
        ("queue_pending", queue.get("pending", 0)),
        ("queue_processing", queue.get("processing", 0)),
        ("active_tasks", health.get("active_tasks", 0)),
        ("uptime_s", health.get("uptime", 0)),
    ]
    label_w = max(len(k) for k, _ in fields) + 1
    for key, value in fields:
        print(f"{key + ':':<{label_w + 1}}{value}")
    return 0 if status == "healthy" else 1


def render_verbose(health: dict, workers: dict) -> None:
    """Render `/health.gpu_status` and `/workers/status` as ASCII tables."""
    gpus = health.get("gpu_status", {}) or {}
    gpu_rows: list[list[str]] = []
    for dev in sorted(gpus, key=_device_key):
        g = gpus[dev] or {}
        gpu_rows.append(
            [
                dev,
                g.get("name", "?"),
                g.get("memory_total", "?"),
                g.get("memory_used_percent", "?"),
                "yes" if g.get("available") else "no",
            ]
        )
    render_table(
        "/health gpus",
        ["device", "name", "memory_total", "memory_used", "available"],
        gpu_rows,
    )

    worker_rows: list[list[str]] = []
    for wid, info in sorted(workers.items(), key=lambda item: _device_key(item[0])):
        info = info or {}
        worker_rows.append(
            [
                wid,
                info.get("device", "?"),
                info.get("status", "?"),
                info.get("last_heartbeat", "?"),
                info.get("node_id", "?"),
                info.get("hostname", "?"),
            ]
        )
    render_table(
        "/workers/status",
        ["worker_id", "device", "status", "last_heartbeat", "node_id", "hostname"],
        worker_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Format a reward-node check report from /health and /workers/status JSON on stdin.",
    )
    parser.add_argument("--base", required=True, help="API base URL for display only")
    parser.add_argument("--verbose", action="store_true", help="Also render ASCII tables")
    args = parser.parse_args()

    payload = json.load(sys.stdin)
    health = payload.get("health") or {}
    workers = payload.get("workers") or {}
    exit_code = render_summary(args.base, health, workers)
    if args.verbose:
        render_verbose(health, workers)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
