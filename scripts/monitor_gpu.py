#!/usr/bin/env python3
"""Sweep GPU memory usage across 192.168.16.50-76 every 10 minutes and plot trends.

Usage:
    ./monitor_gpu.py          # run forever, sweep every 10 min
    ./monitor_gpu.py --once   # single sweep + update plot, then exit
"""

import subprocess
import time
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

USER = "chenshuailin"
SUBNET = "192.168.16"
START, END = 50, 76
SSH_OPTS = "-o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "gpu_mem_log.csv")
PLOT_FILE = os.path.join(SCRIPT_DIR, "gpu_mem_trend.png")
RETENTION_DAYS = 1

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def sweep():
    """Query all machines, return {ip: max_mem_pct}."""
    results = {}
    for i in range(START, END + 1):
        ip = f"{SUBNET}.{i}"
        try:
            r = subprocess.run(
                f"ssh {SSH_OPTS} {USER}@{ip} "
                f'"nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"',
                shell=True,
                capture_output=True,
                timeout=10,
            )
            if r.returncode != 0 or not r.stdout.strip():
                continue

            max_pct = 0.0
            for line in r.stdout.decode().strip().splitlines():
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                used = float(parts[0].strip())
                total = float(parts[1].strip())
                if total > 0:
                    pct = (used / total) * 100
                    max_pct = max(max_pct, pct)

            results[ip] = max_pct
        except Exception:
            continue
    return results


def load_and_prune():
    """Load CSV, drop rows older than RETENTION_DAYS, rewrite file.
    Returns {ip: [(datetime, pct), ...]}.
    """
    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
    history = defaultdict(list)

    if not os.path.exists(DATA_FILE):
        return history

    rows = []
    with open(DATA_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                ts = datetime.fromisoformat(parts[0])
            except ValueError:
                continue

            if ts < cutoff:
                continue  # skip old row

            rows.append(line)
            for i in range(1, len(parts) - 1, 2):
                try:
                    ip = parts[i]
                    pct = float(parts[i + 1])
                    history[ip].append((ts, pct))
                except (IndexError, ValueError):
                    continue

    # Rewrite pruned file
    with open(DATA_FILE, "w") as f:
        for row in rows:
            f.write(row + "\n")

    return history


def append_row(ts, results):
    row = ts.isoformat()
    for ip, pct in sorted(results.items()):
        row += f",{ip},{pct:.1f}"
    with open(DATA_FILE, "a") as f:
        f.write(row + "\n")


def _plot_axis(ax, history, group_ips, title, colors, color_idx):
    for ip in group_ips:
        if ip not in history:
            continue
        points = sorted(history[ip], key=lambda x: x[0])
        times = [p[0] for p in points]
        vals = [p[1] for p in points]
        color = colors[color_idx % len(colors)]
        color_idx += 1
        ax.plot(times, vals, marker=".", markersize=4, linewidth=1.2, color=color, label=ip, alpha=0.85)

    ax.set_ylim(-5, 105)
    ax.set_ylabel("Max GPU Mem (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ncols = 2 if len(group_ips) > 8 else 1
    ax.legend(fontsize=7, ncol=ncols, loc="upper right")


def plot(history):
    if not history:
        return

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(16, 15), sharex=True)
    colors = list(plt.cm.tab10.colors) + list(plt.cm.Set3.colors)
    sorted_ips = sorted(history.keys(), key=lambda x: int(x.split(".")[-1]))

    group_a = [ip for ip in sorted_ips if int(ip.split(".")[-1]) <= 59]
    group_b = [ip for ip in sorted_ips if 60 <= int(ip.split(".")[-1]) <= 69]
    group_c = [ip for ip in sorted_ips if int(ip.split(".")[-1]) >= 70]

    _plot_axis(ax0, history, group_a, "192.168.16.50-59", colors, 0)
    _plot_axis(ax1, history, group_b, "192.168.16.60-69", colors, 0)
    _plot_axis(ax2, history, group_c, "192.168.16.70-76", colors, 0)

    ax2.set_xlabel("Time")
    fig.suptitle("GPU Memory Usage Trend (max across 8 GPUs per machine)", fontsize=13, y=1.01)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Plot saved: {PLOT_FILE}")


def run_sweep():
    ts = datetime.now()
    print(f"[{ts.strftime('%H:%M:%S')}] Sweeping {SUBNET}.{START}-{END}...", end=" ", flush=True)
    results = sweep()
    append_row(ts, results)

    if results:
        idle = [ip for ip, pct in results.items() if pct < 20]
        busy = [ip for ip, pct in results.items() if pct >= 20]
        print(f"OK: {len(results)} machines | idle: {len(idle)} | busy: {len(busy)}")
        if idle:
            print(f"  Idle: {', '.join(idle)}")
    else:
        print("no reachable GPU machines")

    history = load_and_prune()
    plot(history)


def main():
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    print(f"Data log: {DATA_FILE}")
    print(f"Plot:     {PLOT_FILE}")
    print(f"Retention: {RETENTION_DAYS} days")

    if "--once" in sys.argv:
        print("Single-sweep mode.\n")
        run_sweep()
    else:
        print("Sweeping every 10 minutes. Press Ctrl+C to stop.\n")
        while True:
            run_sweep()
            time.sleep(600)


if __name__ == "__main__":
    main()
