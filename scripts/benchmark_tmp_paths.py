#!/usr/bin/env python3
"""Benchmark candidate temp/cache roots for KernelGym compile workloads.

The benchmark is intentionally lightweight and focuses on the operations that
matter for KernelGym compile/cache dirs:

- sequential write/read throughput for one larger file
- many small file create/read/delete throughput

Example:
    python scripts/benchmark_tmp_paths.py \
      --path /tmp/kernelgym_bench \
      --path /dev/shm/kernelgym_bench \
      --path /data/ssd1/kernelgym_bench
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def mib_per_sec(size_mib: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return size_mib / seconds


def files_per_sec(count: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return count / seconds


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_dd_write(path: Path, size_mib: int) -> tuple[float, int]:
    t0 = time.time()
    proc = subprocess.run(
        [
            "dd",
            "if=/dev/zero",
            f"of={path}",
            "bs=1M",
            f"count={size_mib}",
            "conv=fdatasync",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return time.time() - t0, proc.returncode


def run_dd_read(path: Path, size_mib: int) -> tuple[float, int]:
    t0 = time.time()
    proc = subprocess.run(
        [
            "dd",
            f"if={path}",
            "of=/dev/null",
            "bs=4M",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return time.time() - t0, proc.returncode


def benchmark_small_files(root: Path, file_count: int, file_size: int) -> dict:
    small_dir = root / "small"
    if small_dir.exists():
        shutil.rmtree(small_dir)
    small_dir.mkdir(parents=True)

    payload = b"0" * file_size

    t0 = time.time()
    for i in range(file_count):
        with open(small_dir / f"f{i:05d}.bin", "wb") as f:
            f.write(payload)
    create_s = time.time() - t0

    t0 = time.time()
    total_bytes = 0
    for i in range(file_count):
        with open(small_dir / f"f{i:05d}.bin", "rb") as f:
            total_bytes += len(f.read())
    read_s = time.time() - t0

    t0 = time.time()
    shutil.rmtree(small_dir)
    delete_s = time.time() - t0

    return {
        "small_create_files_s": round(files_per_sec(file_count, create_s) or 0.0, 1),
        "small_read_files_s": round(files_per_sec(file_count, read_s) or 0.0, 1),
        "small_delete_s": round(delete_s, 3),
        "small_total_bytes": total_bytes,
    }


def benchmark_path(path: Path, size_mib: int, file_count: int, file_size: int) -> dict:
    ensure_parent(path)
    path.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage(path.parent)
    result = {
        "path": str(path),
        "parent": str(path.parent),
        "free_gib": round(usage.free / (1024**3), 1),
    }

    if usage.free < max(2 * 1024**3, size_mib * 2 * 1024**2):
        result["error"] = "free space too low for benchmark"
        return result

    data_file = path / "dd.bin"
    try:
        write_s, write_rc = run_dd_write(data_file, size_mib)
        read_s, read_rc = run_dd_read(data_file, size_mib)

        result.update(
            {
                "write_rc": write_rc,
                "read_rc": read_rc,
                "write_mib_s": round(mib_per_sec(size_mib, write_s) or 0.0, 1),
                "read_mib_s": round(mib_per_sec(size_mib, read_s) or 0.0, 1),
            }
        )
        result.update(benchmark_small_files(path, file_count=file_count, file_size=file_size))
    finally:
        try:
            if data_file.exists():
                data_file.unlink()
        except Exception:
            pass
        try:
            if path.exists() and not any(path.iterdir()):
                path.rmdir()
        except Exception:
            pass

    return result


def rank_score(item: dict) -> tuple:
    if item.get("error"):
        return (-1, -1, -1)
    return (
        item.get("small_create_files_s", 0.0),
        item.get("write_mib_s", 0.0),
        item.get("small_read_files_s", 0.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark candidate temp/cache paths for KernelGym.")
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        required=True,
        help="Candidate benchmark path. Pass multiple times.",
    )
    parser.add_argument(
        "--size-mib",
        type=int,
        default=1024,
        help="Sequential read/write test size in MiB. Default: 1024",
    )
    parser.add_argument(
        "--small-files",
        type=int,
        default=4000,
        help="Number of small files for metadata test. Default: 4000",
    )
    parser.add_argument(
        "--small-size",
        type=int,
        default=4096,
        help="Size of each small file in bytes. Default: 4096",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [
        benchmark_path(Path(raw).expanduser(), args.size_mib, args.small_files, args.small_size)
        for raw in args.paths
    ]
    ranked = sorted(results, key=rank_score, reverse=True)

    if args.json:
        print(json.dumps(ranked, indent=2))
        return 0

    print("KernelGym tmp path benchmark")
    print(f"Sequential file size: {args.size_mib} MiB")
    print(f"Small files: {args.small_files} x {args.small_size} bytes")
    print("")
    for idx, item in enumerate(ranked, start=1):
        print(f"[{idx}] {item['path']}")
        if item.get("error"):
            print(f"  error: {item['error']}")
            continue
        print(f"  free_gib: {item['free_gib']}")
        print(f"  write_mib_s: {item['write_mib_s']}")
        print(f"  read_mib_s: {item['read_mib_s']}")
        print(f"  small_create_files_s: {item['small_create_files_s']}")
        print(f"  small_read_files_s: {item['small_read_files_s']}")
        print(f"  small_delete_s: {item['small_delete_s']}")
        print("")

    if ranked and not ranked[0].get("error"):
        print(f"Recommended KERNELGYM_TMPDIR={ranked[0]['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
