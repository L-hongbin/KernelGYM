#!/usr/bin/env python3
"""Move and prune KernelGym core dump files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kernelgym.utils.core_dumps import (
    DEFAULT_CORE_DUMP_KEEP,
    list_core_dumps,
    move_root_core_dumps,
    prune_core_dumps,
    resolve_core_dump_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage KernelGym core dump retention")
    parser.add_argument("--dir", default=None, help="Core dump directory; defaults to KERNELGYM_CORE_DUMP_DIR")
    parser.add_argument("--keep", type=int, default=DEFAULT_CORE_DUMP_KEEP, help="Number of newest core dumps to keep")
    parser.add_argument(
        "--move-root",
        action="store_true",
        help="Move core dump files from the repo root into --dir before pruning",
    )
    return parser.parse_args()


def _format_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
        size /= 1024
    return f"{path.stat().st_size}B"


def main() -> int:
    args = parse_args()
    target = resolve_core_dump_dir(args.dir)
    if args.move_root:
        moved, deleted = move_root_core_dumps(target, args.keep)
    else:
        target.mkdir(parents=True, exist_ok=True)
        moved = []
        deleted = prune_core_dumps(target, args.keep)

    remaining = list_core_dumps(target)
    print(f"core_dump_dir={target}")
    print(f"moved={len(moved)}")
    print(f"deleted={len(deleted)}")
    print(f"remaining={len(remaining)}")
    for path in remaining:
        print(f"{path.name}\t{_format_size(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
