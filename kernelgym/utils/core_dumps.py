"""Core dump placement and retention helpers."""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_DUMP_DIR_ENV = "KERNELGYM_CORE_DUMP_DIR"
CORE_DUMP_KEEP_ENV = "KERNELGYM_CORE_DUMP_KEEP"
DEFAULT_CORE_DUMP_DIR = "logs/core_dumps"
DEFAULT_CORE_DUMP_KEEP = 10


def hostname() -> str:
    return os.environ.get("HOSTNAME") or socket.gethostname() or "local"


def resolve_core_dump_dir(value: str | os.PathLike[str] | None = None) -> Path:
    raw = str(value or os.environ.get(CORE_DUMP_DIR_ENV) or DEFAULT_CORE_DUMP_DIR)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def core_dump_keep(value: str | int | None = None) -> int:
    raw = value if value is not None else os.environ.get(CORE_DUMP_KEEP_ENV)
    if raw in (None, ""):
        return DEFAULT_CORE_DUMP_KEEP
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_CORE_DUMP_KEEP


def ensure_core_dump_dir(value: str | os.PathLike[str] | None = None) -> Path:
    directory = resolve_core_dump_dir(value)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_core_dump_file(path: Path) -> bool:
    name = path.name
    if not path.is_file():
        return False
    return name == "core" or name.startswith("core.") or (name.startswith("core-") and name.endswith(".core"))


def list_core_dumps(directory: str | os.PathLike[str] | None = None) -> list[Path]:
    root = resolve_core_dump_dir(directory)
    if not root.exists():
        return []
    return sorted(
        (path for path in root.iterdir() if is_core_dump_file(path)),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )


def prune_core_dumps(
    directory: str | os.PathLike[str] | None = None,
    keep: str | int | None = None,
) -> list[Path]:
    retain = core_dump_keep(keep)
    files = list_core_dumps(directory)
    deleted: list[Path] = []
    for path in files[retain:]:
        try:
            path.unlink()
            deleted.append(path)
        except FileNotFoundError:
            continue
    return deleted


def move_root_core_dumps(
    directory: str | os.PathLike[str] | None = None,
    keep: str | int | None = None,
) -> tuple[list[Path], list[Path]]:
    target = ensure_core_dump_dir(directory)
    moved: list[Path] = []
    for path in sorted(PROJECT_ROOT.iterdir(), key=lambda item: (item.stat().st_mtime_ns, item.name)):
        if not is_core_dump_file(path) or path.parent == target:
            continue
        destination = target / path.name
        if destination.exists():
            destination = target / f"{path.stem}.{path.stat().st_mtime_ns}{path.suffix}"
        shutil.move(str(path), str(destination))
        moved.append(destination)
    deleted = prune_core_dumps(target, keep)
    return moved, deleted


def prepare_core_dump_dir(
    directory: str | os.PathLike[str] | None = None,
    keep: str | int | None = None,
    *,
    chdir: bool = False,
) -> Path:
    target = ensure_core_dump_dir(directory)
    prune_core_dumps(target, keep)
    if chdir:
        os.chdir(target)
    return target
