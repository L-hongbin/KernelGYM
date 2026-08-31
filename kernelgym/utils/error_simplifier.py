"""Helpers for removing deployment-specific paths from error messages."""

from __future__ import annotations

import os
import re
import sys
import sysconfig
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _remove_path_prefix(error_message: str, prefix: os.PathLike[str] | str) -> str:
    normalized_prefix = os.fspath(prefix).rstrip("/\\")
    if not normalized_prefix:
        return error_message
    return re.sub(rf"{re.escape(normalized_prefix)}[/\\]", "", error_message)


def simplify_error_message(
    error_message: str,
    *,
    work_dir: os.PathLike[str] | str | None,
    enabled: bool = True,
) -> str:
    """Simplify error output while preserving useful relative paths and diagnostics."""
    if not enabled or not error_message:
        return error_message

    prefixes: set[str] = set()
    if work_dir is not None:
        prefixes.add(os.fspath(work_dir))
    if sys.prefix != sys.base_prefix:
        prefixes.add(sys.prefix)
    for path_name in ("purelib", "platlib"):
        site_packages = sysconfig.get_path(path_name)
        if site_packages:
            prefixes.add(site_packages)

    source_root = os.fspath(_SOURCE_ROOT)
    prefixes.discard(source_root)
    for prefix in sorted(prefixes, key=len, reverse=True):
        error_message = _remove_path_prefix(error_message, prefix)

    # Keep the source root last. A virtualenv may live inside the checkout; if
    # the repository prefix were removed first, /repo/.venv/... would become
    # .venv/... and could no longer match the complete virtualenv prefix.
    return _remove_path_prefix(error_message, source_root)
