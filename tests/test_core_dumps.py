from __future__ import annotations

import os

from kernelgym.utils import core_dumps


def _touch(path, timestamp: int) -> None:
    path.write_bytes(b"core")
    os.utime(path, (timestamp, timestamp))


def test_prune_core_dumps_keeps_newest(tmp_path) -> None:
    for index in range(12):
        _touch(tmp_path / f"core-python-{index}.core", index)

    deleted = core_dumps.prune_core_dumps(tmp_path, keep=10)

    assert len(deleted) == 2
    assert len(core_dumps.list_core_dumps(tmp_path)) == 10
    assert not (tmp_path / "core-python-0.core").exists()
    assert not (tmp_path / "core-python-1.core").exists()


def test_move_root_core_dumps_prunes_after_move(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(core_dumps, "PROJECT_ROOT", tmp_path)
    for index in range(3):
        _touch(tmp_path / f"core-python-{index}.core", index)

    moved, deleted = core_dumps.move_root_core_dumps(tmp_path / "cores", keep=2)

    assert len(moved) == 3
    assert len(deleted) == 1
    assert not list(tmp_path.glob("core-python-*.core"))
    assert len(core_dumps.list_core_dumps(tmp_path / "cores")) == 2
