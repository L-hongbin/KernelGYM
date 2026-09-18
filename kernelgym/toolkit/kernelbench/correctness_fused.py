"""Lazy CUDA diagnostics with one bounded-per-tile device-to-host transfer.

Only the detailed comparison imports this module. The native library has no
PyTorch C++ ABI dependency; buffers and the current stream are owned by PyTorch.
Unsupported layouts/dtypes and unavailable builds use the existing comparator.
"""

from __future__ import annotations

import ctypes
import fcntl
import functools
import hashlib
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch

logger = logging.getLogger(__name__)
_DTYPES = {torch.float32: 0, torch.float64: 1, torch.float16: 2, torch.bfloat16: 3}


@functools.lru_cache(maxsize=8)
def _library(capability: tuple[int, int]):
    """Cache on disk across evaluator processes, serialize concurrent builds."""
    source = Path(__file__).with_suffix(".cu")
    compiler = shutil.which("nvcc")
    if compiler is None:
        compiler_path = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
        compiler = str(compiler_path) if compiler_path.is_file() else None
    if compiler is None:
        return None
    try:
        identity = (
            source.read_bytes()
            + str(capability).encode()
            + str(Path(compiler).resolve()).encode()
            + str(Path(compiler).stat().st_mtime_ns).encode()
        )
        key = hashlib.sha256(identity).hexdigest()[:24]
        cache = Path(os.environ.get("KERNELGYM_CORRECTNESS_CACHE_DIR", str(Path(__file__).parents[3] / ".native")))
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f"correctness-{key}.so"
        with (cache / f"correctness-{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                # Same filesystem for atomic publication; no partially built .so
                # is visible to another fresh evaluator process.
                with tempfile.TemporaryDirectory(prefix="correctness-build-", dir=cache) as directory:
                    built = Path(directory) / "diagnostics.so"
                    arch = f"sm_{capability[0]}{capability[1]}"
                    subprocess.run(
                        [
                            compiler,
                            "-O3",
                            "--shared",
                            "-Xcompiler",
                            "-fPIC",
                            "--fmad=false",
                            "-arch=" + arch,
                            str(source),
                            "-o",
                            str(built),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    built.replace(target)
        lib = ctypes.CDLL(str(target))
        lib.kg_correctness.argtypes = (
            [ctypes.c_void_p] * 3
            + [ctypes.c_int]
            + [ctypes.c_int64] * 3
            + [
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_void_p,
            ]
        )
        lib.kg_correctness.restype = ctypes.c_int
        return lib
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "Fused correctness unavailable; using PyTorch diagnostics: %s", getattr(exc, "stderr", None) or exc
        )
        return None


def compare(reference, candidate, *, atol, rtol, output_path):
    """Return the same internal diagnostic tuple, or None to select fallback."""
    if (
        reference.device.type != "cuda"
        or reference.device != candidate.device
        or reference.dtype not in _DTYPES
        or candidate.dtype != reference.dtype
        or reference.shape != candidate.shape
        or not reference.is_contiguous()
        or not candidate.is_contiguous()
        or reference.numel() == 0
        or not math.isfinite(atol)
        or not math.isfinite(rtol)
        or atol < 0
        or rtol < 0
    ):
        return None
    shape = tuple(reference.shape)
    rows, columns = shape[-2:] if len(shape) >= 2 else (1, reference.numel())
    prefix = reference.numel() // (rows * columns)
    tile_rows, tile_columns = (rows + 31) // 32, (columns + 31) // 32
    tiles = prefix * tile_rows * tile_columns
    # Keep host summaries bounded for pathological long/skinny tensors. The
    # fallback already handles all supported tensor sizes and arbitrary strides.
    if tiles > 65536 or reference.numel() >= 2**53:
        return None
    with torch.cuda.device(reference.device):
        lib = _library(torch.cuda.get_device_capability(reference.device))
        if lib is None:
            return None
        summaries = torch.empty((tiles, 19), dtype=torch.float64, device=reference.device)
        status = lib.kg_correctness(
            reference.data_ptr(),
            candidate.data_ptr(),
            summaries.data_ptr(),
            _DTYPES[reference.dtype],
            rows,
            columns,
            prefix,
            atol,
            rtol,
            torch.cuda.current_stream(reference.device).cuda_stream,
        )
        if status:
            # CUDA failures must go through the existing runtime-error handling.
            raise RuntimeError(f"Fused correctness CUDA launch failed (CUDA error code {status})")
        data = summaries.cpu().numpy()
    return _assemble(data, reference, output_path, rows, columns, prefix, tile_rows, tile_columns)


def _assemble(data, reference, output_path, rows, columns, prefix, tile_rows, tile_columns):
    from .correctness import (
        _coordinate_record,
        _empty_tensor_mismatch_diagnostics,
        _flat_index_to_coordinate,
        _unit_correctness_summary,
    )

    total = reference.numel()
    shape = reference.shape
    sums = data[:, 1:9].sum(axis=0)
    maximum = float(data[:, 0].max())
    mean = float(sums[0] / total)
    bad, nan_count, inf_count = (int(x) for x in sums[1:4])
    counts = {1: total - bad}
    diagnostics = _empty_tensor_mismatch_diagnostics(reference, output_path)
    if not bad:
        counts.update(dict.fromkeys((2, 4, 8, 16), total))
        return True, maximum, mean, counts, total, nan_count, inf_count, diagnostics
    cumulative = 0
    for multiplier, count in zip((2, 4, 8, 16), sums[4:8]):
        cumulative += int(count)
        counts[multiplier] = cumulative
        if cumulative == total:
            counts.update({m: total for m in (2, 4, 8, 16) if m > multiplier})
            break
    scores = data[:, 13:19:2].reshape(-1)
    indices = data[:, 14:19:2].reshape(-1)
    order = np.lexsort((indices, -scores))[: min(3, bad)]
    diagnostics["first"] = _coordinate_record(output_path, int(data[:, 9].min()), shape)
    diagnostics["last"] = _coordinate_record(output_path, int(data[:, 10].max()), shape)
    diagnostics["top"] = [(float(scores[i]), _coordinate_record(output_path, int(indices[i]), shape)) for i in order]
    if len(shape) >= 2:
        row_bits = data[:, 11].astype(np.uint32).reshape(prefix, tile_rows, tile_columns)
        column_bits = data[:, 12].astype(np.uint32).reshape(prefix, tile_rows, tile_columns)
        shifts = np.arange(32, dtype=np.uint32)
        row_bad = ((np.bitwise_or.reduce(row_bits, axis=2)[..., None] >> shifts) & 1).astype(bool)
        row_bad = row_bad.reshape(prefix, tile_rows * 32)[:, :rows].reshape(tuple(shape[:-1]))
        column_bad = ((np.bitwise_or.reduce(column_bits, axis=(0, 1))[:, None] >> shifts) & 1).astype(bool)
        column_bad = column_bad.reshape(-1)[:columns]
        batch_count = shape[0] if len(shape) >= 3 else 1
        batch_bad = row_bad.reshape(batch_count, -1).any(axis=1)
        tile_bad = data[:, 2].reshape(prefix, tile_rows, tile_columns) > 0

        def summary(mask, record):
            # Count on CPU and materialize only the bounded coordinate list.
            positions = np.flatnonzero(mask)
            units = [record(_flat_index_to_coordinate(int(i), mask.shape)) for i in positions[:8]]
            return _unit_correctness_summary(mask.size, int(positions.size), units)

        def tile_record(index):
            p, r, c = index
            return {
                "batch_index": _flat_index_to_coordinate(p, shape[:-2]),
                "M": [r * 32, min((r + 1) * 32, rows)],
                "N": [c * 32, min((c + 1) * 32, columns)],
            }

        names = (
            ("B", "M", "N")
            if len(shape) == 3
            else ("M", "N")
            if len(shape) == 2
            else tuple(f"axis_{axis}" for axis in range(len(shape)))
        )
        bounds = {}
        for axis, name in enumerate(names):
            if axis == len(shape) - 1:
                occupied = column_bad
            else:
                occupied = row_bad.any(axis=tuple(d for d in range(row_bad.ndim) if d != axis))
            positions = np.flatnonzero(occupied)
            bounds[name] = [int(positions[0]), int(positions[-1])]
        diagnostics["localizations"] = [
            {
                "output_path": output_path,
                "shape": list(shape),
                "mismatch_bounds": bounds,
                "batch": summary(batch_bad, lambda i: {"batch": i[0]}),
                "row": summary(row_bad, lambda i: {"batch_index": i[:-1], "row": i[-1]}),
                "tile": summary(tile_bad, tile_record),
            }
        ]
    return False, maximum, mean, counts, total, nan_count, inf_count, diagnostics
