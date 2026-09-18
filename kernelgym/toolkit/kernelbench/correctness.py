"""KernelBench correctness helpers (toolkit layer)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable, TypeVar

import torch
import torch.nn as nn

from kernelgym.toolkit.kernelbench.exec_types import (
    KernelExecResult,
    get_error_name,
    set_seed,
)
from kernelgym.toolkit.kernelbench.execution_policy import (
    prepare_model_for_execution,
    record_execution_policy,
    tf32_execution_context,
)
from kernelgym.toolkit.kernelbench.input_perturbation import (
    CORRECTNESS_INPUT_PERTURBATIONS,
    PERTURBATION_ORIGINAL,
    apply_input_perturbation,
    capture_random_input_origins,
)
from kernelgym.toolkit.kernelbench.profiling import (
    aten_operator_profiling_context,
    extract_aten_operator_metrics,
)

logger = logging.getLogger(__name__)
_CORRECTNESS_EARLY_STOP_ENV = "KERNELGYM_CORRECTNESS_EARLY_STOP"
_CORRECTNESS_MAX_WALL_S_ENV = "KERNELGYM_CORRECTNESS_MAX_WALL_S"
_CORRECTNESS_PASS_ON_BUDGET_ENV = "KERNELGYM_CORRECTNESS_PASS_ON_BUDGET"
_CORRECTNESS_BUDGET_MIN_PASS_TRIALS_ENV = "KERNELGYM_CORRECTNESS_BUDGET_MIN_PASS_TRIALS"
_CORRECTNESS_GPU_INPUTS_ENV = "KERNELGYM_CORRECTNESS_GPU_INPUTS"
T = TypeVar("T")


def get_tolerance_for_dtype(dtype: torch.dtype) -> float:
    """Match KernelBench fp32 tolerance for integral outputs."""
    tolerances = {
        torch.float64: 1e-4,
        torch.float32: 1e-3,
        torch.float16: 1e-2,
        torch.bfloat16: 1e-2,
        # Complex dtypes: comparison runs on abs() magnitudes (see
        # _compare_tensors_inplace), so mirror the real-dtype tolerance of the
        # matching precision.
        torch.complex128: 1e-4,
        torch.complex64: 1e-3,
        torch.bool: 0.0,
        torch.uint8: 1e-4,
        torch.int8: 1e-4,
        torch.int16: 1e-4,
        torch.int32: 1e-4,
        torch.int64: 1e-4,
    }
    # torch.complex32 / torch.chalf may be absent on older torch builds.
    half_complex = getattr(torch, "complex32", None)
    if half_complex is not None:
        tolerances[half_complex] = 1e-2
    if dtype not in tolerances:
        raise ValueError(f"Unsupported correctness tolerance dtype: {dtype}")
    return tolerances[dtype]


def _env_optional_str(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value


def _env_flag(name: str, *, default: bool) -> bool:
    value = _env_optional_str(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_parsed(name: str, parser: Callable[[str], T]) -> T | None:
    value = _env_optional_str(name)
    if value is None:
        return None
    try:
        return parser(value)
    except (TypeError, ValueError):
        return None


def _env_positive_float(name: str) -> float | None:
    parsed = _env_parsed(name, float)
    return parsed if parsed is not None and parsed > 0 else None


def _env_positive_int(name: str) -> int | None:
    parsed = _env_parsed(name, int)
    return parsed if parsed is not None and parsed > 0 else None


@contextmanager
def _input_generation_device_context(device: Any, *, enabled: bool = True):
    if not enabled or device is None:
        yield
        return
    if isinstance(device, int):
        cuda_device = torch.device("cuda", device)
    else:
        target = torch.device(device)
        if target.type != "cuda":
            yield
            return
        cuda_device = target
    previous_device = None
    if hasattr(torch, "get_default_device"):
        previous_device = torch.get_default_device()
    torch.set_default_device(cuda_device)
    try:
        yield
    finally:
        torch.set_default_device(previous_device or "cpu")


def _move_input_to_device(value: Any, *, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if device is None:
            return value
        return value.cuda(device=device)
    if isinstance(value, list):
        return [_move_input_to_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_input_to_device(item, device=device) for item in value)
    if isinstance(value, dict):
        return {key: _move_input_to_device(item, device=device) for key, item in value.items()}
    return value


def _clone_output_on_device(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_output_on_device(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_output_on_device(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_output_on_device(item) for key, item in value.items()}
    return value


def _zero_poison_like(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return torch.zeros_like(value, memory_format=torch.preserve_format)
    if isinstance(value, list):
        return [_zero_poison_like(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_zero_poison_like(item) for item in value)
    if isinstance(value, dict):
        return {key: _zero_poison_like(item) for key, item in value.items()}
    return None


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _outputs_are_finite(value: Any) -> bool:
    for tensor in _iter_tensors(value):
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all().item()):
            return False
    return True


def _tensor_storage_id(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        return tensor.data_ptr()


def _output_aliases_inputs(output: Any, inputs: Any) -> bool:
    input_storage_ids = {_tensor_storage_id(tensor) for tensor in _iter_tensors(inputs)}
    if not input_storage_ids:
        return False
    return any(_tensor_storage_id(tensor) in input_storage_ids for tensor in _iter_tensors(output))


ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS = (1, 2, 4, 8, 16)
MISMATCH_COORDINATE_TOP_K = 3
OUTPUT_LOCALIZATION_TILE_SIZE = 32
OUTPUT_LOCALIZATION_MAX_MISMATCH_UNITS = 8


def _candidate_nonfinite_count_tensors(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not candidate.is_floating_point() and not candidate.is_complex():
        zero = torch.zeros((), dtype=torch.int64, device=candidate.device)
        return zero, zero
    return (
        torch.isnan(candidate).sum(dtype=torch.int64),
        torch.isinf(candidate).sum(dtype=torch.int64),
    )


def _flat_index_to_coordinate(flat_index: int, shape: torch.Size) -> list[int]:
    coordinate: list[int] = []
    for dimension in reversed(shape):
        coordinate.append(flat_index % dimension)
        flat_index //= dimension
    return list(reversed(coordinate))


def _coordinate_record(output_path: str, flat_index: int, shape: torch.Size) -> dict[str, Any]:
    return {
        "output_path": output_path,
        "coordinate": _flat_index_to_coordinate(flat_index, shape),
    }


def _unit_correctness_summary(
    total: int,
    mismatch_count: int,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    correctness = (total - mismatch_count) / total if total else 1.0
    return {
        "correctness": round(correctness, 6),
        "correct": total - mismatch_count,
        "total": total,
        "mismatch_count": mismatch_count,
        "mismatches": mismatches,
        "mismatches_truncated": mismatch_count > len(mismatches),
    }


def _perfect_tensor_output_space_localization(shape: torch.Size, output_path: str) -> dict[str, Any] | None:
    if len(shape) < 2:
        return None
    rows, columns = shape[-2:]
    prefix_count = 1
    for dimension in shape[:-2]:
        prefix_count *= dimension
    batch_count = shape[0] if len(shape) >= 3 else 1
    row_count = prefix_count * rows
    tile_count = (
        prefix_count
        * ((rows + OUTPUT_LOCALIZATION_TILE_SIZE - 1) // OUTPUT_LOCALIZATION_TILE_SIZE)
        * ((columns + OUTPUT_LOCALIZATION_TILE_SIZE - 1) // OUTPUT_LOCALIZATION_TILE_SIZE)
    )
    return {
        "output_path": output_path,
        "shape": list(shape),
        "mismatch_bounds": {},
        "batch": _unit_correctness_summary(batch_count, 0, []),
        "row": _unit_correctness_summary(row_count, 0, []),
        "tile": _unit_correctness_summary(tile_count, 0, []),
    }


def _empty_tensor_mismatch_diagnostics(tensor: torch.Tensor, output_path: str) -> dict[str, Any]:
    localization = _perfect_tensor_output_space_localization(tensor.shape, output_path)
    return {
        "first": None,
        "last": None,
        "top": [],
        "localizations": [] if localization is None else [localization],
    }


def _limited_nonzero_indices(mask: torch.Tensor) -> tuple[list[list[int]], int]:
    indices = torch.nonzero(mask, as_tuple=False)
    mismatch_count = indices.shape[0]
    return indices[:OUTPUT_LOCALIZATION_MAX_MISMATCH_UNITS].tolist(), mismatch_count


def _mismatch_axis_bounds(mismatch_mask: torch.Tensor, row_bad: torch.Tensor) -> dict[str, list[int]]:
    rank = mismatch_mask.ndim
    if rank == 3:
        axis_names = ("B", "M", "N")
    elif rank == 2:
        axis_names = ("M", "N")
    else:
        axis_names = tuple(f"axis_{axis}" for axis in range(rank))

    bounds: dict[str, list[int]] = {}
    for axis, axis_name in enumerate(axis_names):
        if axis < rank - 1:
            reduce_dimensions = tuple(dimension for dimension in range(rank - 1) if dimension != axis)
            occupied = row_bad.any(dim=reduce_dimensions) if reduce_dimensions else row_bad
        else:
            occupied = mismatch_mask.any(dim=tuple(range(rank - 1)))
        positions = torch.nonzero(occupied, as_tuple=False).flatten()
        first_position, last_position = positions[[0, -1]].tolist()
        bounds[axis_name] = [int(first_position), int(last_position)]
    return bounds


def _tensor_output_space_localization(mismatch_mask: torch.Tensor, output_path: str) -> dict[str, Any] | None:
    """Summarize batch, row, and 32x32 tile verifiers for rank-2+ tensor outputs."""
    if mismatch_mask.ndim < 2:
        return None

    mismatch_mask = mismatch_mask.bool()
    shape = tuple(mismatch_mask.shape)

    row_bad = mismatch_mask.reshape(-1, shape[-1]).any(dim=1).reshape(shape[:-1])
    row_indices, failed_rows = _limited_nonzero_indices(row_bad)
    row_units = [
        {
            "batch_index": index[:-1],
            "row": index[-1],
        }
        for index in row_indices
    ]

    batch_count = shape[0] if mismatch_mask.ndim >= 3 else 1
    batch_bad = row_bad.reshape(batch_count, -1).any(dim=1)
    batch_indices, failed_batches = _limited_nonzero_indices(batch_bad)
    batch_units = [{"batch": index[0]} for index in batch_indices]

    rows, columns = shape[-2:]
    tile_size = OUTPUT_LOCALIZATION_TILE_SIZE
    row_tile_count = (rows + tile_size - 1) // tile_size
    column_tile_count = (columns + tile_size - 1) // tile_size
    prefix_shape = shape[:-2]
    prefix_count = mismatch_mask.numel() // (rows * columns)
    tiled_mask = mismatch_mask.reshape(prefix_count, rows, columns)
    pad_rows = row_tile_count * tile_size - rows
    pad_columns = column_tile_count * tile_size - columns
    if pad_rows or pad_columns:
        tiled_mask = torch.nn.functional.pad(tiled_mask, (0, pad_columns, 0, pad_rows))
    tile_bad = tiled_mask.reshape(
        prefix_count,
        row_tile_count,
        tile_size,
        column_tile_count,
        tile_size,
    ).any(dim=(2, 4))
    tile_indices, failed_tiles = _limited_nonzero_indices(tile_bad)
    tile_units: list[dict[str, Any]] = []
    for prefix_flat, row_tile, column_tile in tile_indices:
        row_start = row_tile * tile_size
        column_start = column_tile * tile_size
        tile_units.append(
            {
                "batch_index": _flat_index_to_coordinate(prefix_flat, torch.Size(prefix_shape)),
                "M": [row_start, min(row_start + tile_size, rows)],
                "N": [column_start, min(column_start + tile_size, columns)],
            }
        )

    return {
        "output_path": output_path,
        "shape": list(shape),
        "mismatch_bounds": _mismatch_axis_bounds(mismatch_mask, row_bad),
        "batch": _unit_correctness_summary(batch_bad.numel(), failed_batches, batch_units),
        "row": _unit_correctness_summary(row_bad.numel(), failed_rows, row_units),
        "tile": _unit_correctness_summary(tile_bad.numel(), failed_tiles, tile_units),
    }


def _normalize_difference_inplace(
    difference: torch.Tensor,
    tolerance: torch.Tensor,
    *,
    tolerance_can_be_zero: bool,
) -> torch.Tensor:
    exact_zero_at_zero_tolerance = None
    if tolerance_can_be_zero:
        exact_zero_at_zero_tolerance = difference.eq(0).logical_and_(tolerance.eq(0))
    difference.div_(tolerance)
    finite_max = torch.finfo(difference.dtype).max
    difference.nan_to_num_(nan=finite_max, posinf=finite_max, neginf=finite_max)
    if exact_zero_at_zero_tolerance is not None:
        difference.masked_fill_(exact_zero_at_zero_tolerance, 0)
    return difference


def _tensor_mismatch_coordinate_diagnostics(
    normalized_error: torch.Tensor,
    mismatch_mask: torch.Tensor,
    *,
    mismatch_count: int,
    output_path: str,
) -> dict[str, Any]:
    if mismatch_count == 0:
        return _empty_tensor_mismatch_diagnostics(normalized_error, output_path)

    flat_error = normalized_error.reshape(-1)
    top_count = min(MISMATCH_COORDINATE_TOP_K, mismatch_count)
    top_values, top_indices = torch.topk(flat_error, k=top_count, largest=True, sorted=True)
    flat_mask = mismatch_mask.reshape(-1)
    first_index_tensor = torch.max(flat_mask, dim=0).indices
    reverse_last_index_tensor = torch.max(torch.flip(flat_mask, dims=(0,)), dim=0).indices
    packed_diagnostics = torch.cat(
        (
            top_values.to(dtype=torch.float64),
            top_indices.to(dtype=torch.float64),
            first_index_tensor.reshape(1).to(dtype=torch.float64),
            reverse_last_index_tensor.reshape(1).to(dtype=torch.float64),
        )
    ).tolist()
    top_scores = packed_diagnostics[:top_count]
    top_flat_indices = packed_diagnostics[top_count : 2 * top_count]
    top_records = [
        (float(score), _coordinate_record(output_path, int(index), normalized_error.shape))
        for score, index in zip(top_scores, top_flat_indices)
    ]

    localization = _tensor_output_space_localization(mismatch_mask, output_path)
    first_index = int(packed_diagnostics[-2])
    last_index = flat_mask.numel() - 1 - int(packed_diagnostics[-1])
    return {
        "first": _coordinate_record(output_path, first_index, normalized_error.shape),
        "last": _coordinate_record(output_path, last_index, normalized_error.shape),
        "top": top_records,
        "localizations": [] if localization is None else [localization],
    }


def _collect_base_difference_stats(
    difference: torch.Tensor,
    tolerance: torch.Tensor,
    candidate_nan_count: torch.Tensor,
    candidate_inf_count: torch.Tensor,
) -> tuple[float, float, int, torch.Tensor, int, int]:
    """Collect base scalar diagnostics and the reusable 1x mask with one host sync."""
    mismatch_mask = difference.le(tolerance).logical_not_()
    device_stats = torch.stack(
        (
            difference.max().to(dtype=torch.float64),
            difference.sum(dtype=torch.float64),
            mismatch_mask.sum(dtype=torch.float64),
            candidate_nan_count.to(dtype=torch.float64),
            candidate_inf_count.to(dtype=torch.float64),
        )
    )
    max_difference, total_difference, mismatch_count, nan_count, inf_count = device_stats.tolist()
    return (
        float(max_difference),
        float(total_difference / difference.numel()),
        int(mismatch_count),
        mismatch_mask,
        int(nan_count),
        int(inf_count),
    )


def _count_correctness_curve_from_normalized_error(
    normalized_error: torch.Tensor,
    mismatch_count: int,
) -> dict[int, int]:
    """Count the tolerance curve, stopping after the first 100% point."""
    total_elements = normalized_error.numel()
    counts = {1: total_elements - mismatch_count}
    if mismatch_count == 0:
        return dict.fromkeys(ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS, total_elements)

    for multiplier in ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS[1:]:
        counts[multiplier] = int(torch.count_nonzero(normalized_error <= multiplier).item())
        if counts[multiplier] == total_elements:
            for remaining_multiplier in ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS:
                if remaining_multiplier > multiplier:
                    counts[remaining_multiplier] = total_elements
            return counts
    return counts


def _compare_tensors_inplace_with_diagnostics(
    output: torch.Tensor,
    output_new: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    output_path: str = "output",
) -> tuple[bool, float, float, dict[int, int], int, int, int, dict[str, Any]]:
    if output.is_cuda and _env_flag('KERNELGYM_FUSED_CORRECTNESS', default=True):
        from .correctness_fused import compare

        result = compare(output, output_new, atol=atol, rtol=rtol, output_path=output_path)
        if result is not None:
            return result
    return _compare_tensors_inplace_with_diagnostics_torch(
        output, output_new, atol=atol, rtol=rtol, output_path=output_path,
    )


def _compare_tensors_inplace_with_diagnostics_torch(
    output: torch.Tensor,
    output_new: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    output_path: str = 'output',
) -> tuple[bool, float, float, dict[int, int], int, int, int, dict[str, Any]]:
    """Destructively compare tensors and collect bounded mismatch diagnostics."""
    if output.numel() == 0:
        return (
            True,
            0.0,
            0.0,
            dict.fromkeys(ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS, 0),
            0,
            0,
            0,
            _empty_tensor_mismatch_diagnostics(output, output_path),
        )

    if output.dtype in {torch.bool, torch.uint8}:
        nan_count = 0
        inf_count = 0
        output.ne_(output_new)
        mismatch_count = output.sum(dtype=torch.float64).item()
        avg_diff = mismatch_count / output.numel()
        max_diff = 1.0 if mismatch_count else 0.0
        correct_count = output.numel() - int(mismatch_count)
        curve_counts = dict.fromkeys(ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS, correct_count)
        coordinates = _empty_tensor_mismatch_diagnostics(output, output_path)
        if mismatch_count:
            normalized_error = output.to(dtype=torch.float32)
            normalized_error.mul_(torch.finfo(normalized_error.dtype).max)
            coordinates = _tensor_mismatch_coordinate_diagnostics(
                normalized_error,
                output.bool(),
                mismatch_count=int(mismatch_count),
                output_path=output_path,
            )
        return (
            mismatch_count == 0,
            float(max_diff),
            float(avg_diff),
            curve_counts,
            output.numel(),
            nan_count,
            inf_count,
            coordinates,
        )

    if not output.is_floating_point() and not output.is_complex():
        candidate_nan_count, candidate_inf_count = _candidate_nonfinite_count_tensors(output_new)
        tolerance = output.abs().to(dtype=torch.float64).mul_(rtol).add_(atol)
        output.sub_(output_new).abs_()
        max_diff, avg_diff, mismatch_count, mismatch_mask, nan_count, inf_count = _collect_base_difference_stats(
            output,
            tolerance,
            candidate_nan_count,
            candidate_inf_count,
        )
        coordinates = _empty_tensor_mismatch_diagnostics(output, output_path)
        if mismatch_count:
            normalized_error = output.to(dtype=torch.float64)
            _normalize_difference_inplace(normalized_error, tolerance, tolerance_can_be_zero=atol == 0)
            curve_counts = _count_correctness_curve_from_normalized_error(normalized_error, mismatch_count)
            coordinates = _tensor_mismatch_coordinate_diagnostics(
                normalized_error,
                mismatch_mask,
                mismatch_count=mismatch_count,
                output_path=output_path,
            )
        else:
            curve_counts = dict.fromkeys(ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS, output.numel())
        return (
            mismatch_count == 0,
            float(max_diff),
            float(avg_diff),
            curve_counts,
            output.numel(),
            nan_count,
            inf_count,
            coordinates,
        )

    if output.is_complex():
        # In-place abs_() is unsupported for complex tensors (|z| is real), so
        # compare on out-of-place real magnitudes.
        candidate_nan_count, candidate_inf_count = _candidate_nonfinite_count_tensors(output_new)
        diff = (output - output_new).abs()
        tolerance = output.abs().mul_(rtol).add_(atol)
        max_diff, avg_diff, mismatch_count, mismatch_mask, nan_count, inf_count = _collect_base_difference_stats(
            diff,
            tolerance,
            candidate_nan_count,
            candidate_inf_count,
        )
        coordinates = _empty_tensor_mismatch_diagnostics(output, output_path)
        if mismatch_count:
            normalized_error = _normalize_difference_inplace(
                diff,
                tolerance,
                tolerance_can_be_zero=atol == 0,
            )
            curve_counts = _count_correctness_curve_from_normalized_error(normalized_error, mismatch_count)
            coordinates = _tensor_mismatch_coordinate_diagnostics(
                normalized_error,
                mismatch_mask,
                mismatch_count=mismatch_count,
                output_path=output_path,
            )
        else:
            curve_counts = dict.fromkeys(ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS, output.numel())
        return (
            mismatch_count == 0,
            float(max_diff),
            float(avg_diff),
            curve_counts,
            output.numel(),
            nan_count,
            inf_count,
            coordinates,
        )

    # Keep the reference intact until its tolerance has been computed. The
    # candidate buffer becomes the absolute-difference/normalized-error buffer;
    # one Boolean mismatch mask is retained and shared by every diagnostic.
    candidate_nan_count, candidate_inf_count = _candidate_nonfinite_count_tensors(output_new)
    output_new.sub_(output).abs_()
    output.abs_().mul_(rtol).add_(atol)
    max_diff, avg_diff, mismatch_count, mismatch_mask, nan_count, inf_count = _collect_base_difference_stats(
        output_new,
        output,
        candidate_nan_count,
        candidate_inf_count,
    )
    coordinates = _empty_tensor_mismatch_diagnostics(output, output_path)
    if mismatch_count:
        normalized_error = _normalize_difference_inplace(
            output_new,
            output,
            tolerance_can_be_zero=atol == 0,
        )
        curve_counts = _count_correctness_curve_from_normalized_error(normalized_error, mismatch_count)
        coordinates = _tensor_mismatch_coordinate_diagnostics(
            normalized_error,
            mismatch_mask,
            mismatch_count=mismatch_count,
            output_path=output_path,
        )
    else:
        curve_counts = dict.fromkeys(ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS, output.numel())
    return (
        mismatch_count == 0,
        float(max_diff),
        float(avg_diff),
        curve_counts,
        output.numel(),
        nan_count,
        inf_count,
        coordinates,
    )


def _compare_tensors_inplace_with_correctness_curve(
    output: torch.Tensor,
    output_new: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[bool, float, float, dict[int, int], int]:
    """Backward-compatible comparison exposing the tolerance curve."""
    outputs_close, max_diff, avg_diff, curve_counts, total_elements, _nan_count, _inf_count, _coordinates = (
        _compare_tensors_inplace_with_diagnostics(output, output_new, atol=atol, rtol=rtol)
    )
    return outputs_close, max_diff, avg_diff, curve_counts, total_elements


def _compare_tensors_inplace_with_element_counts(
    output: torch.Tensor,
    output_new: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[bool, float, float, int, int]:
    """Backward-compatible comparison exposing only the base-tolerance count."""
    outputs_close, max_diff, avg_diff, curve_counts, total_elements = _compare_tensors_inplace_with_correctness_curve(
        output, output_new, atol=atol, rtol=rtol
    )
    return outputs_close, max_diff, avg_diff, curve_counts[1], total_elements


def _compare_tensors_inplace(
    output: torch.Tensor,
    output_new: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[bool, float, float]:
    """Destructively compare two tensors without allocating detailed diagnostics."""
    if output.numel() == 0:
        return True, 0.0, 0.0

    if output.dtype in {torch.bool, torch.uint8}:
        output.ne_(output_new)
        mismatch_count = output.sum(dtype=torch.float64).item()
        avg_diff = mismatch_count / output.numel()
        max_diff = 1.0 if mismatch_count else 0.0
        return mismatch_count == 0, float(max_diff), float(avg_diff)

    if not output.is_floating_point() and not output.is_complex():
        outputs_close = torch.allclose(output, output_new, atol=atol, rtol=rtol)
        output.sub_(output_new).abs_()
        max_diff = output.max().item()
        avg_diff = output.sum(dtype=torch.float64).item() / output.numel()
        return bool(outputs_close), float(max_diff), float(avg_diff)

    if output.is_complex():
        diff = (output - output_new).abs()
        max_diff = diff.max().item()
        avg_diff = diff.mean().item()
        tolerance = output_new.abs().mul_(rtol).add_(atol)
        max_over_tolerance = diff.sub_(tolerance).max().item()
        return max_over_tolerance <= 0, float(max_diff), float(avg_diff)

    output.sub_(output_new).abs_()
    max_diff = output.max().item()
    avg_diff = output.mean().item()
    output_new.abs_().mul_(rtol).add_(atol)
    output.sub_(output_new)
    max_over_tolerance = output.max().item()
    return max_over_tolerance <= 0, float(max_diff), float(avg_diff)


def _describe_structure(value: Any) -> Any:
    """Structural shape description for a (possibly nested) output."""
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return type(value).__name__, [_describe_structure(v) for v in value]
    if isinstance(value, dict):
        return {key: _describe_structure(v) for key, v in value.items()}
    return type(value).__name__


def _structures_match(a: Any, b: Any) -> bool:
    """Recursively check that two (possibly nested) outputs share structure/shape."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_structures_match(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_structures_match(a[k], b[k]) for k in a)
    return type(a) is type(b)


def _first_tensor(value: Any) -> torch.Tensor | None:
    for tensor in _iter_tensors(value):
        return tensor
    return None


def _child_output_path(parent: str, key: Any, *, sequence: bool) -> str:
    if sequence:
        return f"{parent}[{key}]"
    if isinstance(key, str) and key.isidentifier():
        return f"{parent}.{key}"
    return f"{parent}[{key!r}]"


def _compare_outputs_inplace_with_diagnostics(
    output: Any,
    output_new: Any,
    *,
    output_path: str = "output",
) -> tuple[bool, float, float, dict[int, int], int, int, int, dict[str, Any]]:
    """Recursive, destructive comparison over nested tensor/list/tuple/dict outputs.

    Each tensor leaf is compared with its own dtype tolerance via
    ``_compare_tensors_inplace``; results aggregate to (all_close, max_diff,
    mean_of_per_leaf_avg, curve_correct_counts, total_element_count, nan_count,
    inf_count, mismatch_coordinates).
    Non-tensor leaves compare by equality and count as one element.
    """
    if isinstance(output, torch.Tensor):
        tolerance = get_tolerance_for_dtype(output.dtype)
        return _compare_tensors_inplace_with_diagnostics(
            output,
            output_new,
            atol=tolerance,
            rtol=tolerance,
            output_path=output_path,
        )
    if isinstance(output, (list, tuple)):
        items = [
            (a, b, _child_output_path(output_path, index, sequence=True))
            for index, (a, b) in enumerate(zip(output, output_new))
        ]
    elif isinstance(output, dict):
        items = [
            (output[key], output_new[key], _child_output_path(output_path, key, sequence=False)) for key in output
        ]
    else:
        close = bool(output == output_new)
        curve_counts = dict.fromkeys(ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS, int(close))
        record = {"output_path": output_path, "coordinate": []}
        coordinates = {
            "first": None if close else record,
            "last": None if close else record,
            "top": [] if close else [(float("inf"), record)],
            "localizations": [],
        }
        return close, 0.0, 0.0, curve_counts, 1, 0, 0, coordinates

    close = True
    max_diff = 0.0
    avg_sum = 0.0
    leaves = 0
    curve_counts = dict.fromkeys(ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS, 0)
    total_elements = 0
    nan_count = 0
    inf_count = 0
    coordinates: dict[str, Any] = {"first": None, "last": None, "top": [], "localizations": []}
    for a, b, child_path in items:
        leaf_close, leaf_max, leaf_avg, leaf_curve_counts, leaf_total, leaf_nan, leaf_inf, leaf_coordinates = (
            _compare_outputs_inplace_with_diagnostics(a, b, output_path=child_path)
        )
        close = close and leaf_close
        max_diff = max(max_diff, leaf_max)
        avg_sum += leaf_avg
        leaves += 1
        for multiplier in ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS:
            curve_counts[multiplier] += leaf_curve_counts[multiplier]
        total_elements += leaf_total
        nan_count += leaf_nan
        inf_count += leaf_inf
        if coordinates["first"] is None and leaf_coordinates["first"] is not None:
            coordinates["first"] = leaf_coordinates["first"]
        if leaf_coordinates["last"] is not None:
            coordinates["last"] = leaf_coordinates["last"]
        coordinates["top"].extend(leaf_coordinates["top"])
        coordinates["localizations"].extend(leaf_coordinates["localizations"])
    coordinates["top"] = sorted(coordinates["top"], key=lambda item: item[0], reverse=True)[:MISMATCH_COORDINATE_TOP_K]
    return (
        close,
        max_diff,
        (avg_sum / leaves if leaves else 0.0),
        curve_counts,
        total_elements,
        nan_count,
        inf_count,
        coordinates,
    )


def _compare_outputs_inplace_with_correctness_curve(
    output: Any,
    output_new: Any,
) -> tuple[bool, float, float, dict[int, int], int]:
    """Backward-compatible nested comparison exposing the tolerance curve."""
    outputs_close, max_diff, avg_diff, curve_counts, total_elements, _nan_count, _inf_count, _coordinates = (
        _compare_outputs_inplace_with_diagnostics(output, output_new)
    )
    return outputs_close, max_diff, avg_diff, curve_counts, total_elements


def _compare_outputs_inplace_with_element_counts(output: Any, output_new: Any) -> tuple[bool, float, float, int, int]:
    """Backward-compatible nested comparison exposing the base-tolerance count."""
    outputs_close, max_diff, avg_diff, curve_counts, total_elements = _compare_outputs_inplace_with_correctness_curve(
        output, output_new
    )
    return outputs_close, max_diff, avg_diff, curve_counts[1], total_elements


def _compare_outputs_inplace(output: Any, output_new: Any) -> tuple[bool, float, float]:
    """Recursive, destructive comparison without allocating detailed diagnostics."""
    if isinstance(output, torch.Tensor):
        tolerance = get_tolerance_for_dtype(output.dtype)
        return _compare_tensors_inplace(output, output_new, atol=tolerance, rtol=tolerance)
    if isinstance(output, (list, tuple)):
        items = list(zip(output, output_new))
    elif isinstance(output, dict):
        items = [(output[key], output_new[key]) for key in output]
    else:
        return bool(output == output_new), 0.0, 0.0

    close = True
    max_diff = 0.0
    avg_sum = 0.0
    leaves = 0
    for a, b in items:
        leaf_close, leaf_max, leaf_avg = _compare_outputs_inplace(a, b)
        close = close and leaf_close
        max_diff = max(max_diff, leaf_max)
        avg_sum += leaf_avg
        leaves += 1
    return close, max_diff, (avg_sum / leaves if leaves else 0.0)


def _format_element_correctness(correct_elements: int, total_elements: int) -> str:
    percentage = 100.0 * correct_elements / total_elements if total_elements else 100.0
    return f"{percentage:.2f}%"


def _format_element_correctness_curve(curve_counts: dict[int, int], total_elements: int) -> dict[str, str]:
    curve: dict[str, str] = {}
    for multiplier in ELEMENT_CORRECTNESS_CURVE_MULTIPLIERS:
        curve[f"{multiplier}x"] = _format_element_correctness(curve_counts[multiplier], total_elements)
        if curve_counts[multiplier] == total_elements:
            break
    return curve


def _format_element_correctness_curve_text(curve: dict[str, str]) -> str:
    points = ",".join(f"{multiplier}:{percentage}" for multiplier, percentage in curve.items())
    return f"{{{points}}}"


def _format_mismatch_coordinate(coordinates: dict[str, Any]) -> dict[str, Any]:
    return {
        "first": coordinates["first"],
        "last": coordinates["last"],
        f"top_{MISMATCH_COORDINATE_TOP_K}": [record for _score, record in coordinates["top"]],
    }


def _format_output_space_localization(coordinates: dict[str, Any]) -> dict[str, Any] | None:
    tensor_localizations = coordinates["localizations"]
    if not tensor_localizations:
        return None

    result: dict[str, Any] = {
        "tile_shape": [OUTPUT_LOCALIZATION_TILE_SIZE, OUTPUT_LOCALIZATION_TILE_SIZE],
        "tensors": tensor_localizations,
    }
    for unit_name in ("batch", "row", "tile"):
        correct = sum(localization[unit_name]["correct"] for localization in tensor_localizations)
        total = sum(localization[unit_name]["total"] for localization in tensor_localizations)
        result[f"{unit_name}_correctness"] = round(correct / total, 6) if total else 1.0
        result[f"{unit_name}_correct"] = correct
        result[f"{unit_name}_total"] = total
    return result


def _format_mismatch_localization(coordinates: dict[str, Any]) -> dict[str, Any]:
    result = {"element": _format_mismatch_coordinate(coordinates)}
    output_space = _format_output_space_localization(coordinates)
    if output_space is not None:
        result["output_space"] = output_space
    return result


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate: bool = False,
    max_length: int = 200,
):
    if verbose:
        logger.warning("[Exception %s] %s", exception_type, exception_msg)

    metadata[exception_type] = exception_msg
    return metadata


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: Callable[[], Any],
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Any = None,
    stop_on_first_failure: bool | None = None,
    max_wall_time_s: float | None = None,
    pass_on_time_budget: bool | None = None,
    budget_min_pass_trials: int | None = None,
    stage_update_fn: Callable[[str], None] | None = None,
    detect_aten_fallback: bool = False,
    enable_input_perturbations: bool = False,
    return_detail_correctness: bool = False,
) -> KernelExecResult:
    pass_count = 0
    trials_run = 0
    skipped_reference_perturbations = 0
    total_correct_trials = (
        max(num_correct_trials, len(CORRECTNESS_INPUT_PERTURBATIONS))
        if enable_input_perturbations and num_correct_trials > 0
        else num_correct_trials
    )
    expected_pass_count = total_correct_trials
    correctness_start = perf_counter()
    trial_durations: list[float] = []
    reference_trial_durations: list[float] = []
    custom_trial_durations: list[float] = []
    compare_trial_durations: list[float] = []
    input_generation_durations: list[float] = []
    input_transfer_durations: list[float] = []
    reference_alias_clone_durations: list[float] = []

    def _set_substage(name: str, *, trial: int | None = None) -> None:
        if trial is not None:
            metadata["correctness_current_trial"] = trial
        metadata["correctness_current_substage"] = name
        if stage_update_fn is not None:
            stage_update_fn(f"kernel.correctness.{name}")

    if stop_on_first_failure is None:
        stop_on_first_failure = _env_flag(_CORRECTNESS_EARLY_STOP_ENV, default=True)
    if max_wall_time_s is None:
        max_wall_time_s = _env_positive_float(_CORRECTNESS_MAX_WALL_S_ENV)
    if pass_on_time_budget is None:
        pass_on_time_budget = _env_flag(_CORRECTNESS_PASS_ON_BUDGET_ENV, default=False)
    if budget_min_pass_trials is None:
        budget_min_pass_trials = _env_positive_int(_CORRECTNESS_BUDGET_MIN_PASS_TRIALS_ENV) or 1
    budget_min_pass_trials = max(1, min(int(budget_min_pass_trials), total_correct_trials))
    generate_inputs_on_gpu = _env_flag(_CORRECTNESS_GPU_INPUTS_ENV, default=True)

    metadata["correctness_early_stop_enabled"] = bool(stop_on_first_failure)
    metadata["correctness_budget_pass_on_success_enabled"] = bool(pass_on_time_budget)
    metadata["correctness_budget_min_pass_trials"] = budget_min_pass_trials
    metadata["correctness_inplace_compare_enabled"] = True
    metadata["correctness_reference_alias_clone_trials"] = []
    metadata["correctness_reference_cache_poison_enabled"] = True
    metadata["correctness_forward_seed_reset_enabled"] = True
    metadata["correctness_tolerance_source"] = "kernelbench_precision_or_fp32_integral"
    metadata["aten_detection_enabled"] = bool(detect_aten_fallback)
    metadata["correctness_inputs_generated_on_gpu"] = bool(generate_inputs_on_gpu)
    metadata["correctness_input_perturbations_enabled"] = bool(enable_input_perturbations)
    metadata["correctness_requested_trials"] = int(num_correct_trials)
    metadata["correctness_effective_trials"] = int(total_correct_trials)
    if enable_input_perturbations:
        metadata["correctness_input_perturbation_trials"] = []
    metadata["correctness_reference_skipped_perturbations"] = []
    metadata["correctness_candidate_forward_completed"] = False
    metadata["correctness_candidate_forward_completed_trials"] = []
    metadata["correctness_output_mismatch"] = False
    record_execution_policy(metadata)
    if max_wall_time_s is not None:
        metadata["correctness_max_wall_s"] = max_wall_time_s

    def _record_trial_metadata() -> None:
        metadata["correctness_trials"] = f"({pass_count} / {expected_pass_count})"
        metadata["correctness_trials_run"] = trials_run
        metadata["correctness_reference_skipped_perturbation_count"] = skipped_reference_perturbations
        metadata["correctness_trial_s"] = trial_durations
        metadata["correctness_reference_trial_s"] = reference_trial_durations
        metadata["correctness_custom_trial_s"] = custom_trial_durations
        metadata["correctness_compare_trial_s"] = compare_trial_durations
        metadata["correctness_input_generation_trial_s"] = input_generation_durations
        metadata["correctness_input_transfer_trial_s"] = input_transfer_durations
        metadata["correctness_reference_alias_clone_trial_s"] = reference_alias_clone_durations

    def _result(*, correctness: bool) -> KernelExecResult:
        """Build a result without dropping hard decoy evidence on failure exits."""

        return KernelExecResult(
            compiled=True,
            correctness=correctness,
            decoy_kernel=bool(metadata.get("policy_violation")),
            metadata=metadata,
        )

    def _record_aten_trial_metrics(aten_metrics: dict[str, Any], trial: int) -> None:
        trial_record = {"trial": trial, **aten_metrics}
        trial_records = metadata.setdefault("aten_detection_trials", [])
        trial_records.append(trial_record)
        metadata["aten_detection_trials_run"] = len(trial_records)
        metadata["aten_allowlist_version"] = aten_metrics.get("aten_allowlist_version")
        metadata["aten_detection_valid"] = all(bool(record.get("aten_detection_valid")) for record in trial_records)

        errors = [
            {"trial": record["trial"], "error": record.get("aten_detection_error")}
            for record in trial_records
            if not record.get("aten_detection_valid")
        ]
        if errors:
            metadata["aten_detection_errors"] = errors

        merged_ops: dict[str, dict[str, Any]] = {}
        for record in trial_records:
            for item in record.get("aten_ops", []):
                name = item["name"]
                merged = merged_ops.setdefault(
                    name,
                    {
                        "name": name,
                        "normalized_name": item["normalized_name"],
                        "count": 0,
                        "cpu_time_us": 0.0,
                        "allowed": item["allowed"],
                    },
                )
                merged["count"] += item["count"]
                merged["cpu_time_us"] += item["cpu_time_us"]

        aten_ops = [merged_ops[name] for name in sorted(merged_ops)]
        allowed_ops = [item for item in aten_ops if item["allowed"]]
        forbidden_ops = [item for item in aten_ops if not item["allowed"]]
        metadata["aten_ops"] = aten_ops
        metadata["allowed_aten_ops"] = allowed_ops
        metadata["forbidden_aten_ops"] = forbidden_ops
        metadata["forbidden_aten_op_names"] = [item["name"] for item in forbidden_ops]
        metadata["aten_operator_count"] = sum(item["count"] for item in aten_ops)
        metadata["allowed_aten_operator_count"] = sum(item["count"] for item in allowed_ops)
        metadata["forbidden_aten_operator_count"] = sum(item["count"] for item in forbidden_ops)

        if forbidden_ops:
            metadata["policy_violation"] = True
            metadata["policy_violation_reason"] = "DISALLOWED_ATEN_COMPUTE"
            metadata["decoy_reason"] = "DISALLOWED_ATEN_COMPUTE"
            logger.warning(
                "[ATen Detection] Forbidden candidate operators: %s",
                metadata["forbidden_aten_op_names"],
            )

    def _maybe_finish_on_time_budget() -> KernelExecResult | None:
        if max_wall_time_s is None or trials_run == 0:
            return None
        elapsed_before_trial = perf_counter() - correctness_start
        if elapsed_before_trial < max_wall_time_s:
            return None

        metadata["correctness_time_budget_exceeded"] = True
        _record_trial_metadata()
        all_completed_trials_passed = pass_count + skipped_reference_perturbations == trials_run
        if pass_on_time_budget and all_completed_trials_passed:
            if pass_count >= budget_min_pass_trials:
                metadata["correctness_budget_passed_early"] = True
                metadata["correctness_issue_name"] = "correctness_time_budget_passed_early"
                metadata["correctness_issue"] = (
                    f"Correctness time budget reached after {pass_count} passing trials; accepted early"
                )
                return _result(correctness=True)
            metadata["correctness_time_budget_overrun_to_min_pass"] = True
            return None

        metadata["correctness_issue_name"] = "correctness_time_budget_exceeded"
        metadata["correctness_issue"] = (
            f"Correctness time budget exceeded after {trials_run} / {total_correct_trials} trials"
        )
        return _result(correctness=False)

    torch.manual_seed(seed)
    correctness_trial_seeds = [int(torch.randint(0, 2**32 - 1, (1,)).item()) for _ in range(total_correct_trials)]

    def _record_runtime_exception(
        exception: Exception,
        *,
        trial: int,
        trial_start: float,
        trial_seed: int,
        perturbation: str,
    ) -> KernelExecResult:
        nonlocal metadata, trials_run
        trials_run = trial + 1
        trial_durations.append(perf_counter() - trial_start)
        logger.warning("[Error] Exception happened during correctness check")
        logger.warning("Error in launching kernel for ModelNew: %s", exception)

        metadata = register_and_format_exception("runtime_error", exception, metadata, truncate=False)
        metadata["runtime_error_name"] = get_error_name(exception)
        metadata["correctness_failed_trial"] = trial
        # Internal-only replay input; result serialization strips this field.
        metadata["correctness_failed_trial_seed"] = int(trial_seed)
        metadata["correctness_failed_input_perturbation"] = perturbation
        metadata["correctness_runtime_error_stage"] = metadata.get("correctness_current_substage")
        _record_trial_metadata()
        return _result(correctness=False)

    def _record_reference_perturbation_skip(
        *,
        trial: int,
        trial_start: float,
        trial_seed: int,
        perturbation: str,
        reason: str,
    ) -> None:
        nonlocal expected_pass_count, skipped_reference_perturbations, trials_run
        expected_pass_count -= 1
        skipped_reference_perturbations += 1
        trials_run = trial + 1
        trial_durations.append(perf_counter() - trial_start)
        metadata["correctness_reference_skipped_perturbations"].append(
            {
                "trial": trial,
                "seed": int(trial_seed),
                "perturbation": perturbation,
                "reason": reason,
            }
        )
        _record_trial_metadata()

    with tf32_execution_context(metadata, stage="correctness"), torch.no_grad():
        set_seed(seed)
        model = prepare_model_for_execution(original_model_instance.cuda(device=device))
        set_seed(seed)
        model_new = prepare_model_for_execution(new_model_instance.cuda(device=device))

        for trial in range(total_correct_trials):
            if trial > 0:
                budget_result = _maybe_finish_on_time_budget()
                if budget_result is not None:
                    return budget_result

            trial_start = perf_counter()
            trial_seed = correctness_trial_seeds[trial]
            perturbation = (
                CORRECTNESS_INPUT_PERTURBATIONS[trial % len(CORRECTNESS_INPUT_PERTURBATIONS)]
                if enable_input_perturbations
                else PERTURBATION_ORIGINAL
            )
            if verbose:
                logger.info(
                    "[Eval] Generating Random Input with seed %s, perturbation=%s",
                    trial_seed,
                    perturbation,
                )

            set_seed(trial_seed)
            _set_substage("input_generation", trial=trial)
            input_generation_start = perf_counter()
            with _input_generation_device_context(device, enabled=generate_inputs_on_gpu):
                if enable_input_perturbations:
                    with capture_random_input_origins() as origins:
                        inputs = get_inputs_fn()
                    inputs, perturbation_summary = apply_input_perturbation(inputs, origins, perturbation)
                else:
                    inputs = get_inputs_fn()
                    perturbation_summary = {
                        "name": PERTURBATION_ORIGINAL,
                        "detected_input_kinds": {},
                        "transforms": {},
                        "transformed_tensor_count": 0,
                    }
            if enable_input_perturbations:
                metadata["correctness_input_perturbation_trials"].append(
                    {"trial": trial, "seed": int(trial_seed), **perturbation_summary}
                )
            input_generation_durations.append(perf_counter() - input_generation_start)

            _set_substage("input_transfer", trial=trial)
            input_transfer_start = perf_counter()
            inputs = _move_input_to_device(inputs, device=device)
            input_transfer_durations.append(perf_counter() - input_transfer_start)

            if verbose:
                first_input_device = getattr(inputs[0], "device", None) if inputs else None
                logger.debug("device: %s", device)
                logger.debug("inputs: %s", first_input_device)

            # Reseed immediately before each forward so models with an
            # RNG-consuming op (e.g. torch.bernoulli) draw from the same RNG
            # state in both runs; otherwise the reference forward advances the
            # RNG and the candidate mismatches even when identical.
            _set_substage("reference_forward", trial=trial)
            # Input generation may consume RNG. Start both forwards from the
            # same per-trial Torch RNG state so stochastic PyTorch operations
            # are comparable when ModelNew uses the same RNG implementation.
            set_seed(trial_seed)
            reference_start = perf_counter()
            try:
                output = model(*inputs)
                torch.cuda.synchronize(device=device)
            except Exception as reference_exc:
                reference_trial_durations.append(perf_counter() - reference_start)
                if enable_input_perturbations and perturbation != PERTURBATION_ORIGINAL:
                    _record_reference_perturbation_skip(
                        trial=trial,
                        trial_start=trial_start,
                        trial_seed=trial_seed,
                        perturbation=perturbation,
                        reason=f"{type(reference_exc).__name__}: {reference_exc}",
                    )
                    del inputs
                    continue
                raise
            reference_trial_durations.append(perf_counter() - reference_start)

            if (
                enable_input_perturbations
                and perturbation != PERTURBATION_ORIGINAL
                and not _outputs_are_finite(output)
            ):
                _record_reference_perturbation_skip(
                    trial=trial,
                    trial_start=trial_start,
                    trial_seed=trial_seed,
                    perturbation=perturbation,
                    reason="reference output contains NaN or Inf",
                )
                del inputs, output
                continue

            _set_substage("reference_alias_clone", trial=trial)
            alias_clone_start = perf_counter()
            if _output_aliases_inputs(output, inputs):
                output = _clone_output_on_device(output)
                metadata["correctness_reference_alias_clone_trials"].append(trial)
                reference_alias_clone_durations.append(perf_counter() - alias_clone_start)
            else:
                reference_alias_clone_durations.append(0.0)

            poison_scratch = _zero_poison_like(output)
            if any(True for _ in _iter_tensors(poison_scratch)):
                torch.cuda.synchronize(device=device)
            del poison_scratch

            try:
                _set_substage("custom_forward", trial=trial)
                set_seed(trial_seed)
                custom_start = perf_counter()
                profile_aten_operators = bool(detect_aten_fallback)
                with aten_operator_profiling_context(profile_aten_operators) as aten_prof:
                    output_new = model_new(*inputs)
                    torch.cuda.synchronize(device=device)
                metadata["correctness_candidate_forward_completed"] = True
                metadata["correctness_candidate_forward_completed_trials"].append(trial)
                if profile_aten_operators:
                    aten_metrics = extract_aten_operator_metrics(aten_prof)
                    _record_aten_trial_metrics(aten_metrics, trial)
                custom_trial_durations.append(perf_counter() - custom_start)
                trials_run = trial + 1
                del inputs

                if not _structures_match(output, output_new):
                    metadata["correctness_output_mismatch"] = True
                    # Internal-only replay input, independent of detailed diagnostics.
                    metadata["correctness_failed_trial_seed"] = int(trial_seed)
                    expected_shape = _describe_structure(output)
                    got_shape = _describe_structure(output_new)
                    compare_trial_durations.append(0.0)
                    trial_durations.append(perf_counter() - trial_start)
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        (
                            f"Output shape mismatch under input perturbation {perturbation}: "
                            f"Expected {expected_shape}, got {got_shape}"
                        ),
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "output_structure_mismatch"
                    metadata["correctness_failed_trial"] = trial
                    metadata["correctness_failed_input_perturbation"] = perturbation
                    _record_trial_metadata()
                    if verbose:
                        logger.warning(
                            "[FAIL] trial %s: Output shape mismatch: Expected %s, got %s",
                            trial,
                            expected_shape,
                            got_shape,
                        )
                    return _result(correctness=False)

                _set_substage("compare", trial=trial)
                compare_start = perf_counter()
                reference_leaf = _first_tensor(output)
                if reference_leaf is not None:
                    tolerance = get_tolerance_for_dtype(reference_leaf.dtype)
                    metadata["correctness_atol"] = tolerance
                    metadata["correctness_rtol"] = tolerance
                if return_detail_correctness:
                    (
                        outputs_close,
                        max_diff,
                        avg_diff,
                        curve_counts,
                        total_elements,
                        nan_count,
                        inf_count,
                        mismatch_coordinates,
                    ) = _compare_outputs_inplace_with_diagnostics(output, output_new)
                else:
                    outputs_close, max_diff, avg_diff = _compare_outputs_inplace(output, output_new)
                compare_trial_durations.append(perf_counter() - compare_start)
                trial_durations.append(perf_counter() - trial_start)

                if not outputs_close:
                    metadata["correctness_output_mismatch"] = True
                    metadata["correctness_failed_trial_seed"] = int(trial_seed)
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    if return_detail_correctness:
                        element_correctness_curve = _format_element_correctness_curve(curve_counts, total_elements)
                        element_correctness_text = element_correctness_curve["1x"]
                        metadata.setdefault("element_correctness_curve", []).append(element_correctness_curve)
                        metadata.setdefault("nan_count", []).append(nan_count)
                        metadata.setdefault("inf_count", []).append(inf_count)
                        metadata.setdefault("mismatch_localization", []).append(
                            _format_mismatch_localization(mismatch_coordinates)
                        )
                        metadata["correctness_issue"] = (
                            f"Numerical output mismatch under input perturbation {perturbation}: "
                            f"max_difference={max_diff:.6g}, avg_difference={avg_diff:.6g}, "
                            f"element_correctness={element_correctness_text}, "
                            "element_correctness_curve="
                            f"{_format_element_correctness_curve_text(element_correctness_curve)}, "
                            f"atol={metadata.get('correctness_atol')}, rtol={metadata.get('correctness_rtol')}"
                        )
                    else:
                        metadata["correctness_issue"] = (
                            f"Numerical output mismatch under input perturbation {perturbation}: "
                            f"max_difference={max_diff:.6g}, avg_difference={avg_diff:.6g}, "
                            f"atol={metadata.get('correctness_atol')}, rtol={metadata.get('correctness_rtol')}"
                        )
                    metadata["correctness_issue_name"] = "numerical_mismatch"
                    metadata["correctness_failed_trial"] = trial
                    metadata["correctness_failed_input_perturbation"] = perturbation
                    if verbose:
                        logger.warning("[FAIL] trial %s: Output mismatch", trial)
                    if stop_on_first_failure:
                        metadata["correctness_early_stopped"] = True
                        _record_trial_metadata()
                        return _result(correctness=False)
                else:
                    pass_count += 1
                    if verbose:
                        logger.info("[PASS] trial %s: New Model matches Model", trial)
                del output, output_new

            except Exception as e:
                if len(custom_trial_durations) < len(reference_trial_durations):
                    custom_trial_durations.append(perf_counter() - custom_start)
                return _record_runtime_exception(
                    e,
                    trial=trial,
                    trial_start=trial_start,
                    trial_seed=trial_seed,
                    perturbation=perturbation,
                )

    if verbose:
        logger.info("[Eval] Pass count: %s, num_correct_trials: %s", pass_count, total_correct_trials)

    _record_trial_metadata()

    if pass_count == expected_pass_count and expected_pass_count > 0:
        return _result(correctness=True)
    return _result(correctness=False)
