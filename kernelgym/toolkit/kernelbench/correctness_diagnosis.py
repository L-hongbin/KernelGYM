"""High-confidence correctness diagnosis from already-collected metadata."""

from __future__ import annotations

from typing import Any


def _latest(value: Any) -> Any:
    if isinstance(value, list):
        return value[-1] if value else None
    return value


def _percentage(value: Any) -> float | None:
    if not isinstance(value, str) or not value.endswith("%"):
        return None
    try:
        return float(value[:-1])
    except ValueError:
        return None


def _diagnosis(
    category: str,
    confidence: float,
    text: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "category": category,
        "confidence": confidence,
        "text": text,
        "evidence": evidence,
    }


def _localized_diagnosis(metadata: dict[str, Any]) -> dict[str, Any] | None:
    mismatch_localization = _latest(metadata.get("mismatch_localization"))
    localization = (
        mismatch_localization.get("output_space")
        if isinstance(mismatch_localization, dict)
        else None
    )
    if not isinstance(localization, dict):
        return None
    tensors = localization.get("tensors")
    if not isinstance(tensors, list):
        return None

    # A strict terminal-tile pattern is the strongest output-space signal for
    # an incorrect tail mask, ceil-div grid, or final-tile index calculation.
    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        shape = tensor.get("shape")
        tile = tensor.get("tile")
        if not isinstance(shape, list) or len(shape) < 2 or not isinstance(tile, dict):
            continue
        mismatches = tile.get("mismatches")
        mismatch_count = tile.get("mismatch_count")
        total = tile.get("total")
        if (
            not isinstance(mismatches, list)
            or not isinstance(mismatch_count, int)
            or not isinstance(total, int)
            or tile.get("mismatches_truncated")
            or mismatch_count != len(mismatches)
            or not (0 < mismatch_count < total)
        ):
            continue
        for axis, dimension in (("M", shape[-2]), ("N", shape[-1])):
            ranges = [item.get(axis) for item in mismatches if isinstance(item, dict)]
            if len(ranges) != mismatch_count or not ranges:
                continue
            if all(
                isinstance(bounds, list)
                and len(bounds) == 2
                and isinstance(bounds[1], int)
                and bounds[1] == dimension
                for bounds in ranges
            ):
                path = str(tensor.get("output_path", "output"))
                return _diagnosis(
                    "tail_or_boundary_error",
                    0.92,
                    (
                        f"Mismatch is confined to terminal {axis}-axis tile(s) of {path}; "
                        "inspect tail masking, ceil-div grid sizing, and boundary indexing."
                    ),
                    [
                        f"{mismatch_count}/{total} tiles fail and every failed tile touches {axis}={dimension}",
                        f"{total - mismatch_count} preceding/non-terminal tiles are correct",
                    ],
                )

    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        path = str(tensor.get("output_path", "output"))
        for unit_name, category, confidence, hint in (
            (
                "batch",
                "batch_indexing_error",
                0.88,
                "inspect batch stride, batch program-id mapping, and batch offsets",
            ),
            (
                "row",
                "row_localized_error",
                0.82,
                "inspect row stride, row program-id mapping, and leading-dimension indexing",
            ),
            (
                "tile",
                "tile_localized_error",
                0.78,
                "inspect tile program-id mapping, tile strides, and tile masks",
            ),
        ):
            units = tensor.get(unit_name)
            if not isinstance(units, dict):
                continue
            mismatch_count = units.get("mismatch_count")
            total = units.get("total")
            mismatches = units.get("mismatches")
            if (
                isinstance(mismatch_count, int)
                and isinstance(total, int)
                and isinstance(mismatches, list)
                and not units.get("mismatches_truncated")
                and mismatch_count == len(mismatches)
                and 0 < mismatch_count < total
            ):
                return _diagnosis(
                    category,
                    confidence,
                    f"Mismatch is localized to {mismatch_count}/{total} {unit_name} units of {path}; {hint}.",
                    [f"{total - mismatch_count}/{total} {unit_name} units are fully correct"],
                )
    return None


def diagnose_correctness_failure(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Return a diagnosis only for strong patterns already present in metadata."""

    if metadata.get("runtime_error") or not metadata.get("correctness_output_mismatch"):
        return None

    issue_name = metadata.get("correctness_issue_name")
    if issue_name == "output_structure_mismatch":
        return _diagnosis(
            "output_contract_error",
            0.99,
            "Candidate output structure or shape differs from the reference; inspect output allocation and return structure.",
            [str(metadata.get("correctness_issue", "output structure mismatch"))],
        )
    if issue_name != "numerical_mismatch":
        return None

    nan_count = _latest(metadata.get("nan_count"))
    inf_count = _latest(metadata.get("inf_count"))
    nan_count = nan_count if isinstance(nan_count, int) else 0
    inf_count = inf_count if isinstance(inf_count, int) else 0
    if nan_count or inf_count:
        return _diagnosis(
            "nonfinite_output",
            0.98,
            (
                f"Candidate output contains {nan_count} NaN and {inf_count} Inf values; "
                "inspect invalid arithmetic, divide-by-zero, overflow, and missing initialization."
            ),
            [f"nan_count={nan_count}", f"inf_count={inf_count}"],
        )

    localized = _localized_diagnosis(metadata)
    if localized is not None:
        return localized

    curve = _latest(metadata.get("element_correctness_curve"))
    if not isinstance(curve, dict) or not curve:
        return None
    parsed = [(str(multiplier), _percentage(value)) for multiplier, value in curve.items()]
    if any(value is None for _multiplier, value in parsed):
        return None
    points = [(multiplier, float(value)) for multiplier, value in parsed if value is not None]
    first_multiplier, first_value = points[0]
    last_multiplier, last_value = points[-1]
    if first_value < 100.0 and last_value == 100.0:
        return _diagnosis(
            "tolerance_sensitive_numerical_error",
            0.90,
            (
                f"All elements pass at {last_multiplier} tolerance but only {first_value:.2f}% pass at "
                f"{first_multiplier}; inspect numerical precision, approximation, and accumulation order."
            ),
            [f"element correctness rises from {first_value:.2f}% to 100.00%"],
        )
    values = [value for _multiplier, value in points]
    if first_value <= 50.0 and max(values) - min(values) <= 0.01:
        return _diagnosis(
            "large_magnitude_or_indexing_error",
            0.80,
            (
                f"Element correctness remains {first_value:.2f}% through {last_multiplier} tolerance; "
                "this is unlikely to be a small precision error, so inspect indexing, permutation, masking, or core logic."
            ),
            [f"tolerance curve is flat from {first_multiplier} through {last_multiplier}"],
        )
    return None


def maybe_record_correctness_diagnosis(
    metadata: dict[str, Any],
    *,
    return_detail_correctness: bool,
    sanitizer_dispatched: bool,
) -> dict[str, Any] | None:
    """Attach a diagnosis only in detailed mode and when Sanitizer did not run."""

    if not return_detail_correctness or sanitizer_dispatched:
        return None
    diagnosis = diagnose_correctness_failure(metadata)
    if diagnosis is not None:
        metadata["correctness_diagnosis"] = diagnosis
    return diagnosis
