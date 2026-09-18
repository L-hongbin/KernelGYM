from __future__ import annotations

from kernelgym.schema.result import KernelEvaluationResult
from kernelgym.toolkit.kernelbench.correctness_diagnosis import (
    diagnose_correctness_failure,
    maybe_record_correctness_diagnosis,
)
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult


def _numerical_metadata(**updates):
    metadata = {
        "correctness_output_mismatch": True,
        "correctness_issue_name": "numerical_mismatch",
        "correctness_issue": "Numerical output mismatch",
        "nan_count": [0],
        "inf_count": [0],
    }
    metadata.update(updates)
    return metadata


def test_diagnoses_output_contract_mismatch() -> None:
    diagnosis = diagnose_correctness_failure(
        {
            "correctness_output_mismatch": True,
            "correctness_issue_name": "output_structure_mismatch",
            "correctness_issue": "Expected (4, 8), got (4, 7)",
        }
    )

    assert diagnosis is not None
    assert diagnosis["category"] == "output_contract_error"
    assert diagnosis["confidence"] == 0.99
    assert "version" not in diagnosis


def test_nonfinite_output_has_priority_over_localization() -> None:
    diagnosis = diagnose_correctness_failure(
        _numerical_metadata(
            nan_count=[2],
            inf_count=[1],
            mismatch_localization=[{"element": {}, "output_space": {"tensors": []}}],
        )
    )

    assert diagnosis is not None
    assert diagnosis["category"] == "nonfinite_output"
    assert "2 NaN and 1 Inf" in diagnosis["text"]


def test_diagnoses_terminal_tile_pattern() -> None:
    diagnosis = diagnose_correctness_failure(
        _numerical_metadata(
            mismatch_localization=[
                {
                    "element": {},
                    "output_space": {
                        "tensors": [
                            {
                                "output_path": "output",
                                "shape": [64, 64],
                                "tile": {
                                    "mismatch_count": 2,
                                    "total": 4,
                                    "mismatches": [
                                        {"M": [0, 32], "N": [32, 64]},
                                        {"M": [32, 64], "N": [32, 64]},
                                    ],
                                    "mismatches_truncated": False,
                                },
                            }
                        ]
                    },
                }
            ]
        )
    )

    assert diagnosis is not None
    assert diagnosis["category"] == "tail_or_boundary_error"
    assert "terminal N-axis" in diagnosis["text"]


def test_diagnoses_partial_batch_pattern() -> None:
    diagnosis = diagnose_correctness_failure(
        _numerical_metadata(
            mismatch_localization=[
                {
                    "element": {},
                    "output_space": {
                        "tensors": [
                            {
                                "output_path": "output",
                                "shape": [4, 32, 32],
                                "tile": {
                                    "mismatch_count": 4,
                                    "total": 4,
                                    "mismatches": [],
                                    "mismatches_truncated": True,
                                },
                                "batch": {
                                    "mismatch_count": 1,
                                    "total": 4,
                                    "mismatches": [{"batch": 2}],
                                    "mismatches_truncated": False,
                                },
                            }
                        ]
                    },
                }
            ]
        )
    )

    assert diagnosis is not None
    assert diagnosis["category"] == "batch_indexing_error"


def test_diagnoses_tolerance_sensitive_error() -> None:
    diagnosis = diagnose_correctness_failure(
        _numerical_metadata(
            element_correctness_curve=[
                {"1x": "61.00%", "2x": "72.00%", "4x": "83.00%", "8x": "100.00%"}
            ]
        )
    )

    assert diagnosis is not None
    assert diagnosis["category"] == "tolerance_sensitive_numerical_error"
    assert "8x tolerance" in diagnosis["text"]


def test_diagnoses_flat_low_tolerance_curve() -> None:
    diagnosis = diagnose_correctness_failure(
        _numerical_metadata(
            element_correctness_curve=[
                {"1x": "12.50%", "2x": "12.50%", "4x": "12.50%", "8x": "12.50%", "16x": "12.50%"}
            ]
        )
    )

    assert diagnosis is not None
    assert diagnosis["category"] == "large_magnitude_or_indexing_error"


def test_ambiguous_curve_does_not_emit_diagnosis() -> None:
    diagnosis = diagnose_correctness_failure(
        _numerical_metadata(
            element_correctness_curve=[
                {"1x": "61.00%", "2x": "72.00%", "4x": "83.00%", "8x": "91.00%", "16x": "95.00%"}
            ]
        )
    )

    assert diagnosis is None


def test_detailed_gate_and_sanitizer_dispatch_suppress_diagnosis() -> None:
    metadata = _numerical_metadata(
        element_correctness_curve=[{"1x": "0.00%", "2x": "0.00%", "4x": "0.00%"}]
    )

    assert (
        maybe_record_correctness_diagnosis(
            metadata,
            return_detail_correctness=False,
            sanitizer_dispatched=False,
        )
        is None
    )
    assert (
        maybe_record_correctness_diagnosis(
            metadata,
            return_detail_correctness=True,
            sanitizer_dispatched=True,
        )
        is None
    )
    assert "correctness_diagnosis" not in metadata


def test_diagnosis_is_appended_to_correctness_error_message() -> None:
    metadata = _numerical_metadata(
        nan_count=[2], inf_count=[1],
    )
    diagnosis = maybe_record_correctness_diagnosis(
        metadata, return_detail_correctness=True, sanitizer_dispatched=False,
    )
    assert diagnosis is not None
    assert "text" not in diagnosis
    assert "2 NaN and 1 Inf" in metadata["correctness_issue"]
    issue = metadata["correctness_issue"]
    maybe_record_correctness_diagnosis(
        metadata, return_detail_correctness=True, sanitizer_dispatched=False,
    )
    assert metadata["correctness_issue"] == issue
    result = KernelEvaluationResult.from_kernel_exec_result(
        "child",
        "parent",
        KernelExecResult(compiled=True, correctness=False, metadata=metadata),
    )

    assert result.error_code is not None
    assert result.error_code.value == "CORRECTNESS_ERROR"
    assert result.error_message == f"Kernel produced incorrect results: {issue}"
    assert result.error_message.count("Diagnosis:") == 1
    assert "text" not in result.metadata["correctness_diagnosis"]


def test_sanitizer_issue_uses_sanitizer_message_instead_of_diagnosis() -> None:
    result = KernelEvaluationResult.from_kernel_exec_result(
        "child",
        "parent",
        KernelExecResult(
            compiled=True,
            correctness=False,
            metadata={
                "correctness_issue": "Numerical output mismatch",
                "correctness_diagnosis": {"text": "This must not be returned."},
            },
            runtime_sanitizer={
                "status": "issues_found",
                "check_results": [{"issues": [{"message": "Invalid global write"}]}],
            },
        ),
    )

    assert result.error_message == "Runtime Sanitizer detected an unsafe CUDA kernel: Invalid global write"
    assert "Diagnosis" not in result.error_message
