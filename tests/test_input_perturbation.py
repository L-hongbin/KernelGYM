from __future__ import annotations

import torch

from kernelgym.common import ErrorCode
from kernelgym.config import settings
from kernelgym.schema.result import KernelEvaluationResult
from kernelgym.schema.task import EvaluationTask
from kernelgym.server.api.models import EvaluationRequest
from kernelgym.server.request_defaults import apply_runtime_defaults
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult
from kernelgym.toolkit.kernelbench.input_perturbation import (
    PERTURBATION_SCALE_DOWN,
    PERTURBATION_SCALE_UP,
    PERTURBATION_SIGN_CHALLENGE,
    apply_input_perturbation,
    capture_random_input_origins,
)
from kernelgym.workflow.kernelbench import KernelBenchWorkflowController
from kernelgym.workflow.kernelbench_helpers import _create_paired_tasks


def _captured_inputs() -> tuple[list[torch.Tensor], object]:
    torch.manual_seed(1234)
    with capture_random_input_origins() as origins:
        inputs = [
            torch.rand(64),
            torch.randn(64),
            torch.randint(0, 8, (64,)),
        ]
    return inputs, origins


def test_sign_challenge_uses_distribution_specific_transforms() -> None:
    inputs, origins = _captured_inputs()
    transformed, summary = apply_input_perturbation(inputs, origins, PERTURBATION_SIGN_CHALLENGE)

    assert torch.equal(transformed[0], -inputs[0])
    assert torch.equal(transformed[1], inputs[1].abs())
    assert transformed[2] is inputs[2]
    assert summary["detected_input_kinds"] == {"torch.rand": 1, "torch.randn": 1}
    assert summary["transforms"] == {"negate": 1, "absolute": 1}


def test_scale_perturbations_only_change_recognized_float_inputs() -> None:
    for perturbation, factor in ((PERTURBATION_SCALE_UP, 3.0), (PERTURBATION_SCALE_DOWN, 0.01)):
        inputs, origins = _captured_inputs()
        transformed, summary = apply_input_perturbation(inputs, origins, perturbation)

        assert torch.allclose(transformed[0], inputs[0] * factor)
        assert torch.allclose(transformed[1], inputs[1] * factor)
        assert transformed[2] is inputs[2]
        assert summary["transformed_tensor_count"] == 2


def test_input_perturbation_api_defaults_disabled_and_propagates() -> None:
    request = EvaluationRequest(
        task_id="input-perturbation-default",
        reference_code="class Model:\n    pass",
        kernel_code="class ModelNew:\n    pass",
    )
    assert request.enable_correctness_input_perturbations is None
    assert settings.enable_correctness_input_perturbations is False

    payload = apply_runtime_defaults(
        request.model_dump(),
        workflow_name="kernelbench",
        split_compile_and_execute=False,
        enable_correctness_input_perturbations=False,
    )
    assert payload["enable_correctness_input_perturbations"] is False

    task = EvaluationTask(
        task_id="input-perturbation-enabled",
        reference_code="class Model: pass",
        kernel_code="class ModelNew: pass",
        enable_correctness_input_perturbations=True,
    )
    _, kernel_task = _create_paired_tasks(task)
    assert kernel_task.enable_correctness_input_perturbations is True
    options = KernelBenchWorkflowController()._kernel_execution_options(task, compile_only=False)
    assert options["enable_correctness_input_perturbations"] is True
    compile_only_options = KernelBenchWorkflowController()._kernel_execution_options(task, compile_only=True)
    assert compile_only_options["enable_correctness_input_perturbations"] is False


def test_numerical_correctness_error_is_returned_to_api_result() -> None:
    result = KernelExecResult(
        compiled=True,
        correctness=False,
        metadata={
            "correctness_issue_name": "numerical_mismatch",
            "correctness_failed_trial_seed": 839703249,
            "correctness_issue": (
                "Numerical output mismatch under input perturbation sign_challenge: "
                "max_difference=1, avg_difference=0.5, atol=0.0001, rtol=0.0001"
            ),
        },
    )

    public_result = KernelEvaluationResult.from_kernel_exec_result("task", "task", result)

    assert public_result.error_code == ErrorCode.CORRECTNESS_ERROR
    assert "sign_challenge" in str(public_result.error_message)
    assert "max_difference=1" in str(public_result.error_message)
    assert public_result.metadata["correctness_failed_trial_seed"] == 839703249
    assert "correctness_failed_trial_seed" not in public_result.to_dict()["metadata"]
