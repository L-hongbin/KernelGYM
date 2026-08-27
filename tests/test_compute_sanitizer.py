import subprocess
from pathlib import Path
from types import SimpleNamespace

from kernelgym.config import settings
from kernelgym.schema.result import EvaluationResult, KernelEvaluationResult
from kernelgym.schema.task import EvaluationTask
from kernelgym.server.api.models import EvaluationRequest, EvaluationResponse
from kernelgym.server.request_defaults import apply_runtime_defaults
from kernelgym.toolkit.kernelbench import compute_sanitizer
from kernelgym.toolkit.kernelbench import pipeline as kernelbench_pipeline
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult
from kernelgym.workflow.kernelbench import KernelBenchWorkflowController
from kernelgym.workflow.kernelbench_helpers import _create_paired_tasks

MEMCHECK_OUTPUT = """
========= Invalid __global__ write of size 4 bytes
=========     at 0x120 in unsafe.cu:17:oob_kernel(float*)
=========     by thread (3,0,0) in block (4,0,0)
=========     Address 0x7f00 is out of bounds
========= ERROR SUMMARY: 1 error
"""

RACECHECK_OUTPUT = """
========= Race reported between Write access at race.cu:21:race_kernel(float*)
=========     by thread (1,0,0) in block (0,0,0)
========= Hazard at race.cu:21:race_kernel(float*)
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
"""

SYNCCHECK_OUTPUT = """
========= Barrier error detected. Invalid arguments
=========     by thread (16,0,0) in block (0,0,0)
=========     Device Frame:sanitizer_sync_kernel(float*, float*, int)+0x3c0 in generated.cu:67
========= ERROR SUMMARY: 1 error
"""

CLEAN_OUTPUT = """
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
"""


def test_parse_memcheck_extracts_agent_fields() -> None:
    parsed = compute_sanitizer.parse_compute_sanitizer_output(MEMCHECK_OUTPUT, tool="memcheck")

    assert parsed["detected_issue_count"] == 1
    issue = parsed["issues"][0]
    assert issue["hazard_type"] == "invalid_global_write"
    assert issue["access_type"] == "write"
    assert issue["access_size_bytes"] == 4
    assert issue["kernel_info"] == [{"name": "oob_kernel(float*)", "source": "file unsafe.cu line 17"}]
    assert "kernel" not in issue and "source" not in issue
    assert issue["occurrence_count"] == 1
    assert "tool" not in issue
    assert issue["threads"] == {
        "x": [3, 3],
        "y": [0, 0],
        "z": [0, 0],
    }
    assert issue["blocks"]["x"] == [4, 4]
    assert issue["addresses"] == {"ranges": ["0x7f00", "0x7f00"]}


def test_parse_racecheck_extracts_hazard() -> None:
    parsed = compute_sanitizer.parse_compute_sanitizer_output(RACECHECK_OUTPUT, tool="racecheck")

    assert parsed["detected_issue_count"] == 1
    assert parsed["issues"][0]["hazard_type"] == "shared_memory_race"
    assert parsed["issues"][0]["kernel_info"] == [{"name": "race_kernel(float*)", "source": "file race.cu line 21"}]


def test_parse_synccheck_handles_source_at_end_of_device_frame() -> None:
    parsed = compute_sanitizer.parse_compute_sanitizer_output(SYNCCHECK_OUTPUT, tool="synccheck")

    assert parsed["detected_issue_count"] == 1
    issue = parsed["issues"][0]
    assert issue["hazard_type"] == "synchronization_error"
    assert issue["threads"]["x"] == [16, 16]
    assert issue["kernel_info"] == [
        {
            "name": "sanitizer_sync_kernel(float*, float*, int)",
            "source": "file generated.cu line 67",
        }
    ]


def test_parse_memcheck_aggregates_repeated_threads_blocks_and_addresses() -> None:
    diagnostics = []
    base_address = 0x7FAC4E000FA0
    for thread_x in range(232, 256):
        address = base_address + (thread_x - 232) * 4
        diagnostics.extend(
            [
                "========= Invalid __global__ write of size 4 bytes",
                "=========     at sanitizer_oob_kernel+0x110 in generated.cu:12",
                f"=========     by thread ({thread_x},0,0) in block (3,0,0)",
                f"=========     Access at {hex(address)} is out of bounds",
            ]
        )
    diagnostics.append("========= ERROR SUMMARY: 24 errors")

    parsed = compute_sanitizer.parse_compute_sanitizer_output("\n".join(diagnostics), tool="memcheck", max_issues=2)

    assert parsed["detected_issue_count"] == 24
    assert parsed["observed_issue_count"] == 24
    assert parsed["parsed_issue_count"] == 24
    assert parsed["unique_issue_count"] == 1
    assert parsed["returned_issue_count"] == 1
    assert parsed["aggregation_complete"] is True
    assert parsed["issues_truncated"] is False
    issue = parsed["issues"][0]
    assert issue["occurrence_count"] == 24
    assert "tool" not in issue
    assert issue["threads"]["x"] == [232, 255]
    assert issue["blocks"]["x"] == [3, 3]
    assert issue["addresses"] == {"ranges": [hex(base_address), hex(base_address + 23 * 4)]}
    assert len(issue["representative_occurrences"]) == 2


def test_parse_memcheck_secondarily_aggregates_equivalent_kernel_locations() -> None:
    output = """
========= Invalid __global__ write of size 4 bytes
=========     at aggregate_oob_1+0x60 in generated.cu:6
=========     by thread (0,0,0) in block (0,0,0)
=========     Access at 0x7f00 is out of bounds
========= Invalid __global__ write of size 4 bytes
=========     at aggregate_oob_2+0x60 in generated.cu:7
=========     by thread (0,0,0) in block (0,0,0)
=========     Access at 0x7f00 is out of bounds
========= ERROR SUMMARY: 2 errors
"""
    parsed = compute_sanitizer.parse_compute_sanitizer_output(output, tool="memcheck", max_issues=20)

    assert parsed["detected_issue_count"] == 2
    assert parsed["unique_issue_count"] == 1
    assert parsed["returned_issue_count"] == 1
    issue = parsed["issues"][0]
    assert issue["occurrence_count"] == 2
    assert issue["kernel_info"] == [
        {"name": "aggregate_oob_1", "source": "file generated.cu line 6"},
        {"name": "aggregate_oob_2", "source": "file generated.cu line 7"},
    ]
    assert "kernel" not in issue and "source" not in issue
    assert "raw_excerpt" not in issue


def test_parse_memcheck_keeps_distinct_source_lines_as_separate_groups() -> None:
    output = """
========= Invalid __global__ write of size 4 bytes
=========     at kernel_a+0x10 in generated.cu:12
=========     by thread (1,0,0) in block (0,0,0)
========= Invalid __global__ write of size 4 bytes
=========     at kernel_a+0x20 in generated.cu:18
=========     by thread (2,0,0) in block (0,0,0)
========= ERROR SUMMARY: 2 errors
"""
    parsed = compute_sanitizer.parse_compute_sanitizer_output(output, tool="memcheck", max_issues=20)

    assert parsed["detected_issue_count"] == 2
    assert parsed["unique_issue_count"] == 2
    assert [issue["kernel_info"] for issue in parsed["issues"]] == [
        [{"name": "kernel_a", "source": "file generated.cu line 12"}],
        [{"name": "kernel_a", "source": "file generated.cu line 18"}],
    ]


def test_parse_compute_sanitizer_returns_at_most_four_unique_issue_groups() -> None:
    diagnostics = []
    for line in range(10, 16):
        diagnostics.extend(
            [
                "========= Invalid __global__ write of size 4 bytes",
                f"=========     at kernel_{line}+0x10 in generated.cu:{line}",
                f"=========     by thread ({line},0,0) in block (0,0,0)",
            ]
        )
    diagnostics.append("========= ERROR SUMMARY: 6 errors")

    parsed = compute_sanitizer.parse_compute_sanitizer_output("\n".join(diagnostics), tool="memcheck", max_issues=20)

    assert parsed["detected_issue_count"] == 6
    assert parsed["unique_issue_count"] == 6
    assert parsed["returned_issue_count"] == 4
    assert parsed["issue_groups_truncated"] is True
    assert parsed["issues_truncated"] is True
    assert [issue["kernel_info"] for issue in parsed["issues"]] == [
        [{"name": f"kernel_{line}", "source": f"file generated.cu line {line}"}] for line in range(10, 14)
    ]


def test_build_command_uses_argv_and_supported_filters(tmp_path: Path) -> None:
    command = compute_sanitizer.build_compute_sanitizer_command(
        sanitizer_path="/opt/compute-sanitizer",
        tool="racecheck",
        payload_path=tmp_path / "payload.json",
        kernel_names=["kernel;touch /tmp/not-run"],
        max_kernels=3,
    )

    assert command[:3] == ["/opt/compute-sanitizer", "--tool", "racecheck"]
    assert command[command.index("--error-exitcode") + 1] == "86"
    assert command[command.index("--launch-count") + 1] == "3"
    assert command[command.index("--print-limit") + 1] == "5000"
    assert "kns=kernel;touch /tmp/not-run" in command
    assert command[-3:] == [
        "-m",
        "kernelgym.toolkit.kernelbench.compute_sanitizer_runner",
        str(tmp_path / "payload.json"),
    ]


def test_run_compute_sanitizer_aggregates_clean_and_failed_tools(monkeypatch, tmp_path: Path) -> None:
    fake_tool = tmp_path / "compute-sanitizer"
    fake_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    monkeypatch.setattr(compute_sanitizer, "_tool_version", lambda _path: "test-version")

    executed_tools = []

    def fake_run(command, **_kwargs):
        tool = command[command.index("--tool") + 1]
        executed_tools.append(tool)
        if tool == "memcheck":
            return SimpleNamespace(returncode=86, stdout="", stderr=MEMCHECK_OUTPUT)
        return SimpleNamespace(returncode=0, stdout="", stderr=CLEAN_OUTPUT)

    monkeypatch.setattr(compute_sanitizer, "_run_sanitizer_command", fake_run)
    result = compute_sanitizer.run_compute_sanitizer(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="cuda_agent",
        entry_point="Model",
        device="cuda:2",
        kernel_names=["oob_kernel"],
        sanitizer_path=str(fake_tool),
        timeout_s=10,
        max_kernels=2,
        max_issues=5,
        mode="full",
    )

    assert result["status"] == "issues_found"
    assert result["passed"] is False
    assert result["requested_checks"] == [
        "memcheck",
        "synccheck",
        "racecheck",
        "initcheck",
    ]
    assert result["executed_checks"] == result["requested_checks"]
    assert [item["check"] for item in result["check_results"]] == result["requested_checks"]
    assert {
        "tools",
        "requested_tools",
        "executed_tools",
        "primary_tool",
        "issue_count_by_tool",
    }.isdisjoint(result)
    assert result["measurement_complete"] is True
    assert result["detected_issue_count"] == 1
    assert result["primary_check"] == "memcheck"
    assert result["primary_detected_issue_count"] == 1
    assert result["issue_count_by_check"] == {
        "memcheck": 1,
        "synccheck": 0,
        "racecheck": 0,
        "initcheck": 0,
    }
    assert [item["input_generation"] for item in result["check_results"]] == [
        "gpu",
        "gpu",
        "gpu",
        "cpu_then_h2d",
    ]
    assert result["mode"] == "full"
    assert result["run_all_checks"] is True
    assert executed_tools == ["memcheck", "synccheck", "racecheck", "initcheck"]
    assert [item["status"] for item in result["check_results"]] == [
        "issues_found",
        "clean",
        "clean",
        "clean",
    ]


def test_initcheck_uses_cpu_generated_inputs_with_kernel_filter(monkeypatch, tmp_path: Path) -> None:
    fake_tool = tmp_path / "compute-sanitizer"
    fake_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    observed_env = {}

    def fake_run(command, **kwargs):
        tool = command[command.index("--tool") + 1]
        observed_env[tool] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr=CLEAN_OUTPUT)

    monkeypatch.setattr(compute_sanitizer, "_run_sanitizer_command", fake_run)
    result = compute_sanitizer.run_compute_sanitizer(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="tvm_ffi",
        entry_point="Model",
        device="cuda:0",
        kernel_names=["candidate_kernel"],
        sanitizer_path=str(fake_tool),
        mode="full",
        timeout_s=10,
        max_kernels=2,
        max_issues=5,
        generate_inputs_on_gpu=True,
    )

    assert observed_env["initcheck"]["KERNELGYM_COMPUTE_SANITIZER_TOOL"] == "initcheck"
    assert result["check_results"][-1]["input_generation"] == "cpu_then_h2d"
    assert result["check_results"][-1]["input_values_exactly_replayed"] is False
    assert all(item["input_values_exactly_replayed"] is True for item in result["check_results"][:-1])


def test_full_mode_reports_primary_count_instead_of_summing_tool_reports(monkeypatch, tmp_path: Path) -> None:
    fake_tool = tmp_path / "compute-sanitizer"
    fake_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_tool.chmod(0o755)

    def fake_run(command, **_kwargs):
        tool = command[command.index("--tool") + 1]
        count = 24 if tool == "memcheck" else (1000 if tool == "initcheck" else 0)
        output = f"========= ERROR SUMMARY: {count} errors\n"
        return SimpleNamespace(returncode=86 if count else 0, stdout="", stderr=output)

    monkeypatch.setattr(compute_sanitizer, "_run_sanitizer_command", fake_run)
    result = compute_sanitizer.run_compute_sanitizer(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="tvm_ffi",
        entry_point="Model",
        device="cuda:0",
        kernel_names=["candidate_kernel"],
        sanitizer_path=str(fake_tool),
        mode="full",
        primary_tool="memcheck",
        timeout_s=10,
        max_kernels=2,
        max_issues=5,
    )

    assert result["detected_issue_count"] == 24
    assert result["primary_detected_issue_count"] == 24
    assert result["issue_count_by_check"]["initcheck"] == 1000
    assert result["issues_truncated"] is True


def test_compute_sanitizer_mode_normalization_and_error_classification() -> None:
    assert compute_sanitizer.classify_compute_sanitizer_error("barrier error from __syncwarp") == "synccheck"
    assert compute_sanitizer.classify_compute_sanitizer_error("CUDA error: misaligned address") == "memcheck"
    assert compute_sanitizer.classify_compute_sanitizer_error("CUDA error: unspecified launch failure") is None
    for mode in (*compute_sanitizer.FULL_SANITIZER_TOOLS, "full"):
        assert compute_sanitizer.normalize_compute_sanitizer_execution_mode(mode) == mode

    assert kernelbench_pipeline._select_compute_sanitizer_execution_mode("barrier error from __syncwarp", None) == (
        "synccheck",
        "synccheck",
    )
    assert kernelbench_pipeline._select_compute_sanitizer_execution_mode(
        "CUDA error: unspecified launch failure", None
    ) == ("full", None)
    assert kernelbench_pipeline._select_compute_sanitizer_execution_mode("race hazard", "error_based") == (
        "racecheck",
        "racecheck",
    )
    assert kernelbench_pipeline._select_compute_sanitizer_execution_mode("race hazard", "full") == (
        "full",
        "racecheck",
    )

    assert compute_sanitizer.normalize_compute_sanitizer_execution_mode("MEMCHECK") == "memcheck"
    assert compute_sanitizer.normalize_compute_sanitizer_execution_mode("full") == "full"
    try:
        compute_sanitizer.normalize_compute_sanitizer_execution_mode("error_based")
    except ValueError:
        pass
    else:
        raise AssertionError("error_based must not be an execution mode")


def test_run_compute_sanitizer_runs_selected_mode_for_host_python_error(monkeypatch, tmp_path: Path) -> None:
    fake_tool = tmp_path / "compute-sanitizer"
    fake_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    monkeypatch.setattr(compute_sanitizer, "_tool_version", lambda _path: "test-version")
    executed = []
    host_error = """
========= COMPUTE-SANITIZER
========= Target application returned an error
========= ERROR SUMMARY: 0 errors
Traceback (most recent call last):
RuntimeError: intentional correctness-stage runtime failure after CUDA launch
"""

    def fake_run(command, **_kwargs):
        executed.append(command[command.index("--tool") + 1])
        return SimpleNamespace(returncode=1, stdout="", stderr=host_error)

    monkeypatch.setattr(compute_sanitizer, "_run_sanitizer_command", fake_run)
    result = compute_sanitizer.run_compute_sanitizer(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="tvm_ffi",
        entry_point="Model",
        device="cuda:0",
        kernel_names=["candidate_kernel"],
        sanitizer_path=str(fake_tool),
        mode="memcheck",
        timeout_s=10,
        max_kernels=2,
        max_issues=5,
    )

    assert executed == ["memcheck"]
    assert result["status"] == "clean"
    assert result["passed"] is True
    assert result["measurement_complete"] is True
    assert result["check_results"][0]["status"] == "clean"
    assert "error" not in result["check_results"][0]
    assert result["check_results"][0]["process_completed"] is True
    assert result["check_results"][0]["target_application_failed"] is True
    assert result["check_results"][0]["sanitizer_issue_found"] is False
    assert result["executed_checks"] == ["memcheck"]


def test_zero_issue_summary_must_be_explicit() -> None:
    assert compute_sanitizer._has_explicit_zero_issue_summary(
        "Target application returned an error\\nERROR SUMMARY: 0 errors"
    )
    assert compute_sanitizer._has_explicit_zero_issue_summary(
        "Target application returned an error\\nRACECHECK SUMMARY: 0 hazards displayed"
    )
    assert not compute_sanitizer._has_explicit_zero_issue_summary("Target application returned an error")


def test_pipeline_only_triggers_sanitizer_for_candidate_runtime_failure() -> None:
    assert kernelbench_pipeline._is_candidate_correctness_runtime_failure(
        {
            "runtime_error": "CUDA error: illegal memory access",
            "correctness_runtime_error_stage": "custom_forward",
        }
    )
    assert not kernelbench_pipeline._is_candidate_correctness_runtime_failure({"correctness_issue": "Output mismatch"})
    assert not kernelbench_pipeline._is_candidate_correctness_runtime_failure(
        {
            "runtime_error": "reference failed",
            "correctness_runtime_error_stage": "reference_forward",
        }
    )


def test_run_compute_sanitizer_single_mode_replays_seed(monkeypatch, tmp_path: Path) -> None:
    fake_tool = tmp_path / "compute-sanitizer"
    fake_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    monkeypatch.setattr(compute_sanitizer, "_tool_version", lambda _path: "test-version")
    executed = []

    def fake_run(command, **_kwargs):
        executed.append(command[command.index("--tool") + 1])
        payload = Path(command[-1]).read_text(encoding="utf-8")
        assert '"input_seed": 1234' in payload
        assert '"input_perturbation": "sign_challenge"' in payload
        assert '"model_seed": 42' in payload
        return SimpleNamespace(returncode=86, stdout="", stderr=MEMCHECK_OUTPUT)

    monkeypatch.setattr(compute_sanitizer, "_run_sanitizer_command", fake_run)
    result = compute_sanitizer.run_compute_sanitizer(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="cuda_agent",
        entry_point="Model",
        device="cuda:0",
        kernel_names=["oob_kernel"],
        sanitizer_path=str(fake_tool),
        mode="memcheck",
        timeout_s=10,
        max_kernels=2,
        max_issues=5,
        input_seed=1234,
        input_perturbation="sign_challenge",
        model_seed=42,
    )

    assert executed == ["memcheck"]
    assert result["status"] == "issues_found"
    assert result["measurement_complete"] is True
    assert result["executed_checks"] == ["memcheck"]
    assert result["replayed_input_perturbation"] == "sign_challenge"


def test_run_compute_sanitizer_timeout_is_fail_open_metadata(monkeypatch, tmp_path: Path) -> None:
    fake_tool = tmp_path / "compute-sanitizer"
    fake_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    monkeypatch.setattr(compute_sanitizer, "_tool_version", lambda _path: "test-version")
    monkeypatch.setattr(
        compute_sanitizer,
        "_run_sanitizer_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd="sanitizer", timeout=4)),
    )

    result = compute_sanitizer.run_compute_sanitizer(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="cuda_agent",
        entry_point="Model",
        device="cuda:0",
        kernel_names=[],
        sanitizer_path=str(fake_tool),
        mode="memcheck",
        timeout_s=4,
        max_kernels=1,
        max_issues=5,
    )

    assert result["status"] == "error"
    assert result["passed"] is None
    assert result["measurement_complete"] is False
    assert result["check_results"][0]["status"] == "timeout"


def test_sanitizer_issues_found_is_a_runtime_failure_response() -> None:
    sanitizer = {
        "status": "issues_found",
        "check_results": [{"issues": [{"message": "Invalid __global__ write"}]}],
    }
    result = KernelEvaluationResult.from_kernel_exec_result(
        "task_kernel",
        "task",
        KernelExecResult(
            compiled=True,
            correctness=False,
            runtime_sanitizer=sanitizer,
        ),
    )

    assert result.status == "failed"
    assert result.runtime_sanitizer == sanitizer
    assert result.error_code.value == "RUNTIME_ERROR"
    assert "Invalid __global__ write" in result.error_message

    monolithic = EvaluationResult.from_kernel_exec_result(
        "task",
        KernelExecResult(compiled=True, correctness=False, runtime_sanitizer=sanitizer),
        1.0,
    )
    assert monolithic.status == "failed"
    assert monolithic.error_code.value == "RUNTIME_ERROR"
    assert monolithic.runtime_sanitizer == sanitizer


def test_kernel_only_workflow_result_preserves_runtime_sanitizer() -> None:
    sanitizer = {
        "status": "issues_found",
        "executed_checks": ["memcheck"],
        "check_results": [{"issues": [{"message": "Invalid __global__ write"}]}],
    }
    task = EvaluationTask(
        task_id="sanitizer-kernel-only",
        reference_code="class Model: pass",
        kernel_code="class ModelNew: pass",
    )
    kernel_result = KernelEvaluationResult(
        task_id="sanitizer-kernel-only_kernel",
        base_task_id=task.task_id,
        compiled=True,
        correctness=False,
        decoy_kernel=False,
        kernel_runtime=-1.0,
        metadata={},
        runtime_sanitizer=sanitizer,
        status="failed",
    )

    response = KernelBenchWorkflowController()._kernel_only_result(task, kernel_result)

    assert response["runtime_sanitizer"] == sanitizer


def test_public_result_omits_runtime_sanitizer_when_not_triggered() -> None:
    result = EvaluationResult(
        task_id="sanitizer-skipped",
        compiled=True,
        correctness=True,
        decoy_kernel=False,
        reference_runtime=1.0,
        kernel_runtime=1.0,
        speedup=1.0,
        metadata={},
        runtime_sanitizer={
            "status": "skipped",
            "reason": "disabled",
            "measurement_complete": False,
        },
    )

    public_result = result.to_dict()
    response = EvaluationResponse(**public_result)

    assert "runtime_sanitizer" not in public_result
    assert "runtime_sanitizer" not in response.model_dump()


def test_api_defaults_and_workflow_propagate_sanitizer() -> None:
    request = EvaluationRequest(
        task_id="sanitizer-default",
        reference_code="class Model:\n    pass",
        kernel_code="class ModelNew:\n    pass",
    )
    assert request.enable_compute_sanitizer is False
    assert request.compute_sanitizer_mode == "error_based"
    for mode in ("error_based", "full"):
        mode_request = EvaluationRequest(**{**request.model_dump(), "compute_sanitizer_mode": mode})
        assert mode_request.compute_sanitizer_mode == mode

    for internal_mode in ("memcheck", "synccheck", "racecheck", "initcheck"):
        try:
            EvaluationRequest(
                **{
                    **request.model_dump(),
                    "compute_sanitizer_mode": internal_mode,
                }
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"internal check {internal_mode!r} must be rejected in the payload")

    payload = apply_runtime_defaults(
        {**request.model_dump(), "enable_compute_sanitizer": None},
        workflow_name="kernelbench",
        split_compile_and_execute=False,
        enable_ncu=False,
        enable_compute_sanitizer=True,
        compute_sanitizer_profile_version="v1",
    )
    assert payload["enable_compute_sanitizer"] is True
    assert payload["_compute_sanitizer_profile_version"] == "v1"

    task = EvaluationTask(
        task_id="sanitizer-propagation",
        reference_code="class Model: pass",
        kernel_code="class ModelNew: pass",
        enable_compute_sanitizer=False,
        compute_sanitizer_mode="full",
    )
    _, kernel_task = _create_paired_tasks(task)
    assert kernel_task.enable_compute_sanitizer is False
    assert kernel_task.compute_sanitizer_mode == "full"

    controller = KernelBenchWorkflowController()
    assert controller._kernel_execution_options(task, compile_only=True)["enable_compute_sanitizer"] is False
    assert controller._kernel_execution_options(task, compile_only=False)["compute_sanitizer_mode"] == "full"
    default_options = controller._kernel_execution_options(
        EvaluationTask(
            task_id="sanitizer-default-options",
            reference_code="class Model: pass",
            kernel_code="class ModelNew: pass",
        ),
        compile_only=False,
    )
    assert default_options["enable_compute_sanitizer"] is settings.enable_compute_sanitizer
