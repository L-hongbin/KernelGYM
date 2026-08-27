import subprocess
from pathlib import Path
from types import SimpleNamespace

from kernelgym.config import settings
from kernelgym.schema.task import EvaluationTask
from kernelgym.server.api.models import EvaluationRequest
from kernelgym.server.request_defaults import apply_runtime_defaults
from kernelgym.toolkit.kernelbench import ncu_profiler
from kernelgym.workflow.kernelbench import KernelBenchWorkflowController
from kernelgym.workflow.kernelbench_helpers import _create_paired_tasks

SAMPLE_CSV = """==PROF== Connected to process 123
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"1","123","python","host","vector_add(float const*, float*)","1","7","(256, 1, 1)","(16, 1, 1)","NVIDIA H100","9.0","Command line profiler metrics","gpu__time_duration.sum","usecond","12.50"
"1","123","python","host","vector_add(float const*, float*)","1","7","(256, 1, 1)","(16, 1, 1)","NVIDIA H100","9.0","Command line profiler metrics","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","73.25"
"""


WIDE_SAMPLE_CSV = """"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","device__attribute_architecture","gpu__time_duration.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed"
"","","","","","","","","","","","","usecond","%"
"1","123","python","host","vector_add(float const*, float*)","1","7","(256, 1, 1)","(16, 1, 1)","NVIDIA H100","9.0","Hopper","12.50","73.25"
"""


def test_default_ncu_metrics_include_l1_l2_hit_rates(monkeypatch) -> None:
    from kernelgym.config.settings import Settings

    monkeypatch.delenv("NCU_METRICS", raising=False)
    defaults = Settings(_env_file=None)

    assert "l1tex__t_sector_hit_rate.pct" in defaults.ncu_metrics
    assert "lts__t_sector_hit_rate.pct" in defaults.ncu_metrics
    assert defaults.ncu_profile_version == "v1"
    assert defaults.enable_ncu is False


def test_parse_ncu_csv_groups_metrics_by_kernel() -> None:
    kernels = ncu_profiler.parse_ncu_csv(SAMPLE_CSV)

    assert len(kernels) == 1
    assert kernels[0]["kernel_name"] == "vector_add(float const*, float*)"
    assert kernels[0]["block_size"] == "(256, 1, 1)"
    assert kernels[0]["grid_size"] == "(16, 1, 1)"
    assert kernels[0]["metrics"]["gpu__time_duration.sum"] == {
        "value": 12.5,
        "unit": "usecond",
    }
    assert kernels[0]["metrics"][
        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    ] == {
        "value": 73.25,
        "unit": "%",
    }


def test_parse_ncu_csv_supports_wide_metric_columns() -> None:
    kernels = ncu_profiler.parse_ncu_csv(
        WIDE_SAMPLE_CSV,
        metric_names=[
            "gpu__time_duration.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
    )

    assert len(kernels) == 1
    assert kernels[0]["kernel_name"] == "vector_add(float const*, float*)"
    assert kernels[0]["metrics"] == {
        "gpu__time_duration.sum": {"value": 12.5, "unit": "usecond"},
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": {
            "value": 73.25,
            "unit": "%",
        },
    }
    assert "device__attribute_architecture" not in kernels[0]["metrics"]


def test_build_ncu_command_uses_argv_and_escapes_kernel_regex(tmp_path: Path) -> None:
    command = ncu_profiler.build_ncu_command(
        ncu_path="/opt/ncu",
        report_base=tmp_path / "report",
        payload_path=tmp_path / "payload.json",
        metrics=["gpu__time_duration.sum"],
        kernel_names=["kernel.*;touch /tmp/not-run"],
        max_kernels=3,
    )

    assert command[0] == "/opt/ncu"
    assert command[command.index("--launch-count") + 1] == "3"
    regex_arg = command[command.index("--kernel-name") + 1]
    assert regex_arg.startswith("regex:")
    assert r"\.\*;touch\ /tmp/not\-run" in regex_arg
    assert command[-3:] == [
        "-m",
        "kernelgym.toolkit.kernelbench.ncu_runner",
        str(tmp_path / "payload.json"),
    ]


def test_select_kernel_names_is_deduplicated_and_bounded() -> None:
    names = ncu_profiler.select_kernel_names(
        {
            "custom_kernel_in_profiling": ["slow", "fast", "slow"],
            "custom_kernel_names": ["fallback"],
        },
        max_kernels=2,
    )

    assert names == ["slow", "fast"]


def test_run_ncu_profile_exports_and_parses_csv(monkeypatch, tmp_path: Path) -> None:
    fake_ncu = tmp_path / "ncu"
    fake_ncu.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ncu.chmod(0o755)
    monkeypatch.setattr(ncu_profiler, "_ncu_version", lambda _path: "NCU test")

    def fake_run(command, **_kwargs):
        if "--import" in command:
            csv_path = Path(command[command.index("--log-file") + 1])
            csv_path.write_text(WIDE_SAMPLE_CSV, encoding="utf-8")
        else:
            report_base = Path(command[command.index("--export") + 1])
            report_base.with_suffix(".ncu-rep").write_bytes(b"report")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ncu_profiler.subprocess, "run", fake_run)
    result = ncu_profiler.run_ncu_profile(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="cuda",
        entry_point="Model",
        device="cuda:3",
        kernel_names=["vector_add"],
        ncu_path=str(fake_ncu),
        metrics=["gpu__time_duration.sum"],
        timeout_s=10,
        max_kernels=2,
        warmup=1,
        profile_version="v1",
    )

    assert result["status"] == "ok"
    assert result["tool_version"] == "NCU test"
    assert result["profiled_kernel_count"] == 1
    assert result["kernels"][0]["metrics"]["gpu__time_duration.sum"]["value"] == 12.5


def test_run_ncu_profile_timeout_is_fail_open_metadata(monkeypatch, tmp_path: Path) -> None:
    fake_ncu = tmp_path / "ncu"
    fake_ncu.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ncu.chmod(0o755)
    monkeypatch.setattr(ncu_profiler, "_ncu_version", lambda _path: "NCU test")

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="ncu", timeout=4)

    monkeypatch.setattr(ncu_profiler.subprocess, "run", timeout)
    result = ncu_profiler.run_ncu_profile(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="cuda",
        entry_point="Model",
        device="cuda:0",
        kernel_names=[],
        ncu_path=str(fake_ncu),
        metrics=[],
        timeout_s=4,
        max_kernels=1,
        warmup=0,
        profile_version="v1",
    )

    assert result["status"] == "timeout"
    assert result["profiled_kernel_count"] == 0
    assert "timed out" in result["error"]


def test_api_defaults_disable_ncu_and_null_inherits_runtime_enablement() -> None:
    request = EvaluationRequest(
        task_id="ncu-default",
        reference_code="class Model:\n    pass",
        kernel_code="class ModelNew:\n    pass",
    )
    assert request.enable_ncu is False

    payload = apply_runtime_defaults(
        request.model_dump(),
        workflow_name="kernelbench",
        split_compile_and_execute=False,
        enable_ncu=True,
        ncu_profile_version="v1",
    )
    assert payload["enable_ncu"] is False
    assert "_ncu_profile_version" not in payload

    inherit_request = EvaluationRequest(
        task_id="ncu-inherit",
        reference_code="class Model:\n    pass",
        kernel_code="class ModelNew:\n    pass",
        enable_ncu=None,
    )
    payload = apply_runtime_defaults(
        inherit_request.model_dump(),
        workflow_name="kernelbench",
        split_compile_and_execute=False,
        enable_ncu=True,
        ncu_profile_version="v1",
    )
    assert payload["enable_ncu"] is True
    assert payload["_ncu_profile_version"] == "v1"


def test_workflow_propagates_ncu_to_kernel_task_and_compile_only_disables_it() -> None:
    task = EvaluationTask(
        task_id="ncu-propagation",
        reference_code="class Model: pass",
        kernel_code="class ModelNew: pass",
        enable_ncu=False,
    )
    _, kernel_task = _create_paired_tasks(task)
    assert kernel_task.enable_ncu is False

    controller = KernelBenchWorkflowController()
    default_options = controller._kernel_execution_options(
        EvaluationTask(
            task_id="ncu-default-options",
            reference_code="class Model: pass",
            kernel_code="class ModelNew: pass",
        ),
        compile_only=False,
    )
    assert default_options["enable_ncu"] is settings.enable_ncu
    assert controller._kernel_execution_options(task, compile_only=True)["enable_ncu"] is False


def test_select_kernel_names_falls_back_to_compile_artifact_hints() -> None:
    names = ncu_profiler.select_kernel_names(
        {
            "compile_artifact": {
                "profiling_hints": {
                    "custom_kernel_names": ["prod_tanh_kernel"],
                }
            }
        },
        max_kernels=2,
    )

    assert names == ["prod_tanh_kernel"]


def test_run_ncu_profile_falls_back_to_export_stdout(
    monkeypatch, tmp_path: Path
) -> None:
    fake_ncu = tmp_path / "ncu"
    fake_ncu.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ncu.chmod(0o755)
    monkeypatch.setattr(ncu_profiler, "_ncu_version", lambda _path: "NCU test")

    def fake_run(command, **_kwargs):
        if "--import" in command:
            return SimpleNamespace(returncode=0, stdout=SAMPLE_CSV, stderr="")
        report_base = Path(command[command.index("--export") + 1])
        report_base.with_suffix(".ncu-rep").write_bytes(b"report")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ncu_profiler.subprocess, "run", fake_run)
    result = ncu_profiler.run_ncu_profile(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="cuda",
        entry_point="Model",
        device="cuda:0",
        kernel_names=["vector_add"],
        ncu_path=str(fake_ncu),
        metrics=["gpu__time_duration.sum"],
        timeout_s=10,
        max_kernels=2,
        warmup=1,
        profile_version="v1",
    )

    assert result["status"] == "ok"
    assert result["csv_source"] == "stdout"
    assert result["profiled_kernel_count"] == 1


def test_run_ncu_profile_returns_export_diagnostics_for_unparseable_csv(
    monkeypatch, tmp_path: Path
) -> None:
    fake_ncu = tmp_path / "ncu"
    fake_ncu.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ncu.chmod(0o755)
    monkeypatch.setattr(ncu_profiler, "_ncu_version", lambda _path: "NCU test")

    def fake_run(command, **_kwargs):
        if "--import" in command:
            csv_path = Path(command[command.index("--log-file") + 1])
            csv_path.write_text("unexpected csv format", encoding="utf-8")
            return SimpleNamespace(
                returncode=0, stdout="export stdout", stderr="export stderr"
            )
        report_base = Path(command[command.index("--export") + 1])
        report_base.with_suffix(".ncu-rep").write_bytes(b"report")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ncu_profiler.subprocess, "run", fake_run)
    result = ncu_profiler.run_ncu_profile(
        original_model_src="class Model: pass",
        custom_model_src="class ModelNew: pass",
        artifact=None,
        backend="cuda",
        entry_point="Model",
        device="cuda:0",
        kernel_names=[],
        ncu_path=str(fake_ncu),
        metrics=[],
        timeout_s=10,
        max_kernels=1,
        warmup=0,
        profile_version="v1",
    )

    assert result["status"] == "no_matching_kernel"
    assert result["csv_source"] == "none"
    assert result["csv_size_bytes"] == len("unexpected csv format")
    assert result["csv_head"] == "unexpected csv format"
    assert result["export_stdout_tail"] == "export stdout"
    assert result["export_stderr_tail"] == "export stderr"
