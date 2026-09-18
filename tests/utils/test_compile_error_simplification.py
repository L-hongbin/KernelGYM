import pytest

from kernelgym.backend.kernelbench import cuda_agent_backend, tvm_ffi_backend
from kernelgym.backend.kernelbench.cuda_agent_backend import KernelBenchCudaAgentBackend
from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend
from kernelgym.schema.result import KernelEvaluationResult
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult
from kernelgym.utils import error_simplifier
from kernelgym.utils.error_simplifier import simplify_error_message

NUM_GPUS = 0


@pytest.mark.parametrize("parsed", [False, True])
def test_compile_only_failure_uses_shared_error_message_format(monkeypatch, parsed) -> None:
    from benchmarks.kernels.tvm_ffi_vector_add import KERNEL_CODE, REFERENCE_CODE
    from kernelgym.schema.task import KernelEvaluationTask
    from kernelgym.toolkit.kernelbench import toolkit
    from kernelgym.utils.error_classifier import classify_compile_error_metadata

    original = 'generated.cu:9: error: identifier "missing" is undefined' if parsed else 'compiler launch failed'
    metadata = {"compilation_error": original, **classify_compile_error_metadata(original)}
    monkeypatch.setattr(toolkit, "set_seed", lambda *args: None)
    monkeypatch.setattr(toolkit.kernelbench_pipeline, "eval_kernel_against_ref", lambda **kwargs: KernelExecResult(
        compiled=False, metadata=metadata,
    ))
    task = KernelEvaluationTask(
        task_id="compile", base_task_id="parent", reference_code=REFERENCE_CODE,
        kernel_code=KERNEL_CODE, backend="tvm_ffi", task_stage="compile", device="cpu",
    )
    result = toolkit.KernelBenchToolkit().evaluate_kernel_only(task)
    assert result.compiled is False
    assert result.metadata["compile_only"] is True
    assert result.metadata["compilation_error"] == original
    if parsed:
        assert "First diagnostic:" in result.error_message
    else:
        assert result.error_message == f"Kernel compilation failed: {original}"


@pytest.mark.parametrize("error", [None, "ptxas error : Arguments mismatch for instruction 'mma'"])
def test_public_compile_artifact_omits_status_and_error_without_mutating_internal_artifact(error) -> None:
    from kernelgym.toolkit.kernelbench.pipeline import _copy_compile_artifact_metadata

    artifact = {"compiled": error is None, "error": error, "backend": "tvm_ffi"}
    original = dict(artifact)
    metadata = {"compilation_error": error} if error is not None else {}
    _copy_compile_artifact_metadata(metadata, artifact)
    assert metadata["compile_artifact"] == {"backend": "tvm_ffi"}
    assert artifact == original
    if error is not None:
        assert metadata["compilation_error"] == error


def test_load_failure_keeps_top_level_compiled_false_and_internal_compile_success() -> None:
    from kernelgym.toolkit.kernelbench.pipeline import _copy_compile_artifact_metadata

    artifact = {"compiled": True, "error": None, "backend": "tvm_ffi"}
    metadata = {"compilation_error": "Failed to load shared library"}
    _copy_compile_artifact_metadata(metadata, artifact)
    result = KernelEvaluationResult.from_kernel_exec_result(
        "child", "parent", KernelExecResult(compiled=False, metadata=metadata),
    )
    assert result.compiled is False
    assert "compiled" not in result.metadata["compile_artifact"]
    assert "error" not in result.metadata["compile_artifact"]
    assert artifact["compiled"] is True
    assert result.metadata["compilation_error"] == "Failed to load shared library"


def test_compile_error_message_summarizes_details_without_changing_original() -> None:
    from kernelgym.utils.error_classifier import classify_compile_error_metadata

    lines = [f'generated.cu:{i}: error: identifier "value_{i}" is undefined' for i in range(10)]
    original = "Compiler output:\n" + "\n".join(lines + lines)
    metadata = {"compilation_error": original, **classify_compile_error_metadata(original)}
    result = KernelEvaluationResult.from_kernel_exec_result(
        "child", "parent", KernelExecResult(compiled=False, metadata=metadata),
    )
    assert result.metadata["compilation_error"] == original
    assert result.metadata["compilation_error_detail"] == metadata["compilation_error_detail"]
    assert result.error_message == (
        "Kernel compilation failed: 10 unique diagnostics (8 shown); undefined_identifier=10. "
        f"First diagnostic: {lines[0]}"
    )


@pytest.mark.parametrize("details", [None, {}, {"other": []}, {"other": {"count": 0, "errors": []}}])
def test_compile_error_message_keeps_full_log_without_parsed_details(details) -> None:
    original = "ninja exited with status 1\n" + "unparsed linker output\n" * 1000
    result = KernelEvaluationResult.from_kernel_exec_result(
        "child", "parent", KernelExecResult(compiled=False, metadata={
            "compilation_error": original, "compilation_error_detail": details,
        }),
    )
    assert result.metadata["compilation_error"] == original
    assert result.error_message == f"Kernel compilation failed: {original}"


def test_compile_error_summary_bounds_first_diagnostic() -> None:
    from kernelgym.schema.result import _compile_error_summary

    diagnostic = "error: " + "x" * 1000
    assert _compile_error_summary({"other": [diagnostic]}).endswith(diagnostic[:300] + "...")


def test_compile_error_paths_are_made_relative(monkeypatch, tmp_path) -> None:
    site_packages = tmp_path / "project" / ".venv" / "lib" / "python3.12" / "site-packages"
    venv_root = site_packages.parents[2]
    work_dir = tmp_path / "compile_cache" / "hash" / "kernelgym_backend_hash"
    monkeypatch.setattr(
        error_simplifier.sysconfig,
        "get_path",
        lambda name: str(site_packages) if name in {"purelib", "platlib"} else None,
    )
    monkeypatch.setattr(error_simplifier.sys, "prefix", str(venv_root))
    monkeypatch.setattr(error_simplifier.sys, "base_prefix", "/usr")
    error_message = (
        f"{venv_root}/bin/python "
        f"-I{site_packages}/tvm_ffi/include "
        f"{work_dir}/kernels/generated_binding.cpp:28:64: error: "
        "'struct DLDataType' has no member named 'bytes'"
    )

    simplified = simplify_error_message(error_message, work_dir=work_dir)

    assert str(site_packages) not in simplified
    assert str(venv_root) not in simplified
    assert str(work_dir) not in simplified
    assert "bin/python" in simplified
    assert "-Itvm_ffi/include" in simplified
    assert "kernels/generated_binding.cpp:28:64: error:" in simplified


def test_error_simplification_can_be_disabled(monkeypatch, tmp_path) -> None:
    site_packages = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    work_dir = tmp_path / "work"
    monkeypatch.setattr(error_simplifier.sysconfig, "get_path", lambda _name: str(site_packages))
    error_message = f"{site_packages}/torch/include {work_dir}/kernels/generated.cu:10: error"

    assert simplify_error_message(error_message, work_dir=work_dir, enabled=False) == error_message


def test_source_root_is_removed_after_nested_venv_paths(monkeypatch, tmp_path) -> None:
    source_root = tmp_path / "KernelGYM"
    venv_root = source_root / ".venv"
    site_packages = venv_root / "lib" / "python3.12" / "site-packages"
    monkeypatch.setattr(error_simplifier, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(error_simplifier.sys, "prefix", str(venv_root))
    monkeypatch.setattr(error_simplifier.sys, "base_prefix", "/usr")
    monkeypatch.setattr(
        error_simplifier.sysconfig,
        "get_path",
        lambda name: str(site_packages) if name in {"purelib", "platlib"} else None,
    )
    error_message = (
        f'  File "{source_root}/kernelgym/toolkit/kernelbench/correctness.py", line 670\n'
        f"{venv_root}/bin/python {site_packages}/torch/nn/modules/module.py"
    )

    simplified = simplify_error_message(error_message, work_dir=None)

    assert str(source_root) not in simplified
    assert ".venv/" not in simplified
    assert 'File "kernelgym/toolkit/kernelbench/correctness.py", line 670' in simplified
    assert "bin/python torch/nn/modules/module.py" in simplified


@pytest.mark.parametrize(
    ("backend_module", "backend_class", "precheck_name"),
    (
        (cuda_agent_backend, KernelBenchCudaAgentBackend, "precheck_cuda_agent_submission"),
        (tvm_ffi_backend, KernelBenchTvmFfiBackend, "precheck_tvm_ffi_submission"),
    ),
)
@pytest.mark.parametrize("enabled", (True, False))
def test_backends_honor_simplify_error_control(
    monkeypatch,
    tmp_path,
    backend_module,
    backend_class,
    precheck_name: str,
    enabled: bool,
) -> None:
    backend = backend_class()
    work_dir = tmp_path / backend_class.__name__
    sources = {"kernels/generated.cu": "__global__ void kernel() {}"}
    raw_error = f"{work_dir}/kernels/generated.cu:10: error: intentional"

    monkeypatch.setattr(backend, "_normalize_device", lambda _device: "cpu")
    monkeypatch.setattr(backend, "_parse_embedded_sources", lambda _code: (sources, "class ModelNew: pass"))
    monkeypatch.setattr(
        backend_module,
        precheck_name,
        lambda *_args, **_kwargs: ("", None, {"passed": True}),
    )
    monkeypatch.setattr(backend, "_artifact_cache_key", lambda **_kwargs: "cache-key")
    monkeypatch.setattr(backend, "_create_work_dir", lambda: work_dir)
    monkeypatch.setattr(backend, "_write_runtime_scaffold", lambda *_args: None)
    monkeypatch.setattr(backend, "_materialize_sources", lambda *_args: None)
    collected_sources = ([], []) if backend_class is KernelBenchTvmFfiBackend else []
    monkeypatch.setattr(backend, "_collect_compile_sources", lambda *_args: collected_sources)
    monkeypatch.setattr(backend, "_build_extension", lambda *_args, **_kwargs: {"compiled": False, "error": raw_error})

    result = backend.compile(
        "class ModelNew: pass",
        device="cpu",
        cuda_sources=sources,
        simplify_error=enabled,
    )

    if enabled:
        assert result["error"] == "kernels/generated.cu:10: error: intentional"
    else:
        assert result["error"] == raw_error


@pytest.mark.parametrize("enabled", (True, False))
def test_runtime_error_traceback_honors_simplify_error(monkeypatch, tmp_path, enabled: bool) -> None:
    site_packages = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    work_dir = tmp_path / "work"
    monkeypatch.setattr(error_simplifier.sysconfig, "get_path", lambda _name: str(site_packages))

    try:
        raise RuntimeError(
            f"runtime failure in {work_dir}/kernels/generated.py and {site_packages}/tvm_ffi/runtime.py"
        )
    except RuntimeError as exc:
        exec_result = KernelExecResult(
            compiled=True,
            correctness=False,
            metadata={
                "runtime_error": exc,
                "runtime_error_name": "builtins.RuntimeError",
                "_error_work_dir": str(work_dir),
                "_simplify_error_enabled": enabled,
            },
        )

    result = KernelEvaluationResult.from_kernel_exec_result(
        "runtime-error_kernel",
        "runtime-error",
        exec_result,
        verbose_errors=True,
    )
    runtime_error = result.metadata["runtime_error"]

    assert "Traceback (most recent call last)" in runtime_error
    assert (str(work_dir) not in runtime_error) is enabled
    assert (str(site_packages) not in runtime_error) is enabled
    assert "kernels/generated.py" in runtime_error
    assert "tvm_ffi/runtime.py" in runtime_error
    assert "_error_work_dir" not in result.metadata
    assert "_simplify_error_enabled" not in result.metadata


@pytest.mark.parametrize("enabled", (True, False))
def test_runtime_sanitizer_raw_output_tail_honors_simplify_error(monkeypatch, tmp_path, enabled: bool) -> None:
    source_root = tmp_path / "KernelGYM"
    work_dir = tmp_path / "compile_cache" / "artifact" / "kernelgym_tvm_ffi_hash"
    monkeypatch.setattr(error_simplifier, "_SOURCE_ROOT", source_root)
    raw_output_tail = (
        f'  File "{source_root}/kernelgym/toolkit/kernelbench/compute_sanitizer_runner.py", line 110\n'
        f'  File "{work_dir}/model_new.py", line 11'
    )
    exec_result = KernelExecResult(
        compiled=True,
        correctness=False,
        metadata={
            "_error_work_dir": str(work_dir),
            "_simplify_error_enabled": enabled,
        },
        runtime_sanitizer={
            "status": "clean",
            "check_results": [{"check": "memcheck", "status": "clean", "raw_output_tail": raw_output_tail}],
        },
    )

    result = KernelEvaluationResult.from_kernel_exec_result(
        "runtime-sanitizer_kernel",
        "runtime-sanitizer",
        exec_result,
        verbose_errors=True,
    )
    simplified_tail = result.runtime_sanitizer["check_results"][0]["raw_output_tail"]

    if enabled:
        assert str(source_root) not in simplified_tail
        assert str(work_dir) not in simplified_tail
        assert "kernelgym/toolkit/kernelbench/compute_sanitizer_runner.py" in simplified_tail
        assert "model_new.py" in simplified_tail
    else:
        assert simplified_tail == raw_output_tail
