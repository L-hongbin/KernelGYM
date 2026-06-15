"""Deploy-time preflight: validate_runtime must abort if a CUDA math library
the TVM-FFI build links is missing from the environment.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _load_validate_runtime():
    spec = importlib.util.spec_from_file_location("validate_runtime_script", ROOT / "scripts" / "validate_runtime.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_runtime = _load_validate_runtime()

from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend  # noqa: E402


def _patch_report(monkeypatch, report):
    monkeypatch.setattr(
        KernelBenchTvmFfiBackend,
        "_resolve_cuda_math_libs",
        classmethod(lambda cls: report),
    )


def test_check_passes_when_all_libs_resolve(monkeypatch) -> None:
    _patch_report(
        monkeypatch,
        [
            ("nvidia.cublas", "libcublas", True, "/x/libcublas.so.12"),
            ("nvidia.cudnn", "libcudnn", True, "/y/libcudnn.so.9"),
        ],
    )
    # Must not raise.
    validate_runtime._check_cuda_math_libs()


def test_check_aborts_when_a_required_lib_is_missing(monkeypatch) -> None:
    _patch_report(
        monkeypatch,
        [
            ("nvidia.cublas", "libcublas", True, "/x/libcublas.so.12"),
            ("nvidia.cudnn", "libcudnn", True, None),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        validate_runtime._check_cuda_math_libs()
    # The error names the offending library so deploy logs are actionable.
    assert "libcudnn" in str(excinfo.value)


def test_check_warns_but_does_not_abort_on_missing_optional(monkeypatch, capsys) -> None:
    _patch_report(
        monkeypatch,
        [
            ("nvidia.cublas", "libcublas", True, "/x/libcublas.so.12"),
            ("nvidia.cusparselt", "libcusparseLt", False, None),
        ],
    )
    # A missing optional library must not block deploy — only warn.
    validate_runtime._check_cuda_math_libs()
    out = capsys.readouterr().out
    assert "WARNING" in out and "libcusparseLt" in out


def test_real_environment_resolves_all_required_libs() -> None:
    # In a correctly provisioned reward venv every *required* library resolves.
    report = KernelBenchTvmFfiBackend._resolve_cuda_math_libs()
    missing_required = [base for _pkg, base, required, path in report if required and path is None]
    assert missing_required == [], f"unresolved required CUDA math libs in this env: {missing_required}"
