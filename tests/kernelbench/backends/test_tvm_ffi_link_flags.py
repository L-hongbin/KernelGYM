"""Bug #1 regression: the TVM-FFI build must link cuBLAS/cuBLASLt/cuDNN/cuFFT so
model-generated extensions that call those host APIs do not crash the worker with
``undefined symbol: cublasCreate_v2``.

Covers ``KernelBenchTvmFfiBackend._cuda_math_link_flags`` selection/exclusion
logic with a synthetic wheel layout (no GPU, no nvcc).
"""

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend  # noqa: E402


def _make_fake_wheels(tmp_path):
    layout = {
        "nvidia.cublas": ["libcublas.so.12", "libcublasLt.so.12", "libnvblas.so.12"],
        "nvidia.cudnn": [
            "libcudnn.so.9",
            "libcudnn_cnn.so.9",
            "libcudnn_engines_precompiled.so.9",
        ],
        "nvidia.cufft": ["libcufft.so.11", "libcufftw.so.11"],
        "nvidia.cusparse": ["libcusparse.so.12"],
        "nvidia.cusparselt": ["libcusparseLt.so.0"],
        "nvidia.cusolver": ["libcusolver.so.11", "libcusolverMg.so.11"],
        "nvidia.curand": ["libcurand.so.10"],
        "nvidia.cuda_nvrtc": ["libnvrtc.so.12", "libnvrtc-builtins.so.12.9"],
    }
    pkg_dirs = {}
    for package, libs in layout.items():
        pkg_dir = tmp_path / package.replace(".", "_")
        lib_dir = pkg_dir / "lib"
        lib_dir.mkdir(parents=True)
        for name in libs:
            (lib_dir / name).write_bytes(b"")
        pkg_dirs[package] = pkg_dir
    return pkg_dirs


def test_link_flags_pick_the_right_libraries(monkeypatch, tmp_path) -> None:
    pkg_dirs = _make_fake_wheels(tmp_path)

    class _FakeSpec:
        def __init__(self, location):
            self.submodule_search_locations = [str(location)]

    def _fake_find_spec(name):
        return _FakeSpec(pkg_dirs[name]) if name in pkg_dirs else None

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    flags = KernelBenchTvmFfiBackend._cuda_math_link_flags()
    link_inputs = [f for f in flags if not f.startswith("-Wl,-rpath,")]
    rpaths = [f for f in flags if f.startswith("-Wl,-rpath,")]
    names = sorted(Path(p).name for p in link_inputs)

    assert names == [
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
        "libcufft.so.11",
        "libcurand.so.10",
        "libcusolver.so.11",
        "libcusparse.so.12",
        "libcusparseLt.so.0",
        "libnvrtc.so.12",
    ]
    # A base name must resolve only its own library, never a longer-named
    # sibling in the same dir: libcublas !-> libcublasLt/libnvblas; libcudnn
    # !-> the heavy engine sublibs; libcufft !-> cufftw; libcusolver !->
    # cusolverMg; libnvrtc !-> nvrtc-builtins.
    for unwanted in (
        "libnvblas.so.12",
        "libcudnn_cnn.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcufftw.so.11",
        "libcusolverMg.so.11",
        "libnvrtc-builtins.so.12.9",
    ):
        assert unwanted not in names
    # One rpath per distinct lib dir (cublas is referenced twice -> deduped):
    # cublas, cudnn, cufft, cusparse, cusparselt, cusolver, curand, cuda_nvrtc.
    assert len(rpaths) == 8
    # Link inputs are absolute paths to real files (full-path linking).
    assert all(Path(p).is_file() for p in link_inputs)


def test_link_flags_graceful_when_wheels_absent(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    # No vendored libs found -> no flags, build falls back to prior behavior.
    assert KernelBenchTvmFfiBackend._cuda_math_link_flags() == []


def test_strict_link_flag_is_always_enabled(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    # Even a minimal/non-wheel CUDA layout must reject unresolved symbols at
    # link time instead of letting dlopen kill a CUDA-owning worker later.
    assert KernelBenchTvmFfiBackend._link_flags() == ["-Wl,-z,defs"]


def test_artifact_cache_key_is_fenced_by_strict_link_policy(monkeypatch) -> None:
    inputs = {
        "model_code": "class ModelNew: pass",
        "cuda_sources": {"kernel.cu": 'extern "C" __global__ void kernel() {}'},
        "entry_point": "ModelNew",
    }
    monkeypatch.setattr(KernelBenchTvmFfiBackend, "_link_flags", staticmethod(lambda: []))
    permissive_key = KernelBenchTvmFfiBackend._artifact_cache_key(**inputs)
    monkeypatch.setattr(
        KernelBenchTvmFfiBackend,
        "_link_flags",
        staticmethod(lambda: ["-Wl,-z,defs"]),
    )

    assert KernelBenchTvmFfiBackend._artifact_cache_key(**inputs) != permissive_key


def test_build_extension_passes_strict_link_policy(monkeypatch, tmp_path) -> None:
    captured = {}

    class _FakeCpp:
        @staticmethod
        def build(**kwargs):
            captured.update(kwargs)
            return tmp_path / "build" / "extension.so"

    monkeypatch.setattr(
        KernelBenchTvmFfiBackend,
        "_import_tvm_ffi",
        staticmethod(lambda: (object(), _FakeCpp)),
    )
    monkeypatch.setattr(
        KernelBenchTvmFfiBackend,
        "_cuda_math_link_flags",
        staticmethod(lambda: ["/fake/libcublas.so.12", "-Wl,-rpath,/fake"]),
    )

    result = KernelBenchTvmFfiBackend._build_extension(
        tmp_path,
        [str(tmp_path / "binding.cpp")],
        [str(tmp_path / "kernel.cu")],
    )

    assert result["compiled"] is True
    assert captured["extra_ldflags"] == [
        "/fake/libcublas.so.12",
        "-Wl,-rpath,/fake",
        "-Wl,-z,defs",
    ]


@pytest.mark.skipif(
    os.environ.get("KERNELGYM_RUN_TVM_FFI_LINK_INTEGRATION") != "1",
    reason="requires the target CUDA/tvm-ffi compiler image",
)
def test_real_tvm_ffi_extension_links_under_strict_policy(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "benchmarks"))
    from kernels.tvm_ffi_vector_add import KERNEL_CODE

    backend = KernelBenchTvmFfiBackend()
    sources, model_code = backend._parse_embedded_sources(KERNEL_CODE)
    result = backend.compile(
        model_code,
        cuda_sources=sources,
        device="cuda:0",
        entry_point="ModelNew",
        enable_compile_artifact_cache=False,
    )
    try:
        assert result.get("compiled") is True, result.get("error")
    finally:
        work_dir = result.get("work_dir")
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
