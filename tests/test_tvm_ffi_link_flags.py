"""Bug #1 regression: the TVM-FFI build must link cuBLAS/cuBLASLt/cuDNN/cuFFT so
model-generated extensions that call those host APIs do not crash the worker with
``undefined symbol: cublasCreate_v2``.

Covers ``KernelBenchTvmFfiBackend._cuda_math_link_flags`` selection/exclusion
logic with a synthetic wheel layout (no GPU, no nvcc).
"""

import importlib.util
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
