"""KernelBench model loading helpers (toolkit layer)."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


_MODEL_TMPDIR_ENV = "KERNELGYM_MODEL_TMPDIR"
_MODEL_DEFAULT_TMPDIR = "/dev/shm/kernelgym/work/model_loading"
_FAST_RW_ROOT = Path("/dev/shm")


def _model_tmpdir() -> str:
    path = Path(os.environ.get(_MODEL_TMPDIR_ENV, _MODEL_DEFAULT_TMPDIR))
    try:
        resolved_path = path.resolve(strict=False)
        resolved_root = _FAST_RW_ROOT.resolve(strict=False)
    except OSError:
        resolved_path = path.absolute()
        resolved_root = _FAST_RW_ROOT.absolute()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"{_MODEL_TMPDIR_ENV} must be under /dev/shm for fast local I/O: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(f"{_MODEL_TMPDIR_ENV} is not writable/executable: {path}")
    return str(path)


def load_original_model_and_inputs(
    model_original_src: str, context: dict, entry_point: str = "Model"
) -> Tuple[nn.Module, callable, callable]:
    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None
    try:
        exec(model_original_src, context)
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get(entry_point)

    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model_with_tempfile(model_custom_src: str, entry_point: str = "ModelNew"):
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        dir=_model_tmpdir(),
    ) as tmp_file:
        tmp_file.write(model_custom_src)
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    return ModelNew, temp_file


def load_custom_model(model_custom_src: str, context: dict, build_directory: str = None) -> nn.Module:
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_custom_src = (
            f"import os\nos.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
):
    del curr_context
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)
    if tempfile:
        tempfile.close()
        os.remove(tempfile.name)
