"""Validation helpers for KernelBench toolkit."""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from kernelgym.common import ErrorCode


def validate_code(code: str, entry_point: str = "Model") -> Tuple[bool, str]:
    """Basic validation of PyTorch code."""
    try:
        if not code:
            return False, "Code is required"
        if f"class {entry_point}" not in code:
            return False, f"Code must contain a '{entry_point}' class"
        return True, ""
    except Exception as exc:
        return False, f"Code validation error: {exc}"


def early_kernel_validation(
    kernel_code: str,
    backend: str = "triton",
    entry_point: str = "Model",
) -> Tuple[bool, str, Optional[ErrorCode]]:
    """Perform early kernel code validation without GPU resources."""
    try:
        kernel_entry_point = f"{entry_point}New"
        is_valid, error_msg = validate_code(kernel_code, kernel_entry_point)
        if not is_valid:
            return False, error_msg, ErrorCode.VALIDATION_ERROR

        try:
            compile(kernel_code, "<string>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error in kernel code: {str(e)}", ErrorCode.SYNTAX_ERROR

        if backend == "triton":
            required_imports = ["import triton", "from triton import"]
            if not any(imp in kernel_code for imp in required_imports):
                return False, "Kernel code must import triton for triton backend", ErrorCode.IMPORT_ERROR
        elif backend == "cuda":
            cuda_indicators = [
                "torch.cuda",
                "cuda_kernel",
                "@cuda.jit",
                "from numba import cuda",
            ]
            if not any(indicator in kernel_code for indicator in cuda_indicators):
                return False, "Kernel code must contain CUDA kernel code for cuda backend", ErrorCode.IMPORT_ERROR

        kernel_patterns = [
            "@triton.jit",
            "def.*kernel.*\\(",
            "torch\\.cuda",
            "\\.cuda\\(",
        ]

        import re

        has_kernel_pattern = any(re.search(pattern, kernel_code) for pattern in kernel_patterns)
        if not has_kernel_pattern and backend == "triton":
            return True, "", None

        try:
            test_code = f"""
            import torch
            import torch.nn as nn
            {kernel_code}

            try:
                model = {kernel_entry_point}()
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate {kernel_entry_point}: {{e}}")
            """
            compile(test_code, "<test>", "exec")
        except Exception as e:
            error_str = str(e)
            if "Failed to instantiate" in error_str:
                return False, error_str, ErrorCode.INSTANTIATION_ERROR

        return True, "", None

    except Exception as e:
        return False, f"Early validation error: {str(e)}", ErrorCode.VALIDATION_ERROR


def _find_register_binding_semicolon_issue(
    source_map: dict[str, str],
) -> Optional[Tuple[str, int]]:
    """Detect ``REGISTER_BINDING(...)`` calls missing the required trailing semicolon."""
    marker = "REGISTER_BINDING("

    for file_name, content in source_map.items():
        if not file_name.endswith(".cpp") or marker not in content:
            continue

        search_start = 0
        while True:
            marker_index = content.find(marker, search_start)
            if marker_index == -1:
                break

            paren_depth = 0
            closing_index: Optional[int] = None
            i = marker_index + len("REGISTER_BINDING")
            while i < len(content):
                char = content[i]
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        closing_index = i
                        break
                i += 1

            if closing_index is None:
                line_no = content.count("\n", 0, marker_index) + 1
                return file_name, line_no

            next_index = closing_index + 1
            while next_index < len(content) and content[next_index].isspace():
                next_index += 1

            if next_index >= len(content) or content[next_index] != ";":
                line_no = content.count("\n", 0, marker_index) + 1
                return file_name, line_no

            search_start = closing_index + 1

    return None


def precheck_cuda_agent_submission(
    model_code: str,
    cuda_sources: dict[str, str],
    *,
    entry_point: str = "ModelNew",
    source_mode: str = "files",
) -> Tuple[str, Optional[ErrorCode], dict[str, Any]]:
    """Run a cheap structure precheck before expensive CUDA compilation.

    The checks are intentionally conservative:
    - validate the Python model skeleton and syntax
    - confirm the model imports and calls ``cuda_extension``
    - confirm CUDA/C++ source files are present
    - detect whether the binding layer looks like a Python extension or TVM-FFI
    - apply mode-specific rules while keeping a compatibility hook for TVM-FFI
    """
    try:
        normalized_mode = str(source_mode or "files").strip().lower()
        source_map = cuda_sources or {}

        precheck: dict[str, Any] = {
            "passed": False,
            "source_mode": normalized_mode,
            "cuda_source_count": len(source_map),
            "detected_forward_calls": [],
            "binding_api": "unknown",
            "binding_files": [],
        }

        def _fail(message: str, code: Optional[ErrorCode]) -> Tuple[str, Optional[ErrorCode], dict[str, Any]]:
            formatted_message = f"Precheck failed: {message}"
            precheck["passed"] = False
            precheck["error_message"] = formatted_message
            precheck["error_code"] = code.name if code is not None else None
            return formatted_message, code, precheck

        is_valid, error_msg = validate_code(model_code, entry_point)
        if not is_valid:
            return _fail(error_msg, ErrorCode.VALIDATION_ERROR)

        try:
            compile(model_code, "<string>", "exec")
        except SyntaxError as exc:
            return _fail(f"Syntax error in model code: {exc}", ErrorCode.SYNTAX_ERROR)

        if "cuda_extension" not in model_code:
            return _fail("model_new.py must use cuda_extension", ErrorCode.IMPORT_ERROR)

        # forward_calls = sorted(set(re.findall(r"cuda_extension\.([A-Za-z_]\w*)\s*\(", model_code)))
        # precheck["detected_forward_calls"] = forward_calls
        # if not forward_calls:
        #     return _fail(
        #         "model_new.py must call at least one cuda_extension.* function",
        #         ErrorCode.VALIDATION_ERROR,
        #     )

        if not source_map:
            return _fail("CUDA sources are required for CUDA-Agent compilation", ErrorCode.VALIDATION_ERROR)

        cu_files = [name for name in source_map if name.endswith(".cu")]
        cpp_files = [name for name in source_map if name.endswith(".cpp")]
        precheck["cu_files"] = cu_files
        precheck["cpp_files"] = cpp_files
        precheck["binding_files"] = [name for name in cpp_files if "binding" in name.lower()]

        if not cu_files:
            return _fail(
                "CUDA-Agent sources must include at least one .cu file",
                ErrorCode.VALIDATION_ERROR,
            )

        combined_sources = "\n".join(str(content) for content in source_map.values())
        combined_cpp = "\n".join(
            str(content) for name, content in source_map.items() if name.endswith(".cpp")
        )

        python_ext_markers = [
            'REGISTER_BINDING(',
            'pybind11::module',
            'm.def(',
            '#include "../binding_registry.h"',
            '#include <torch/types.h>',
            'torch::Tensor',
        ]
        tvm_ffi_markers = [
            'TVM_FFI_DLL_EXPORT_TYPED_FUNC(',
            '#include <tvm/ffi/function.h>',
            '#include <tvm/ffi/container/tensor.h>',
            'tvm::ffi::Tensor',
            'tvm::ffi::TensorView',
        ]

        has_python_ext_markers = any(marker in combined_sources for marker in python_ext_markers)
        has_tvm_ffi_markers = any(marker in combined_sources for marker in tvm_ffi_markers)

        if has_python_ext_markers and has_tvm_ffi_markers:
            binding_api = "mixed"
        elif has_tvm_ffi_markers:
            binding_api = "tvm_ffi"
        elif has_python_ext_markers:
            binding_api = "python_extension"
        else:
            binding_api = "unknown"
        precheck["binding_api"] = binding_api

        # missing_exports = [
        #     name for name in forward_calls if name not in combined_cpp and name not in combined_sources
        # ]
        # precheck["missing_forward_exports"] = missing_exports

        if normalized_mode in {"files", "inline"}:
            required_python_ext_markers = {
                '#include "../binding_registry.h"': "binding source must include ../binding_registry.h",
            }
            # 'REGISTER_BINDING(': "binding source must use REGISTER_BINDING(...)",
            for marker, message in required_python_ext_markers.items():
                if marker not in combined_cpp:
                    return _fail(message, ErrorCode.VALIDATION_ERROR)

            if binding_api == "tvm_ffi":
                return _fail(
                    "source_mode files/inline expects a pybind/binding_registry style binding, but TVM-FFI exports were detected",
                    ErrorCode.VALIDATION_ERROR,
                )

            if binding_api == "unknown":
                return _fail(
                    "No supported Python extension binding pattern detected in CUDA sources",
                    ErrorCode.VALIDATION_ERROR,
                )

            register_binding_issue = _find_register_binding_semicolon_issue(source_map)
            if register_binding_issue is not None:
                issue_file, issue_line = register_binding_issue
                return _fail(
                    f"{issue_file}:{issue_line} has REGISTER_BINDING(...) without a trailing ';'",
                    ErrorCode.SYNTAX_ERROR,
                )
        elif normalized_mode == "tvm_ffi":
            if binding_api == "python_extension":
                return _fail(
                    "source_mode='tvm_ffi' requires TVM-FFI exports instead of REGISTER_BINDING/pybind bindings",
                    ErrorCode.VALIDATION_ERROR,
                )
            if binding_api == "unknown":
                return _fail(
                    "source_mode='tvm_ffi' requires TVM_FFI_DLL_EXPORT_TYPED_FUNC(...) or tvm::ffi Tensor bindings",
                    ErrorCode.VALIDATION_ERROR,
                )

        # if missing_exports and normalized_mode != "tvm_ffi":
        #     return _fail(
        #         f"model_new.py calls cuda_extension functions that were not found in binding sources: {missing_exports}",
        #         ErrorCode.VALIDATION_ERROR,
        #     )

        precheck["passed"] = True
        precheck["error_message"] = ""
        precheck["error_code"] = None
        return "", None, precheck
    except Exception as exc:
        return (
            f"Precheck failed: CUDA-Agent precheck error: {exc}",
            ErrorCode.VALIDATION_ERROR,
            {
                "passed": False,
                "error_message": f"Precheck failed: CUDA-Agent precheck error: {exc}",
                "error_code": ErrorCode.VALIDATION_ERROR.name,
            },
        )
