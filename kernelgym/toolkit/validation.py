"""Validation helpers for KernelBench toolkit."""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from kernelgym.common import ErrorCode
from kernelgym.toolkit.kernelbench.static_checker import (
    detect_extension_calls,
    mask_native_comments,
    mask_native_noncode,
    validate_kernel_static,
)


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


def _find_register_binding_semicolon_issue(source_map: dict[str, str]) -> tuple[str, int] | None:
    marker = "REGISTER_BINDING("
    for filename, content in source_map.items():
        search_start = 0
        while True:
            marker_index = content.find(marker, search_start)
            if marker_index == -1:
                break

            depth = 0
            closing_index: int | None = None
            for index in range(marker_index, len(content)):
                char = content[index]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        closing_index = index
                        break

            if closing_index is None:
                line_no = content.count("\n", 0, marker_index) + 1
                return filename, line_no

            next_index = closing_index + 1
            while next_index < len(content) and content[next_index].isspace():
                next_index += 1

            if next_index >= len(content) or content[next_index] != ";":
                line_no = content.count("\n", 0, marker_index) + 1
                return filename, line_no

            search_start = closing_index + 1

    return None


def _has_native_include(code: str, header: str) -> bool:
    pattern = rf'(?m)^\s*#\s*include\s*[<"]{re.escape(header)}[>"]'
    comments_only = mask_native_comments(code)
    noncode_masked = mask_native_noncode(code)
    directive = re.compile(r"^\s*#\s*include\b")
    return any(
        directive.search(masked_line) is not None and re.search(pattern, source_line) is not None
        for source_line, masked_line in zip(comments_only.splitlines(), noncode_masked.splitlines())
    )


def _extract_tvm_ffi_exports(source_map: dict[str, str]) -> list[str]:
    export_pattern = re.compile(
        r"\bTVM_FFI_DLL_EXPORT_TYPED_FUNC\s*\(\s*([A-Za-z_]\w*)\s*,",
        re.MULTILINE,
    )
    exports: set[str] = set()
    for name, content in source_map.items():
        if not name.lower().endswith((".cpp", ".cc", ".cxx")):
            continue
        exports.update(export_pattern.findall(mask_native_noncode(str(content))))
    return sorted(exports)


def _run_submission_static_check(
    precheck: dict[str, Any],
    model_code: str,
    source_map: dict[str, str],
    *,
    precision: str,
    allowed_extension_modules: set[str],
) -> Tuple[str, Optional[ErrorCode], dict[str, Any]] | None:
    result = validate_kernel_static(
        model_code,
        precision=precision,
        source_map=source_map,
        allowed_extension_modules=allowed_extension_modules,
    )
    precheck["static_check"] = result.to_dict()
    if result.valid:
        return None
    message = "Precheck failed: static check failed: " + "; ".join(result.errors)
    precheck["error_message"] = message
    precheck["error_code"] = ErrorCode.VALIDATION_ERROR.name
    return message, ErrorCode.VALIDATION_ERROR, precheck


def precheck_cuda_agent_submission(
    model_code: str,
    cuda_sources: dict[str, str],
    *,
    entry_point: str = "ModelNew",
    precision: str = "fp32",
) -> Tuple[str, Optional[ErrorCode], dict[str, Any]]:
    """Run a cheap structure precheck before CUDA-Agent compilation."""

    source_map = cuda_sources or {}
    precheck: dict[str, Any] = {
        "passed": False,
        "cuda_source_count": len(source_map),
        "binding_files": [],
        "static_check": None,
    }

    def _fail(message: str, code: Optional[ErrorCode]) -> Tuple[str, Optional[ErrorCode], dict[str, Any]]:
        formatted = f"Precheck failed: {message}"
        precheck["error_message"] = formatted
        precheck["error_code"] = code.name if code is not None else None
        return formatted, code, precheck

    try:
        is_valid, error_msg = validate_code(model_code, entry_point)
        if not is_valid:
            return _fail(error_msg, ErrorCode.VALIDATION_ERROR)

        try:
            compile(model_code, "<string>", "exec")
        except SyntaxError as exc:
            return _fail(f"Syntax error in model code: {exc}", ErrorCode.SYNTAX_ERROR)

        try:
            imported_extension, detected_calls = detect_extension_calls(model_code, "cuda_extension")
        except SyntaxError as exc:
            return _fail(f"Syntax error in model code: {exc}", ErrorCode.SYNTAX_ERROR)

        if not imported_extension:
            return _fail(
                "model_new.py must import cuda_extension",
                ErrorCode.IMPORT_ERROR,
            )

        if not source_map:
            return _fail(
                "CUDA sources are required for CUDA-Agent compilation",
                ErrorCode.VALIDATION_ERROR,
            )

        cu_files = [name for name in source_map if name.lower().endswith(".cu")]
        cpp_files = [name for name in source_map if name.lower().endswith((".cpp", ".cc", ".cxx"))]
        header_files = [name for name in source_map if name.lower().endswith((".h", ".hpp", ".hh", ".cuh"))]
        precheck["cu_files"] = cu_files
        precheck["cpp_files"] = cpp_files
        precheck["header_files"] = header_files

        if not cu_files:
            return _fail(
                "CUDA-Agent sources must include at least one .cu file",
                ErrorCode.VALIDATION_ERROR,
            )

        binding_candidates = [name for name in cpp_files if "binding" in name.lower() or "bind" in name.lower()]
        precheck["binding_files"] = binding_candidates
        if not binding_candidates:
            return _fail(
                "CUDA-Agent sources must include a binding .cpp file",
                ErrorCode.VALIDATION_ERROR,
            )

        combined_cpp = "\n".join(mask_native_noncode(str(source_map[name])) for name in cpp_files)
        uses_pybind11_module = re.search(r"\bPYBIND11_MODULE\s*\(", combined_cpp) is not None
        precheck["binding_mode"] = "pybind11_module" if uses_pybind11_module else "register_binding"

        if not uses_pybind11_module:
            if not any(
                _has_native_include(str(source_map[name]), header)
                for name in cpp_files
                for header in ("../binding_registry.h", "binding_registry.h")
            ):
                return _fail(
                    "Binding source must include binding_registry.h",
                    ErrorCode.VALIDATION_ERROR,
                )

            if "REGISTER_BINDING(" not in combined_cpp:
                return _fail(
                    "Binding source must register functions with REGISTER_BINDING(...)",
                    ErrorCode.VALIDATION_ERROR,
                )

            binding_issue = _find_register_binding_semicolon_issue(
                {name: mask_native_noncode(str(content)) for name, content in source_map.items()}
            )
            if binding_issue is not None:
                issue_file, issue_line = binding_issue
                return _fail(
                    f"{issue_file}:{issue_line} has REGISTER_BINDING(...) without a trailing ';'",
                    ErrorCode.SYNTAX_ERROR,
                )

        precheck["detected_extension_calls"] = detected_calls
        static_failure = _run_submission_static_check(
            precheck,
            model_code,
            source_map,
            precision=precision,
            allowed_extension_modules={"cuda_extension"},
        )
        if static_failure is not None:
            return static_failure

        precheck["passed"] = True
        precheck["error_message"] = ""
        precheck["error_code"] = None
        return "", None, precheck
    except Exception as exc:
        return _fail(f"CUDA-Agent precheck error: {exc}", ErrorCode.VALIDATION_ERROR)


def precheck_tvm_ffi_submission(
    model_code: str,
    cuda_sources: dict[str, str],
    *,
    entry_point: str = "ModelNew",
    precision: str = "fp32",
) -> Tuple[str, Optional[ErrorCode], dict[str, Any]]:
    """Run a cheap structure precheck before TVM-FFI CUDA compilation."""

    source_map = cuda_sources or {}
    precheck: dict[str, Any] = {
        "passed": False,
        "cuda_source_count": len(source_map),
        "binding_files": [],
        "binding_mode": "tvm_ffi",
        "static_check": None,
    }

    def _fail(message: str, code: Optional[ErrorCode]) -> Tuple[str, Optional[ErrorCode], dict[str, Any]]:
        formatted = f"Precheck failed: {message}"
        precheck["error_message"] = formatted
        precheck["error_code"] = code.name if code is not None else None
        return formatted, code, precheck

    try:
        is_valid, error_msg = validate_code(model_code, entry_point)
        if not is_valid:
            return _fail(error_msg, ErrorCode.VALIDATION_ERROR)

        try:
            compile(model_code, "<string>", "exec")
        except SyntaxError as exc:
            return _fail(f"Syntax error in model code: {exc}", ErrorCode.SYNTAX_ERROR)

        try:
            imported_extension, detected_calls = detect_extension_calls(
                model_code,
                "tvm_ffi_extension",
            )
        except SyntaxError as exc:
            return _fail(f"Syntax error in model code: {exc}", ErrorCode.SYNTAX_ERROR)

        precheck["detected_extension_calls"] = detected_calls
        if not imported_extension:
            return _fail(
                "model_new.py must import or reference tvm_ffi_extension",
                ErrorCode.IMPORT_ERROR,
            )
        if not detected_calls:
            return _fail(
                "model_new.py must call at least one tvm_ffi_extension function",
                ErrorCode.VALIDATION_ERROR,
            )

        if not source_map:
            return _fail(
                "CUDA sources are required for TVM-FFI compilation",
                ErrorCode.VALIDATION_ERROR,
            )

        cu_files = [name for name in source_map if name.lower().endswith(".cu")]
        cpp_files = [name for name in source_map if name.lower().endswith((".cpp", ".cc", ".cxx"))]
        header_files = [name for name in source_map if name.lower().endswith((".h", ".hpp", ".hh", ".cuh"))]
        precheck["cu_files"] = cu_files
        precheck["cpp_files"] = cpp_files
        precheck["header_files"] = header_files

        if not cu_files:
            return _fail(
                "TVM-FFI sources must include at least one .cu file",
                ErrorCode.VALIDATION_ERROR,
            )

        binding_candidates = [name for name in cpp_files if "binding" in name.lower() or "bind" in name.lower()]
        precheck["binding_files"] = binding_candidates
        if not binding_candidates:
            return _fail(
                "TVM-FFI sources must include a binding .cpp file",
                ErrorCode.VALIDATION_ERROR,
            )

        combined_cpp = "\n".join(mask_native_noncode(str(source_map[name])) for name in cpp_files)
        forbidden_markers = (
            "PYBIND11_MODULE",
            "REGISTER_BINDING(",
        )
        for marker in forbidden_markers:
            if marker in combined_cpp:
                return _fail(
                    f"TVM-FFI binding source must not use pybind11 marker {marker}",
                    ErrorCode.VALIDATION_ERROR,
                )

        if any(
            _has_native_include(str(source_map[name]), header)
            for name in cpp_files
            for header in ("../binding_registry.h", "binding_registry.h")
        ):
            return _fail(
                "TVM-FFI binding source must not use pybind11 marker binding_registry.h",
                ErrorCode.VALIDATION_ERROR,
            )

        forbidden_cuda_header = next(
            (
                header
                for name in cpp_files
                for header in ("cuda_runtime.h", "cuda.h")
                if _has_native_include(str(source_map[name]), header)
            ),
            None,
        )
        forbidden_cuda_marker = forbidden_cuda_header or ("cudaStream_t" if "cudaStream_t" in combined_cpp else None)
        if forbidden_cuda_marker is not None:
            return _fail(
                "TVM-FFI host binding source must keep CUDA runtime headers/types out of "
                f"binding .cpp files; use an opaque void* stream handle instead of {forbidden_cuda_marker}",
                ErrorCode.VALIDATION_ERROR,
            )

        if not any(
            _has_native_include(str(source_map[name]), header)
            for name in cpp_files
            for header in ("tvm/ffi/tvm_ffi.h", "tvm/ffi/function.h", "tvm/ffi/container/tensor.h")
        ):
            return _fail(
                "TVM-FFI binding source must include a tvm/ffi header",
                ErrorCode.VALIDATION_ERROR,
            )

        exported_functions = _extract_tvm_ffi_exports(source_map)
        precheck["exported_functions"] = exported_functions
        if not exported_functions:
            return _fail(
                "TVM-FFI binding source must export functions with TVM_FFI_DLL_EXPORT_TYPED_FUNC(...)",
                ErrorCode.VALIDATION_ERROR,
            )

        missing_exports = sorted(set(detected_calls) - set(exported_functions))
        if missing_exports:
            return _fail(
                "TVM-FFI model calls are not exported: " + ", ".join(missing_exports),
                ErrorCode.VALIDATION_ERROR,
            )

        static_failure = _run_submission_static_check(
            precheck,
            model_code,
            source_map,
            precision=precision,
            allowed_extension_modules={"tvm_ffi_extension"},
        )
        if static_failure is not None:
            return static_failure

        precheck["passed"] = True
        precheck["error_message"] = ""
        precheck["error_code"] = None
        return "", None, precheck
    except Exception as exc:
        return _fail(f"TVM-FFI precheck error: {exc}", ErrorCode.VALIDATION_ERROR)
