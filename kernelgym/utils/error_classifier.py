"""Error classification utilities for KernelGym."""

import re
from typing import Optional

from kernelgym.common import ErrorCode


TVM_FFI_API_TENSOR_ACCESSOR = "tvm_ffi_api_tensor_accessor"
TVM_FFI_API_SHAPEVIEW = "tvm_ffi_api_shapeview"
TVM_FFI_API_DTYPE = "tvm_ffi_api_dtype"
TVM_FFI_API_EXPORT_MACRO = "tvm_ffi_api_export_macro"
TVM_FFI_API_WRONG_BINDING_FRAMEWORK = "tvm_ffi_api_wrong_binding_framework"
TVM_FFI_API_MISUSE = "tvm_ffi_api_misuse"
OTHER_COMPILE_ERROR = "other"
FAILURE_PROMPT_OVERLONG = "prompt_overlong"
FAILURE_TIMEOUT = "timeout"
FAILURE_OOM = "oom"
FAILURE_VALIDATION = "validation_precheck"
FAILURE_DECOY = "decoy_or_no_custom_compute"
FAILURE_COMPILE = "compile_error"
FAILURE_MODEL_RUNTIME = "model_runtime_python"
FAILURE_RUNTIME = "runtime_error_non_oom"
FAILURE_CORRECTNESS = "correctness_mismatch"
FAILURE_OTHER = "other"

_TVM_FFI_MARKER_RE = re.compile(
    r"\btvm::ffi\b|\btvm_ffi\b|\btvm-ffi\b|\btvm/ffi\b|\btvmffi\b",
)
_TVM_FFI_API_MARKER_RE = re.compile(
    r"\btvm::ffi\b|\btvm/ffi\b|\btvm_ffi_dll_export_typed_func\b",
)

_TVM_FFI_TENSOR_ACCESSOR_PATTERNS = (
    r"\btvm::ffi::tensor::shape\s*\(",
    r"\btvm::ffi::tensor::ndim\s*(?:\(|')",
    r"\btvm::ffi::tensor\b.*no member named '(?:data|data_ptr|shape_data|num_dims|dl_tensor)'",
    r"\bclass tvm::ffi::tensor\b.*no member named '(?:data|data_ptr|shape_data|num_dims|dl_tensor)'",
    r"\binvalid use of member function .*tvm::ffi::tensor::ndim",
    r"\bcannot convert 'tvm::ffi::tensor::ndim' from type",
    r"\btvm::ffi::tensor::get\(\) const' is protected",
    r"\bconst class tvm::ffi::object' has no member named 'ndim'",
    r"\binvalid types '<unresolved overloaded function type>\[int\]' for array subscript",
)
_TVM_FFI_SHAPEVIEW_PATTERNS = (
    r"\btvm::ffi::shapeview\b",
    r"\boperator==.*shapeview",
    r"\bshapeview.*operator==",
)
_TVM_FFI_DTYPE_PATTERNS = (
    r"\bstruct dldatatype' has no member named '(?:type_code|type|device_type)'",
    r"\bdldatatype\b.*\boperator==",
    r"\boperator==.*\bdldatatype\b",
)
_TVM_FFI_EXPORT_MACRO_PATTERNS = (
    r"\btvm_ffi_dll_export_typed_func\b",
    r"\bpasting \"_registrar_\"",
    r"\bdecltype' cannot resolve address of overloaded function",
)
_TVM_FFI_WRONG_BINDING_PATTERNS = (
    r"\bpybind11::",
    r"\bat::tensor\b",
    r"\btorch::",
    r"\bc10::",
    r"\baten/",
    r"\brequest for member 'item' in .*\.size\(",
)


def _normalize_error_text(error_message: str) -> str:
    """Normalize compiler diagnostics enough for stable regex matching."""
    return (
        error_message.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .lower()
    )


def _has_pattern(error_text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, error_text, re.DOTALL) for pattern in patterns)


def classify_compile_error_detail(
    error_message: str,
    backend: Optional[str] = None,
) -> str:
    """Classify detailed compile-error causes used by offline failure analysis.

    The broad public ErrorCode intentionally stays stable. This helper returns a
    finer string category for compiler diagnostics, with explicit TVM-FFI API
    buckets before the generic ``other`` fallback.
    """
    if not error_message:
        return OTHER_COMPILE_ERROR

    error_text = _normalize_error_text(str(error_message))
    backend_key = (backend or "").lower().replace("-", "_")
    has_tvm_ffi_marker = bool(_TVM_FFI_MARKER_RE.search(error_text))
    has_tvm_ffi_api_marker = bool(_TVM_FFI_API_MARKER_RE.search(error_text))
    is_tvm_ffi_backend = backend_key == "tvm_ffi"

    if _has_pattern(error_text, _TVM_FFI_EXPORT_MACRO_PATTERNS) and (has_tvm_ffi_marker or is_tvm_ffi_backend):
        return TVM_FFI_API_EXPORT_MACRO

    if _has_pattern(error_text, _TVM_FFI_WRONG_BINDING_PATTERNS) and (has_tvm_ffi_marker or is_tvm_ffi_backend):
        return TVM_FFI_API_WRONG_BINDING_FRAMEWORK

    if _has_pattern(error_text, _TVM_FFI_DTYPE_PATTERNS) and (has_tvm_ffi_marker or is_tvm_ffi_backend):
        return TVM_FFI_API_DTYPE

    if _has_pattern(error_text, _TVM_FFI_SHAPEVIEW_PATTERNS):
        return TVM_FFI_API_SHAPEVIEW

    if _has_pattern(error_text, _TVM_FFI_TENSOR_ACCESSOR_PATTERNS) and (has_tvm_ffi_marker or is_tvm_ffi_backend):
        return TVM_FFI_API_TENSOR_ACCESSOR

    if has_tvm_ffi_api_marker and " error:" in error_text:
        return TVM_FFI_API_MISUSE

    return OTHER_COMPILE_ERROR


def _is_true(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _is_false(value: object) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no"}
    return False


def _collect_metadata_error_text(metadata: Optional[dict]) -> str:
    if not metadata:
        return ""
    parts: list[str] = []
    for key in (
        "error",
        "error_message",
        "compilation_error",
        "runtime_error",
        "model_load_error",
        "validation_error",
        "compilation_error_name",
        "runtime_error_name",
        "model_load_error_name",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    return "\n".join(parts)


def classify_failure_detail(
    error_message: str = "",
    *,
    status: Optional[str] = None,
    compiled: Optional[object] = None,
    correctness: Optional[object] = None,
    decoy_kernel: Optional[object] = None,
    error_code: Optional[object] = None,
    backend: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Classify a failed reward/eval result into a stable fine-grained bucket."""
    metadata_text = _collect_metadata_error_text(metadata)
    error_text = _normalize_error_text("\n".join(part for part in (error_message, metadata_text) if part))
    status_key = (status or "").strip().lower()
    error_code_value = getattr(error_code, "value", error_code)
    error_code_key = str(error_code_value or "").strip().upper()

    if "prompt overlong" in error_text or "prompt too long" in error_text:
        return FAILURE_PROMPT_OVERLONG

    if status_key == "timeout" or "timeout" in error_text or "timed out" in error_text:
        return FAILURE_TIMEOUT

    if _is_true(decoy_kernel) or "decoy kernel" in error_text or "reward hacking" in error_text:
        return FAILURE_DECOY

    if (
        "validation failed" in error_text
        or "client validation failed" in error_text
        or "precheck failed" in error_text
        or "pre-check error" in error_text
        or "missing class modelnew" in error_text
        or "sections are missing" in error_text
        or error_code_key == ErrorCode.VALIDATION_ERROR.value
    ):
        return FAILURE_VALIDATION

    if "outofmemoryerror" in error_text or "out of memory" in error_text or "cuda oom" in error_text:
        return FAILURE_OOM

    if (
        "nameerror" in error_text
        or re.search(r"\bname ['\"][^'\"]+['\"] is not defined", error_text)
        or "modulenotfounderror" in error_text
        or "no module named" in error_text
        or "attributeerror" in error_text
        or re.search(r"\b(?:object|module)\b.*\bhas no attribute\b", error_text)
        or "model_load_error" in error_text
    ):
        return FAILURE_MODEL_RUNTIME

    if (
        error_code_key == ErrorCode.COMPILATION_ERROR.value
        or "kernel compilation error" in error_text
        or "compilation failed" in error_text
        or "compilation_error" in error_text
        or (_is_false(compiled) and error_text)
    ):
        detail = classify_compile_error_detail(error_text, backend=backend)
        return detail if detail != OTHER_COMPILE_ERROR else FAILURE_COMPILE

    if error_code_key == ErrorCode.RUNTIME_ERROR.value or "runtime_error" in error_text:
        return FAILURE_RUNTIME

    if (
        error_code_key == ErrorCode.CORRECTNESS_ERROR.value
        or _is_false(correctness)
        or "incorrect" in error_text
        or "mismatch" in error_text
        or "correctness" in error_text
    ):
        return FAILURE_CORRECTNESS

    return FAILURE_OTHER


def classify_error(error_message: str, context: Optional[str] = None) -> ErrorCode:
    """Classify error message into appropriate error code."""
    if not error_message:
        return ErrorCode.UNKNOWN_ERROR

    error_lower = error_message.lower()

    validation_patterns = [
        r"validation failed",
        r"dangerous pattern detected",
        r"code must contain",
        r"invalid code format",
        r"missing.*class",
        r"invalid.*entry.*point",
        r"code validation error",
    ]
    if any(re.search(pattern, error_lower) for pattern in validation_patterns):
        return ErrorCode.VALIDATION_ERROR

    compilation_patterns = [
        r"compilation failed",
        r"compile.*error",
        r"syntax error",
        r"nvcc.*error",
        r"cuda.*compilation",
        r"triton.*compilation",
        r"kernel.*compilation",
        r"build.*failed",
        r"linker.*error",
    ]
    if any(re.search(pattern, error_lower) for pattern in compilation_patterns):
        return ErrorCode.COMPILATION_ERROR

    timeout_patterns = [
        r"timeout",
        r"task.*timed.*out",
        r"execution.*timeout",
        r"time.*limit.*exceeded",
        r"hung.*task",
        r"stuck.*task",
    ]
    if any(re.search(pattern, error_lower) for pattern in timeout_patterns):
        return ErrorCode.TIMEOUT_ERROR

    runtime_patterns = [
        r"runtime error",
        r"kernel.*execution.*failed",
        r"cuda.*runtime",
        r"out of memory",
        r"device.*error",
        r"gpu.*error",
        r"invalid.*device",
        r"cuda.*error",
        r"execution.*failed",
    ]
    if any(re.search(pattern, error_lower) for pattern in runtime_patterns):
        return ErrorCode.RUNTIME_ERROR

    correctness_patterns = [
        r"correctness.*check.*failed",
        r"output.*mismatch",
        r"result.*incorrect",
        r"assertion.*failed",
        r"accuracy.*test.*failed",
        r"numerical.*error",
        r"precision.*error",
    ]
    if any(re.search(pattern, error_lower) for pattern in correctness_patterns):
        return ErrorCode.CORRECTNESS_ERROR

    system_patterns = [
        r"system.*error",
        r"internal.*server.*error",
        r"redis.*error",
        r"database.*error",
        r"connection.*failed",
        r"service.*unavailable",
        r"initialization.*failed",
    ]
    if any(re.search(pattern, error_lower) for pattern in system_patterns):
        return ErrorCode.SYSTEM_ERROR

    resource_patterns = [
        r"resource.*error",
        r"insufficient.*memory",
        r"queue.*full",
        r"no.*available.*workers",
        r"gpu.*unavailable",
        r"device.*busy",
        r"memory.*exhausted",
        r"disk.*space",
        r"resource.*exhausted",
    ]
    if any(re.search(pattern, error_lower) for pattern in resource_patterns):
        return ErrorCode.RESOURCE_ERROR

    if context:
        context_lower = context.lower()
        if "validation" in context_lower:
            return ErrorCode.VALIDATION_ERROR
        if "compilation" in context_lower or "compile" in context_lower:
            return ErrorCode.COMPILATION_ERROR
        if "runtime" in context_lower or "execution" in context_lower:
            return ErrorCode.RUNTIME_ERROR
        if "correctness" in context_lower:
            return ErrorCode.CORRECTNESS_ERROR
        if "timeout" in context_lower:
            return ErrorCode.TIMEOUT_ERROR
        if "system" in context_lower:
            return ErrorCode.SYSTEM_ERROR
        if "resource" in context_lower:
            return ErrorCode.RESOURCE_ERROR

    return ErrorCode.UNKNOWN_ERROR


def get_error_description(error_code: ErrorCode) -> str:
    """Get human-readable description for error code."""
    descriptions = {
        ErrorCode.VALIDATION_ERROR: "Code validation failed - invalid or dangerous code detected",
        ErrorCode.COMPILATION_ERROR: "Kernel compilation failed - syntax or build errors",
        ErrorCode.RUNTIME_ERROR: "Kernel runtime error - execution or GPU errors",
        ErrorCode.CORRECTNESS_ERROR: "Correctness check failed - output doesn't match reference",
        ErrorCode.TIMEOUT_ERROR: "Task timeout - execution took too long",
        ErrorCode.SYSTEM_ERROR: "System error - internal service or infrastructure issue",
        ErrorCode.RESOURCE_ERROR: "Resource error - insufficient memory or unavailable resources",
        ErrorCode.UNKNOWN_ERROR: "Unknown error - unclassified error type",
    }
    return descriptions.get(error_code, "Unknown error type")


def get_error_category(error_code: ErrorCode) -> str:
    """Get error category for grouping similar errors."""
    categories = {
        ErrorCode.VALIDATION_ERROR: "input",
        ErrorCode.COMPILATION_ERROR: "compilation",
        ErrorCode.RUNTIME_ERROR: "runtime",
        ErrorCode.CORRECTNESS_ERROR: "correctness",
        ErrorCode.TIMEOUT_ERROR: "timeout",
        ErrorCode.SYSTEM_ERROR: "system",
        ErrorCode.RESOURCE_ERROR: "resource",
        ErrorCode.UNKNOWN_ERROR: "unknown",
    }
    return categories.get(error_code, "unknown")
