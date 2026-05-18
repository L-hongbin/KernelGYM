"""Static reward-hacking checks for KernelBench submissions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


@dataclass(frozen=True)
class StaticCheckResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    precision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "precision": self.precision,
        }


def _strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    lines: list[str] = []
    for line in code.splitlines():
        hash_index = line.find("#")
        slash_index = line.find("//")
        cut_points = [idx for idx in (hash_index, slash_index) if idx >= 0]
        if cut_points:
            line = line[: min(cut_points)]
        lines.append(line)
    return "\n".join(lines)


def _search_patterns(
    code: str,
    patterns: list[str],
    message: str,
) -> tuple[bool, str]:
    stripped = _strip_comments(code)
    for pattern in patterns:
        if re.search(pattern, stripped):
            return True, message
    return False, ""


def check_code_bypass(code: str) -> tuple[bool, str]:
    return _search_patterns(
        code,
        [
            r"\btry\s*:",
            r"\bexcept\s*:",
            r"\bexcept\s+\w+",
            r"\bpass\b",
        ],
        "Contains try/except or pass fallback bypass pattern",
    )


def check_timing_event_patch(code: str) -> tuple[bool, str]:
    return _search_patterns(
        code,
        [
            r"torch\.cuda\.Event\.record\s*=",
            r"torch\.cuda\.Event\.elapsed_time\s*=",
            r"torch\.cuda\.synchronize\s*=",
            r"torch\.cuda\.Event\s*=",
            r"time\.perf_counter\s*=",
            r"time\.time\s*=",
        ],
        "Reassigns timing or CUDA synchronization function",
    )


def check_thread_injection(code: str) -> tuple[bool, str]:
    return _search_patterns(
        code,
        [
            r"threading\.Thread\s*\(",
            r"import\s+threading\b",
            r"from\s+threading\s+import\b",
            r"multiprocessing\.(Process|Pool|Manager|Queue|Pipe)",
            r"import\s+multiprocessing\b",
            r"concurrent\.futures\b",
            r"ThreadPoolExecutor\b",
            r"ProcessPoolExecutor\b",
        ],
        "Uses threading or multiprocessing inside submitted kernel code",
    )


def check_lazy_eval(code: str) -> tuple[bool, str]:
    return _search_patterns(
        code,
        [
            r"_make_subclass\b",
            r"class\s+\w+.*\(torch\.Tensor\)",
            r"class\s+\w+.*\(Tensor\)",
            r"torch\.Tensor\.__new__",
        ],
        "Defines or constructs lazy/fake tensor objects",
    )


def check_stream_injection(code: str) -> tuple[bool, str]:
    return _search_patterns(
        code,
        [
            r"torch\.cuda\.Stream\s*\(",
            r"\bcuda\.Stream\s*\(",
            r"with\s+torch\.cuda\.stream\b",
            r"\.wait_stream\s*\(",
            r"\.record_stream\s*\(",
        ],
        "Uses explicit CUDA stream control",
    )


_FRAMEWORK_COMPUTE_PATTERNS = [
    r"\bat::(matmul|mm|bmm|einsum|conv[123]?d|softmax|log_softmax|relu|gelu|layer_norm|batch_norm|sum|mean|max|min|prod|cumprod|cumsum|exp|sqrt|rsqrt|norm|cross_entropy|nll_loss)\s*\(",
    r"\btorch::(matmul|mm|bmm|einsum|conv[123]?d|softmax|log_softmax|relu|gelu|layer_norm|batch_norm|sum|mean|max|min|prod|cumprod|cumsum|exp|sqrt|rsqrt|norm|cross_entropy|nll_loss)\s*\(",
    r"\btorch\.(matmul|mm|bmm|einsum|conv[123]?d|softmax|log_softmax|relu|gelu|layer_norm|batch_norm|sum|mean|max|min|prod|cumprod|cumsum|exp|sqrt|rsqrt|norm|cross_entropy|nll_loss)\s*\(",
    r"\.\s*(matmul|mm|bmm|softmax|log_softmax|relu|gelu|sum|mean|max|min|prod|cumprod|cumsum|exp|sqrt|rsqrt|norm)\s*\(",
]


def check_framework_compute(code: str) -> tuple[bool, str]:
    return _search_patterns(
        code,
        _FRAMEWORK_COMPUTE_PATTERNS,
        "Uses PyTorch/ATen compute instead of custom CUDA kernels",
    )


_FP32_TO_FP16_PATTERNS = [
    r"__float2half(_rn)?\s*\(",
    r"\(\s*__half\s*\)\s*[\w\->\.]+",
    r"static_cast\s*<\s*(__half|half)\s*>\s*\(",
    r"\bCUBLAS_COMPUTE_(16F|32F_FAST_16F)\b",
    r"\bCUDA_R_16F\b",
    r"tl\.astype\s*\(\s*[^,]+,\s*tl\.float16\s*\)",
    r"NumericConverter\s*<\s*half_t\s*,\s*float\s*>",
    r"LinearCombination\s*<\s*half_t",
    r"type_convert\s*<\s*half_t\s*>\s*\(",
    r"tk::half\s*\(",
    r"\.half\s*\(",
    r"\.float16\s*\(",
    r"\.to\s*\(\s*torch\.(float16|half)\b",
    r"\.to\s*\(\s*dtype\s*=\s*torch\.(float16|half)\b",
]


def _normalize_precision(precision: str) -> str:
    normalized = precision.strip().lower()
    precision_map = {
        "fp32": "fp32",
        "float32": "fp32",
        "torch.float32": "fp32",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "torch.float16": "fp16",
        "torch.half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
    }
    return precision_map.get(normalized, normalized)


def check_precision_downgrade(code: str, precision: str = "fp32") -> tuple[bool, str]:
    if _normalize_precision(precision) != "fp32":
        return False, ""
    return _search_patterns(
        code,
        _FP32_TO_FP16_PATTERNS,
        "Precision downgrade detected: required FP32 but code uses FP16",
    )


CheckFn = Callable[[str], tuple[bool, str]]
PrecisionCheckFn = Callable[[str, str], tuple[bool, str]]


_CHECKS: dict[str, CheckFn | PrecisionCheckFn] = {
    "code_bypass": check_code_bypass,
    "timing_event_patch": check_timing_event_patch,
    "thread_injection": check_thread_injection,
    "lazy_eval": check_lazy_eval,
    "stream_injection": check_stream_injection,
    "framework_compute": check_framework_compute,
    "precision_downgrade": check_precision_downgrade,
}

_PRECISION_CHECKS = {"precision_downgrade"}

DEFAULT_FORBIDDEN_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",
    "lazy_eval",
    "framework_compute",
    "precision_downgrade",
]

DEFAULT_WARNING_CHECKS = [
    "stream_injection",
]


def validate_kernel_static(
    code: str,
    *,
    precision: str = "fp32",
    forbidden: list[str] | None = None,
    warnings: list[str] | None = None,
) -> StaticCheckResult:
    forbidden_checks = list(DEFAULT_FORBIDDEN_CHECKS if forbidden is None else forbidden)
    warning_checks = list(DEFAULT_WARNING_CHECKS if warnings is None else warnings)
    all_checks = list(dict.fromkeys(forbidden_checks + warning_checks))

    errors: list[str] = []
    warnings_out: list[str] = []
    normalized_precision = _normalize_precision(precision)

    for check_name in all_checks:
        check = _CHECKS.get(check_name)
        if check is None:
            continue
        if check_name in _PRECISION_CHECKS:
            has_issue, message = check(code, normalized_precision)  # type: ignore[misc]
        else:
            has_issue, message = check(code)  # type: ignore[misc]
        if not has_issue:
            continue
        formatted = f"{check_name}: {message}"
        if check_name in forbidden_checks:
            errors.append(formatted)
        else:
            warnings_out.append(formatted)

    return StaticCheckResult(
        valid=not errors,
        errors=errors,
        warnings=warnings_out,
        precision=normalized_precision,
    )
