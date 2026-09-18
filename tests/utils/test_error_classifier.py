import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_path = str(REPO_ROOT)
if repo_root_path not in sys.path:
    sys.path.insert(0, repo_root_path)

from kernelgym.utils.error_classifier import (
    FUNCTION_ARGUMENT_MISMATCH,
    INSTRUCTION_ARGUMENT_MISMATCH,
    INCOMPLETE_TYPE,
    INVALID_DECLARATION,
    INVALID_TYPE_CONVERSION,
    MISSING_HEADER,
    OTHER_COMPILE_ERROR,
    SYNTAX_ERROR,
    TVM_FFI_API_DTYPE,
    UNDEFINED_IDENTIFIER,
    classify_compile_error_metadata,
    classify_compile_error_detail,
    classify_failure_detail,
    extract_compile_error_excerpt,
)

NUM_GPUS = 0


@pytest.mark.parametrize("marker", ["error:", "error :", "error   :", "error\t:", "Error   :"])
def test_ptxas_instruction_argument_mismatch_is_extracted(marker: str) -> None:
    line = f"ptxas /tmp/tmpxft_0036d2c4_00000000-6_kernel.ptx, line 454; {marker} Arguments mismatch for instruction 'mma'"
    assert classify_compile_error_detail(line) == INSTRUCTION_ARGUMENT_MISMATCH
    assert extract_compile_error_excerpt(line) == line
    assert classify_compile_error_metadata(line + "\n" + line) == {
        "compilation_error_detail": {
            INSTRUCTION_ARGUMENT_MISMATCH: {"count": 1, "truncated": False, "errors": [line]},
        },
    }


def test_ptxas_unclassified_spaced_error_is_preserved() -> None:
    line = "ptxas kernel.ptx, line 12; error   : Unexpected assembly problem"
    assert classify_compile_error_metadata(line) == {
        "compilation_error_detail": {
            OTHER_COMPILE_ERROR: {"count": 1, "truncated": False, "errors": [line]},
        },
    }


def test_ptxas_warning_and_source_text_do_not_become_errors() -> None:
    text = "ptxas warning : warning only\n  7 | const char* s = \"error   : source literal\";"
    assert classify_compile_error_metadata(text) == {}


@pytest.mark.parametrize(
    "error_message",
    (
        "error: ‘struct DLDataType’ has no member named ‘bytes’",
        'error: class DLDataType has no member named "bytes"',
        "error: no member named 'bytes' in 'DLDataType'",
        "error: 'struct DLDataType' has no member named 'device_type'",
    ),
)
def test_tvm_ffi_dtype_member_errors_are_classified(error_message: str) -> None:
    assert (
        classify_compile_error_detail(error_message, backend="tvm_ffi")
        == TVM_FFI_API_DTYPE
    )


def test_non_tvm_backend_does_not_classify_generic_dtype_member_error() -> None:
    error_message = "error: 'struct DLDataType' has no member named 'bytes'"

    assert (
        classify_compile_error_detail(error_message, backend="cuda")
        == OTHER_COMPILE_ERROR
    )


@pytest.mark.parametrize("backend", (None, "cuda_agent", "tvm_ffi", "triton"))
@pytest.mark.parametrize(
    "error_message",
    (
        'generated.cu(33): error: identifier "xxx" is undefined',
        "generated.cu:33:25: error: use of undeclared identifier 'xxx'",
        "generated.cu:33:25: error: 'xxx' was not declared in this scope",
        "generated_binding.cpp:33:25: error: 'TensorView' has not been declared",
        "generated_binding.cpp:33:25: error: unknown type name 'TensorView'",
        "generated_binding.cpp:33:25: error: 'TensorView' does not name a type",
    ),
)
def test_undefined_identifier_errors_are_backend_independent(
    backend: str | None, error_message: str
) -> None:
    assert (
        classify_compile_error_detail(error_message, backend=backend)
        == UNDEFINED_IDENTIFIER
    )


@pytest.mark.parametrize("backend", (None, "cuda_agent", "tvm_ffi", "triton"))
@pytest.mark.parametrize(
    "error_message",
    (
        "generated_binding.cpp:1:10: fatal error: cuda_fp16.h: No such file or directory",
        "generated.cu:1:10: fatal error: 'missing_header.cuh' file not found",
        "fatal error C1083: Cannot open include file: 'cuda_fp16.h': No such file or directory",
        'catastrophic error: cannot open source file "missing_header.cuh"',
    ),
)
def test_missing_header_errors_are_backend_independent(
    backend: str | None, error_message: str
) -> None:
    assert (
        classify_compile_error_detail(error_message, backend=backend) == MISSING_HEADER
    )


def test_linker_undefined_reference_is_not_an_undefined_identifier() -> None:
    error_message = "ld: generated.o: undefined reference to `missing_launcher'"

    assert (
        classify_compile_error_detail(error_message, backend="cuda_agent")
        == OTHER_COMPILE_ERROR
    )


@pytest.mark.parametrize("backend", (None, "cuda_agent", "tvm_ffi", "triton"))
@pytest.mark.parametrize(
    "error_message",
    (
        "generated_binding.cpp:33:25: error: too few arguments to function "
        "'void check_tensor_cuda_f32_2d(Tensor, int, int)'",
        "generated.cu(33): error: too many arguments in function call",
        "generated_binding.cpp:33:25: error: no matching function for call to 'launch_kernel'",
        'generated.cu(33): error: no instance of overloaded function "launch_kernel" matches the argument list',
        "error: no matching function for call to 'launch_kernel'\n"
        "note: candidate expects 3 arguments, 2 provided",
        "error: no matching member function for call to 'check'\n"
        "note: candidate function not viable: requires 3 arguments, but 2 were provided",
        "error C2660: 'launch_kernel': function does not take 3 arguments",
    ),
)
def test_function_argument_mismatch_is_backend_independent(backend: str | None, error_message: str) -> None:
    assert classify_compile_error_detail(error_message, backend=backend) == FUNCTION_ARGUMENT_MISMATCH


@pytest.mark.parametrize("backend", (None, "cuda_agent", "tvm_ffi", "triton"))
@pytest.mark.parametrize(
    "error_message",
    (
        "generated_binding.cpp:33:25: error: invalid conversion from 'void*' to 'float*'",
        "generated_binding.cpp:33:25: error: cannot convert 'void*' to 'const float*'",
        'generated.cu(33): error: no suitable conversion function from "void *" to "float *" exists',
        'generated.cu(33): error: argument of type "void *" is incompatible with parameter of type "float *"',
        "error: cannot initialize a parameter of type 'float *' with an lvalue of type 'void *'",
        'error: a value of type "void *" cannot be used to initialize an entity of type "float *"',
    ),
)
def test_invalid_type_conversion_is_backend_independent(backend: str | None, error_message: str) -> None:
    assert classify_compile_error_detail(error_message, backend=backend) == INVALID_TYPE_CONVERSION


@pytest.mark.parametrize(
    "error_message",
    (
        "ptxas fatal : too many resources requested for launch",
        "warning: conversion from 'long' to 'int' may change value",
        "runtime error: invalid argument",
        "error: invalid conversion specifier '%q'",
        "ld: generated.o: undefined reference to `launch_kernel'",
        "note: candidate expects 3 arguments, 2 provided",
        "note: candidate function not viable: requires 3 arguments, but 2 were provided",
    ),
)
def test_ambiguous_argument_and_conversion_text_stays_other(error_message: str) -> None:
    assert classify_compile_error_detail(error_message, backend="cuda_agent") == OTHER_COMPILE_ERROR


@pytest.mark.parametrize("backend", (None, "cuda_agent", "tvm_ffi", "triton"))
@pytest.mark.parametrize(
    "error_message",
    (
        "generated_binding.cpp:33:25: error: variable or field 'output' declared void",
        "generated.cu(33): error: this declaration has no storage class or type specifier",
        "generated_binding.cpp:33:25: error: duplicate parameter name 'input'",
        "generated.cu:33:25: error: variable length array cannot have static storage duration",
        "generated.cu:33:25: error: variable length array declaration cannot have 'static' storage duration",
        "generated.cu:33:25: error: variable length array declaration not allowed at file scope",
        "generated.cu:33:25: error: variably modified 'buffer' at file scope",
        "generated.cu:33:25: error: variably modified 'buffer' must have automatic storage duration",
    ),
)
def test_invalid_declaration_is_backend_independent(backend: str | None, error_message: str) -> None:
    assert classify_compile_error_detail(error_message, backend=backend) == INVALID_DECLARATION


@pytest.mark.parametrize(
    "error_message",
    (
        "note: previous declaration is here",
        "warning: declaration of 'index' shadows a local variable",
        "ld: duplicate symbol '_launch_kernel' in generated.o and binding.o",
        "generated.cu:33:25: error: storage size of 'buffer' isn't known",
        "generated.cu:33:25: error: redefinition of 'output'",
    ),
)
def test_ambiguous_declaration_text_stays_other(error_message: str) -> None:
    assert classify_compile_error_detail(error_message, backend="cuda_agent") == OTHER_COMPILE_ERROR


@pytest.mark.parametrize("backend", (None, "cuda_agent", "tvm_ffi", "triton"))
@pytest.mark.parametrize(
    ("error_message", "expected"),
    (
        ("generated_binding.cpp:7:75: error: expected primary-expression before ')' token", SYNTAX_ERROR),
        ('generated.cu:31:9: error: incomplete type "__nv_bfloat16" is not allowed', INCOMPLETE_TYPE),
        (
            "generated_binding.cpp:9:36: error: aggregate 'incomplete_value_for_test' "
            "has incomplete type and cannot be defined",
            INCOMPLETE_TYPE,
        ),
    ),
)
def test_syntax_and_incomplete_type_errors_are_backend_independent(
    backend: str | None,
    error_message: str,
    expected: str,
) -> None:
    assert classify_compile_error_detail(error_message, backend=backend) == expected


@pytest.mark.parametrize(
    ("backend", "error_message", "expected_detail", "expected_excerpt"),
    (
        (
            "tvm_ffi",
            "ninja failed\n"
            "generated_binding.cpp:26:57: error: ‘struct DLDataType’ has no member named ‘bytes’\n"
            "   26 | output.dtype().bytes == 4",
            TVM_FFI_API_DTYPE,
            "generated_binding.cpp:26:57: error: ‘struct DLDataType’ has no member named ‘bytes’\n"
            "   26 | output.dtype().bytes == 4",
        ),
        (
            "cuda_agent",
            'generated.cu(33): error: identifier "xxx" is undefined',
            UNDEFINED_IDENTIFIER,
            'generated.cu(33): error: identifier "xxx" is undefined',
        ),
        (
            "tvm_ffi",
            "generated_binding.cpp:1:10: fatal error: cuda_fp16.h: No such file or directory",
            MISSING_HEADER,
            "generated_binding.cpp:1:10: fatal error: cuda_fp16.h: No such file or directory",
        ),
        (
            "cuda_agent",
            "generated_binding.cpp:33:25: error: no matching function for call to 'launch_kernel'",
            FUNCTION_ARGUMENT_MISMATCH,
            "generated_binding.cpp:33:25: error: no matching function for call to 'launch_kernel'",
        ),
        (
            "cuda_agent",
            "generated_binding.cpp:33:25: error: invalid conversion from 'void*' to 'float*'",
            INVALID_TYPE_CONVERSION,
            "generated_binding.cpp:33:25: error: invalid conversion from 'void*' to 'float*'",
        ),
        (
            "triton",
            "generated.cu:33:25: error: duplicate parameter name 'input'",
            INVALID_DECLARATION,
            "generated.cu:33:25: error: duplicate parameter name 'input'",
        ),
    ),
)
def test_compile_error_metadata_includes_matched_diagnostic_line(
    backend: str,
    error_message: str,
    expected_detail: str,
    expected_excerpt: str,
) -> None:
    metadata = classify_compile_error_metadata(error_message, backend=backend)
    assert metadata == {
        "compilation_error_detail": {
            expected_detail: {"count": 1, "truncated": False, "errors": [expected_excerpt]},
        },
    }


def test_compile_error_excerpt_ignores_source_and_note_lines() -> None:
    error_message = (
        "note: candidate function not viable: requires 3 arguments, but 2 were provided\n"
        "  20 | const char* text = \"error: invalid conversion from 'void*' to 'float*'\";\n"
        "generated_binding.cpp:21:7: error: invalid conversion from 'void*' to 'float*'"
    )

    assert extract_compile_error_excerpt(error_message, backend="cuda_agent") == (
        "generated_binding.cpp:21:7: error: invalid conversion from 'void*' to 'float*'"
    )


@pytest.mark.parametrize(
    ("error_message", "expected_excerpt"),
    (
        (
            'generated.cu(33): error: identifier "G" is undefined\n'
            "      __attribute__((shared)) float As[G_TILE][G_TILE_P];\n"
            "                                       ^\n\n"
            '1 error detected in the compilation of "generated.cu".',
            'generated.cu(33): error: identifier "G" is undefined\n'
            "      __attribute__((shared)) float As[G_TILE][G_TILE_P];",
        ),
        (
            "generated_binding.cpp:8:33: error: expected primary-expression before ';' token\n"
            "    8 |     int syntax_error_for_test = ;\n"
            "      |                                 ^\n"
            "ninja: build stopped: subcommand failed.",
            "generated_binding.cpp:8:33: error: expected primary-expression before ';' token\n"
            "    8 |     int syntax_error_for_test = ;",
        ),
    ),
)
def test_compile_error_excerpt_includes_source_and_omits_caret(
    error_message: str,
    expected_excerpt: str,
) -> None:
    assert extract_compile_error_excerpt(error_message, backend="cuda_agent") == expected_excerpt


@pytest.mark.parametrize("error_message", ["", "   ", "Build completed successfully", "ninja: build stopped: subcommand failed."])
def test_compile_error_detail_is_omitted_without_diagnostics(error_message: str) -> None:
    assert classify_compile_error_metadata(error_message, backend="tvm_ffi") == {}


def test_unclassified_compile_error_is_grouped_as_other() -> None:
    error_message = "generated.cu:20:3: error: expected ';' before '}' token"

    metadata = classify_compile_error_metadata(error_message, backend="cuda_agent")
    assert metadata == {
        "compilation_error_detail": {
            OTHER_COMPILE_ERROR: {"count": 1, "truncated": False, "errors": [error_message]},
        },
    }


def test_compile_error_metadata_groups_multiple_unique_diagnostics_by_type() -> None:
    syntax_error = "generated_binding.cpp:7:75: error: expected primary-expression before ')' token"
    syntax_excerpt = f"{syntax_error}\n    7 | int value = ;"
    incomplete_type = 'generated.cu:31:9: error: incomplete type "__nv_bfloat16" is not allowed'
    incomplete_excerpt = f'{incomplete_type}\n          __nv_bfloat16 value;'
    error_message = "\n".join(
        (
            syntax_excerpt,
            incomplete_excerpt,
            syntax_excerpt,
        )
    )

    metadata = classify_compile_error_metadata(error_message, backend="tvm_ffi")
    assert metadata == {
        "compilation_error_detail": {
            SYNTAX_ERROR: {"count": 1, "truncated": False, "errors": [syntax_excerpt]},
            INCOMPLETE_TYPE: {"count": 1, "truncated": False, "errors": [incomplete_excerpt]},
        }
    }


@pytest.mark.parametrize("count", [8, 9, 25])
def test_compile_error_detail_limits_each_category_and_counts_all(count: int) -> None:
    undefined = [f'generated.cu:{i}: error: identifier "value_{i}" is undefined' for i in range(count)]
    incomplete = [f'generated.cu:{i + 100}: error: incomplete type "Type_{i}" is not allowed' for i in range(count)]
    metadata = classify_compile_error_metadata("\n".join(undefined + incomplete + undefined))
    assert metadata == {
        "compilation_error_detail": {
            UNDEFINED_IDENTIFIER: {"count": count, "truncated": count > 8, "errors": undefined[:8]},
            INCOMPLETE_TYPE: {"count": count, "truncated": count > 8, "errors": incomplete[:8]},
        },
    }


@pytest.mark.parametrize(
    ("error_message", "expected"),
    (
        (
            'generated.cu(33): error: identifier "xxx" is undefined',
            UNDEFINED_IDENTIFIER,
        ),
        (
            "generated_binding.cpp:1:10: fatal error: cuda_fp16.h: No such file or directory",
            MISSING_HEADER,
        ),
    ),
)
def test_failure_detail_exposes_generic_compile_error_category(
    error_message: str, expected: str
) -> None:
    assert (
        classify_failure_detail(error_message, compiled=False, backend="cuda_agent")
        == expected
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
