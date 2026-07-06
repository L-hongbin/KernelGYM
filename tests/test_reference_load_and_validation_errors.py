import pytest

from kernelgym.server.api.server import _sanitize_validation_errors
from kernelgym.toolkit.kernelbench.loading import OriginalModelLoadError, load_original_model_and_inputs


def test_original_model_import_failure_preserves_root_cause() -> None:
    source = """
import definitely_missing_reference_dep

class Model:
    pass

def get_init_inputs():
    return []

def get_inputs():
    return []
"""

    with pytest.raises(OriginalModelLoadError) as exc_info:
        load_original_model_and_inputs(source, {})

    assert "definitely_missing_reference_dep" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert exc_info.value.__cause__.__class__.__name__ == "ModuleNotFoundError"


def test_validation_error_sanitizer_omits_large_raw_input() -> None:
    raw_output = "x" * 100001
    errors = [
        {
            "type": "value_error",
            "loc": ("body", "kernel_code"),
            "msg": "Value error, Code length exceeds limit: length=100001, limit=100000",
            "input": raw_output,
            "ctx": {"error": ValueError("Code length exceeds limit: length=100001, limit=100000")},
        }
    ]

    sanitized = _sanitize_validation_errors(errors)

    assert sanitized == [
        {
            "type": "value_error",
            "loc": ("body", "kernel_code"),
            "msg": "Value error, Code length exceeds limit: length=100001, limit=100000",
            "ctx": {"error": "Code length exceeds limit: length=100001, limit=100000"},
            "input_type": "str",
            "input_length": 100001,
            "max_length": 100000,
        }
    ]
    assert raw_output not in repr(sanitized)
