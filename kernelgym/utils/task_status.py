"""Helpers for deriving public task status from result payloads."""

from __future__ import annotations

from typing import Any, Mapping

from kernelgym.common import ErrorCode, TaskStatus


def _value(value: Any) -> str:
    if isinstance(value, (ErrorCode, TaskStatus)):
        return value.value
    return str(value or "")


def is_timeout_error_code(error_code: Any) -> bool:
    return _value(error_code).upper() == ErrorCode.TIMEOUT_ERROR.value


def task_status_from_result_payload(payload: Mapping[str, Any]) -> TaskStatus:
    raw_status = _value(payload.get("status")).lower()
    if raw_status == TaskStatus.TIMEOUT.value:
        return TaskStatus.TIMEOUT
    if is_timeout_error_code(payload.get("error_code")):
        return TaskStatus.TIMEOUT
    if raw_status == TaskStatus.FAILED.value:
        return TaskStatus.FAILED
    return TaskStatus.COMPLETED
