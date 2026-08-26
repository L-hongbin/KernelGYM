"""Isolated NVIDIA Compute Sanitizer trials for candidate CUDA kernels."""

from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Sequence

SUPPORTED_TOOLS = ("memcheck", "racecheck", "synccheck", "initcheck")
FULL_SANITIZER_TOOLS = ("memcheck", "synccheck", "racecheck", "initcheck")
SANITIZER_MODE_FULL = "full"
SUPPORTED_SANITIZER_EXECUTION_MODES = (*FULL_SANITIZER_TOOLS, SANITIZER_MODE_FULL)
SANITIZER_ERROR_EXIT_CODE = 86

_ARTIFACT_KEYS = {
    "compiled",
    "device",
    "entry_point",
    "backend",
    "build_dir",
    "work_dir",
    "so_path",
    "module_name",
    "code",
    "persistent_work_dir",
    "profiling_hints",
}
_PREFIX_RE = re.compile(r"^=+\s?(.*)$")
_SOURCE_RE = re.compile(r"(?P<file>[^\s()]+\.(?:cu|cuh|cpp|cc|cxx)):(?P<line>\d+)")
_THREAD_BLOCK_RE = re.compile(r"by thread \((?P<thread>[^)]+)\) in block \((?P<block>[^)]+)\)", re.IGNORECASE)
_RACE_THREAD_RE = re.compile(r"(?:Read|Write) Thread \((?P<thread>[^)]+)\)", re.IGNORECASE)
_BLOCK_RE = re.compile(r"in block \((?P<block>[^)]+)\)", re.IGNORECASE)
_ADDRESS_RE = re.compile(r"\b(?:Access at|Address(?: at)?)\s+(?P<address>0x[0-9a-f]+)\b", re.IGNORECASE)
_HEX_ADDRESS_RE = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
MAX_PARSED_ISSUE_OCCURRENCES = 5000
MAX_RETURNED_UNIQUE_ISSUES = 4
REPRESENTATIVE_OCCURRENCES_PER_ISSUE = 2
_KERNEL_AT_RE = re.compile(
    r"\bat\s+(?P<kernel>[A-Za-z_$][\w$]*(?:\([^)]*\))?)(?:\+0x[0-9a-f]+)?\s+in\s+",
    re.IGNORECASE,
)
_DEVICE_FRAME_KERNEL_RE = re.compile(
    r"Device Frame:\s*(?P<kernel>[A-Za-z_$][\w$]*(?:\([^)]*\))?)(?:\+0x[0-9a-f]+)?\s+in\s+",
    re.IGNORECASE,
)
_ACCESS_RE = re.compile(
    r"(?P<kind>Invalid|Uninitialized)\s+(?P<space>__\w+__)(?:\s+memory)?\s+"
    r"(?P<access>read|write)\s+of\s+size\s+(?P<size>\d+)\s+bytes?",
    re.IGNORECASE,
)
_ERROR_SUMMARY_RE = re.compile(r"ERROR SUMMARY:\s*(\d+)\s+errors?", re.IGNORECASE)
_RACE_SUMMARY_RE = re.compile(r"RACECHECK SUMMARY:\s*(\d+)\s+hazards?", re.IGNORECASE)
_DIAGNOSTIC_STARTS = (
    "Error: Potential ",
    "Invalid ",
    "Uninitialized ",
    "Barrier error detected",
    "Barrier error: ",
    "Race reported",
    "Hazard at",
    "Program hit cudaError",
    "Leaked ",
)
_TARGET_APPLICATION_ERROR_RE = re.compile(r"Target application (?:returned an error|terminated)", re.IGNORECASE)


def _has_explicit_zero_issue_summary(output: str) -> bool:
    """Return true only when Compute Sanitizer explicitly reports zero issues."""

    summary_counts = [int(match.group(1)) for match in _ERROR_SUMMARY_RE.finditer(output)]
    summary_counts.extend(int(match.group(1)) for match in _RACE_SUMMARY_RE.finditer(output))
    return bool(summary_counts) and all(count == 0 for count in summary_counts)


def classify_compute_sanitizer_error(
    runtime_error: Exception | str,
) -> Optional[str]:
    """Return the most relevant tool only when the runtime error is specific."""

    lowered = str(runtime_error).lower()
    if any(token in lowered for token in ("uninitialized", "uninitialised")):
        return "initcheck"
    elif re.search(r"\b(?:data\s+race|race(?:check)?|hazard)\b", lowered):
        return "racecheck"
    elif any(
        token in lowered
        for token in (
            "barrier",
            "syncwarp",
            "syncthreads",
            "synchronization",
            "synchronisation",
        )
    ):
        return "synccheck"
    elif any(
        token in lowered
        for token in (
            "illegal memory access",
            "illegal address",
            "misaligned address",
            "out of bounds",
            "out-of-bounds",
            "device-side assert",
            "device side assert",
            "cudaerrorillegaladdress",
            "cudaerrormisalignedaddress",
            "invalid __global__",
            "invalid __local__",
            "invalid __shared__",
        )
    ):
        return "memcheck"
    return None


def normalize_compute_sanitizer_execution_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_SANITIZER_EXECUTION_MODES:
        raise ValueError(
            f"Unsupported Compute Sanitizer execution mode: {mode!r}; "
            f"expected one of {SUPPORTED_SANITIZER_EXECUTION_MODES}"
        )
    return normalized


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _artifact_payload(artifact: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(artifact, dict):
        return None
    return _json_safe({key: value for key, value in artifact.items() if key in _ARTIFACT_KEYS})


def _clean_output_lines(output: str) -> list[str]:
    cleaned: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        prefix_match = _PREFIX_RE.match(line)
        if prefix_match:
            line = prefix_match.group(1).strip()
        if line:
            cleaned.append(line)
    return cleaned


def _coordinates(value: str) -> Dict[str, Optional[int]]:
    parts = [part.strip() for part in value.split(",")]
    parsed: list[Optional[int]] = []
    for part in parts[:3]:
        try:
            parsed.append(int(part))
        except ValueError:
            parsed.append(None)
    parsed.extend([None] * (3 - len(parsed)))
    return dict(zip(("x", "y", "z"), parsed))


def _hazard_type(tool: str, message: str) -> str:
    lowered = message.lower()
    access = _ACCESS_RE.search(message)
    if access:
        return "_".join(
            (
                access.group("kind").lower(),
                access.group("space").strip("_").lower(),
                access.group("access").lower(),
            )
        )
    if "race" in lowered or "hazard" in lowered:
        return "shared_memory_race"
    if "barrier" in lowered or "sync" in lowered:
        return "synchronization_error"
    if "cudaerror" in lowered:
        return "cuda_api_error"
    if "leak" in lowered:
        return "device_memory_leak"
    if tool == "racecheck":
        return "shared_memory_race"
    if tool == "synccheck":
        return "synchronization_error"
    return f"{tool}_error"


def _parse_issue(tool: str, group: list[str]) -> Dict[str, Any]:
    message = group[0]
    text = "\n".join(group)
    source_match = _SOURCE_RE.search(text)
    location: Optional[Dict[str, Any]] = None
    kernel: Optional[str] = None
    if source_match:
        location = {
            "file": source_match.group("file"),
            "line": int(source_match.group("line")),
        }
        trailing_lines = text[source_match.end() :].splitlines()
        remainder = trailing_lines[0].lstrip(": ") if trailing_lines else ""
        if remainder:
            kernel = remainder
    kernel_match = _KERNEL_AT_RE.search(text)
    if kernel_match:
        kernel = kernel_match.group("kernel")
    device_frame_match = _DEVICE_FRAME_KERNEL_RE.search(text)
    if device_frame_match:
        kernel = device_frame_match.group("kernel")

    thread_block = _THREAD_BLOCK_RE.search(text)
    race_thread = _RACE_THREAD_RE.search(text)
    block_match = _BLOCK_RE.search(text)
    thread_value = (
        thread_block.group("thread") if thread_block else (race_thread.group("thread") if race_thread else "")
    )
    block_value = thread_block.group("block") if thread_block else (block_match.group("block") if block_match else "")
    access = _ACCESS_RE.search(message)
    address_match = _ADDRESS_RE.search(text)
    return {
        "tool": tool,
        "hazard_type": _hazard_type(tool, message),
        "message": message,
        "kernel": kernel,
        "source": location,
        "thread": _coordinates(thread_value) if thread_value else None,
        "block": _coordinates(block_value) if block_value else None,
        "address": address_match.group("address").lower() if address_match else None,
        "access_type": access.group("access").lower() if access else None,
        "memory_space": access.group("space").strip("_").lower() if access else None,
        "access_size_bytes": int(access.group("size")) if access else None,
    }


def _range_bounds(values: Sequence[int]) -> Optional[list[int]]:
    sorted_values = sorted(set(values))
    if not sorted_values:
        return None
    return [sorted_values[0], sorted_values[-1]]


def _coordinate_ranges(occurrences: Sequence[Dict[str, Any]], field: str) -> Optional[Dict[str, Any]]:
    coordinates = [item.get(field) for item in occurrences if item.get(field)]
    if not coordinates:
        return None
    result: Dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        values = [item.get(axis) for item in coordinates if item.get(axis) is not None]
        ranges = _range_bounds(values)
        if ranges is not None:
            result[axis] = ranges
    return result or None


def _address_ranges(occurrences: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    values = sorted({int(item["address"], 16) for item in occurrences if isinstance(item.get("address"), str)})
    if not values:
        return None
    return {"ranges": [hex(values[0]), hex(values[-1])]}


def _issue_signature(issue: Dict[str, Any]) -> tuple[Any, ...]:
    source = issue.get("source") if isinstance(issue.get("source"), dict) else {}
    message = _HEX_ADDRESS_RE.sub("<address>", str(issue.get("message") or ""))
    return (
        issue.get("tool"),
        issue.get("hazard_type"),
        message,
        issue.get("kernel"),
        source.get("file"),
        source.get("line"),
        issue.get("access_type"),
        issue.get("memory_space"),
        issue.get("access_size_bytes"),
    )


def _freeze_issue_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_issue_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_issue_value(item) for item in value)
    return value


def _cross_kernel_issue_signature(issue: Dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (key, _freeze_issue_value(value))
        for key, value in sorted(issue.items())
        if key not in {"kernel", "source", "raw_excerpt", "kernel_info"}
    )


def _kernel_info(issue: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"name": issue.get("kernel")}
    source = issue.get("source") if isinstance(issue.get("source"), dict) else {}
    file_name = source.get("file")
    line = source.get("line")
    if file_name is not None and line is not None:
        info["source"] = f"file {file_name} line {line}"
    elif file_name is not None:
        info["source"] = f"file {file_name}"
    elif line is not None:
        info["source"] = f"line {line}"
    return info


def _aggregate_equivalent_kernel_locations(
    issues: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    buckets: Dict[tuple[Any, ...], list[Dict[str, Any]]] = {}
    for issue in issues:
        buckets.setdefault(_cross_kernel_issue_signature(issue), []).append(issue)

    merged: list[Dict[str, Any]] = []
    for bucket in buckets.values():
        issue = dict(bucket[0])
        issue.pop("kernel", None)
        issue.pop("source", None)
        issue["kernel_info"] = [_kernel_info(item) for item in bucket]
        issue["occurrence_count"] = sum(int(item.get("occurrence_count") or 0) for item in bucket)
        if len(bucket) > 1:
            issue.pop("raw_excerpt", None)
        merged.append(issue)
    return merged


def aggregate_compute_sanitizer_issues(
    occurrences: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Group repeated reports while preserving their affected execution range."""

    buckets: Dict[tuple[Any, ...], list[Dict[str, Any]]] = {}
    for occurrence in occurrences:
        buckets.setdefault(_issue_signature(occurrence), []).append(occurrence)

    aggregated: list[Dict[str, Any]] = []
    for bucket in buckets.values():
        first = bucket[0]
        issue = {key: value for key, value in first.items() if key not in {"tool", "thread", "block", "address"}}
        issue["occurrence_count"] = len(bucket)
        issue["threads"] = _coordinate_ranges(bucket, "thread")
        issue["blocks"] = _coordinate_ranges(bucket, "block")
        issue["addresses"] = _address_ranges(bucket)
        issue["representative_occurrences"] = [
            {key: occurrence.get(key) for key in ("thread", "block", "address") if occurrence.get(key) is not None}
            for occurrence in bucket[:REPRESENTATIVE_OCCURRENCES_PER_ISSUE]
        ]
        aggregated.append(issue)
    return _aggregate_equivalent_kernel_locations(aggregated)


def parse_compute_sanitizer_output(
    output: str,
    *,
    tool: str,
    max_issues: int = 20,
) -> Dict[str, Any]:
    """Parse occurrences, aggregate duplicate diagnostics, then bound groups."""

    lines = _clean_output_lines(output)
    groups: list[list[str]] = []
    observed_issue_count = 0
    current: list[str] = []
    for line in lines:
        is_start = line.startswith(_DIAGNOSTIC_STARTS)
        if line.startswith("Hazard at") and current and current[0].startswith("Race reported"):
            current.append(line)
            continue
        if is_start:
            if current:
                observed_issue_count += 1
                if len(groups) < MAX_PARSED_ISSUE_OCCURRENCES:
                    groups.append(current)
            current = [line]
        elif current:
            if _ERROR_SUMMARY_RE.search(line) or _RACE_SUMMARY_RE.search(line):
                observed_issue_count += 1
                if len(groups) < MAX_PARSED_ISSUE_OCCURRENCES:
                    groups.append(current)
                current = []
            else:
                current.append(line)
    if current:
        observed_issue_count += 1
        if len(groups) < MAX_PARSED_ISSUE_OCCURRENCES:
            groups.append(current)

    issue_group_limit = min(MAX_RETURNED_UNIQUE_ISSUES, max(1, int(max_issues)))
    raw_excerpt_signatures: set[tuple[Any, ...]] = set()
    occurrences = []
    for group in groups:
        occurrence = _parse_issue(tool, group)
        signature = _issue_signature(occurrence)
        if signature not in raw_excerpt_signatures and len(raw_excerpt_signatures) < issue_group_limit:
            occurrence["raw_excerpt"] = "\n".join(group)[:4000]
            raw_excerpt_signatures.add(signature)
        occurrences.append(occurrence)
    aggregated = aggregate_compute_sanitizer_issues(occurrences)
    returned_issues = aggregated[:issue_group_limit]
    summary_counts = [int(match.group(1)) for match in _ERROR_SUMMARY_RE.finditer(output)]
    summary_counts.extend(int(match.group(1)) for match in _RACE_SUMMARY_RE.finditer(output))
    summary_count = max(summary_counts, default=0)
    detected_issue_count = max(summary_count, observed_issue_count)
    occurrences_truncated = detected_issue_count > len(occurrences)
    issue_groups_truncated = len(aggregated) > len(returned_issues)
    return {
        "summary_error_count": summary_count,
        "observed_issue_count": observed_issue_count,
        "parsed_issue_count": len(occurrences),
        "detected_issue_count": detected_issue_count,
        "unique_issue_count": len(aggregated),
        "returned_issue_count": len(returned_issues),
        "occurrences_truncated": occurrences_truncated,
        "issue_groups_truncated": issue_groups_truncated,
        "issues_truncated": occurrences_truncated or issue_groups_truncated,
        "aggregation_complete": not occurrences_truncated,
        "issues": returned_issues,
    }


def build_compute_sanitizer_command(
    *,
    sanitizer_path: str,
    tool: str,
    payload_path: Path,
    kernel_names: Sequence[str],
    max_kernels: int,
) -> list[str]:
    if tool not in SUPPORTED_TOOLS:
        raise ValueError(f"Unsupported Compute Sanitizer tool: {tool}")
    command = [
        sanitizer_path,
        "--tool",
        tool,
        "--target-processes",
        "all",
        "--error-exitcode",
        str(SANITIZER_ERROR_EXIT_CODE),
        "--check-exit-code",
        "yes",
        "--destroy-on-device-error",
        "kernel",
        "--demangle",
        "simple",
        "--show-backtrace",
        "device",
        "--print-limit",
        str(MAX_PARSED_ISSUE_OCCURRENCES),
        "--launch-count",
        str(max(1, max_kernels)),
    ]
    if tool == "memcheck":
        command.extend(["--report-api-errors", "explicit"])
    elif tool == "racecheck":
        command.extend(["--racecheck-detect-level", "warn", "--racecheck-report", "all"])
    for name in kernel_names:
        clean_name = str(name).strip()
        if clean_name:
            command.extend(["--kernel-name", f"kns={clean_name}"])
    command.extend(
        [
            sys.executable,
            "-m",
            "kernelgym.toolkit.kernelbench.compute_sanitizer_runner",
            str(payload_path),
        ]
    )
    return command


def _resolve_tool_path(configured_path: str) -> Optional[str]:
    candidate = Path(configured_path)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(configured_path)


def _device_index(device: Any) -> int:
    if isinstance(device, int):
        return device
    match = re.search(r"(?::|^)(\d+)$", str(device))
    return int(match.group(1)) if match else 0


def _sanitizer_environment(device: Any) -> Dict[str, str]:
    env = dict(os.environ)
    index = _device_index(device)
    visible = [item.strip() for item in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    env["CUDA_VISIBLE_DEVICES"] = (
        visible[index] if visible and index < len(visible) else (visible[0] if visible else str(index))
    )
    return env


@lru_cache(maxsize=8)
def _tool_version(sanitizer_path: str) -> str:
    try:
        completed = subprocess.run(
            [sanitizer_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "unknown"
    lines = [line.strip() for line in (completed.stdout + completed.stderr).splitlines() if line.strip()]
    return lines[-1] if lines else "unknown"


def _run_sanitizer_command(
    command: Sequence[str],
    *,
    env: Dict[str, str],
    timeout_s: int,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=max(1, timeout_s))
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd=list(command),
            timeout=max(1, timeout_s),
            output=stdout,
            stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def skipped_compute_sanitizer_result(reason: str) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "passed": None,
        "measurement_complete": False,
        "reason": reason,
        "requested_checks": [],
        "detected_issue_count": 0,
        "check_results": [],
    }


def run_compute_sanitizer(
    *,
    original_model_src: str,
    custom_model_src: str,
    artifact: Optional[Dict[str, Any]],
    backend: str,
    entry_point: str,
    device: Any,
    kernel_names: Sequence[str],
    sanitizer_path: str,
    timeout_s: int,
    max_kernels: int,
    max_issues: int,
    mode: str,
    primary_tool: Optional[str] = None,
    input_seed: Optional[int] = None,
    input_perturbation: Optional[str] = None,
    model_seed: int = 42,
    generate_inputs_on_gpu: bool = True,
) -> Dict[str, Any]:
    """Run one sanitizer check or the full suite in an isolated process."""

    started = perf_counter()
    resolved_path = _resolve_tool_path(sanitizer_path)
    execution_mode = normalize_compute_sanitizer_execution_mode(mode)
    requested_tools = list(FULL_SANITIZER_TOOLS) if execution_mode == SANITIZER_MODE_FULL else [execution_mode]
    if primary_tool is not None and primary_tool not in requested_tools:
        raise ValueError(
            f"Primary Compute Sanitizer tool {primary_tool!r} was not requested; "
            f"requested tools are {requested_tools}"
        )
    result: Dict[str, Any] = {
        "status": "unavailable" if resolved_path is None else "starting",
        "passed": None,
        "measurement_complete": False,
        "tool_path": resolved_path,
        "tool_version": (_tool_version(resolved_path) if resolved_path else "unavailable"),
        "requested_checks": requested_tools,
        "kernel_filter": [str(name) for name in kernel_names],
        "detected_issue_count": 0,
        "check_results": [],
        "mode": execution_mode,
        "run_all_checks": execution_mode == SANITIZER_MODE_FULL,
        "replayed_input_seed": input_seed,
        "replayed_input_perturbation": input_perturbation,
    }
    if resolved_path is None:
        result["error"] = f"Compute Sanitizer executable not found: {sanitizer_path}"
        result["wall_time_s"] = perf_counter() - started
        return result

    temp_root = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    try:
        with tempfile.TemporaryDirectory(prefix="kernelgym_sanitizer_", dir=temp_root) as temp_dir:
            payload_path = Path(temp_dir) / "payload.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "reference_code": original_model_src,
                        "kernel_code": custom_model_src,
                        "artifact": _artifact_payload(artifact),
                        "backend": backend,
                        "entry_point": entry_point,
                        "device": "cuda:0",
                        "input_seed": input_seed,
                        "input_perturbation": input_perturbation,
                        "model_seed": int(model_seed),
                        "generate_inputs_on_gpu": bool(generate_inputs_on_gpu),
                    }
                ),
                encoding="utf-8",
            )
            for tool in requested_tools:
                tool_started = perf_counter()
                # Candidate-kernel filtering excludes PyTorch RNG kernels. For
                # initcheck, generate inputs on CPU and copy them to the device so
                # their initialization is visible without instrumenting framework
                # kernels that are outside the candidate filter.
                tool_generates_inputs_on_gpu = bool(generate_inputs_on_gpu) and tool != "initcheck"
                sanitizer_env = _sanitizer_environment(device)
                sanitizer_env["KERNELGYM_COMPUTE_SANITIZER_TOOL"] = tool
                command = build_compute_sanitizer_command(
                    sanitizer_path=resolved_path,
                    tool=tool,
                    payload_path=payload_path,
                    kernel_names=kernel_names,
                    max_kernels=max_kernels,
                )
                try:
                    completed = _run_sanitizer_command(
                        command,
                        env=sanitizer_env,
                        timeout_s=timeout_s,
                    )
                    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
                    parsed = parse_compute_sanitizer_output(output, tool=tool, max_issues=max_issues)
                    target_application_failed = bool(_TARGET_APPLICATION_ERROR_RE.search(output))
                    replayed_failure_without_issues = target_application_failed and _has_explicit_zero_issue_summary(
                        output
                    )
                    if parsed["detected_issue_count"] > 0 or completed.returncode == SANITIZER_ERROR_EXIT_CODE:
                        status = "issues_found"
                    elif completed.returncode == 0 or replayed_failure_without_issues:
                        status = "clean"
                    else:
                        status = "error"
                    check_result = {
                        "check": tool,
                        "status": status,
                        "passed": status == "clean",
                        "process_completed": True,
                        "target_application_failed": target_application_failed,
                        "sanitizer_issue_found": parsed["detected_issue_count"] > 0,
                        "input_generation": ("gpu" if tool_generates_inputs_on_gpu else "cpu_then_h2d"),
                        "input_values_exactly_replayed": not (bool(generate_inputs_on_gpu) and tool == "initcheck"),
                        "return_code": completed.returncode,
                        **parsed,
                        "raw_output_tail": output[-12000:],
                        "wall_time_s": perf_counter() - tool_started,
                    }
                    if status == "error":
                        check_result["error"] = output[-4000:] or f"runner exited with code {completed.returncode}"
                except subprocess.TimeoutExpired as exc:
                    check_result = {
                        "check": tool,
                        "status": "timeout",
                        "passed": None,
                        "process_completed": False,
                        "target_application_failed": None,
                        "sanitizer_issue_found": False,
                        "input_generation": ("gpu" if tool_generates_inputs_on_gpu else "cpu_then_h2d"),
                        "input_values_exactly_replayed": not (bool(generate_inputs_on_gpu) and tool == "initcheck"),
                        "return_code": None,
                        "summary_error_count": 0,
                        "parsed_issue_count": 0,
                        "detected_issue_count": 0,
                        "issues_truncated": False,
                        "issues": [],
                        "error": f"Compute Sanitizer {tool} timed out after {exc.timeout}s",
                        "wall_time_s": perf_counter() - tool_started,
                    }
                result["check_results"].append(check_result)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_s"] = perf_counter() - started
        return result

    statuses = [item["status"] for item in result["check_results"]]
    result["executed_checks"] = [item["check"] for item in result["check_results"]]
    issue_count_by_check = {
        item["check"]: int(item.get("detected_issue_count", 0)) for item in result["check_results"]
    }
    selected_primary_check = primary_tool
    if selected_primary_check is None:
        selected_primary_check = next(
            (tool for tool in requested_tools if issue_count_by_check.get(tool, 0) > 0),
            requested_tools[0] if len(requested_tools) == 1 else None,
        )
    primary_detected_issue_count = (
        issue_count_by_check.get(selected_primary_check, 0) if selected_primary_check is not None else 0
    )
    result["primary_check"] = selected_primary_check
    result["primary_detected_issue_count"] = primary_detected_issue_count
    result["detected_issue_count"] = primary_detected_issue_count
    result["issue_count_by_check"] = issue_count_by_check
    result["issues_truncated"] = any(bool(item.get("issues_truncated")) for item in result["check_results"])
    result["measurement_complete"] = (
        bool(statuses)
        and all(status in {"clean", "issues_found"} for status in statuses)
        and len(statuses) == len(requested_tools)
    )
    if "issues_found" in statuses:
        result["status"] = "issues_found"
        result["passed"] = False
    elif statuses and all(status == "clean" for status in statuses):
        result["status"] = "clean"
        result["passed"] = True
    elif "clean" in statuses:
        result["status"] = "partial"
    else:
        result["status"] = "error"
    result["wall_time_s"] = perf_counter() - started
    return result
