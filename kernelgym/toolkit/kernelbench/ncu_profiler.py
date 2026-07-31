"""Nsight Compute collection for already-verified KernelBench candidates."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

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


def select_kernel_names(metadata: Dict[str, Any], max_kernels: int) -> List[str]:
    """Return stable, bounded custom-kernel names for NCU filtering."""

    candidates: List[Any] = []
    candidates.extend(metadata.get("custom_kernel_in_profiling") or [])
    candidates.extend(metadata.get("custom_kernel_names") or [])
    compile_artifact = metadata.get("compile_artifact")
    if isinstance(compile_artifact, dict):
        profiling_hints = compile_artifact.get("profiling_hints")
        if isinstance(profiling_hints, dict):
            candidates.extend(profiling_hints.get("custom_kernel_names") or [])
    candidates.extend(metadata.get("triton_profiler_matches") or [])

    result: List[str] = []
    seen = set()
    for candidate in candidates:
        name = str(candidate).strip()
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            result.append(name[:512])
        if len(result) >= max(1, max_kernels):
            break
    return result


def build_ncu_command(
    *,
    ncu_path: str,
    report_base: Path,
    payload_path: Path,
    metrics: Sequence[str],
    kernel_names: Sequence[str],
    max_kernels: int,
) -> List[str]:
    command = [
        ncu_path,
        "--target-processes",
        "all",
        "--profile-from-start",
        "no",
        "--force-overwrite",
        "--export",
        str(report_base),
        "--launch-count",
        str(max(1, max_kernels)),
    ]
    clean_metrics = [metric.strip() for metric in metrics if metric and metric.strip()]
    if clean_metrics:
        command.extend(["--metrics", ",".join(clean_metrics)])
    if kernel_names:
        expression = "|".join(re.escape(name) for name in kernel_names)
        command.extend(["--kernel-name-base", "demangled", "--kernel-name", f"regex:{expression}"])
    command.extend([sys.executable, "-m", "kernelgym.toolkit.kernelbench.ncu_runner", str(payload_path)])
    return command


def _parse_metric_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return None
    normalized = value.replace(",", "")
    try:
        parsed = float(normalized)
    except ValueError:
        return value
    if parsed.is_integer() and not any(char in normalized.lower() for char in (".", "e")):
        return int(parsed)
    return parsed


def _new_kernel_result(item: Dict[str, str], kernel_name: str) -> Dict[str, Any]:
    return {
        "id": item.get("ID", "").strip(),
        "kernel_name": kernel_name,
        "device": item.get("Device", "").strip(),
        "context": item.get("Context", "").strip(),
        "stream": item.get("Stream", "").strip(),
        "block_size": item.get("Block Size", "").strip(),
        "grid_size": item.get("Grid Size", "").strip(),
        "metrics": {},
    }


def parse_ncu_csv(
    text: str,
    max_kernels: int = 8,
    metric_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Parse NCU raw CSV in long-form or one-row-per-kernel wide-form."""

    rows = list(csv.reader(io.StringIO(text)))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if "Kernel Name" in [column.strip().lstrip("\ufeff") for column in row]
        ),
        None,
    )
    if header_index is None:
        return []

    header = [column.strip().lstrip("\ufeff") for column in rows[header_index]]
    long_form = "Metric Name" in header and "Metric Value" in header
    grouped: Dict[tuple[str, ...], Dict[str, Any]] = {}
    order: List[tuple[str, ...]] = []

    if long_form:
        for row in rows[header_index + 1 :]:
            if len(row) != len(header):
                continue
            item = dict(zip(header, row))
            metric_name = item.get("Metric Name", "").strip()
            kernel_name = item.get("Kernel Name", "").strip()
            if not metric_name or not kernel_name:
                continue
            key = (
                item.get("ID", "").strip(),
                kernel_name,
                item.get("Context", "").strip(),
                item.get("Stream", "").strip(),
            )
            if key not in grouped:
                if len(order) >= max(1, max_kernels):
                    continue
                grouped[key] = _new_kernel_result(item, kernel_name)
                order.append(key)
            grouped[key]["metrics"][metric_name] = {
                "value": _parse_metric_value(item.get("Metric Value", "")),
                "unit": item.get("Metric Unit", "").strip(),
            }
        return [grouped[key] for key in order]

    requested_metrics = {
        str(metric).strip() for metric in (metric_names or []) if str(metric).strip()
    }
    if requested_metrics:
        wide_metric_columns = [
            column for column in header if column in requested_metrics
        ]
    else:
        launch_columns = {
            "ID",
            "Process ID",
            "Process Name",
            "Host Name",
            "Kernel Name",
            "Context",
            "Stream",
            "Block Size",
            "Grid Size",
            "Device",
            "CC",
        }
        wide_metric_columns = [
            column
            for column in header
            if column not in launch_columns
            and not column.startswith(("c2clink__", "device__attribute_"))
        ]
    if not wide_metric_columns:
        return []

    metric_units: Dict[str, str] = {}
    for row in rows[header_index + 1 :]:
        if len(row) != len(header):
            continue
        item = dict(zip(header, row))
        kernel_name = item.get("Kernel Name", "").strip()
        if not kernel_name:
            for metric_name in wide_metric_columns:
                unit = item.get(metric_name, "").strip()
                if unit and isinstance(_parse_metric_value(unit), str):
                    metric_units[metric_name] = unit
            continue

        if len(order) >= max(1, max_kernels):
            break
        metrics = {
            metric_name: {
                "value": _parse_metric_value(item.get(metric_name, "")),
                "unit": metric_units.get(metric_name, ""),
            }
            for metric_name in wide_metric_columns
            if item.get(metric_name, "").strip()
        }
        if not metrics:
            continue
        key = (
            item.get("ID", "").strip(),
            kernel_name,
            item.get("Context", "").strip(),
            item.get("Stream", "").strip(),
        )
        grouped[key] = _new_kernel_result(item, kernel_name)
        grouped[key]["metrics"] = metrics
        order.append(key)

    return [grouped[key] for key in order]


def _resolve_ncu_path(configured_path: str) -> Optional[str]:
    candidate = Path(configured_path)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    located = shutil.which(configured_path)
    return located


def _device_index(device: Any) -> int:
    if isinstance(device, int):
        return device
    match = re.search(r"(?::|^)(\d+)$", str(device))
    return int(match.group(1)) if match else 0


def _ncu_environment(device: Any) -> Dict[str, str]:
    env = dict(os.environ)
    index = _device_index(device)
    visible = [item.strip() for item in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible:
        selected = visible[index] if index < len(visible) else visible[0]
    else:
        selected = str(index)
    env["CUDA_VISIBLE_DEVICES"] = selected
    return env


@lru_cache(maxsize=8)
def _ncu_version(ncu_path: str) -> str:
    try:
        completed = subprocess.run(
            [ncu_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "unknown"
    lines = [line.strip() for line in (completed.stdout + completed.stderr).splitlines() if line.strip()]
    return lines[-1] if lines else "unknown"


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if text else ""


def _head(text: str, limit: int = 2000) -> str:
    return text[:limit] if text else ""


def _failure_status(output: str) -> str:
    lowered = output.lower()
    if "err_nvgpuctrperm" in lowered or "permission to access nvidia gpu performance counters" in lowered:
        return "permission_denied"
    if "unknown metric" in lowered or "failed to find metric" in lowered or "not supported on this device" in lowered:
        return "unsupported_metrics"
    if "no kernels were profiled" in lowered or "no kernels to profile" in lowered:
        return "no_matching_kernel"
    return "error"


def run_ncu_profile(
    *,
    original_model_src: str,
    custom_model_src: str,
    artifact: Optional[Dict[str, Any]],
    backend: str,
    entry_point: str,
    device: Any,
    kernel_names: Sequence[str],
    ncu_path: str,
    metrics: Sequence[str],
    timeout_s: int,
    max_kernels: int,
    warmup: int,
    profile_version: str,
) -> Dict[str, Any]:
    """Launch NCU around a dedicated runner. Failures are returned as metadata."""

    started = perf_counter()
    resolved_ncu = _resolve_ncu_path(ncu_path)
    base_result: Dict[str, Any] = {
        "status": "unavailable" if resolved_ncu is None else "starting",
        "profile_version": profile_version,
        "tool_version": _ncu_version(resolved_ncu) if resolved_ncu else "unavailable",
        "requested_metrics": list(metrics),
        "kernel_filter": list(kernel_names),
        "profiled_kernel_count": 0,
        "kernels": [],
    }
    if resolved_ncu is None:
        base_result["error"] = f"NCU executable not found: {ncu_path}"
        base_result["wall_time_s"] = perf_counter() - started
        return base_result

    temp_root = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    try:
        with tempfile.TemporaryDirectory(prefix="kernelgym_ncu_", dir=temp_root) as temp_dir:
            root = Path(temp_dir)
            payload_path = root / "payload.json"
            report_base = root / "profile"
            csv_path = root / "profile.csv"
            payload = {
                "reference_code": original_model_src,
                "kernel_code": custom_model_src,
                "artifact": _artifact_payload(artifact),
                "backend": backend,
                "entry_point": entry_point,
                "device": "cuda:0",
                "warmup": max(0, warmup),
            }
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            command = build_ncu_command(
                ncu_path=resolved_ncu,
                report_base=report_base,
                payload_path=payload_path,
                metrics=metrics,
                kernel_names=kernel_names,
                max_kernels=max_kernels,
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=_ncu_environment(device),
                timeout=max(1, timeout_s),
            )
            profile_output = f"{completed.stdout}\n{completed.stderr}"
            if completed.returncode != 0:
                base_result["status"] = _failure_status(profile_output)
                base_result["error"] = _tail(profile_output)
                return base_result

            report_path = report_base.with_suffix(".ncu-rep")
            if not report_path.exists():
                reports = list(root.glob("*.ncu-rep"))
                report_path = reports[0] if reports else report_path
            if not report_path.exists():
                base_result["status"] = _failure_status(profile_output)
                base_result["error"] = _tail(profile_output) or "NCU produced no report"
                return base_result

            export = subprocess.run(
                [
                    resolved_ncu,
                    "--import",
                    str(report_path),
                    "--csv",
                    "--page",
                    "raw",
                    "--log-file",
                    str(csv_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=max(10, min(max(1, timeout_s), 30)),
            )
            export_output = f"{export.stdout}\n{export.stderr}"
            if export.returncode != 0:
                base_result["status"] = _failure_status(export_output)
                base_result["error"] = _tail(export_output) or "Failed to export NCU CSV"
                return base_result

            csv_text = (
                csv_path.read_text(encoding="utf-8", errors="replace")
                if csv_path.exists()
                else ""
            )
            export_candidates = (
                ("log_file", csv_text),
                ("stdout", export.stdout),
                ("stderr", export.stderr),
            )
            kernels: List[Dict[str, Any]] = []
            csv_source = "none"
            for source, text in export_candidates:
                kernels = parse_ncu_csv(text, max_kernels, metrics)
                if kernels:
                    csv_source = source
                    break

            base_result["kernels"] = kernels
            base_result["profiled_kernel_count"] = len(kernels)
            base_result["status"] = "ok" if kernels else "no_matching_kernel"
            base_result["csv_source"] = csv_source
            if not kernels:
                base_result["csv_size_bytes"] = len(csv_text.encode("utf-8"))
                base_result["csv_head"] = _head(csv_text)
                base_result["export_stdout_tail"] = _tail(export.stdout)
                base_result["export_stderr_tail"] = _tail(export.stderr)
                base_result["error"] = (
                    "NCU export contained no parseable kernel metrics"
                )
            return base_result
    except subprocess.TimeoutExpired as exc:
        base_result["status"] = "timeout"
        base_result["error"] = f"NCU timed out after {exc.timeout}s"
        return base_result
    except Exception as exc:
        base_result["status"] = "error"
        base_result["error"] = f"{type(exc).__name__}: {exc}"
        return base_result
    finally:
        base_result["wall_time_s"] = perf_counter() - started


def skipped_ncu_result(status: str, profile_version: str) -> Dict[str, Any]:
    return {
        "status": status,
        "profile_version": profile_version,
        "profiled_kernel_count": 0,
        "kernels": [],
        "wall_time_s": 0.0,
    }
