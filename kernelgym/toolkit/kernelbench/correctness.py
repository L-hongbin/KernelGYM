"""KernelBench correctness helpers (toolkit layer)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable, TypeVar

import torch
import torch.nn as nn

from kernelgym.toolkit.kernelbench.exec_types import (
    KernelExecResult,
    get_error_name,
    set_seed,
)
from kernelgym.toolkit.kernelbench.profiling import (
    aten_operator_profiling_context,
    extract_aten_operator_metrics,
)


logger = logging.getLogger(__name__)
_CORRECTNESS_EARLY_STOP_ENV = "KERNELGYM_CORRECTNESS_EARLY_STOP"
_CORRECTNESS_MAX_WALL_S_ENV = "KERNELGYM_CORRECTNESS_MAX_WALL_S"
_CORRECTNESS_PASS_ON_BUDGET_ENV = "KERNELGYM_CORRECTNESS_PASS_ON_BUDGET"
_CORRECTNESS_BUDGET_MIN_PASS_TRIALS_ENV = "KERNELGYM_CORRECTNESS_BUDGET_MIN_PASS_TRIALS"
_CORRECTNESS_GPU_INPUTS_ENV = "KERNELGYM_CORRECTNESS_GPU_INPUTS"
_CORRECTNESS_DISABLE_TF32_ENV = "KERNELGYM_CORRECTNESS_DISABLE_TF32"
T = TypeVar("T")


_FP32_TOL_ENV = "KERNELGYM_FP32_TOL"


def get_tolerance_for_dtype(dtype: torch.dtype) -> float:
    """Match KernelBench fp32 tolerance for integral outputs.

    The float32 tolerance can be overridden via ``KERNELGYM_FP32_TOL`` (opt-in;
    default is the KernelBench-standard 1e-4) — used to study how much of the
    correctness gap is fp32 accumulation vs the strict default tolerance. All
    other dtypes are unchanged.
    """
    tolerances = {
        torch.float32: 1e-4,
        torch.float16: 1e-2,
        torch.bfloat16: 1e-2,
        torch.bool: 0.0,
        torch.uint8: 1e-4,
        torch.int8: 1e-4,
        torch.int16: 1e-4,
        torch.int32: 1e-4,
        torch.int64: 1e-4,
    }
    if dtype not in tolerances:
        raise ValueError(f"Unsupported correctness tolerance dtype: {dtype}")
    if dtype == torch.float32:
        override = os.environ.get(_FP32_TOL_ENV)
        if override is not None and override.strip():
            try:
                value = float(override)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
    return tolerances[dtype]


def _env_optional_str(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value


def _env_flag(name: str, *, default: bool) -> bool:
    value = _env_optional_str(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_parsed(name: str, parser: Callable[[str], T]) -> T | None:
    value = _env_optional_str(name)
    if value is None:
        return None
    try:
        return parser(value)
    except (TypeError, ValueError):
        return None


def _env_positive_float(name: str) -> float | None:
    parsed = _env_parsed(name, float)
    return parsed if parsed is not None and parsed > 0 else None


def _env_positive_int(name: str) -> int | None:
    parsed = _env_parsed(name, int)
    return parsed if parsed is not None and parsed > 0 else None


def _read_backend_attr(root: Any, path: tuple[str, ...]) -> tuple[bool, Any]:
    obj = root
    for name in path[:-1]:
        try:
            obj = getattr(obj, name)
        except Exception:
            return False, None
        if obj is None:
            return False, None
    try:
        return True, getattr(obj, path[-1])
    except Exception:
        return False, None


def _write_backend_attr(root: Any, path: tuple[str, ...], value: Any) -> bool:
    obj = root
    for name in path[:-1]:
        try:
            obj = getattr(obj, name)
        except Exception:
            return False
        if obj is None:
            return False
    try:
        setattr(obj, path[-1], value)
        return True
    except Exception:
        return False


@contextmanager
def _true_fp32_correctness_context(metadata: dict):
    """Force CUDA float32 ops to true fp32 during correctness, then restore."""
    enabled = _env_flag(_CORRECTNESS_DISABLE_TF32_ENV, default=True)
    metadata["correctness_tf32_disabled"] = bool(enabled)
    if not enabled:
        yield
        return

    # PyTorch 2.9+ prefers per-op fp32_precision settings. Avoid mixing those
    # with legacy allow_tf32 flags when they are available; newer PyTorch can
    # raise if the old and new APIs imply different cuDNN policies.
    has_cudnn_conv_precision, _ = _read_backend_attr(torch.backends, ("cudnn", "conv", "fp32_precision"))
    has_matmul_precision, _ = _read_backend_attr(torch.backends, ("cuda", "matmul", "fp32_precision"))

    attr_targets = []
    if has_cudnn_conv_precision:
        attr_targets.append(("cudnn.conv.fp32_precision", ("cudnn", "conv", "fp32_precision"), "ieee"))
    else:
        attr_targets.append(("cudnn.allow_tf32", ("cudnn", "allow_tf32"), False))
    if has_matmul_precision:
        attr_targets.append(("cuda.matmul.fp32_precision", ("cuda", "matmul", "fp32_precision"), "ieee"))
    else:
        attr_targets.append(("cuda.matmul.allow_tf32", ("cuda", "matmul", "allow_tf32"), False))
    saved_attrs: list[tuple[tuple[str, ...], Any]] = []
    before: dict[str, str] = {}
    applied: dict[str, str] = {}
    for label, path, value in attr_targets:
        exists, old_value = _read_backend_attr(torch.backends, path)
        if not exists:
            continue
        before[label] = str(old_value)
        if _write_backend_attr(torch.backends, path, value):
            saved_attrs.append((path, old_value))
            applied[label] = str(value)

    old_matmul_precision = None
    if (
        not has_matmul_precision
        and hasattr(torch, "get_float32_matmul_precision")
        and hasattr(torch, "set_float32_matmul_precision")
    ):
        try:
            old_matmul_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            before["float32_matmul_precision"] = str(old_matmul_precision)
            applied["float32_matmul_precision"] = "highest"
        except Exception:
            old_matmul_precision = None

    metadata["correctness_tf32_state_before"] = before
    metadata["correctness_tf32_state_forced"] = applied

    try:
        yield
    finally:
        if old_matmul_precision is not None:
            try:
                torch.set_float32_matmul_precision(old_matmul_precision)
            except Exception:
                pass
        for path, old_value in reversed(saved_attrs):
            _write_backend_attr(torch.backends, path, old_value)


@contextmanager
def _input_generation_device_context(device: Any, *, enabled: bool = True):
    if not enabled or device is None:
        yield
        return
    if isinstance(device, int):
        cuda_device = torch.device("cuda", device)
    else:
        target = torch.device(device)
        if target.type != "cuda":
            yield
            return
        cuda_device = target
    previous_device = None
    if hasattr(torch, "get_default_device"):
        previous_device = torch.get_default_device()
    torch.set_default_device(cuda_device)
    try:
        yield
    finally:
        torch.set_default_device(previous_device or "cpu")


def _move_input_to_device(value: Any, *, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if device is None:
            return value
        return value.cuda(device=device)
    if isinstance(value, list):
        return [_move_input_to_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_input_to_device(item, device=device) for item in value)
    if isinstance(value, dict):
        return {key: _move_input_to_device(item, device=device) for key, item in value.items()}
    return value


def _clone_output_on_device(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_output_on_device(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_output_on_device(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_output_on_device(item) for key, item in value.items()}
    return value


def _zero_poison_like(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return torch.zeros_like(value, memory_format=torch.preserve_format)
    if isinstance(value, list):
        return [_zero_poison_like(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_zero_poison_like(item) for item in value)
    if isinstance(value, dict):
        return {key: _zero_poison_like(item) for key, item in value.items()}
    return None


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _tensor_storage_id(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        return tensor.data_ptr()


def _output_aliases_inputs(output: Any, inputs: Any) -> bool:
    input_storage_ids = {_tensor_storage_id(tensor) for tensor in _iter_tensors(inputs)}
    if not input_storage_ids:
        return False
    return any(_tensor_storage_id(tensor) in input_storage_ids for tensor in _iter_tensors(output))


def _compare_tensors_inplace(
    output: torch.Tensor,
    output_new: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[bool, float, float]:
    """Destructively compare two tensors without allocating a full diff tensor."""
    if output.numel() == 0:
        return True, 0.0, 0.0

    if output.dtype in {torch.bool, torch.uint8}:
        output.ne_(output_new)
        mismatch_count = output.sum(dtype=torch.float64).item()
        avg_diff = mismatch_count / output.numel()
        max_diff = 1.0 if mismatch_count else 0.0
        return mismatch_count == 0, float(max_diff), float(avg_diff)

    if not output.is_floating_point() and not output.is_complex():
        outputs_close = torch.allclose(output, output_new, atol=atol, rtol=rtol)
        output.sub_(output_new).abs_()
        max_diff = output.max().item()
        avg_diff = output.sum(dtype=torch.float64).item() / output.numel()
        return bool(outputs_close), float(max_diff), float(avg_diff)

    output.sub_(output_new).abs_()
    max_diff = output.max().item()
    avg_diff = output.mean().item()

    # Match torch.allclose(input, other): abs(input - other) <= atol + rtol * abs(other).
    output_new.abs_().mul_(rtol).add_(atol)
    output.sub_(output_new)
    max_over_tolerance = output.max().item()
    return max_over_tolerance <= 0, float(max_diff), float(avg_diff)


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate: bool = False,
    max_length: int = 200,
):
    if verbose:
        logger.warning("[Exception %s] %s", exception_type, exception_msg)

    metadata[exception_type] = exception_msg
    return metadata


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: Callable[[], Any],
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Any = None,
    stop_on_first_failure: bool | None = None,
    max_wall_time_s: float | None = None,
    pass_on_time_budget: bool | None = None,
    budget_min_pass_trials: int | None = None,
    stage_update_fn: Callable[[str], None] | None = None,
    detect_aten_fallback: bool = False,
) -> KernelExecResult:
    pass_count = 0
    trials_run = 0
    correctness_start = perf_counter()
    trial_durations: list[float] = []
    reference_trial_durations: list[float] = []
    custom_trial_durations: list[float] = []
    compare_trial_durations: list[float] = []
    input_generation_durations: list[float] = []
    input_transfer_durations: list[float] = []
    reference_alias_clone_durations: list[float] = []

    def _set_substage(name: str, *, trial: int | None = None) -> None:
        if trial is not None:
            metadata["correctness_current_trial"] = trial
        metadata["correctness_current_substage"] = name
        if stage_update_fn is not None:
            stage_update_fn(f"kernel.correctness.{name}")

    if stop_on_first_failure is None:
        stop_on_first_failure = _env_flag(_CORRECTNESS_EARLY_STOP_ENV, default=True)
    if max_wall_time_s is None:
        max_wall_time_s = _env_positive_float(_CORRECTNESS_MAX_WALL_S_ENV)
    if pass_on_time_budget is None:
        pass_on_time_budget = _env_flag(_CORRECTNESS_PASS_ON_BUDGET_ENV, default=False)
    if budget_min_pass_trials is None:
        budget_min_pass_trials = _env_positive_int(_CORRECTNESS_BUDGET_MIN_PASS_TRIALS_ENV) or 1
    budget_min_pass_trials = max(1, min(int(budget_min_pass_trials), num_correct_trials))
    generate_inputs_on_gpu = _env_flag(_CORRECTNESS_GPU_INPUTS_ENV, default=True)

    metadata["correctness_early_stop_enabled"] = bool(stop_on_first_failure)
    metadata["correctness_budget_pass_on_success_enabled"] = bool(pass_on_time_budget)
    metadata["correctness_budget_min_pass_trials"] = budget_min_pass_trials
    metadata["correctness_inplace_compare_enabled"] = True
    metadata["correctness_reference_alias_clone_trials"] = []
    metadata["correctness_reference_cache_poison_enabled"] = True
    metadata["correctness_forward_seed_reset_enabled"] = True
    metadata["correctness_tolerance_source"] = "kernelbench_precision_or_fp32_integral"
    metadata["aten_detection_enabled"] = bool(detect_aten_fallback)
    if max_wall_time_s is not None:
        metadata["correctness_max_wall_s"] = max_wall_time_s

    def _record_trial_metadata() -> None:
        metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"
        metadata["correctness_trials_run"] = trials_run
        metadata["correctness_trial_s"] = trial_durations
        metadata["correctness_reference_trial_s"] = reference_trial_durations
        metadata["correctness_custom_trial_s"] = custom_trial_durations
        metadata["correctness_compare_trial_s"] = compare_trial_durations
        metadata["correctness_input_generation_trial_s"] = input_generation_durations
        metadata["correctness_input_transfer_trial_s"] = input_transfer_durations
        metadata["correctness_reference_alias_clone_trial_s"] = reference_alias_clone_durations

    def _record_aten_trial_metrics(aten_metrics: dict[str, Any], trial: int) -> None:
        trial_record = {"trial": trial, **aten_metrics}
        trial_records = metadata.setdefault("aten_detection_trials", [])
        trial_records.append(trial_record)
        metadata["aten_detection_trials_run"] = len(trial_records)
        metadata["aten_allowlist_version"] = aten_metrics.get("aten_allowlist_version")
        metadata["aten_detection_valid"] = all(bool(record.get("aten_detection_valid")) for record in trial_records)

        errors = [
            {"trial": record["trial"], "error": record.get("aten_detection_error")}
            for record in trial_records
            if not record.get("aten_detection_valid")
        ]
        if errors:
            metadata["aten_detection_errors"] = errors

        merged_ops: dict[str, dict[str, Any]] = {}
        for record in trial_records:
            for item in record.get("aten_ops", []):
                name = item["name"]
                merged = merged_ops.setdefault(
                    name,
                    {
                        "name": name,
                        "normalized_name": item["normalized_name"],
                        "count": 0,
                        "cpu_time_us": 0.0,
                        "allowed": item["allowed"],
                    },
                )
                merged["count"] += item["count"]
                merged["cpu_time_us"] += item["cpu_time_us"]

        aten_ops = [merged_ops[name] for name in sorted(merged_ops)]
        allowed_ops = [item for item in aten_ops if item["allowed"]]
        forbidden_ops = [item for item in aten_ops if not item["allowed"]]
        metadata["aten_ops"] = aten_ops
        metadata["allowed_aten_ops"] = allowed_ops
        metadata["forbidden_aten_ops"] = forbidden_ops
        metadata["forbidden_aten_op_names"] = [item["name"] for item in forbidden_ops]
        metadata["aten_operator_count"] = sum(item["count"] for item in aten_ops)
        metadata["allowed_aten_operator_count"] = sum(item["count"] for item in allowed_ops)
        metadata["forbidden_aten_operator_count"] = sum(item["count"] for item in forbidden_ops)

        if forbidden_ops:
            metadata["policy_violation"] = True
            metadata["policy_violation_reason"] = "DISALLOWED_ATEN_COMPUTE"
            metadata["decoy_reason"] = "DISALLOWED_ATEN_COMPUTE"
            logger.warning(
                "[ATen Detection] Forbidden candidate operators: %s",
                metadata["forbidden_aten_op_names"],
            )

    def _maybe_finish_on_time_budget() -> KernelExecResult | None:
        if max_wall_time_s is None or trials_run == 0:
            return None
        elapsed_before_trial = perf_counter() - correctness_start
        if elapsed_before_trial < max_wall_time_s:
            return None

        metadata["correctness_time_budget_exceeded"] = True
        _record_trial_metadata()
        all_completed_trials_passed = pass_count == trials_run
        if pass_on_time_budget and all_completed_trials_passed:
            if pass_count >= budget_min_pass_trials:
                metadata["correctness_budget_passed_early"] = True
                metadata["correctness_issue_name"] = "correctness_time_budget_passed_early"
                metadata["correctness_issue"] = (
                    f"Correctness time budget reached after {pass_count} passing trials; accepted early"
                )
                return KernelExecResult(
                    compiled=True,
                    correctness=True,
                    decoy_kernel=bool(metadata.get("policy_violation")),
                    metadata=metadata,
                )
            metadata["correctness_time_budget_overrun_to_min_pass"] = True
            return None

        metadata["correctness_issue_name"] = "correctness_time_budget_exceeded"
        metadata["correctness_issue"] = (
            f"Correctness time budget exceeded after {trials_run} / {num_correct_trials} trials"
        )
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

    torch.manual_seed(seed)
    correctness_trial_seeds = [int(torch.randint(0, 2**32 - 1, (1,)).item()) for _ in range(num_correct_trials)]

    def _record_runtime_exception(
        exception: Exception,
        *,
        trial: int,
        trial_start: float,
    ) -> KernelExecResult:
        nonlocal metadata, trials_run
        trials_run = trial + 1
        trial_durations.append(perf_counter() - trial_start)
        logger.warning("[Error] Exception happened during correctness check")
        logger.warning("Error in launching kernel for ModelNew: %s", exception)

        metadata = register_and_format_exception("runtime_error", exception, metadata, truncate=False)
        metadata["runtime_error_name"] = get_error_name(exception)
        metadata["correctness_failed_trial"] = trial
        _record_trial_metadata()
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

    with _true_fp32_correctness_context(metadata), torch.no_grad():
        set_seed(seed)
        model = original_model_instance.cuda(device=device)
        set_seed(seed)
        model_new = new_model_instance.cuda(device=device)
        # Diagnostic-only toggle (default off, matches KernelBench upstream and the
        # official KernelGYM fork, both of which never call .eval()): when set, run
        # the REFERENCE model in eval() mode (e.g. BatchNorm uses running stats
        # instead of live batch stats). Used to quantify how much of a correctness
        # gap is attributable to train-mode-vs-eval-mode reference semantics; must
        # stay opt-in so it never changes default/live scoring behavior.
        if _env_flag("KERNELGYM_CORRECTNESS_REFERENCE_EVAL_MODE", default=False):
            model.eval()
            metadata["correctness_reference_eval_mode"] = True

        for trial in range(num_correct_trials):
            if trial > 0:
                budget_result = _maybe_finish_on_time_budget()
                if budget_result is not None:
                    return budget_result

            trial_start = perf_counter()
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                logger.info("[Eval] Generating Random Input with seed %s", trial_seed)

            set_seed(trial_seed)
            _set_substage("input_generation", trial=trial)
            input_generation_start = perf_counter()
            with _input_generation_device_context(device, enabled=generate_inputs_on_gpu):
                inputs = get_inputs_fn()
            input_generation_durations.append(perf_counter() - input_generation_start)

            _set_substage("input_transfer", trial=trial)
            input_transfer_start = perf_counter()
            inputs = _move_input_to_device(inputs, device=device)
            input_transfer_durations.append(perf_counter() - input_transfer_start)

            if verbose:
                first_input_device = getattr(inputs[0], "device", None) if inputs else None
                logger.debug("device: %s", device)
                logger.debug("inputs: %s", first_input_device)

            _set_substage("reference_forward", trial=trial)
            # Input generation may consume RNG. Start both forwards from the
            # same per-trial Torch RNG state so stochastic PyTorch operations
            # are comparable when ModelNew uses the same RNG implementation.
            set_seed(trial_seed)
            reference_start = perf_counter()
            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            reference_trial_durations.append(perf_counter() - reference_start)

            _set_substage("reference_alias_clone", trial=trial)
            alias_clone_start = perf_counter()
            if _output_aliases_inputs(output, inputs):
                output = _clone_output_on_device(output)
                metadata["correctness_reference_alias_clone_trials"].append(trial)
                reference_alias_clone_durations.append(perf_counter() - alias_clone_start)
            else:
                reference_alias_clone_durations.append(0.0)

            poison_scratch = _zero_poison_like(output)
            if any(True for _ in _iter_tensors(poison_scratch)):
                torch.cuda.synchronize(device=device)
            del poison_scratch

            try:
                _set_substage("custom_forward", trial=trial)
                set_seed(trial_seed)
                custom_start = perf_counter()
                profile_aten_operators = bool(detect_aten_fallback)
                with aten_operator_profiling_context(profile_aten_operators) as aten_prof:
                    output_new = model_new(*inputs)
                    torch.cuda.synchronize(device=device)
                if profile_aten_operators:
                    aten_metrics = extract_aten_operator_metrics(aten_prof)
                    _record_aten_trial_metrics(aten_metrics, trial)
                custom_trial_durations.append(perf_counter() - custom_start)
                trials_run = trial + 1
                del inputs

                if output.shape != output_new.shape:
                    compare_trial_durations.append(0.0)
                    trial_durations.append(perf_counter() - trial_start)
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    metadata["correctness_failed_trial"] = trial
                    _record_trial_metadata()
                    if verbose:
                        logger.warning(
                            "[FAIL] trial %s: Output shape mismatch: Expected %s, got %s",
                            trial,
                            output.shape,
                            output_new.shape,
                        )
                    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

                _set_substage("compare", trial=trial)
                compare_start = perf_counter()
                tolerance = get_tolerance_for_dtype(output.dtype)
                metadata["correctness_atol"] = tolerance
                metadata["correctness_rtol"] = tolerance
                outputs_close, max_diff, avg_diff = _compare_tensors_inplace(
                    output,
                    output_new,
                    atol=tolerance,
                    rtol=tolerance,
                )
                compare_trial_durations.append(perf_counter() - compare_start)
                trial_durations.append(perf_counter() - trial_start)

                if not outputs_close:
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    metadata["correctness_issue_name"] = "correctness_issue"
                    metadata["correctness_failed_trial"] = trial
                    if verbose:
                        logger.warning("[FAIL] trial %s: Output mismatch", trial)
                    if stop_on_first_failure:
                        metadata["correctness_early_stopped"] = True
                        _record_trial_metadata()
                        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
                else:
                    pass_count += 1
                    if verbose:
                        logger.info("[PASS] trial %s: New Model matches Model", trial)
                del output, output_new

            except Exception as e:
                if len(custom_trial_durations) < len(reference_trial_durations):
                    custom_trial_durations.append(perf_counter() - custom_start)
                return _record_runtime_exception(
                    e,
                    trial=trial,
                    trial_start=trial_start,
                )

    if verbose:
        logger.info("[Eval] Pass count: %s, num_correct_trials: %s", pass_count, num_correct_trials)

    _record_trial_metadata()

    if pass_count == num_correct_trials:
        return KernelExecResult(
            compiled=True,
            correctness=True,
            decoy_kernel=bool(metadata.get("policy_violation")),
            metadata=metadata,
        )
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
