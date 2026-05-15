from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from kernelgym.toolkit.kernelbench.loading import load_original_model_and_inputs
from kernelgym.toolkit.kernelbench.timing import (
    time_execution_with_cuda_event,
    time_execution_with_cupti,
    time_execution_with_cupti_flashinfer_bench,
)

VALID_TEST_KEYS = (
    "cuda_event",
    "cupti_direct",
    "cupti_flashinfer_bench",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CUPTI timing against CUDA event timing using a random torch ground_truth sample from parquet."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset-path",
        default="/nfs/FM/lihongbin/datasets/CUDA-Agent-Ops-6K/valid_data_200.parquet",
        help="Path to parquet dataset containing reward_model.ground_truth and extra_info.entry_point.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Optional fixed row index after torch-only filtering. If unset, randomly sampled with --seed.",
    )
    parser.add_argument(
        "--test-order",
        default=",".join(VALID_TEST_KEYS),
        help="Base comma-separated order, e.g. cupti_direct,cuda_event,cupti_flashinfer_bench",
    )
    parser.add_argument(
        "--balanced-order-rounds",
        type=int,
        default=3,
        help="Number of rounds with rotated order for order-bias mitigation.",
    )
    parser.add_argument(
        "--stabilize-iters",
        type=int,
        default=10,
        help="Untimed stabilization iterations before each method measurement.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def _parse_test_order(raw_order: str) -> list[str]:
    order = [x.strip() for x in raw_order.split(",") if x.strip()]
    if set(order) == set(VALID_TEST_KEYS) and len(order) == len(VALID_TEST_KEYS):
        return order
    raise ValueError(
        f"Invalid --test-order: {raw_order}. Must contain each of {VALID_TEST_KEYS} exactly once."
    )


def _build_round_orders(base_order: list[str], rounds: int) -> list[list[str]]:
    n = len(base_order)
    count = max(1, int(rounds))
    orders: list[list[str]] = []
    for i in range(count):
        shift = i % n
        orders.append(base_order[shift:] + base_order[:shift])
    return orders


def _to_device_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cuda(device=device)
    if isinstance(value, list):
        return [_to_device_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device_tree(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _to_device_tree(v, device) for k, v in value.items()}
    return value


def _select_sample_spec(
    dataset_path: str,
    sample_index: int | None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_parquet(path)
    torch_rows = df[df["data_source"].astype(str).str.startswith("torch", na=False)].reset_index(
        drop=False
    )
    if torch_rows.empty:
        raise ValueError("No torch samples found in dataset.")

    if sample_index is None:
        rng = np.random.default_rng(seed)
        sample_index = int(rng.integers(0, len(torch_rows)))
    if sample_index < 0 or sample_index >= len(torch_rows):
        raise IndexError(
            f"sample-index out of range: {sample_index}, expected [0, {len(torch_rows) - 1}]"
        )

    row = torch_rows.iloc[sample_index]
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        raise TypeError("reward_model must be a dict containing ground_truth.")

    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, str):
        raise TypeError("reward_model.ground_truth must be a string.")

    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}

    entry_point = extra_info.get("entry_point") or "Model"

    spec = {
        "ground_truth": ground_truth,
        "entry_point": entry_point,
    }

    metadata = {
        "dataset_path": str(path),
        "torch_filtered_size": int(len(torch_rows)),
        "sample_index_in_torch_rows": int(sample_index),
        "sample_index_in_original_df": int(row["index"]),
        "data_source": row.get("data_source"),
        "ability": row.get("ability"),
        "entry_point": entry_point,
    }
    return spec, metadata


def _build_workload_from_spec(
    spec: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[callable, str]:
    torch.manual_seed(seed)

    context: dict[str, Any] = {}
    loaded = load_original_model_and_inputs(
        spec["ground_truth"],
        context,
        entry_point=spec["entry_point"],
    )
    if loaded is None:
        raise RuntimeError("Failed to compile/execute sampled ground_truth.")

    Model, get_init_inputs, get_inputs = loaded
    if Model is None or not callable(get_init_inputs) or not callable(get_inputs):
        raise RuntimeError("Sampled ground_truth is missing Model/get_init_inputs/get_inputs.")

    init_inputs = _to_device_tree(get_init_inputs(), device=device)
    if isinstance(init_inputs, list):
        model = Model(*init_inputs)
    elif isinstance(init_inputs, tuple):
        model = Model(*init_inputs)
    elif isinstance(init_inputs, dict):
        model = Model(**init_inputs)
    else:
        model = Model(init_inputs)

    model = model.cuda(device=device).eval()

    raw_inputs = _to_device_tree(get_inputs(), device=device)
    if isinstance(raw_inputs, list):
        inputs = tuple(raw_inputs)
    elif isinstance(raw_inputs, tuple):
        inputs = raw_inputs
    else:
        inputs = (raw_inputs,)

    def _run() -> Any:
        with torch.no_grad():
            return model(*inputs)

    model_name = Model.__name__ if hasattr(Model, "__name__") else str(Model)
    return _run, model_name


def _stabilize_workload(fn: callable, device: torch.device, iters: int) -> None:
    count = max(0, int(iters))
    if count == 0:
        return
    torch.cuda.synchronize(device=device)
    for _ in range(count):
        fn()
    torch.cuda.synchronize(device=device)


def _stats(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "var_ms2": float(arr.var()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _measure_or_error(fn: callable) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        samples, _ = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": True,
            "samples_ms": [float(x) for x in samples],
            **_stats(samples),
            "eval_wall_time_ms": float(elapsed_ms),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "eval_wall_time_ms": float(elapsed_ms),
        }


def _aggregate_runs(method_runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok_runs = [r for r in method_runs if r.get("ok")]
    result: dict[str, Any] = {
        "num_rounds": len(method_runs),
        "num_ok_rounds": len(ok_runs),
    }

    if len(ok_runs) == len(method_runs) and len(ok_runs) > 0:
        all_samples: list[float] = []
        for run in ok_runs:
            all_samples.extend(run.get("samples_ms", []))

        wall_times = [float(run["eval_wall_time_ms"]) for run in ok_runs]

        result.update(
            {
                "ok": True,
                "samples_ms": all_samples,
                **_stats(all_samples),
                "eval_wall_time_ms": float(np.mean(wall_times)),
                "eval_wall_time_std_ms": float(np.std(wall_times)),
            }
        )
        return result

    errors = [run.get("error", "unknown error") for run in method_runs if not run.get("ok")]
    result.update(
        {
            "ok": False,
            "error": f"{len(errors)} round(s) failed: {errors}",
            "round_results": method_runs,
        }
    )
    return result


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        pass
    else:
        raise ValueError(f"Expected a CUDA device, got: {device}")

    base_order = _parse_test_order(args.test_order)
    round_orders = _build_round_orders(base_order, args.balanced_order_rounds)

    spec, metadata = _select_sample_spec(
        dataset_path=args.dataset_path,
        sample_index=args.sample_index,
        seed=args.seed,
    )

    metadata["base_test_order"] = base_order
    metadata["round_orders"] = round_orders
    metadata["balanced_order_rounds"] = len(round_orders)
    metadata["stabilize_iters"] = int(args.stabilize_iters)

    runners: dict[str, callable] = {
        "cuda_event": lambda test_fn: time_execution_with_cuda_event(
            test_fn,
            num_warmup=args.warmup,
            num_trials=args.trials,
            verbose=False,
            device=device,
            enable_profiling=False,
        ),
        "cupti_direct": lambda test_fn: time_execution_with_cupti(
            test_fn,
            num_warmup=args.warmup,
            num_trials=args.trials,
            verbose=False,
            device=device,
            enable_profiling=False,
        ),
        "cupti_flashinfer_bench": lambda test_fn: time_execution_with_cupti_flashinfer_bench(
            test_fn,
            num_warmup=args.warmup,
            num_trials=args.trials,
            verbose=False,
            device=device,
            enable_profiling=False,
        ),
    }

    per_method_runs: dict[str, list[dict[str, Any]]] = {k: [] for k in VALID_TEST_KEYS}
    model_name: str | None = None

    for round_order in round_orders:
        for key in round_order:
            test_fn, current_model_name = _build_workload_from_spec(spec, device=device, seed=args.seed)
            if model_name is None:
                model_name = current_model_name
            _stabilize_workload(test_fn, device=device, iters=args.stabilize_iters)
            run_result = _measure_or_error(lambda key=key, test_fn=test_fn: runners[key](test_fn))
            per_method_runs[key].append(run_result)
            torch.cuda.synchronize(device=device)

    metadata["model_name"] = model_name

    cuda_event_result = _aggregate_runs(per_method_runs["cuda_event"])
    cupti_result = _aggregate_runs(per_method_runs["cupti_direct"])
    cupti_flashinfer_bench_result = _aggregate_runs(per_method_runs["cupti_flashinfer_bench"])

    result = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "trials": args.trials,
        "metadata": metadata,
        "cuda_event": cuda_event_result,
        "cupti_direct": cupti_result,
        "cupti_flashinfer_bench": cupti_flashinfer_bench_result,
    }

    if cuda_event_result.get("ok") and cupti_result.get("ok"):
        result["delta_mean_ms_direct_vs_cuda_event"] = (
            cupti_result["mean_ms"] - cuda_event_result["mean_ms"]
        )
        result["delta_var_ms2_direct_vs_cuda_event"] = (
            cupti_result["var_ms2"] - cuda_event_result["var_ms2"]
        )

    if cuda_event_result.get("ok") and cupti_flashinfer_bench_result.get("ok"):
        result["delta_mean_ms_flashinfer_vs_cuda_event"] = (
            cupti_flashinfer_bench_result["mean_ms"] - cuda_event_result["mean_ms"]
        )
        result["delta_var_ms2_flashinfer_vs_cuda_event"] = (
            cupti_flashinfer_bench_result["var_ms2"] - cuda_event_result["var_ms2"]
        )

    if cupti_result.get("ok") and cupti_flashinfer_bench_result.get("ok"):
        result["delta_mean_ms_flashinfer_vs_direct"] = (
            cupti_flashinfer_bench_result["mean_ms"] - cupti_result["mean_ms"]
        )
        result["delta_var_ms2_flashinfer_vs_direct"] = (
            cupti_flashinfer_bench_result["var_ms2"] - cupti_result["var_ms2"]
        )

    if "eval_wall_time_ms" in cuda_event_result and "eval_wall_time_ms" in cupti_result:
        result["delta_eval_wall_time_ms_direct_vs_cuda_event"] = (
            cupti_result["eval_wall_time_ms"] - cuda_event_result["eval_wall_time_ms"]
        )

    if "eval_wall_time_ms" in cuda_event_result and "eval_wall_time_ms" in cupti_flashinfer_bench_result:
        result["delta_eval_wall_time_ms_flashinfer_vs_cuda_event"] = (
            cupti_flashinfer_bench_result["eval_wall_time_ms"]
            - cuda_event_result["eval_wall_time_ms"]
        )

    if "eval_wall_time_ms" in cupti_result and "eval_wall_time_ms" in cupti_flashinfer_bench_result:
        result["delta_eval_wall_time_ms_flashinfer_vs_direct"] = (
            cupti_flashinfer_bench_result["eval_wall_time_ms"] - cupti_result["eval_wall_time_ms"]
        )

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
