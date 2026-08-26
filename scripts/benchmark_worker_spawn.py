#!/usr/bin/env python3
"""Measure KernelGym CUDA subprocess startup and replenishment behavior.

The benchmark deliberately avoids starting Redis, the API, or an outer
KernelGym service.  It has three public modes:

* ``stages`` measures fresh-interpreter Torch/CUDA initialization phases.
* ``constructors`` creates real ``PersistentWorker`` instances concurrently.
* ``pools`` drives real ``SubprocessWorkerPool`` instances with a cheap,
  expected non-CUDA task failure so process turnover can be measured without
  compiling or executing untrusted kernels.

All CUDA-owning children use the production containment and shutdown paths.
Temporary lock, stderr, and stage-metadata files live below ``/dev/shm``.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any


RESULT_PREFIX = "KERNELGYM_SPAWN_BENCH_RESULT="


def _emit(payload: dict[str, Any]) -> None:
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def _read_proc_io() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip())
    except (OSError, ValueError):
        pass
    return values


def _resource_snapshot() -> dict[str, int | float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io_values = _read_proc_io()
    return {
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
        "max_rss_kib": int(usage.ru_maxrss),
        "user_cpu_s": float(usage.ru_utime),
        "system_cpu_s": float(usage.ru_stime),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "fs_input_blocks": int(usage.ru_inblock),
        "fs_output_blocks": int(usage.ru_oublock),
        "read_bytes": io_values.get("read_bytes", 0),
        "write_bytes": io_values.get("write_bytes", 0),
        "rchar": io_values.get("rchar", 0),
        "wchar": io_values.get("wchar", 0),
    }


def _resource_delta(before: dict[str, int | float], after: dict[str, int | float]) -> dict[str, int | float]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}


def _stage_child() -> int:
    process_started = time.perf_counter()
    resources_before = _resource_snapshot()
    phases: dict[str, float] = {}

    start = time.perf_counter()
    import torch

    phases["import_torch_s"] = time.perf_counter() - start

    start = time.perf_counter()
    from kernelgym.backend import get_backend  # noqa: F401
    from kernelgym.toolkit import get_toolkit  # noqa: F401

    phases["import_kernelgym_registries_s"] = time.perf_counter() - start

    start = time.perf_counter()
    torch.cuda.init()
    phases["cuda_init_s"] = time.perf_counter() - start

    start = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    phases["cuda_set_device_s"] = time.perf_counter() - start

    start = time.perf_counter()
    probe = torch.zeros(1, device=device)
    phases["cuda_alloc_s"] = time.perf_counter() - start

    start = time.perf_counter()
    torch.cuda.synchronize(device)
    phases["cuda_sync_s"] = time.perf_counter() - start

    resources_after = _resource_snapshot()
    _emit(
        {
            "kind": "stage_child",
            "ok": True,
            "pid": os.getpid(),
            "python": sys.executable,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torch_file": torch.__file__,
            "torch_c_file": torch._C.__file__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "phases": phases,
            "child_total_s": time.perf_counter() - process_started,
            "resources": _resource_delta(resources_before, resources_after),
            "probe_value": float(probe.item()),
        }
    )
    return 0


def _constructor_child(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    worker = None
    try:
        from kernelgym.worker.subprocess_pool import PersistentWorker

        import_started = time.perf_counter()
        worker = PersistentWorker(
            worker_id=args.worker_id,
            device_id=0,
            pool_size_info="(standalone spawn benchmark)",
            max_tasks_per_worker=1,
        )
        constructor_s = time.perf_counter() - import_started
        result = {
            "kind": "constructor_child",
            "ok": True,
            "worker_id": args.worker_id,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "constructor_s": constructor_s,
            "spawn_slot_wait_s": worker.spawn_slot_wait_s,
            "containment_s": worker.containment_elapsed_s,
            "ready_after_containment_s": worker.ready_after_containment_s,
            "child_init_s": worker.child_init_s,
            "worker_pid": worker.process.pid if worker.process is not None else None,
            "hold_s": args.hold_s,
        }
        if args.hold_s:
            time.sleep(args.hold_s)
        shutdown_started = time.perf_counter()
        result["shutdown_safe"] = worker.shutdown(timeout=30, force=False)
        result["shutdown_s"] = time.perf_counter() - shutdown_started
        result["outer_total_s"] = time.perf_counter() - started
        _emit(result)
        return 0 if result["shutdown_safe"] else 2
    except BaseException as exc:
        if worker is not None:
            try:
                worker.shutdown(timeout=30, force=True)
            except BaseException:
                pass
        _emit(
            {
                "kind": "constructor_child",
                "ok": False,
                "worker_id": args.worker_id,
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "outer_total_s": time.perf_counter() - started,
            }
        )
        return 1


def _parse_child_result(stdout: str) -> dict[str, Any] | None:
    result = None
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
    return result


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _base_child_env(work_root: Path, spawn_concurrency: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["KERNELGYM_WORKER_SPAWN_LOCK_DIR"] = str(work_root / "spawn-locks")
    env["KERNELGYM_WORKER_STDERR_DIR"] = str(work_root / "worker-stderr")
    env["KERNELGYM_STAGE_METADATA_DIR"] = str(work_root / "stage-metadata")
    env["KERNELGYM_CORE_DUMP_DIR"] = str(work_root / "core-dumps")
    env["KERNELGYM_CORE_DUMP_KEEP"] = "2"
    env["KERNELGYM_WORKER_SPAWN_SLOT_TIMEOUT"] = env.get("KERNELGYM_WORKER_SPAWN_SLOT_TIMEOUT", "120")
    if spawn_concurrency is not None:
        env["KERNELGYM_WORKER_SPAWN_CONCURRENCY"] = str(spawn_concurrency)
    for directory in (
        work_root / "spawn-locks",
        work_root / "worker-stderr",
        work_root / "stage-metadata",
        work_root / "core-dumps",
    ):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    return env


def _run_process_batch(
    commands: list[list[str]],
    environments: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    started: list[tuple[subprocess.Popen[str], float, list[str]]] = []
    for command, env in zip(commands, environments, strict=True):
        wall_started = time.perf_counter()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        started.append((process, wall_started, command))

    def collect(
        item: tuple[subprocess.Popen[str], float, list[str]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        process, wall_started, command = item
        stdout, stderr = process.communicate()
        wall_s = time.perf_counter() - wall_started
        parsed = _parse_child_result(stdout)
        if parsed is None:
            parsed = {
                "ok": False,
                "error": "child produced no benchmark result",
            }
        parsed["process_wall_s"] = wall_s
        parsed["returncode"] = process.returncode
        return parsed, {
            "command": command,
            "returncode": process.returncode,
            "stdout_tail": stdout[-8000:],
            "stderr_tail": stderr[-16000:],
        }

    # Collect concurrently so process_wall_s records each child's actual finish
    # time instead of the time at which a sequential communicate() reaches it.
    with ThreadPoolExecutor(max_workers=len(started)) as executor:
        collected = list(executor.map(collect, started))
    return [item[0] for item in collected], [item[1] for item in collected]


def _run_stages(args: argparse.Namespace) -> dict[str, Any]:
    work_root = Path(tempfile.mkdtemp(prefix="kernelgym-stage-bench-", dir="/dev/shm"))
    all_results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    try:
        env_base = _base_child_env(work_root)
        gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
        for batch_index in range(args.repeat):
            commands = [[sys.executable, __file__, "stage-child"] for _ in range(args.concurrency)]
            envs = []
            for child_index in range(args.concurrency):
                env = dict(env_base)
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[child_index % len(gpu_ids)]
                envs.append(env)
            results, batch_diagnostics = _run_process_batch(commands, envs)
            for result in results:
                result["batch"] = batch_index
            all_results.extend(results)
            diagnostics.extend(batch_diagnostics)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    phase_names = sorted(
        {phase for result in all_results if result.get("ok") for phase in (result.get("phases") or {})}
    )
    return {
        "kind": "stages",
        "parameters": {
            "repeat": args.repeat,
            "concurrency": args.concurrency,
            "gpus": args.gpus,
        },
        "summary": {
            phase: _summary([float(result["phases"][phase]) for result in all_results if result.get("ok")])
            for phase in phase_names
        }
        | {
            "child_total_s": _summary([float(result["child_total_s"]) for result in all_results if result.get("ok")]),
            "process_wall_s": _summary(
                [float(result["process_wall_s"]) for result in all_results if result.get("ok")]
            ),
        },
        "results": all_results,
        "diagnostics": diagnostics,
    }


def _run_constructors(args: argparse.Namespace) -> dict[str, Any]:
    work_root = Path(tempfile.mkdtemp(prefix="kernelgym-constructor-bench-", dir="/dev/shm"))
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    try:
        env_base = _base_child_env(work_root, args.spawn_concurrency)
        commands: list[list[str]] = []
        envs: list[dict[str, str]] = []
        for index in range(args.count):
            worker_id = f"bench_constructor_{os.getpid()}_{index}"
            commands.append(
                [
                    sys.executable,
                    __file__,
                    "constructor-child",
                    "--worker-id",
                    worker_id,
                    "--hold-s",
                    str(args.hold_s),
                ]
            )
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % len(gpu_ids)]
            envs.append(env)
        started = time.perf_counter()
        results, diagnostics = _run_process_batch(commands, envs)
        batch_wall_s = time.perf_counter() - started
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    successful = [result for result in results if result.get("ok")]
    return {
        "kind": "constructors",
        "parameters": {
            "count": args.count,
            "spawn_concurrency": args.spawn_concurrency,
            "gpus": args.gpus,
            "hold_s": args.hold_s,
        },
        "batch_wall_s": batch_wall_s,
        "summary": {
            field: _summary([float(result[field]) for result in successful])
            for field in (
                "constructor_s",
                "spawn_slot_wait_s",
                "containment_s",
                "ready_after_containment_s",
                "child_init_s",
                "shutdown_s",
                "process_wall_s",
            )
        },
        "results": results,
        "diagnostics": diagnostics,
    }


async def _pool_child_run(args: argparse.Namespace) -> dict[str, Any]:
    from kernelgym.worker.subprocess_pool import SubprocessWorkerPool

    pool = None
    task_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        init_started = time.perf_counter()
        pool = SubprocessWorkerPool(
            device_id=0,
            pool_size=args.pool_size,
            worker_prefix=args.worker_id,
            max_tasks_per_worker=args.max_tasks_per_worker,
        )
        init_s = time.perf_counter() - init_started
        initial_workers = [
            {
                "worker_id": worker.worker_id,
                "spawn_slot_wait_s": worker.spawn_slot_wait_s,
                "containment_s": worker.containment_elapsed_s,
                "ready_after_containment_s": worker.ready_after_containment_s,
                "child_init_s": worker.child_init_s,
            }
            for worker in pool.workers
        ]
        ready_path = Path(args.barrier_dir) / f"ready-{args.child_index}"
        ready_path.touch()
        go_path = Path(args.barrier_dir) / "go"
        while not go_path.exists():
            await asyncio.sleep(0.05)

        workload_started = time.perf_counter()
        for task_index in range(args.tasks):
            task_started = time.perf_counter()
            result = await pool.execute_task(
                {
                    "task_id": f"bench_task_{args.child_index}_{task_index}",
                    "toolkit": "__kernelgym_spawn_bench_missing_toolkit__",
                    "backend_adapter": "__kernelgym_spawn_bench_missing_backend__",
                },
                timeout=args.task_timeout,
                max_retries=0,
            )
            timing = result.get("pool_timing") or {}
            task_results.append(
                {
                    "task_index": task_index,
                    "success": result.get("success"),
                    "error_type": result.get("error_type"),
                    "wall_s": time.perf_counter() - task_started,
                    **timing,
                }
            )
            if args.task_gap_s:
                await asyncio.sleep(args.task_gap_s)
        workload_s = time.perf_counter() - workload_started
        stats_before_shutdown = pool.get_stats()
        shutdown_started = time.perf_counter()
        shutdown_safe = await pool.shutdown(timeout=120)
        shutdown_s = time.perf_counter() - shutdown_started
        return {
            "kind": "pool_child",
            "ok": shutdown_safe,
            "child_index": args.child_index,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "init_s": init_s,
            "initial_workers": initial_workers,
            "workload_s": workload_s,
            "task_results": task_results,
            "stats_before_shutdown": stats_before_shutdown,
            "shutdown_safe": shutdown_safe,
            "shutdown_s": shutdown_s,
            "outer_total_s": time.perf_counter() - started,
        }
    except BaseException as exc:
        if pool is not None:
            try:
                await pool.shutdown(timeout=120)
            except BaseException:
                pass
        return {
            "kind": "pool_child",
            "ok": False,
            "child_index": args.child_index,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "task_results": task_results,
            "outer_total_s": time.perf_counter() - started,
        }


def _pool_child(args: argparse.Namespace) -> int:
    result = asyncio.run(_pool_child_run(args))
    _emit(result)
    return 0 if result.get("ok") else 1


def _run_pools(args: argparse.Namespace) -> dict[str, Any]:
    work_root = Path(tempfile.mkdtemp(prefix="kernelgym-pool-bench-", dir="/dev/shm"))
    barrier_dir = work_root / "barrier"
    barrier_dir.mkdir(parents=True)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    env_base = _base_child_env(work_root, args.spawn_concurrency)
    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []
    for index in range(args.pool_count):
        commands.append(
            [
                sys.executable,
                __file__,
                "pool-child",
                "--child-index",
                str(index),
                "--worker-id",
                f"bench_pool_{os.getpid()}_{index}",
                "--barrier-dir",
                str(barrier_dir),
                "--pool-size",
                str(args.pool_size),
                "--max-tasks-per-worker",
                str(args.max_tasks_per_worker),
                "--tasks",
                str(args.tasks),
                "--task-timeout",
                str(args.task_timeout),
                "--task-gap-s",
                str(args.task_gap_s),
            ]
        )
        env = dict(env_base)
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % len(gpu_ids)]
        envs.append(env)

    processes: list[tuple[subprocess.Popen[str], float, list[str]]] = []
    batch_started = time.perf_counter()
    try:
        for command, env in zip(commands, envs, strict=True):
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            processes.append((process, time.perf_counter(), command))

        ready_deadline = time.monotonic() + args.ready_timeout
        while len(list(barrier_dir.glob("ready-*"))) < args.pool_count:
            failed = [process.returncode for process, _, _ in processes if process.poll() is not None]
            if failed:
                break
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("pool children did not reach the ready barrier")
            time.sleep(0.1)
        (barrier_dir / "go").touch()

        results: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for process, child_started, command in processes:
            stdout, stderr = process.communicate()
            parsed = _parse_child_result(stdout) or {
                "ok": False,
                "error": "pool child produced no benchmark result",
            }
            parsed["process_wall_s"] = time.perf_counter() - child_started
            parsed["returncode"] = process.returncode
            results.append(parsed)
            diagnostics.append(
                {
                    "command": command,
                    "returncode": process.returncode,
                    "stdout_tail": stdout[-8000:],
                    "stderr_tail": stderr[-24000:],
                }
            )
        batch_wall_s = time.perf_counter() - batch_started
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    task_results = [task for result in results for task in result.get("task_results", [])]
    return {
        "kind": "pools",
        "parameters": {
            "pool_count": args.pool_count,
            "pool_size": args.pool_size,
            "max_tasks_per_worker": args.max_tasks_per_worker,
            "tasks_per_pool": args.tasks,
            "task_gap_s": args.task_gap_s,
            "task_timeout": args.task_timeout,
            "spawn_concurrency": args.spawn_concurrency,
            "gpus": args.gpus,
        },
        "batch_wall_s": batch_wall_s,
        "summary": {
            field: _summary([float(task[field]) for task in task_results if field in task])
            for field in (
                "wall_s",
                "pool_idle_wait_s",
                "pool_execute_s",
                "pool_restart_s",
                "pool_total_s",
            )
        }
        | {
            "pool_init_s": _summary([float(result["init_s"]) for result in results if result.get("ok")]),
            "workload_s": _summary([float(result["workload_s"]) for result in results if result.get("ok")]),
        },
        "results": results,
        "diagnostics": diagnostics,
    }


def _write_output(payload: dict[str, Any], output: str | None) -> None:
    payload["recorded_at_unix_s"] = time.time()
    payload["hostname"] = os.uname().nodename
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stages = subparsers.add_parser("stages")
    stages.add_argument("--repeat", type=int, default=5)
    stages.add_argument("--concurrency", type=int, default=1)
    stages.add_argument("--gpus", default="0,1,2,3")
    stages.add_argument("--output")

    constructors = subparsers.add_parser("constructors")
    constructors.add_argument("--count", type=int, default=8)
    constructors.add_argument("--spawn-concurrency", type=int, required=True)
    constructors.add_argument("--gpus", default="0,1,2,3")
    constructors.add_argument("--hold-s", type=float, default=0.0)
    constructors.add_argument("--output")

    pools = subparsers.add_parser("pools")
    pools.add_argument("--pool-count", type=int, default=4)
    pools.add_argument("--pool-size", type=int, default=2)
    pools.add_argument("--max-tasks-per-worker", type=int, default=1)
    pools.add_argument("--tasks", type=int, default=6)
    pools.add_argument("--task-gap-s", type=float, default=0.0)
    pools.add_argument("--task-timeout", type=int, default=120)
    pools.add_argument("--ready-timeout", type=int, default=600)
    pools.add_argument("--spawn-concurrency", type=int, required=True)
    pools.add_argument("--gpus", default="0,1,2,3")
    pools.add_argument("--output")

    stage_child = subparsers.add_parser("stage-child", help=argparse.SUPPRESS)
    stage_child.set_defaults(internal=True)

    constructor_child = subparsers.add_parser("constructor-child", help=argparse.SUPPRESS)
    constructor_child.add_argument("--worker-id", required=True)
    constructor_child.add_argument("--hold-s", type=float, default=0.0)
    constructor_child.set_defaults(internal=True)

    pool_child = subparsers.add_parser("pool-child", help=argparse.SUPPRESS)
    pool_child.add_argument("--child-index", type=int, required=True)
    pool_child.add_argument("--worker-id", required=True)
    pool_child.add_argument("--barrier-dir", required=True)
    pool_child.add_argument("--pool-size", type=int, required=True)
    pool_child.add_argument("--max-tasks-per-worker", type=int, required=True)
    pool_child.add_argument("--tasks", type=int, required=True)
    pool_child.add_argument("--task-timeout", type=int, required=True)
    pool_child.add_argument("--task-gap-s", type=float, required=True)
    pool_child.set_defaults(internal=True)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "stage-child":
        return _stage_child()
    if args.command == "constructor-child":
        return _constructor_child(args)
    if args.command == "pool-child":
        return _pool_child(args)
    if args.command == "stages":
        payload = _run_stages(args)
    elif args.command == "constructors":
        payload = _run_constructors(args)
    elif args.command == "pools":
        payload = _run_pools(args)
    else:  # pragma: no cover - argparse enforces the command set.
        parser.error(f"unknown command: {args.command}")
    _write_output(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
