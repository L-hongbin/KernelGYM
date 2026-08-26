#!/usr/bin/env python3
"""Inspect or manually clear a KernelGYM GPU safety latch."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
from typing import Any

import redis.asyncio as redis

from kernelgym.config import settings
from kernelgym.utils.gpu_quarantine import clear_gpu_quarantine, read_gpu_quarantine


_UNSAFE_ORPHAN_FAULT_CLASSES = {
    "hard_recovery_failure",
    "pre_fault_reap_failure",
    "unsafe_pool_shutdown",
    "unsafe_process_group_shutdown",
    "worker_topology_corruption",
}


def _decode_hash(data: dict[Any, Any]) -> dict[str, str]:
    return {
        (key.decode(errors="replace") if isinstance(key, bytes) else str(key)): (
            value.decode(errors="replace") if isinstance(value, bytes) else str(value)
        )
        for key, value in data.items()
    }


async def _live_workers_on_device(client: redis.Redis, device: str, hostname: str) -> list[str]:
    live: list[str] = []
    pattern = f"{settings.redis_key_prefix}:worker:*"
    async for raw_key in client.scan_iter(pattern, count=500):
        key = raw_key.decode(errors="replace") if isinstance(raw_key, bytes) else str(raw_key)
        data = _decode_hash(await client.hgetall(key))
        if data.get("device") != device:
            continue
        recorded_host = data.get("hostname", "")
        if hostname and recorded_host and recorded_host != hostname:
            continue
        if data.get("online", "").lower() == "true" or data.get("status", "").lower() == "online":
            live.append(key.rsplit(":worker:", 1)[-1])
    return sorted(live)


async def _live_worker_processes_on_device(client: redis.Redis, device: str, hostname: str) -> list[str]:
    """Return every retained supervisor generation map for the device.

    The session leader can exit while a CUDA-owning descendant remains alive.
    Consequently, ``kill(pid, 0) -> ESRCH`` is not a drain proof.  The monitor
    deletes this map only after it has proven the complete worker session and
    every observed process group absent, so any retained map must block a
    manual quarantine clear.
    """

    live: list[str] = []
    prefix = f"{settings.redis_key_prefix}:worker_process:"
    async for raw_key in client.scan_iter(f"{prefix}*", count=500):
        key = raw_key.decode(errors="replace") if isinstance(raw_key, bytes) else str(raw_key)
        worker_id = key[len(prefix) :] if key.startswith(prefix) else key.rsplit(":", 1)[-1]
        data = _decode_hash(await client.hgetall(key))
        if data.get("device") != device:
            continue
        expected = _decode_hash(await client.hgetall(f"{settings.redis_key_prefix}:expected_worker:{worker_id}"))
        owner = expected.get("hostname", "")
        if hostname and owner and owner != hostname:
            continue
        try:
            pid = int(data.get("pid", ""))
        except (TypeError, ValueError):
            live.append(f"{worker_id}(pid=unverifiable)")
            continue
        if pid <= 0:
            live.append(f"{worker_id}(pid=unverifiable)")
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            live.append(f"{worker_id}(pid={pid},leader-gone,map-retained)")
            continue
        except PermissionError:
            pass
        live.append(f"{worker_id}(pid={pid})")
    return sorted(live)


def _requires_unsafe_orphan_confirmation(record: dict[str, str]) -> bool:
    fault_class = record.get("fault_class", "").lower()
    reason = record.get("reason", "").lower()
    return (
        fault_class in _UNSAFE_ORPHAN_FAULT_CLASSES
        or "reap" in fault_class
        or "unreaped" in reason
        or "could not be confirmed" in reason
    )


def _unsafe_confirmation(hostname: str, device: str) -> str:
    return f"{hostname}/{device}/NO_GPU_PROCESSES"


async def _run(args: argparse.Namespace) -> int:
    client = redis.from_url(settings.redis_url)
    try:
        record = await read_gpu_quarantine(
            client,
            args.worker_id,
            device=args.device,
            hostname=args.hostname,
        )
        if record is None:
            print("No quarantine latch found.")
            return 0
        print(json.dumps(record, indent=2, sort_keys=True))
        if args.command == "inspect":
            return 0

        confirmation = f"{args.hostname}/{args.device}"
        if args.confirm != confirmation:
            print(f"Refusing clear: pass --confirm {confirmation!r} exactly.")
            return 2
        live_workers = await _live_workers_on_device(client, args.device, args.hostname)
        if live_workers:
            print(
                "Refusing clear while matching workers are online: "
                + ", ".join(live_workers)
                + ". Stop them first, then rerun the clear command."
            )
            return 3
        live_processes = await _live_worker_processes_on_device(client, args.device, args.hostname)
        if live_processes:
            print(
                "Refusing clear while supervised worker PIDs remain live or unverifiable: "
                + ", ".join(live_processes)
                + ". Stop and reap them first."
            )
            return 4
        if _requires_unsafe_orphan_confirmation(record):
            unsafe_confirmation = _unsafe_confirmation(args.hostname, args.device)
            if args.confirm_unsafe_orphan != unsafe_confirmation:
                print(
                    "This latch records an unconfirmed CUDA context. After host/GPU process verification, pass "
                    f"--confirm-unsafe-orphan {unsafe_confirmation!r} exactly."
                )
                return 5
        removed = await clear_gpu_quarantine(
            client,
            args.worker_id,
            device=args.device,
            hostname=args.hostname,
        )
        print("Quarantine latch cleared." if removed else "Latch disappeared before clear completed.")
        return 0
    finally:
        await client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "clear"):
        command = subparsers.add_parser(name)
        command.add_argument("--worker-id", required=True)
        command.add_argument("--device", required=True, help="CUDA device such as cuda:0")
        command.add_argument("--hostname", default=socket.gethostname())
        if name == "clear":
            command.add_argument("--confirm", required=True)
            command.add_argument("--confirm-unsafe-orphan", default="")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
