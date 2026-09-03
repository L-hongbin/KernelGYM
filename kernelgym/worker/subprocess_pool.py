"""
Subprocess Worker Pool with CUDA Error Auto-Restart

核心特性：
1. 预先启动一组 worker 进程，复用处理多个任务
2. torch 和 CUDA 只在启动时初始化一次
3. **第一次遇到 CUDA error 时立即关闭 worker 进程**
4. 主进程自动重启新的 worker 进程
5. 大幅降低 spawn 开销（从每任务 2.5s 降至几乎为 0）

Author: KernelGym Team
Date: 2025-10-30
Version: v0.3.3-rc
"""

import json
import math
import os
import re
import fcntl
import stat
import sys
import time
import logging
import traceback
import multiprocessing as mp
import queue
import asyncio
import threading
import uuid
import signal
from contextlib import contextmanager
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass

from kernelgym.utils.core_dumps import (
    CORE_DUMP_DIR_ENV,
    CORE_DUMP_KEEP_ENV,
    prepare_core_dump_dir,
)

logger = logging.getLogger("kernelgym.subprocess_pool")


POOL_HEALTHY = "healthy"
POOL_DEGRADED_CHECK = "degraded_check"
POOL_SUSPECT = "suspect"
POOL_BOOTSTRAP_FAILED = "bootstrap_failed"
POOL_QUARANTINED = "quarantined"

FAULT_NONE = "none"
FAULT_CONTEXT = "context"
FAULT_DEVICE = "device"


def _bounded_env_float(name: str, default: float, *, minimum: float) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _bounded_env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


# Multiprocessing ``spawn`` imports the outer process's main module before the
# target function can attest its PGID.  Under a synchronized eight-GPU recycle
# wave that import/bootstrap phase exceeded the former shared 120-second
# CONTAINED+READY deadline.  Bound node-local constructor concurrency and give
# containment and post-containment CUDA initialization independent clocks.
_WORKER_SPAWN_CONCURRENCY = _bounded_env_int("KERNELGYM_WORKER_SPAWN_CONCURRENCY", 8, minimum=1)
_WORKER_SPAWN_SLOT_TIMEOUT_S = _bounded_env_float("KERNELGYM_WORKER_SPAWN_SLOT_TIMEOUT", 600.0, minimum=30.0)
_WORKER_CONTAINMENT_TIMEOUT_S = _bounded_env_float("KERNELGYM_WORKER_CONTAINMENT_TIMEOUT", 180.0, minimum=10.0)
_WORKER_READY_AFTER_CONTAINMENT_TIMEOUT_S = _bounded_env_float("KERNELGYM_WORKER_READY_TIMEOUT", 90.0, minimum=10.0)
_WORKER_SPAWN_SLOT_YIELD_S = 0.05
_PARENT_CONTAINMENT_ACK_TIMEOUT_S = 30.0
_REPLENISHMENT_DRAIN_TIMEOUT_S = (
    _WORKER_SPAWN_SLOT_TIMEOUT_S + _WORKER_CONTAINMENT_TIMEOUT_S + _WORKER_READY_AFTER_CONTAINMENT_TIMEOUT_S + 30.0
)
_MAX_REPLACEMENT_INFRA_FAILURES = 3
_REPLACEMENT_RETRY_BASE_DELAY_S = 5.0
_STALE_HARD_RECOVERY_EPOCH = -1
_RESULT_PROTOCOL_MAGIC = "kernelgym-result-json-v1"
_PARENT_CONTAINMENT_ACK = "kernelgym-parent-containment-ack-v1"
_MAX_RESULT_MESSAGE_BYTES = 64 * 1024 * 1024


# A failed ``shutdown()`` must not also discard the only safe Process handle
# for the CUDA-owning child.  Keep strong references at process scope so that
# later recovery/shutdown passes can retry the reap even if the original pool
# has already removed the worker from its canonical lists.
_UNREAPED_WORKER_HANDLES_LOCK = threading.Lock()
_UNREAPED_WORKER_HANDLES: Dict[int, Dict[int, Any]] = {}
_ACTIVE_WORKER_IDENTITIES_LOCK = threading.Lock()
# Both registries are keyed by the outer manager SID, not by CUDA device.
# Legacy WorkerManager mode hosts several devices in one session, while the
# service-managed mode has one device per session.  The containment boundary
# is the SID in both cases.
_ACTIVE_WORKER_IDENTITIES: Dict[int, Dict[int, "_LinuxProcessIdentity"]] = {}
_STARTING_WORKER_IDENTITIES: Dict[int, Dict[int, "_LinuxProcessIdentity"]] = {}


@dataclass(frozen=True)
class _LinuxProcessIdentity:
    """Generation-safe process identity read from ``/proc/<pid>/stat``."""

    pid: int
    start_ticks: int
    ppid: int
    pgid: int
    sid: int
    state: str


class _ProcessContainmentError(RuntimeError):
    """Linux could not prove that a CUDA-owning process tree was contained."""


class WorkerIPCProtocolError(RuntimeError):
    """The CUDA subprocess violated the bounded JSON result protocol."""


class WorkerResultChannelClosed(WorkerIPCProtocolError):
    """The child result pipe reached EOF before its expected message."""


def _validate_exact_json_primitives(value: Any, *, max_depth: int = 32, max_nodes: int = 200_000) -> None:
    """Reject candidate-controlled objects without invoking conversion hooks."""

    remaining = [max_nodes]
    active_containers: set[int] = set()

    def _visit(item: Any, depth: int) -> None:
        remaining[0] -= 1
        if remaining[0] < 0:
            raise WorkerIPCProtocolError("result payload exceeds JSON node limit")
        if depth > max_depth:
            raise WorkerIPCProtocolError("result payload exceeds JSON nesting limit")

        item_type = type(item)
        if item is None or item_type is bool or item_type is str or item_type is int:
            return
        if item_type is float:
            if not math.isfinite(item):
                raise WorkerIPCProtocolError("result payload contains a non-finite float")
            return
        if item_type is list:
            identity = id(item)
            if identity in active_containers:
                raise WorkerIPCProtocolError("result payload contains a container cycle")
            active_containers.add(identity)
            try:
                for child in item:
                    _visit(child, depth + 1)
            finally:
                active_containers.remove(identity)
            return
        if item_type is dict:
            identity = id(item)
            if identity in active_containers:
                raise WorkerIPCProtocolError("result payload contains a container cycle")
            active_containers.add(identity)
            try:
                for key, child in item.items():
                    if type(key) is not str:
                        raise WorkerIPCProtocolError("result payload contains a non-string mapping key")
                    _visit(child, depth + 1)
            finally:
                active_containers.remove(identity)
            return
        # Do not call str(), repr(), iteration, dataclass conversion, or a JSON
        # ``default`` hook here.  An arbitrary object never crosses the pipe.
        raise WorkerIPCProtocolError("result payload contains a non-primitive value")

    _visit(value, 0)


def _validate_result_message_schema(kind: str, payload: Dict[str, Any]) -> None:
    """Validate the fixed wire-message variants on both sides of the pipe."""

    if kind == "contained":
        valid = payload.get("status") == "CONTAINED" and all(
            type(payload.get(field)) is int for field in ("pid", "start_ticks", "pgid", "sid")
        )
    elif kind == "ready":
        valid = (
            payload.get("status") == "READY"
            and type(payload.get("init_time")) in {int, float}
            and type(payload.get("device")) is str
            and all(type(payload.get(field)) is int for field in ("pid", "start_ticks", "pgid", "sid"))
        )
    elif kind == "init_failed":
        valid = payload.get("status") == "INIT_FAILED" and all(
            type(payload.get(field)) is str for field in ("init_stage", "error", "traceback")
        )
    elif kind == "task_result":
        success = payload.get("success")
        valid = type(success) is bool
        if valid and success:
            valid = type(payload.get("result")) is dict and type(payload.get("worker_exiting")) is bool
        elif valid:
            valid = all(type(payload.get(field)) is str for field in ("error_type", "error_message"))
    else:
        valid = False
    if not valid:
        raise WorkerIPCProtocolError(f"invalid {kind or 'unknown'} result-message schema")


class _ChildJSONResultChannel:
    """Write-only child result transport that never pickles candidate objects.

    ``multiprocessing.Queue`` pickles arbitrary payload values.  A toolkit
    result containing a hostile ``__reduce__`` object would therefore execute
    in the parent while it dequeued the result.  This channel converts the
    complete payload to JSON bytes in the child and sends only raw bytes over a
    one-way ``multiprocessing.Connection``.
    """

    def __init__(self, connection: Any, max_message_bytes: int = _MAX_RESULT_MESSAGE_BYTES) -> None:
        self._connection = connection
        self._max_message_bytes = max_message_bytes

    def send(self, kind: str, payload: Dict[str, Any]) -> None:
        if type(kind) is not str:
            raise WorkerIPCProtocolError("result message kind must be a string")
        if type(payload) is not dict:
            raise WorkerIPCProtocolError("result payload must be an exact dict")
        _validate_exact_json_primitives(payload)
        _validate_result_message_schema(kind, payload)
        try:
            encoded = json.dumps(
                [_RESULT_PROTOCOL_MAGIC, kind, payload],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise WorkerIPCProtocolError(f"result payload is not JSON-safe: {exc}") from exc
        if len(encoded) > self._max_message_bytes:
            raise WorkerIPCProtocolError(f"encoded result exceeds {self._max_message_bytes} byte safety limit")
        self._connection.send_bytes(encoded)

    def put(self, payload: Dict[str, Any]) -> None:
        """Compatibility shim for trusted tests/helpers sending task results."""

        self.send("task_result", payload)

    def close(self) -> None:
        self._connection.close()


class _ParentJSONResultChannel:
    """Read-only parent result transport; decoding performs no pickle load."""

    def __init__(self, connection: Any, max_message_bytes: int = _MAX_RESULT_MESSAGE_BYTES) -> None:
        self._connection = connection
        self._max_message_bytes = max_message_bytes

    def get(
        self,
        timeout: Optional[float] = None,
        *,
        expected_kinds: frozenset[str],
    ) -> Dict[str, Any]:
        try:
            ready = self._connection.poll(timeout)
        except OSError as exc:
            raise WorkerIPCProtocolError(f"could not poll bounded child result: {exc}") from exc
        if not ready:
            raise queue.Empty
        try:
            encoded = self._connection.recv_bytes(self._max_message_bytes)
        except EOFError as exc:
            raise WorkerResultChannelClosed("child result channel closed before the expected message") from exc
        except OSError as exc:
            raise WorkerIPCProtocolError(f"could not receive bounded child result: {exc}") from exc
        try:
            envelope = json.loads(encoded.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise WorkerIPCProtocolError(f"child result was not valid UTF-8 JSON: {exc}") from exc
        if (
            type(envelope) is not list
            or len(envelope) != 3
            or envelope[0] != _RESULT_PROTOCOL_MAGIC
            or type(envelope[1]) is not str
            or type(envelope[2]) is not dict
        ):
            raise WorkerIPCProtocolError("child result used an invalid protocol envelope")
        kind = envelope[1]
        payload = envelope[2]
        if kind not in expected_kinds:
            expected = ", ".join(sorted(expected_kinds))
            raise WorkerIPCProtocolError(f"unexpected child result kind {kind!r}; expected one of: {expected}")
        _validate_exact_json_primitives(payload)
        _validate_result_message_schema(kind, payload)
        return payload

    def close(self) -> None:
        self._connection.close()


def _receive_worker_message(
    channel: Any,
    *,
    timeout: Optional[float],
    expected_kinds: frozenset[str],
) -> Dict[str, Any]:
    """Read one typed production message while retaining lightweight test fakes."""

    if type(channel) is _ParentJSONResultChannel:
        return channel.get(timeout=timeout, expected_kinds=expected_kinds)
    # Tests and downstream compatibility shims may still expose Queue-like
    # objects.  Production cannot take this branch because its channel type is
    # created privately by ``PersistentWorker``.
    return channel.get(timeout=timeout)


def _read_linux_process_identity(pid: int) -> Optional[_LinuxProcessIdentity]:
    """Return one PID generation, ``None`` only when that PID is gone."""

    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
            raw = stat_file.read().strip()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise _ProcessContainmentError(f"cannot read /proc/{pid}/stat: {exc}") from exc

    # comm is parenthesized and may itself contain spaces or parentheses.  The
    # final ')' is the only reliable delimiter before state (field 3).
    delimiter = raw.rfind(")")
    if delimiter < 0:
        raise _ProcessContainmentError(f"malformed /proc/{pid}/stat")
    fields = raw[delimiter + 2 :].split()
    if len(fields) < 20:
        raise _ProcessContainmentError(f"short /proc/{pid}/stat")
    try:
        return _LinuxProcessIdentity(
            pid=pid,
            state=fields[0],
            ppid=int(fields[1]),
            pgid=int(fields[2]),
            sid=int(fields[3]),
            start_ticks=int(fields[19]),
        )
    except (TypeError, ValueError) as exc:
        raise _ProcessContainmentError(f"invalid /proc/{pid}/stat fields") from exc


def _snapshot_linux_processes() -> Dict[int, _LinuxProcessIdentity]:
    """Take a fail-closed process-table snapshot for descendant discovery."""

    try:
        proc_entries = os.listdir("/proc")
    except OSError as exc:
        raise _ProcessContainmentError(f"cannot enumerate /proc: {exc}") from exc

    snapshot: Dict[int, _LinuxProcessIdentity] = {}
    for entry in proc_entries:
        if not entry.isdigit():
            continue
        identity = _read_linux_process_identity(int(entry))
        if identity is not None:
            snapshot[identity.pid] = identity
    return snapshot


def _register_starting_worker_identity(identity: _LinuxProcessIdentity) -> None:
    with _ACTIVE_WORKER_IDENTITIES_LOCK:
        _STARTING_WORKER_IDENTITIES.setdefault(identity.sid, {})[identity.pid] = identity


def _promote_active_worker_identity(
    starting_identity: _LinuxProcessIdentity,
    confirmed_identity: _LinuxProcessIdentity,
) -> None:
    """Atomically replace an exact-PID startup allowance with a PGID allowance."""

    if (
        starting_identity.pid != confirmed_identity.pid
        or starting_identity.start_ticks != confirmed_identity.start_ticks
        or starting_identity.sid != confirmed_identity.sid
        or confirmed_identity.pgid != confirmed_identity.pid
    ):
        raise _ProcessContainmentError("worker identity changed before containment promotion")
    with _ACTIVE_WORKER_IDENTITIES_LOCK:
        starting = _STARTING_WORKER_IDENTITIES.get(starting_identity.sid)
        registered = starting.get(starting_identity.pid) if starting is not None else None
        if registered is None or registered.start_ticks != starting_identity.start_ticks:
            raise _ProcessContainmentError("worker startup identity was not registered")
        starting.pop(starting_identity.pid, None)
        if not starting:
            _STARTING_WORKER_IDENTITIES.pop(starting_identity.sid, None)
        _ACTIVE_WORKER_IDENTITIES.setdefault(confirmed_identity.sid, {})[confirmed_identity.pid] = confirmed_identity


def _unregister_worker_identity(identity: _LinuxProcessIdentity) -> None:
    with _ACTIVE_WORKER_IDENTITIES_LOCK:
        for registry in (_STARTING_WORKER_IDENTITIES, _ACTIVE_WORKER_IDENTITIES):
            session_identities = registry.get(identity.sid)
            if session_identities is None:
                continue
            registered = session_identities.get(identity.pid)
            if registered is not None and registered.start_ticks == identity.start_ticks:
                session_identities.pop(identity.pid, None)
            if not session_identities:
                registry.pop(identity.sid, None)


def _active_worker_identities(session_id: int) -> list[_LinuxProcessIdentity]:
    with _ACTIVE_WORKER_IDENTITIES_LOCK:
        return list(_ACTIVE_WORKER_IDENTITIES.get(session_id, {}).values())


def _starting_worker_identities(session_id: int) -> list[_LinuxProcessIdentity]:
    with _ACTIVE_WORKER_IDENTITIES_LOCK:
        return list(_STARTING_WORKER_IDENTITIES.get(session_id, {}).values())


def _multiprocessing_resource_tracker_pid() -> Optional[int]:
    try:
        from multiprocessing import resource_tracker

        pid = getattr(resource_tracker._resource_tracker, "_pid", None)
    except (AttributeError, ImportError):
        return None
    return pid if isinstance(pid, int) and pid > 0 else None


def _worker_session_is_contained_after_leader_exit(
    leader: _LinuxProcessIdentity,
) -> bool:
    """Prove that a crashed worker left no unknown process in its outer SID.

    Inner warm workers use independent PGIDs inside one GPU manager session.
    An authenticated sibling and its observable PPid-descendant closure are
    allowed, as are the exact outer manager and multiprocessing resource
    tracker. PGID equality alone is never provenance: Linux can reuse a PID
    while an orphaned group with the same numeric PGID still has members. Any
    other session member may have escaped the crashed leader, so recovery must
    fail closed before opening a fresh CUDA context.
    """

    snapshot = _snapshot_linux_processes()
    outer_pid = os.getpid()
    outer = snapshot.get(outer_pid)
    if outer is None or outer.sid != leader.sid:
        return False

    allowed_pids = {outer_pid}
    tracker_pid = _multiprocessing_resource_tracker_pid()
    if tracker_pid is not None:
        tracker = snapshot.get(tracker_pid)
        if (
            tracker is not None
            and tracker.ppid == outer_pid
            and tracker.pgid == outer.pgid
            and tracker.sid == leader.sid
        ):
            allowed_pids.add(tracker_pid)

    # Before the child proves its dedicated PGID it may be admitted only as an
    # exact PID generation.  The child waits for the parent's promotion ACK
    # before importing Torch/CUDA, so it cannot create a CUDA descendant in
    # this provisional state.
    allowed_starting_pids = {
        registered.pid
        for registered in _starting_worker_identities(leader.sid)
        if (registered.pid != leader.pid or registered.start_ticks != leader.start_ticks)
        and _same_process_generation(registered, snapshot.get(registered.pid))
        and snapshot[registered.pid].sid == leader.sid
    }

    allowed_active_pids: set[int] = set()
    for registered in _active_worker_identities(leader.sid):
        observed = snapshot.get(registered.pid)
        if (
            (registered.pid != leader.pid or registered.start_ticks != leader.start_ticks)
            and _same_process_generation(registered, observed)
            and observed.sid == leader.sid
            and observed.pgid == registered.pgid
        ):
            allowed_active_pids.add(observed.pid)
            allowed_active_pids.update(
                descendant.pid
                for descendant in _descendants_from_snapshot(snapshot, observed.pid).values()
                if descendant.sid == leader.sid
            )

    for member in snapshot.values():
        if (
            member.sid != leader.sid
            or member.pid in allowed_pids
            or member.pid in allowed_starting_pids
            or member.pid in allowed_active_pids
        ):
            continue
        return False
    return True


def _wait_for_worker_session_containment(
    leader: _LinuxProcessIdentity,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + max(0.1, timeout)
    while time.monotonic() < deadline:
        try:
            if _worker_session_is_contained_after_leader_exit(leader):
                return True
        except _ProcessContainmentError:
            pass
        time.sleep(0.02)
    try:
        return _worker_session_is_contained_after_leader_exit(leader)
    except _ProcessContainmentError:
        return False


def _registered_worker_owns_reused_process_group(leader: _LinuxProcessIdentity) -> bool:
    """Return whether ``leader.pgid`` now belongs to another attested generation."""

    for registered in _active_worker_identities(leader.sid):
        if registered.pid != leader.pgid or registered.start_ticks == leader.start_ticks:
            continue
        observed = _read_linux_process_identity(registered.pid)
        if (
            _same_process_generation(registered, observed)
            and observed.pgid == registered.pid
            and observed.sid == leader.sid
        ):
            return True
    return False


def _wait_for_process_group_drain_or_registered_reuse(
    leader: _LinuxProcessIdentity,
    timeout: float,
) -> bool:
    """Prove the old PGID absent, or prove its number belongs to a new worker."""

    deadline = time.monotonic() + max(0.1, timeout)
    while time.monotonic() < deadline:
        try:
            if _process_group_is_drained(leader.pgid) or _registered_worker_owns_reused_process_group(leader):
                return True
        except _ProcessContainmentError:
            pass
        time.sleep(0.02)
    try:
        return _process_group_is_drained(leader.pgid) or _registered_worker_owns_reused_process_group(leader)
    except _ProcessContainmentError:
        return False


def _same_process_generation(
    expected: _LinuxProcessIdentity,
    observed: Optional[_LinuxProcessIdentity],
) -> bool:
    return observed is not None and observed.start_ticks == expected.start_ticks


def _signal_process_generation(identity: _LinuxProcessIdentity, signum: int) -> bool:
    """Signal exactly one captured PID generation, never a reused PID."""

    current = _read_linux_process_identity(identity.pid)
    if not _same_process_generation(identity, current):
        return True
    try:
        os.kill(identity.pid, signum)
    except ProcessLookupError:
        return not _same_process_generation(identity, _read_linux_process_identity(identity.pid))
    except OSError as exc:
        logger.error(f"Could not signal pid={identity.pid} generation={identity.start_ticks}: {exc}")
        return False
    return True


def _descendants_from_snapshot(
    snapshot: Dict[int, _LinuxProcessIdentity],
    root_pid: int,
) -> Dict[int, _LinuxProcessIdentity]:
    """Return the transitive PPid closure rooted at ``root_pid``."""

    descendants: Dict[int, _LinuxProcessIdentity] = {}
    frontier = {root_pid}
    while frontier:
        children = {
            pid: identity
            for pid, identity in snapshot.items()
            if identity.ppid in frontier and pid not in descendants and pid != root_pid
        }
        if not children:
            break
        descendants.update(children)
        frontier = set(children)
    return descendants


def _process_group_is_drained(pgid: int) -> bool:
    """Only kernel ESRCH proves that no process remains in this group."""

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _wait_for_process_group_drain(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.1, timeout)
    while time.monotonic() < deadline:
        if _process_group_is_drained(pgid):
            return True
        time.sleep(0.02)
    return _process_group_is_drained(pgid)


def _freeze_worker_process_tree(
    leader: _LinuxProcessIdentity,
    *,
    timeout: float,
) -> tuple[Dict[int, _LinuxProcessIdentity], bool]:
    """Freeze the dedicated worker PGID and its observable PPid descendants.

    ``killpg(SIGSTOP)`` closes the ordinary fork race for all group members.
    A recursive PID-generation scan additionally catches descendants that
    changed process group but remain in the leader's ancestry.  Any identity,
    enumeration, or stabilization uncertainty fails closed.
    """

    deadline = time.monotonic() + max(0.1, timeout)
    known: Dict[int, _LinuxProcessIdentity] = {leader.pid: leader}
    stable_snapshot: Optional[tuple[tuple[int, int], ...]] = None

    while time.monotonic() < deadline:
        current_leader = _read_linux_process_identity(leader.pid)
        leader_present = _same_process_generation(leader, current_leader)
        if leader_present:
            if current_leader.pgid != leader.pgid or current_leader.sid != leader.sid:
                return known, False
        elif stable_snapshot is None:
            # If the leader escaped before the first freeze, PPid ancestry may
            # already have been reparented and absence cannot be proven.
            return known, False

        try:
            os.killpg(leader.pgid, signal.SIGSTOP)
        except ProcessLookupError:
            if leader_present:
                return known, False
        except OSError:
            return known, False

        snapshot = _snapshot_linux_processes()
        group_members = {pid: item for pid, item in snapshot.items() if item.pgid == leader.pgid}
        if any(item.sid != leader.sid for item in group_members.values()):
            return known, False
        descendants = _descendants_from_snapshot(snapshot, leader.pid) if leader_present else {}
        candidates = {**group_members, **descendants}
        for pid, identity in candidates.items():
            previous = known.get(pid)
            if previous is not None and previous.start_ticks != identity.start_ticks:
                return known, False
            known[pid] = identity
            if identity.state not in {"T", "t", "Z", "X", "x"}:
                if not _signal_process_generation(identity, signal.SIGSTOP):
                    return known, False

        # Re-read every captured generation after signalling.  Generation
        # disappearance is safe; every survivor must be stopped or a zombie.
        survivors: Dict[int, _LinuxProcessIdentity] = {}
        all_frozen = True
        for pid, identity in known.items():
            observed = _read_linux_process_identity(pid)
            if not _same_process_generation(identity, observed):
                continue
            survivors[pid] = observed
            if observed.state not in {"T", "t", "Z", "X", "x"}:
                all_frozen = False
        known.update(survivors)
        marker = tuple(sorted((pid, item.start_ticks) for pid, item in survivors.items()))
        if all_frozen and marker == stable_snapshot:
            return known, True
        stable_snapshot = marker if all_frozen else None
        time.sleep(0.01)

    return known, False


def _kill_and_verify_worker_process_tree(
    leader: _LinuxProcessIdentity,
    *,
    freeze_timeout: float,
    reap_timeout: float,
) -> bool:
    """Freeze, kill, and prove a worker PGID/known descendants are gone."""

    try:
        identities, frozen = _freeze_worker_process_tree(leader, timeout=freeze_timeout)
        if not frozen:
            return False
        try:
            os.killpg(leader.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            return False

        # Individually kill any descendant that changed PGID before the tree
        # was frozen.  The generation check prevents signalling PID reuse.
        for identity in identities.values():
            if identity.pgid != leader.pgid and not _signal_process_generation(identity, signal.SIGKILL):
                return False

        deadline = time.monotonic() + max(0.1, reap_timeout)
        while time.monotonic() < deadline:
            generations_gone = True
            for identity in identities.values():
                observed = _read_linux_process_identity(identity.pid)
                if _same_process_generation(identity, observed) and observed.state not in {"Z", "X", "x"}:
                    generations_gone = False
                    break
            # The multiprocessing leader remains a zombie until its owning
            # Process.join() runs in the caller, so ESRCH cannot be required
            # here.  Prove that every remaining group member is already a
            # zombie; the caller joins the leader and then requires group
            # ESRCH before reporting success.
            snapshot = _snapshot_linux_processes()
            group_quiescent = all(
                item.state in {"Z", "X", "x"} for item in snapshot.values() if item.pgid == leader.pgid
            )
            if generations_gone and group_quiescent:
                return True
            time.sleep(0.02)
    except _ProcessContainmentError as exc:
        logger.error(f"Worker process-tree containment proof failed: {exc}")
    return False


def _retain_unreaped_worker(device_id: int, worker: Any) -> None:
    with _UNREAPED_WORKER_HANDLES_LOCK:
        _UNREAPED_WORKER_HANDLES.setdefault(device_id, {})[id(worker)] = worker


def _release_reaped_worker(device_id: int, worker: Any) -> None:
    with _UNREAPED_WORKER_HANDLES_LOCK:
        device_workers = _UNREAPED_WORKER_HANDLES.get(device_id)
        if device_workers is None:
            return
        device_workers.pop(id(worker), None)
        if not device_workers:
            _UNREAPED_WORKER_HANDLES.pop(device_id, None)


def _snapshot_unreaped_workers(device_id: int) -> List[Any]:
    with _UNREAPED_WORKER_HANDLES_LOCK:
        return list(_UNREAPED_WORKER_HANDLES.get(device_id, {}).values())


def _worker_handle_is_retained(device_id: int, worker: Any) -> bool:
    with _UNREAPED_WORKER_HANDLES_LOCK:
        return id(worker) in _UNREAPED_WORKER_HANDLES.get(device_id, {})


def _record_worker_reap_result(device_id: int, worker: Any, result: Any) -> bool:
    reaped = not isinstance(result, BaseException) and result is True
    if reaped:
        _release_reaped_worker(device_id, worker)
    else:
        _retain_unreaped_worker(device_id, worker)
    return reaped


class CudaFinalSyncError(RuntimeError):
    """CUDA work failed at the task commit barrier.

    CUDA launches are asynchronous, so a toolkit may return a nominal result
    before an illegal access or device-side assert reaches the CPU.  This
    exception makes that final synchronization failure explicit and prevents
    the result from being published as successful.
    """


class GPUQuarantinedError(RuntimeError):
    """Raised when a physical GPU has been fail-closed by the pool."""


class UnsafeGPUContainmentError(Exception):
    """A task's CUDA context could not be proven gone.

    This is deliberately not a ``RuntimeError``: callers must preserve and
    freeze the exact inflight claim instead of routing it through ordinary
    task retry/failure publication.
    """

    def __init__(self, message: str, *, task_id: str = "", worker_id: str = "") -> None:
        super().__init__(message)
        self.task_id = task_id
        self.worker_id = worker_id


class PoolShutdownContainmentError(Exception):
    """The enclosing GPU worker shutdown safely took ownership of a task.

    This control-flow signal must never be converted into a terminal task
    result by the still-unwinding execution coroutine. GPUWorker.stop() owns
    the frozen claim and may finalize it only after pool shutdown succeeds.
    """


class WorkerInitializationError(RuntimeError):
    """A child failed before its READY handshake."""

    CUDA_PROBE_STAGES = frozenset(
        {
            "cuda_init",
            "cuda_set_device",
            "cuda_alloc",
            "cuda_sync",
            "cuda_sync_capture",
            "cuda_identity",
        }
    )

    def __init__(
        self,
        message: str,
        *,
        init_stage: str = "unknown",
        reap_confirmed: bool = True,
    ) -> None:
        super().__init__(message)
        self.init_stage = init_stage
        self.reap_confirmed = reap_confirmed
        self.cuda_probe_failure = init_stage in self.CUDA_PROBE_STAGES or not reap_confirmed


class GPUProbeFailedError(RuntimeError):
    """A real fresh CUDA context/alloc/sync probe failed."""


class WorkerPoolInfrastructureError(RuntimeError):
    """Pool bootstrap failed outside the CUDA health probe."""


def _worker_spawn_lock_dir() -> str:
    configured = os.environ.get("KERNELGYM_WORKER_SPAWN_LOCK_DIR", "").strip()
    if configured:
        if not os.path.isabs(configured):
            raise WorkerInitializationError(
                "KERNELGYM_WORKER_SPAWN_LOCK_DIR must be absolute",
                init_stage="spawn_throttle",
            )
        return configured
    return f"/dev/shm/kernelgym-worker-spawn-{os.getuid()}"


def _prepare_worker_spawn_lock_dir(path: str) -> None:
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
        directory = os.lstat(path)
    except OSError as exc:
        raise WorkerInitializationError(
            f"could not prepare worker spawn lock directory {path}: {exc}",
            init_stage="spawn_throttle",
        ) from exc
    if not stat.S_ISDIR(directory.st_mode) or stat.S_ISLNK(directory.st_mode):
        raise WorkerInitializationError(
            f"worker spawn lock path is not a real directory: {path}",
            init_stage="spawn_throttle",
        )
    if directory.st_uid != os.getuid() or stat.S_IMODE(directory.st_mode) & 0o077:
        raise WorkerInitializationError(
            f"worker spawn lock directory ownership/mode is unsafe: {path}",
            init_stage="spawn_throttle",
        )


@contextmanager
def _host_worker_spawn_slot(worker_id: str):
    """Limit expensive spawn/import/CUDA handshakes across local GPU workers.

    Each slot is a persistent inode in a private directory.  The descriptor is
    CLOEXEC so a spawned CUDA child cannot retain the parent's flock for its
    lifetime.  Lock files are deliberately never unlinked: replacing the inode
    would let another process bypass an existing lock.
    """

    lock_dir = _worker_spawn_lock_dir()
    _prepare_worker_spawn_lock_dir(lock_dir)
    deadline = time.monotonic() + _WORKER_SPAWN_SLOT_TIMEOUT_S
    slot_count = _WORKER_SPAWN_CONCURRENCY
    first_slot = (os.getpid() + threading.get_native_id()) % slot_count
    selected_fd: Optional[int] = None
    selected_slot = -1
    started_waiting = time.monotonic()

    while selected_fd is None:
        for offset in range(slot_count):
            slot = (first_slot + offset) % slot_count
            path = os.path.join(lock_dir, f"slot-{slot}.lock")
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd: Optional[int] = None
            try:
                fd = os.open(path, flags, 0o600)
                os.set_inheritable(fd, False)
                lock_stat = os.fstat(fd)
                if (
                    not stat.S_ISREG(lock_stat.st_mode)
                    or lock_stat.st_nlink < 1
                    or lock_stat.st_uid != os.getuid()
                    or stat.S_IMODE(lock_stat.st_mode) & 0o077
                ):
                    raise OSError(f"unsafe spawn slot inode: {path}")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if fd is not None:
                    os.close(fd)
                continue
            except OSError as exc:
                if fd is not None:
                    os.close(fd)
                raise WorkerInitializationError(
                    f"could not authenticate worker spawn slot {path}: {exc}",
                    init_stage="spawn_throttle",
                ) from exc
            selected_fd = fd
            selected_slot = slot
            break

        if selected_fd is not None:
            break
        if time.monotonic() >= deadline:
            raise WorkerInitializationError(
                f"[{worker_id}] timed out after {_WORKER_SPAWN_SLOT_TIMEOUT_S:.0f}s "
                f"waiting for one of {slot_count} host worker spawn slots",
                init_stage="spawn_throttle",
            )
        time.sleep(0.1)

    wait_s = time.monotonic() - started_waiting
    logger.info(
        "[%s] Acquired host worker spawn slot %s/%s after %.2fs",
        worker_id,
        selected_slot + 1,
        slot_count,
        wait_s,
    )
    try:
        yield wait_s
    finally:
        try:
            fcntl.flock(selected_fd, fcntl.LOCK_UN)
        finally:
            os.close(selected_fd)
        # Let another waiting outer worker win before this process constructs
        # its next pool member.
        time.sleep(_WORKER_SPAWN_SLOT_YIELD_S)


_CUDA_CONTEXT_FAULT_MARKERS = (
    "illegal memory access",
    "device-side assert",
    "misaligned address",
    "illegal instruction",
    "invalid pc",
    "hardware stack error",
    "launch failure",
    "unspecified launch failure",
)

_CUDA_DEVICE_FAULT_MARKERS = (
    "cuda_error_not_initialized",
    "cuda_error_unknown",
    "cuda error: unknown",
    "cuda driver initialization failed",
    "driver initialization failed",
    "initialization error",
    "device lost",
    "system not ready",
    "system not yet initialized",
    "fallen off the bus",
    "gpu has fallen off",
    "uncorrectable ecc",
    "all cuda-capable devices are busy or unavailable",
    "context is destroyed",
    "context destroyed",
    "launch timeout",
    "launch timed out",
    "forward compatibility was attempted",
)


_CUDA_BENIGN_EXIT_MARKERS = (
    "out of memory",
    "profiler_no_cuda_events",
    "nvcc",
    "compilation failed",
    "compilation error",
    "compile error",
)


def _classify_cuda_fault(
    error_type: str,
    error_message: str,
    *,
    final_sync_failed: bool = False,
    is_cuda_error: Optional[bool] = None,
) -> str:
    """Classify faults that require context recycle or physical-GPU gating."""

    text = f"{error_type} {error_message}".lower()
    if error_type in {"TimeoutError", "WorkerProcessCrashed", "WorkerIPCProtocolError"}:
        return FAULT_DEVICE
    if any(marker in text for marker in _CUDA_DEVICE_FAULT_MARKERS) or re.search(r"\bxid(?:\s*[:(]|\s+\d)", text):
        return FAULT_DEVICE
    if final_sync_failed or any(marker in text for marker in _CUDA_CONTEXT_FAULT_MARKERS):
        return FAULT_CONTEXT
    if any(marker in text for marker in _CUDA_BENIGN_EXIT_MARKERS):
        return FAULT_NONE
    if is_cuda_error is None:
        is_cuda_error = "cuda" in text or error_type.lower() == "cudaerror"
    if is_cuda_error:
        # Unknown CUDA runtime failures are not safe to treat as ordinary
        # Python exceptions.  Default to context isolation; explicit device
        # markers above take the stronger path.
        return FAULT_CONTEXT
    return FAULT_NONE


def _strongest_cuda_fault(*severities: Any) -> str:
    """Return the strongest valid child/parent fault classification."""

    ranking = {FAULT_NONE: 0, FAULT_CONTEXT: 1, FAULT_DEVICE: 2}
    return max(
        (severity if severity in ranking else FAULT_NONE for severity in severities),
        key=ranking.__getitem__,
    )


def _capture_trusted_cuda_task_barrier(torch_module: Any, device_id: int) -> Callable[[], None]:
    """Capture immutable references to PyTorch's low-level CUDA barrier.

    Candidate/toolkit Python executes in this same long-lived interpreter and
    can replace attributes such as ``torch.cuda.synchronize`` or even the
    module attribute ``torch._C._cuda_synchronize``.  Looking either attribute
    up after evaluation would therefore let untrusted code turn the commit
    barrier into a no-op.  Capture both C callables before READY/task dispatch
    and close over the objects themselves.
    """

    cuda_c_extension = getattr(torch_module, "_C", None)
    set_device = getattr(cuda_c_extension, "_cuda_setDevice", None)
    synchronize = getattr(cuda_c_extension, "_cuda_synchronize", None)
    if not callable(set_device) or not callable(synchronize):
        raise WorkerInitializationError(
            "PyTorch low-level CUDA synchronization entrypoints are unavailable",
            init_stage="cuda_sync_capture",
        )

    def _trusted_barrier() -> None:
        # A task may change the current CUDA device.  The captured C setter
        # restores the assigned device without consulting monkeypatchable
        # ``torch.cuda`` Python attributes.
        set_device(device_id)
        synchronize()

    return _trusted_barrier


def _strict_cuda_task_barrier(synchronize_cuda: Callable[[], None]) -> None:
    """Wait for all task CUDA work before its result is made visible."""

    try:
        synchronize_cuda()
    except Exception as exc:
        raise CudaFinalSyncError(f"CUDA final synchronize failed: {exc}") from exc


def _synchronize_and_classify_task_error(
    synchronize_cuda: Callable[[], None],
    task_error: BaseException,
) -> Dict[str, Any]:
    """Expose delayed CUDA faults before publishing any task exception.

    A Python/toolkit exception does not prove that earlier asynchronous CUDA
    launches completed safely.  Known sticky CUDA faults skip another driver
    call, but every error initially classified as non-faulting crosses the same
    strict barrier used by successful results.  A barrier failure replaces the
    apparent CPU exception with the real CUDA context fault.
    """

    error_type = type(task_error).__name__
    error_message = str(task_error)
    final_sync_failed = error_type == "CudaFinalSyncError"
    is_cuda_error = (
        "CUDA" in error_type
        or "CUDA" in error_message
        or "cuda" in error_message.lower()
        or error_type in {"CudaError", "CudaFinalSyncError"}
    )
    fault_severity = _classify_cuda_fault(
        error_type,
        error_message,
        final_sync_failed=final_sync_failed,
        is_cuda_error=is_cuda_error,
    )

    if fault_severity == FAULT_NONE:
        try:
            _strict_cuda_task_barrier(synchronize_cuda)
        except CudaFinalSyncError as sync_error:
            error_type = type(sync_error).__name__
            error_message = str(sync_error)
            final_sync_failed = True
            is_cuda_error = True
            fault_severity = FAULT_CONTEXT

    return {
        "error_type": error_type,
        "error_message": error_message,
        "final_sync_failed": final_sync_failed,
        "is_cuda_error": is_cuda_error,
        "fault_severity": fault_severity,
        "is_profiler_error": "PROFILER_NO_CUDA_EVENTS" in error_message,
    }


def _publish_task_result_after_sync(
    synchronize_cuda: Callable[[], None],
    result_queue: Any,
    result: Dict[str, Any],
) -> None:
    """Commit a result only after all asynchronous CUDA work has completed."""

    _strict_cuda_task_barrier(synchronize_cuda)
    result_queue.put(result)


def _commit_task_result(
    synchronize_cuda: Callable[[], None],
    result_queue: Any,
    result: Dict[str, Any],
    *,
    prepare_for_reuse: Optional[Any] = None,
) -> None:
    """Publish a result with no CUDA calls after the commit queue write.

    A reusable process first synchronizes the task itself before touching cache
    maintenance APIs.  It then performs one final barrier immediately before
    publishing, so cleanup cannot hide a newly surfaced sticky CUDA fault.
    """

    if prepare_for_reuse is not None:
        _strict_cuda_task_barrier(synchronize_cuda)
        prepare_for_reuse()
    _publish_task_result_after_sync(synchronize_cuda, result_queue, result)


def _publish_non_cuda_failure_and_count_task(
    result_queue: Any,
    result: Dict[str, Any],
    *,
    tasks_processed: int,
    max_tasks_per_worker: int,
) -> tuple[int, bool]:
    """Publish a synchronized ordinary failure and enforce recycle limits.

    The caller has already crossed the strict CUDA barrier.  The returned flag
    means the process must exit directly: no CUDA cleanup API may run after the
    queue write that made this task result visible.
    """

    tasks_processed += 1
    must_recycle = tasks_processed >= max_tasks_per_worker
    if must_recycle:
        result["worker_exiting"] = True
    result_queue.put(result)
    return tasks_processed, must_recycle


@dataclass(frozen=True)
class _TrustedCudaTaskOperations:
    """Candidate-resistant local operations captured before task dispatch."""

    synchronize: Callable[[], None]
    commit: Callable[[Dict[str, Any], Optional[Callable[[], None]]], None]
    commit_and_wait: Callable[[Dict[str, Any], Optional[Callable[[], None]]], None]
    classify_error: Callable[[BaseException], Dict[str, Any]]
    publish_non_cuda_failure: Callable[[Dict[str, Any], int, int], tuple[int, bool]]
    publish_and_wait: Callable[[Dict[str, Any]], None]


def _capture_trusted_cuda_task_operations(
    synchronize_cuda: Callable[[], None],
    send_task_result: Callable[[Dict[str, Any]], None],
    wait_for_parent_containment: Callable[[], Any],
) -> _TrustedCudaTaskOperations:
    """Close over the entire result boundary before candidate Python runs.

    Capturing only ``torch._C._cuda_synchronize`` is insufficient when the
    loop later resolves helpers such as ``_commit_task_result`` through mutable
    module globals.  These closures retain the C barrier, queue sender, fault
    markers, and exception classes directly.  Ordinary module/setattr
    monkeypatching by evaluated code therefore cannot remove the commit
    barrier.  Deliberate frame/closure/function-code mutation remains outside
    this same-interpreter hardening boundary.
    """

    caught_exception_type = Exception
    sync_error_type = CudaFinalSyncError
    containment_error_type = _ProcessContainmentError
    fault_none = FAULT_NONE
    fault_context = FAULT_CONTEXT
    fault_device = FAULT_DEVICE
    context_markers = tuple(_CUDA_CONTEXT_FAULT_MARKERS)
    device_markers = tuple(_CUDA_DEVICE_FAULT_MARKERS)
    benign_markers = tuple(_CUDA_BENIGN_EXIT_MARKERS)
    xid_search = re.compile(r"\bxid(?:\s*[:(]|\s+\d)").search

    def _classify(
        error_type: str,
        error_message: str,
        *,
        final_sync_failed: bool = False,
        is_cuda_error: Optional[bool] = None,
    ) -> str:
        text_value = f"{error_type} {error_message}".lower()
        if error_type in {"TimeoutError", "WorkerProcessCrashed", "WorkerIPCProtocolError"}:
            return fault_device
        if any(marker in text_value for marker in device_markers) or xid_search(text_value):
            return fault_device
        if final_sync_failed or any(marker in text_value for marker in context_markers):
            return fault_context
        if any(marker in text_value for marker in benign_markers):
            return fault_none
        if is_cuda_error is None:
            is_cuda_error = "cuda" in text_value or error_type.lower() == "cudaerror"
        return fault_context if is_cuda_error else fault_none

    def _strict_synchronize() -> None:
        try:
            synchronize_cuda()
        except caught_exception_type as exc:
            raise sync_error_type(f"CUDA final synchronize failed: {exc}") from exc

    def _classify_error(task_error: BaseException) -> Dict[str, Any]:
        error_type = type(task_error).__name__
        error_message = str(task_error)
        final_sync_failed = error_type == "CudaFinalSyncError"
        is_cuda_error = (
            "CUDA" in error_type
            or "CUDA" in error_message
            or "cuda" in error_message.lower()
            or error_type in {"CudaError", "CudaFinalSyncError"}
        )
        fault_severity = _classify(
            error_type,
            error_message,
            final_sync_failed=final_sync_failed,
            is_cuda_error=is_cuda_error,
        )
        if fault_severity == fault_none:
            try:
                _strict_synchronize()
            except sync_error_type as sync_error:
                error_type = type(sync_error).__name__
                error_message = str(sync_error)
                final_sync_failed = True
                is_cuda_error = True
                fault_severity = fault_context
        return {
            "error_type": error_type,
            "error_message": error_message,
            "final_sync_failed": final_sync_failed,
            "is_cuda_error": is_cuda_error,
            "fault_severity": fault_severity,
            "is_profiler_error": "PROFILER_NO_CUDA_EVENTS" in error_message,
        }

    def _commit(result: Dict[str, Any], prepare_for_reuse: Optional[Callable[[], None]] = None) -> None:
        if prepare_for_reuse is not None:
            _strict_synchronize()
            prepare_for_reuse()
        _strict_synchronize()
        send_task_result(result)

    def _wait_after_publish() -> None:
        # The production callable is an unset Event.wait with no timeout.  If
        # it ever returns unexpectedly, escape this task loop without making a
        # CUDA call or publishing a second message.
        wait_for_parent_containment()
        raise containment_error_type("parent containment wait returned unexpectedly")

    def _commit_and_wait(
        result: Dict[str, Any],
        prepare_for_reuse: Optional[Callable[[], None]] = None,
    ) -> None:
        _commit(result, prepare_for_reuse)
        _wait_after_publish()

    def _publish_non_cuda_failure(
        result: Dict[str, Any],
        tasks_processed: int,
        max_tasks_per_worker: int,
    ) -> tuple[int, bool]:
        tasks_processed += 1
        must_recycle = tasks_processed >= max_tasks_per_worker
        if must_recycle:
            result["worker_exiting"] = True
        send_task_result(result)
        if must_recycle:
            _wait_after_publish()
        return tasks_processed, must_recycle

    def _publish_and_wait(result: Dict[str, Any]) -> None:
        # Any result that retires this worker keeps its leader/ancestry alive
        # until the parent freezes and kills the attested process group.  This
        # applies to sticky faults as well as benign OOM/profiler/max-task
        # recycling; a voluntary early exit would make the containment proof
        # fail closed and randomly quarantine a healthy GPU.
        send_task_result(result)
        _wait_after_publish()

    return _TrustedCudaTaskOperations(
        synchronize=_strict_synchronize,
        commit=_commit,
        commit_and_wait=_commit_and_wait,
        classify_error=_classify_error,
        publish_non_cuda_failure=_publish_non_cuda_failure,
        publish_and_wait=_publish_and_wait,
    )


async def _complete_despite_cancellation(awaitable: Any) -> tuple[Any, bool]:
    """Finish safety-critical async cleanup before reporting caller cancellation."""

    task = asyncio.ensure_future(awaitable)
    cancellation_requested = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation_requested = True
    return task.result(), cancellation_requested


class TaskCancelledError(Exception):
    """Raised when a task is cancelled mid-flight.

    The worker that was running the task must be recycled (its CUDA subprocess
    killed) and the task must NOT be retried.
    """


_STAGE_METADATA_PATH_ENV = "KERNELGYM_STAGE_METADATA_PATH"
_STAGE_METADATA_DIR_ENV = "KERNELGYM_STAGE_METADATA_DIR"
_STAGE_METADATA_DEFAULT_DIR = "/dev/shm/kernelgym/stage_metadata"
_FAST_RW_ROOT = "/dev/shm"

_WORKER_STDERR_DIR_ENV = "KERNELGYM_WORKER_STDERR_DIR"
_WORKER_STDERR_TAIL_BYTES = 2048

# Keeps the faulthandler target file alive for the process lifetime (the
# handler writes to its fd at crash time; a GC'd file object would break it).
_FAULTHANDLER_FILE = None


def _worker_stderr_path(worker_id: str) -> str:
    """Per-pool-slot file that captures the subprocess's native stderr (fd 2).

    Parent and child compute this independently, so it must be deterministic
    from worker_id + environment alone.
    """
    base = os.environ.get(_WORKER_STDERR_DIR_ENV, "")
    if not base:
        import tempfile

        root = _FAST_RW_ROOT if os.path.isdir(_FAST_RW_ROOT) else tempfile.gettempdir()
        base = os.path.join(root, "kernelgym", "worker_stderr")
    return os.path.join(base, f"{worker_id}.stderr")


def _read_stderr_tail(worker_id: str, max_bytes: int = _WORKER_STDERR_TAIL_BYTES) -> str:
    try:
        with open(_worker_stderr_path(worker_id), "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _redirect_native_stderr_to_capture_file(worker_id: str) -> None:
    """In the subprocess: point fd 2 at the capture file, keep Python's stderr.

    The dynamic loader ("symbol lookup error: ... undefined symbol"), CUDA
    asserts, and abort() all write to fd 2 and then kill the process before any
    Python code can report them. Redirecting fd 2 lets the parent recover that
    output after the crash. sys.stderr is rebound to a dup of the ORIGINAL
    stderr so the loop's own print(..., file=sys.stderr) diagnostics still
    reach the worker log. faulthandler adds a Python traceback to the capture
    file on SIGSEGV/SIGABRT.
    """
    global _FAULTHANDLER_FILE
    path = _worker_stderr_path(worker_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    capture_fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    original_fd = os.dup(2)
    os.dup2(capture_fd, 2)
    os.close(capture_fd)
    sys.stderr = os.fdopen(original_fd, "w", buffering=1)
    import faulthandler

    _FAULTHANDLER_FILE = os.fdopen(os.dup(2), "w", buffering=1)
    faulthandler.enable(file=_FAULTHANDLER_FILE)


def _truncate_native_stderr_capture() -> None:
    """In the subprocess: reset the capture file so its content is per-task."""
    try:
        os.ftruncate(2, 0)
        os.lseek(2, 0, os.SEEK_SET)
    except OSError:
        pass


def _require_fast_rw_dir(path_value: str, *, label: str) -> str:
    path = os.path.abspath(path_value)
    root = os.path.abspath(_FAST_RW_ROOT)
    if path != root and not path.startswith(root + os.sep):
        raise ValueError(f"{label} must be under /dev/shm for fast local I/O: {path}")
    os.makedirs(path, exist_ok=True)
    if not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(f"{label} is not writable/executable: {path}")
    return path


def _aggressive_gpu_cleanup(device_id: int, *, strict: bool = False) -> None:
    """
    强力清理 GPU 显存

    这个函数会尝试多种方法来清理显存：
    1. 清空 PyTorch 缓存
    2. 收集 Python 垃圾
    3. 重置 CUDA 峰值内存统计
    4. 清空 Triton 缓存（如果有）
    5. 同步 CUDA 操作

    Args:
        device_id: GPU 设备 ID
    """
    import torch
    import gc

    def _cuda_cleanup_step(label: str, operation: Any) -> None:
        try:
            operation()
        except Exception as exc:
            if strict:
                raise CudaFinalSyncError(f"CUDA pre-commit cleanup failed at {label}: {exc}") from exc

    # Shutdown cleanup is standalone and best-effort, so it synchronizes here.
    # Strict task cleanup is bracketed by _commit_task_result's two barriers.
    if not strict:
        _cuda_cleanup_step("initial synchronize", lambda: torch.cuda.synchronize(device_id))

    # 2. 清空 PyTorch 缓存
    _cuda_cleanup_step("empty_cache", torch.cuda.empty_cache)

    # 3. Python 垃圾回收（释放未引用的张量）
    gc.collect()

    # 4. 再次清空缓存
    _cuda_cleanup_step("second empty_cache", torch.cuda.empty_cache)

    # 5. 重置内存统计（帮助下次分配）
    _cuda_cleanup_step("reset_peak_memory_stats", lambda: torch.cuda.reset_peak_memory_stats(device_id))
    _cuda_cleanup_step(
        "reset_accumulated_memory_stats",
        lambda: torch.cuda.reset_accumulated_memory_stats(device_id),
    )

    # 6. Triton 编译的 kernel 缓存可能残留，但 Triton 没有公开的清理 API；
    #    进程退出时会自动清理。

    # 7. 最终同步
    if not strict:
        _cuda_cleanup_step("final synchronize", lambda: torch.cuda.synchronize(device_id))


@dataclass
class WorkerMetrics:
    """Worker 执行指标"""

    task_execution_time: float  # 任务执行时间
    total_time: float  # 总时间（包括 queue 等待）
    success: bool = True
    error_type: Optional[str] = None


class PersistentWorker:
    """
    持久化的 worker 进程

    特性：
    - 启动时一次性初始化 torch 和 CUDA
    - 通过 Queue 接收任务，并通过单向 JSON-bytes pipe 返回结果
    - **遇到 CUDA error 立即退出（通过特殊标记）**
    - 主进程检测到退出后会重启新的 worker
    """

    def __init__(self, worker_id: str, device_id: int, pool_size_info: str = "", max_tasks_per_worker: int = 100):
        """
        Args:
            worker_id: Worker 标识符（如 "worker_0"）
            device_id: GPU 设备 ID（如 0-7）
            pool_size_info: 用于日志的 pool 大小信息
            max_tasks_per_worker: 每个 worker 最多处理的任务数（防止显存累积）
        """
        self.worker_id = worker_id
        self.device_id = device_id
        self.pool_size_info = pool_size_info
        self.max_tasks_per_worker = max_tasks_per_worker
        self.process: Optional[mp.Process] = None

        # 使用 spawn context 确保完全隔离
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=10)  # 限制队列大小，避免内存爆炸
        result_receiver, result_sender = self.ctx.Pipe(duplex=False)
        self.result_queue = _ParentJSONResultChannel(result_receiver)
        self._child_result_channel: Optional[_ChildJSONResultChannel] = _ChildJSONResultChannel(result_sender)

        self.is_alive_flag = True
        self.tasks_processed = 0
        self.start_time = time.time()
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False
        self._result_channel_closed = False
        self._process_identity: Optional[_LinuxProcessIdentity] = None
        self._starting_process_identity: Optional[_LinuxProcessIdentity] = None
        self._expected_session_id = os.getsid(0)
        self.spawn_slot_wait_s = 0.0
        self.containment_elapsed_s = 0.0
        self.ready_after_containment_s = 0.0
        self.child_init_s = 0.0

        # Hold one cross-process node-local slot through spawn, containment,
        # dependency import, and CUDA READY.  Waiting for the slot is not part
        # of either handshake timeout.
        try:
            with _host_worker_spawn_slot(self.worker_id) as spawn_slot_wait_s:
                self._start_worker(spawn_slot_wait_s=spawn_slot_wait_s)
        except BaseException:
            if self.process is None:
                if self._child_result_channel is not None:
                    self._child_result_channel.close()
                    self._child_result_channel = None
                self._close_result_channel()
                self.task_queue.close()
            raise

    def _start_worker(self, *, spawn_slot_wait_s: float = 0.0):
        """启动 worker 进程"""
        self.spawn_slot_wait_s = spawn_slot_wait_s
        logger.info(
            f"[{self.worker_id}] Starting persistent worker for GPU {self.device_id} {self.pool_size_info} "
            f"(spawn_slot_wait={spawn_slot_wait_s:.2f}s)"
        )
        try:
            prepare_core_dump_dir(os.environ.get(CORE_DUMP_DIR_ENV), os.environ.get(CORE_DUMP_KEEP_ENV))
        except Exception as exc:
            logger.warning(f"[{self.worker_id}] Failed to prepare core dump directory: {exc}")

        child_result_channel = self._child_result_channel
        if child_result_channel is None:
            raise WorkerInitializationError(
                "child JSON result channel is unavailable",
                init_stage="result_channel",
            )
        self.process = self.ctx.Process(
            target=_persistent_worker_loop,
            args=(
                self.worker_id,
                self.device_id,
                self.task_queue,
                child_result_channel,
                self.max_tasks_per_worker,
            ),
            daemon=False,  # 不使用 daemon，确保可以正常清理
        )
        start_error: Optional[BaseException] = None
        try:
            self.process.start()
        except BaseException as exc:
            start_error = exc
        finally:
            # ``spawn`` duplicates the write handle into the child during
            # start().  The parent must not retain a writer or EOF/crash
            # detection on its read-only endpoint becomes ambiguous.
            child_result_channel.close()
            self._child_result_channel = None

        if start_error is not None:
            self.is_alive_flag = False
            try:
                process_pid = self.process.pid if self.process is not None else None
            except (AssertionError, ValueError):
                process_pid = None
            if isinstance(process_pid, int):
                try:
                    reaped = self.shutdown(timeout=5, force=True)
                except BaseException:
                    reaped = False
                if reaped is not True:
                    _retain_unreaped_worker(self.device_id, self)
                    raise WorkerInitializationError(
                        f"[{self.worker_id}] process start failed after a child may have started, "
                        "and child reap was not confirmed",
                        init_stage="process_start",
                        reap_confirmed=False,
                    ) from start_error
            else:
                # No PID was ever assigned, so there is no child/context to
                # retain.  Close the unused read endpoint immediately.
                self._close_result_channel()
                if self.process is not None:
                    try:
                        self.process.close()
                    except (AssertionError, ValueError):
                        pass
            raise start_error

        # Once Process.start() succeeds, every failed READY handshake must
        # force-reap the child before the constructor is allowed to raise.  A
        # constructor exception otherwise prevents the caller from ever
        # obtaining the only Process handle for a possibly live CUDA context.
        try:
            process_pid = self.process.pid if self.process is not None else None
            if not isinstance(process_pid, int):
                raise WorkerInitializationError(
                    "started worker has no process PID",
                    init_stage="process_containment",
                )
            starting_identity = _read_linux_process_identity(process_pid)
            if (
                starting_identity is None
                or starting_identity.pid != process_pid
                or starting_identity.ppid != os.getpid()
                or starting_identity.sid != self._expected_session_id
            ):
                raise WorkerInitializationError(
                    "started worker could not be generation-authenticated inside the outer session",
                    init_stage="process_containment",
                )
            self._starting_process_identity = starting_identity
            _register_starting_worker_identity(starting_identity)

            containment_started = time.monotonic()
            containment_deadline = containment_started + _WORKER_CONTAINMENT_TIMEOUT_S
            try:
                containment_msg = _receive_worker_message(
                    self.result_queue,
                    timeout=max(0.0, containment_deadline - time.monotonic()),
                    expected_kinds=frozenset({"contained", "init_failed"}),
                )
            except queue.Empty as exc:
                raise WorkerInitializationError(
                    f"[{self.worker_id}] Worker containment timeout (>{_WORKER_CONTAINMENT_TIMEOUT_S:.0f}s)",
                    init_stage="handshake_timeout",
                ) from exc
            except WorkerResultChannelClosed as exc:
                raise WorkerInitializationError(
                    f"[{self.worker_id}] Worker result channel closed before CONTAINED",
                    init_stage="handshake_eof",
                ) from exc
            containment_elapsed_s = time.monotonic() - containment_started
            self.containment_elapsed_s = containment_elapsed_s

            if not isinstance(containment_msg, dict):
                raise WorkerInitializationError(
                    f"Worker returned malformed containment handshake: {type(containment_msg).__name__}",
                    init_stage="handshake_protocol",
                )
            if containment_msg.get("status") != "CONTAINED":
                init_stage = str(containment_msg.get("init_stage") or "unknown")
                init_error = str(containment_msg.get("error") or containment_msg)
                raise WorkerInitializationError(
                    f"Worker failed to initialize at {init_stage}: {init_error}",
                    init_stage=init_stage,
                )

            identity = _read_linux_process_identity(process_pid)
            reported_identity = (
                containment_msg.get("pid"),
                containment_msg.get("start_ticks"),
                containment_msg.get("pgid"),
                containment_msg.get("sid"),
            )
            expected_identity = (
                process_pid,
                identity.start_ticks if identity is not None else None,
                process_pid,
                self._expected_session_id,
            )
            if identity is None or reported_identity != expected_identity:
                raise WorkerInitializationError(
                    "worker process identity/PGID attestation failed: "
                    f"reported={reported_identity}, expected={expected_identity}",
                    init_stage="process_containment",
                )
            if identity.pgid != process_pid or identity.sid != self._expected_session_id:
                raise WorkerInitializationError(
                    "worker did not enter its dedicated PGID inside the outer worker session",
                    init_stage="process_containment",
                )
            self._process_identity = identity
            _promote_active_worker_identity(starting_identity, identity)
            self.task_queue.put(_PARENT_CONTAINMENT_ACK)

            ready_started = time.monotonic()
            ready_deadline = ready_started + _WORKER_READY_AFTER_CONTAINMENT_TIMEOUT_S
            try:
                init_msg = _receive_worker_message(
                    self.result_queue,
                    timeout=max(0.0, ready_deadline - time.monotonic()),
                    expected_kinds=frozenset({"ready", "init_failed"}),
                )
            except queue.Empty as exc:
                raise WorkerInitializationError(
                    f"[{self.worker_id}] Worker initialization timeout after CONTAINED "
                    f"(>{_WORKER_READY_AFTER_CONTAINMENT_TIMEOUT_S:.0f}s)",
                    init_stage="handshake_timeout",
                ) from exc
            except WorkerResultChannelClosed as exc:
                raise WorkerInitializationError(
                    f"[{self.worker_id}] Worker result channel closed before READY",
                    init_stage="handshake_eof",
                ) from exc
            ready_elapsed_s = time.monotonic() - ready_started
            self.ready_after_containment_s = ready_elapsed_s

            if not isinstance(init_msg, dict):
                raise WorkerInitializationError(
                    f"Worker returned malformed READY handshake: {type(init_msg).__name__}",
                    init_stage="handshake_protocol",
                )
            if init_msg.get("status") != "READY":
                init_stage = str(init_msg.get("init_stage") or "unknown")
                init_error = str(init_msg.get("error") or init_msg)
                raise WorkerInitializationError(
                    f"Worker failed to initialize at {init_stage}: {init_error}",
                    init_stage=init_stage,
                )

            ready_identity = (
                init_msg.get("pid"),
                init_msg.get("start_ticks"),
                init_msg.get("pgid"),
                init_msg.get("sid"),
            )
            current_identity = _read_linux_process_identity(process_pid)
            if (
                current_identity is None
                or ready_identity != expected_identity
                or not _same_process_generation(identity, current_identity)
            ):
                raise WorkerInitializationError(
                    "READY worker identity changed after containment promotion",
                    init_stage="cuda_identity",
                )
            self.child_init_s = float(init_msg.get("init_time", 0))
            logger.info(
                f"[{self.worker_id}] Worker initialized successfully "
                f"(containment={containment_elapsed_s:.2f}s, ready_after_containment={ready_elapsed_s:.2f}s, "
                f"child_init={self.child_init_s:.2f}s)"
            )
            self.is_alive_flag = True
        except BaseException as handshake_error:
            self.is_alive_flag = False
            shutdown_error: Optional[BaseException] = None
            try:
                reaped = self.shutdown(timeout=5, force=True)
            except BaseException as exc:
                shutdown_error = exc
                reaped = False
            _record_worker_reap_result(self.device_id, self, reaped)

            if reaped is not True:
                error_detail = (
                    f"; shutdown raised {type(shutdown_error).__name__}: {shutdown_error}"
                    if shutdown_error is not None
                    else ""
                )
                init_stage = getattr(handshake_error, "init_stage", "handshake_reap")
                raise WorkerInitializationError(
                    f"[{self.worker_id}] READY handshake failed and child reap was not confirmed"
                    f"{error_detail}: {handshake_error}",
                    init_stage=str(init_stage),
                    reap_confirmed=False,
                ) from handshake_error
            raise

    def execute_task(
        self,
        task_data: Dict[str, Any],
        timeout: int = 60,
        cancel_event: Optional[threading.Event] = None,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
        """
        执行任务

        Args:
            task_data: 任务数据字典
            timeout: 超时时间（秒）
            cancel_event: 若被 set，则中止等待、回收 worker 并抛出 TaskCancelledError
            poll_interval: 轮询结果队列/取消标志的间隔（秒）

        Returns:
            结果字典，包含 success, result/error_type/error_message

        Raises:
            RuntimeError: Worker 已死亡或任务执行失败
            TimeoutError: 任务超时
            TaskCancelledError: 任务在执行中被取消
        """
        if not self.is_alive():
            raise RuntimeError(f"[{self.worker_id}] Worker is not alive")

        # 发送任务
        try:
            self.task_queue.put(task_data, timeout=5)
        except queue.Full:
            raise RuntimeError(f"[{self.worker_id}] Task queue is full")

        # 等待结果：按 poll_interval 轮询，以便及时响应取消请求和超时
        task_id = task_data.get("task_id", "unknown")
        poll = poll_interval if poll_interval and poll_interval > 0 else 0.5
        deadline = time.monotonic() + timeout
        result = None

        def _crash_result(reason: str) -> Dict[str, Any]:
            exitcode = self.process.exitcode if self.process is not None else None
            stderr_tail = _read_stderr_tail(self.worker_id)
            logger.error(
                f"[{self.worker_id}] Worker subprocess became unavailable (exitcode={exitcode}) "
                f"during task {task_id} before returning a result ({reason}); reporting as a crash"
                + (f"; subprocess stderr tail:\n{stderr_tail}" if stderr_tail else "")
            )
            self.is_alive_flag = False
            return {
                "success": False,
                "error_type": "WorkerProcessCrashed",
                "error_message": (
                    f"Worker {self.worker_id} subprocess became unavailable (exitcode={exitcode}) "
                    f"during task {task_id} before returning a result ({reason}); likely a native "
                    "crash in the evaluated kernel or an early process exit"
                    + (f". Subprocess stderr tail:\n{stderr_tail}" if stderr_tail else "")
                ),
                "stderr_tail": stderr_tail,
                "worker_exiting": True,
                "crashed": True,
                "fault_severity": FAULT_DEVICE,
                "device_suspect": True,
            }

        while True:
            if cancel_event is not None and cancel_event.is_set():
                logger.warning(
                    f"[{self.worker_id}] Cancellation requested for task {task_id}; "
                    f"recycling worker to abort in-flight CUDA work"
                )
                # Worker is mid-execution; mark it dead so the pool kills the
                # subprocess and replaces it with a clean one.
                self.is_alive_flag = False
                raise TaskCancelledError(f"[{self.worker_id}] Task {task_id} cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(f"[{self.worker_id}] Task timeout after {timeout}s, task_id={task_id}")
                # 标记 worker 为不可用（可能卡死了）
                self.is_alive_flag = False
                raise TimeoutError(f"[{self.worker_id}] Task {task_id} timeout after {timeout}s")
            try:
                result = _receive_worker_message(
                    self.result_queue,
                    timeout=min(poll, remaining),
                    expected_kinds=frozenset({"task_result"}),
                )
                break
            except queue.Empty:
                # The worker subprocess may have died mid-task (e.g. a native
                # crash in the evaluated kernel: undefined cuBLAS/cuDNN symbol,
                # segfault, illegal memory access). Without this check the loop
                # would keep polling an empty queue until the deadline and then
                # misreport the crash as a TIMEOUT. Detect the death promptly and
                # surface it as a crash result instead.
                if self.process is not None and not self.process.is_alive():
                    # Drain a result the worker may have enqueued just before exit.
                    try:
                        result = _receive_worker_message(
                            self.result_queue,
                            timeout=0.2,
                            expected_kinds=frozenset({"task_result"}),
                        )
                        break
                    except (queue.Empty, WorkerResultChannelClosed):
                        pass
                    result = _crash_result("process exited and the result pipe drained")
                    break
                continue
            except WorkerResultChannelClosed:
                # EOF is definitive: this child can never produce the expected
                # result.  Classify it through the existing crash/containment
                # path instead of treating it as a timeout or IPC device fault.
                result = _crash_result("result channel reached EOF")
                break

        # 检查 worker 是否报告 CUDA error 并准备退出
        if result.get("worker_exiting") is True:
            logger.warning(
                f"[{self.worker_id}] Worker encountered CUDA error and will exit. "
                f"Error: {result.get('error_type', 'Unknown')}: {result.get('error_message', 'N/A')}"
            )
            self.is_alive_flag = False
            # 标记进程将要退出，主进程会重启

        # 更新统计
        self.tasks_processed += 1

        # **关键：检查是否达到任务上限（防止显存累积）**
        if self.tasks_processed >= self.max_tasks_per_worker:
            logger.info(
                f"[{self.worker_id}] Reached max tasks limit ({self.max_tasks_per_worker}). "
                f"Marking for restart to prevent memory accumulation."
            )
            self.is_alive_flag = False
            # 注意：我们不立即关闭进程，而是让它在下次检查时被重启
            # 这样可以先返回当前任务的结果

        return result

    def is_alive(self) -> bool:
        """检查 worker 是否存活"""
        return self.is_alive_flag and self.process is not None and self.process.is_alive()

    def _close_result_channel(self) -> None:
        """Close the parent read FD once, but only after child exit is proven."""

        if getattr(self, "_result_channel_closed", False):
            return
        close_result_channel = getattr(getattr(self, "result_queue", None), "close", None)
        if close_result_channel is not None:
            close_result_channel()
        self._result_channel_closed = True

    def shutdown(self, timeout: int = 10, force: bool = False) -> bool:
        """Serialize shutdown and make a successfully reaped worker idempotent."""

        with self._shutdown_lock:
            if self._shutdown_complete:
                self._close_result_channel()
                _release_reaped_worker(self.device_id, self)
                return True
            try:
                stopped = self._shutdown_impl(timeout=timeout, force=force)
            except BaseException:
                _retain_unreaped_worker(self.device_id, self)
                raise
            _record_worker_reap_result(self.device_id, self, stopped)
            if stopped:
                self._shutdown_complete = True
            return stopped

    def _shutdown_impl(self, timeout: int = 10, force: bool = False) -> bool:
        """关闭 worker 进程

        Ensures the child process is fully reaped (joined) so that the
        CUDA driver releases its GPU memory.  Every kill path is followed
        by ``process.join()`` and ``process.close()``.
        """
        logger.info(f"[{self.worker_id}] Shutting down worker process tree (force={force})...")

        process_stopped = self.process is None
        containment_proven = self.process is None
        session_audit_identity: Optional[_LinuxProcessIdentity] = None
        try:
            if self.process is not None:
                identity = self._process_identity
                starting_identity = getattr(self, "_starting_process_identity", None)
                if identity is None and isinstance(self.process.pid, int):
                    candidate = _read_linux_process_identity(self.process.pid)
                    if (
                        candidate is not None
                        and candidate.pgid == candidate.pid
                        and candidate.sid == self._expected_session_id
                        and (starting_identity is None or _same_process_generation(starting_identity, candidate))
                    ):
                        identity = candidate
                        self._process_identity = candidate
                    elif starting_identity is not None:
                        # The owned child either has not established its PGID,
                        # has exited, or its numeric PID now belongs to another
                        # generation.  Never adopt or signal that generation.
                        # After joining the owned Process below, the full SID
                        # audit can still prove that the trusted bootstrap left
                        # no unknown descendant behind.
                        containment_proven = True
                        session_audit_identity = starting_identity
                        logger.warning(
                            f"[{self.worker_id}] Startup leader generation cannot be adopted; "
                            "requiring owned-process join and complete SID containment proof"
                        )

                if identity is not None:
                    current_identity = _read_linux_process_identity(identity.pid)
                    if _same_process_generation(identity, current_identity):
                        containment_proven = _kill_and_verify_worker_process_tree(
                            identity,
                            freeze_timeout=max(1.0, min(5.0, float(timeout))),
                            reap_timeout=max(3.0, float(timeout)),
                        )
                    else:
                        # A native crash can reap the leader before the parent
                        # enters shutdown.  There is then no live PID
                        # generation that can be frozen or safely signalled.
                        # Treat this only as provisional containment: below we
                        # still join the owned multiprocessing handle and
                        # require kernel ESRCH for its dedicated PGID.  A
                        # surviving same-PGID descendant therefore continues
                        # to fail closed, while an already-drained crash can
                        # proceed to the pool's fresh CUDA-context probe.
                        containment_proven = True
                        session_audit_identity = identity
                        logger.warning(
                            f"[{self.worker_id}] Worker leader generation exited before shutdown; "
                            f"requiring owned-process join and PGID {identity.pgid} drain proof"
                        )
                elif session_audit_identity is None:
                    containment_proven = False

                # Join the multiprocessing leader even after killpg so its
                # exit status/resources are reaped by the owning Process.
                self.process.join(timeout=max(1.0, float(timeout)))
                process_stopped = not self.process.is_alive()
                if not process_stopped:
                    self.process.kill()
                    self.process.join(timeout=10)
                    process_stopped = not self.process.is_alive()
                if identity is not None:
                    if session_audit_identity is identity:
                        group_drained = _wait_for_process_group_drain_or_registered_reuse(
                            identity,
                            max(3.0, float(timeout)),
                        )
                    else:
                        group_drained = _wait_for_process_group_drain(
                            identity.pgid,
                            max(3.0, float(timeout)),
                        )
                    containment_proven = containment_proven and group_drained
                if containment_proven and session_audit_identity is not None:
                    containment_proven = _wait_for_worker_session_containment(
                        session_audit_identity,
                        max(3.0, float(timeout)),
                    )
        except BaseException as exc:
            containment_proven = False
            logger.error(f"[{self.worker_id}] Error during process-tree shutdown: {exc}")
            if self.process is not None:
                try:
                    self.process.kill()
                    self.process.join(timeout=10)
                    process_stopped = not self.process.is_alive()
                except BaseException as kill_err:
                    process_stopped = False
                    logger.error(f"[{self.worker_id}] Failed to kill/join leader in error handler: {kill_err}")

        shutdown_proven = process_stopped and containment_proven

        # Release multiprocessing.Process internal resources (fds, etc.) only
        # after exit is proven.  Closing an alive Process loses the safe handle
        # needed to terminate and reap its CUDA context.
        if shutdown_proven:
            registered_identity = getattr(self, "_process_identity", None) or getattr(
                self, "_starting_process_identity", None
            )
            if registered_identity is not None:
                _unregister_worker_identity(registered_identity)
            if self.process is not None:
                try:
                    self.process.close()
                except Exception as close_err:
                    logger.warning(f"[{self.worker_id}] process.close() failed: {close_err}")
            try:
                self._close_result_channel()
            except Exception as close_err:
                logger.warning(f"[{self.worker_id}] result channel close failed: {close_err}")
        else:
            logger.critical(
                f"[{self.worker_id}] Worker leader/process-group exit was not fully proven; "
                "the physical GPU is not safe to reopen"
            )

        self.is_alive_flag = False
        logger.info(
            f"[{self.worker_id}] Worker shut down "
            f"(processed {self.tasks_processed} tasks in "
            f"{time.time() - self.start_time:.1f}s)"
        )
        return shutdown_proven

    def get_stats(self) -> Dict[str, Any]:
        """获取 worker 统计信息"""
        return {
            "worker_id": self.worker_id,
            "device_id": self.device_id,
            "is_alive": self.is_alive(),
            "tasks_processed": self.tasks_processed,
            "uptime": time.time() - self.start_time,
            "pid": self.process.pid if self.process else None,
        }


class SubprocessWorkerPool:
    """
    Worker Pool 管理器

    职责：
    1. 管理多个 PersistentWorker
    2. 分配任务到空闲的 worker
    3. **自动重启遇到 CUDA error 的 worker**
    4. 负载均衡
    """

    def __init__(
        self, device_id: int, pool_size: int = 2, worker_prefix: str = "pool_worker", max_tasks_per_worker: int = 100
    ):
        """
        Args:
            device_id: GPU 设备 ID
            pool_size: Worker 进程数量（建议 2-4，根据内存大小调整）
            worker_prefix: Worker ID 前缀
            max_tasks_per_worker: 每个 worker 最多处理的任务数（防止显存累积，默认100）
        """
        self.device_id = device_id
        self.pool_size = pool_size
        self.worker_prefix = worker_prefix
        self.max_tasks_per_worker = max_tasks_per_worker

        # Workers 列表
        self.workers: List[PersistentWorker] = []
        self.idle_workers: List[PersistentWorker] = []
        self.busy_workers: List[PersistentWorker] = []
        self.pending_replacements = 0
        self.pending_retirements = 0
        self._retiring_worker_ids: set[int] = set()
        self._top_up_sequence = 0

        # Replenishment constructors run in background threads.  Track their
        # generation independently from capacity accounting so a post-fault
        # validation waits for every pre-fault constructor/context to finish
        # and be reaped before it calls cuInit itself.
        self._ticket_condition = threading.Condition()
        self._active_replenishments: Dict[int, int] = {}
        self._closing = False
        self._shutdown_task: Optional[asyncio.Task[bool]] = None
        self.unsafe_shutdown_reason = ""

        # Physical-GPU admission state.  A context-local CUDA fault permits at
        # most one pre-fault warm spare while a post-fault worker proves that a
        # fresh context can still be created.  Device-level faults stop all
        # admission until a fresh worker passes initialization.
        self.health_state = POOL_HEALTHY
        self.health_reason = ""
        self.health_task_id = ""
        self.health_fault_class = ""
        self.health_scope = "gpu"
        self.health_epoch = 0
        self.pool_generation = 0
        self.hard_recovery_epoch = 0
        self.speculative_dispatches_remaining = 0
        self.consecutive_replacement_failures = 0
        # CUDA probe failures and unconfirmed reap still fail closed on the
        # first occurrence.  Confirmed-reap process/bootstrap failures get a
        # small bounded retry budget before the outer worker is recycled.
        self.max_replacement_failures = _MAX_REPLACEMENT_INFRA_FAILURES

        # 统计
        self.total_tasks_processed = 0
        self.total_workers_restarted = 0
        self.pool_start_time = time.time()

        # 同步锁
        self.lock = asyncio.Lock()
        self._recovery_lock = asyncio.Lock()

        # 初始化 workers
        self._init_workers()

        # Start background zombie reaper thread
        self._reaper_stop = threading.Event()
        try:
            self._reaper_thread = threading.Thread(
                target=self._zombie_reaper_loop,
                daemon=True,
                name=f"zombie-reaper-gpu{device_id}",
            )
            self._reaper_thread.start()
        except Exception as reaper_error:
            rollback_failed = False
            for worker in self._tracked_workers_locked():
                try:
                    shutdown_result = worker.shutdown(10, True)
                    if not _record_worker_reap_result(self.device_id, worker, shutdown_result):
                        rollback_failed = True
                except BaseException as shutdown_error:
                    _record_worker_reap_result(self.device_id, worker, shutdown_error)
                    rollback_failed = True
            self.workers.clear()
            self.idle_workers.clear()
            self.busy_workers.clear()
            message = f"[GPU {device_id}] Zombie reaper thread failed to start: {reaper_error}"
            if rollback_failed:
                raise GPUProbeFailedError(
                    f"{message}; initialized CUDA contexts could not be confirmed reaped"
                ) from reaper_error
            raise WorkerPoolInfrastructureError(message) from reaper_error

        logger.info(f"[GPU {device_id}] Worker pool initialized with {pool_size} workers")

    @property
    def accepting_tasks(self) -> bool:
        """Whether the GPU pool may dequeue another task."""

        if self._closing:
            return False
        if self.health_state == POOL_HEALTHY:
            return True
        return self.health_state == POOL_DEGRADED_CHECK and self.speculative_dispatches_remaining > 0

    def get_health_snapshot(self) -> Dict[str, Any]:
        """Return scheduler/heartbeat-safe physical-GPU health metadata."""

        return {
            "health_state": self.health_state,
            "accepting_tasks": self.accepting_tasks,
            "health_reason": self.health_reason,
            "health_task_id": self.health_task_id,
            "health_fault_class": self.health_fault_class,
            "health_scope": self.health_scope,
            "health_epoch": self.health_epoch,
            "speculative_dispatches_remaining": self.speculative_dispatches_remaining,
            "consecutive_replacement_failures": self.consecutive_replacement_failures,
            "pending_retirements": self.pending_retirements,
        }

    def _register_replenishment_ticket(self, generation: int) -> None:
        with self._ticket_condition:
            self._active_replenishments[generation] = self._active_replenishments.get(generation, 0) + 1

    def _finish_replenishment_ticket(self, generation: int) -> None:
        with self._ticket_condition:
            remaining = self._active_replenishments.get(generation, 0) - 1
            if remaining > 0:
                self._active_replenishments[generation] = remaining
            else:
                self._active_replenishments.pop(generation, None)
            self._ticket_condition.notify_all()

    def _wait_for_older_replenishments(
        self,
        generation: int,
        timeout: float = _REPLENISHMENT_DRAIN_TIMEOUT_S,
    ) -> bool:
        """Wait until all constructors from pre-fault generations are reaped."""

        deadline = time.monotonic() + timeout
        with self._ticket_condition:
            while any(epoch < generation and count > 0 for epoch, count in self._active_replenishments.items()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._ticket_condition.wait(timeout=min(remaining, 1.0))
        return True

    def _wait_for_all_replenishments(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._ticket_condition:
            while self._active_replenishments:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._ticket_condition.wait(timeout=min(remaining, 1.0))
        return True

    def _set_health_locked(
        self,
        state: str,
        *,
        reason: str = "",
        task_id: str = "",
        fault_class: str = "",
        scope: str = "gpu",
    ) -> None:
        self.health_state = state
        self.health_reason = reason
        self.health_task_id = task_id
        self.health_fault_class = fault_class
        self.health_scope = scope
        self.health_epoch += 1
        if state != POOL_DEGRADED_CHECK:
            self.speculative_dispatches_remaining = 0

    def _tracked_workers_locked(self, *extra: PersistentWorker) -> List[PersistentWorker]:
        """Return every worker reference in any pool view, de-duplicated by identity."""

        result: List[PersistentWorker] = []
        seen: set[int] = set()
        retained = _snapshot_unreaped_workers(self.device_id)
        for item in [*self.workers, *self.idle_workers, *self.busy_workers, *retained, *extra]:
            identity = id(item)
            if identity not in seen:
                seen.add(identity)
                result.append(item)
        return result

    def _record_reap_results(
        self,
        workers: List[PersistentWorker],
        results: List[Any],
    ) -> List[PersistentWorker]:
        """Persist handles for every child whose exit was not proven."""

        return [
            worker
            for worker, result in zip(workers, results)
            if not _record_worker_reap_result(self.device_id, worker, result)
        ]

    async def _quarantine_unreaped_context(
        self,
        reason: str,
        *,
        task_id: str = "",
        extra_worker: Optional[PersistentWorker] = None,
    ) -> None:
        """Fail closed when a CUDA-owning process cannot be proven gone."""

        async with self.lock:
            extras = (extra_worker,) if extra_worker is not None else ()
            workers_to_shutdown = self._tracked_workers_locked(*extras)
            self.workers.clear()
            self.idle_workers.clear()
            self.busy_workers.clear()
            self.pool_generation += 1
            self.hard_recovery_epoch += 1
            self.pending_replacements = 0
            self.unsafe_shutdown_reason = reason
            self._set_health_locked(
                POOL_QUARANTINED,
                reason=reason,
                task_id=task_id,
                fault_class="pre_fault_reap_failure",
                scope="gpu",
            )
        if workers_to_shutdown:
            results = await asyncio.gather(
                *(asyncio.to_thread(item.shutdown, 10, True) for item in workers_to_shutdown),
                return_exceptions=True,
            )
            unreaped_workers = self._record_reap_results(workers_to_shutdown, results)
            if unreaped_workers:
                logger.critical(
                    f"[GPU {self.device_id}] One or more CUDA worker processes remain unconfirmed after quarantine"
                )

    async def _contain_unexpected_hard_recovery_failure(
        self,
        error: BaseException,
        *,
        task_id: str,
        extra_workers: tuple[PersistentWorker, ...] = (),
    ) -> None:
        """Fail closed and reap every known context after recovery itself fails.

        Hard recovery is the last containment boundary between an untrusted
        CUDA payload and the physical device.  An infrastructure/programming
        error in that boundary must never leave the pool in SUSPECT (or,
        worse, HEALTHY) with task admission still possible.
        """

        reason = f"unexpected hard-recovery failure: {type(error).__name__}: {error}"
        async with self.lock:
            workers_to_shutdown = self._tracked_workers_locked(*extra_workers)
            self.workers.clear()
            self.idle_workers.clear()
            self.busy_workers.clear()
            self.pool_generation += 1
            self.hard_recovery_epoch += 1
            self.pending_replacements = 0
            self._set_health_locked(
                POOL_QUARANTINED,
                reason=reason,
                task_id=task_id,
                fault_class="hard_recovery_failure",
                scope="gpu",
            )

        shutdown_results: List[Any] = []
        if workers_to_shutdown:
            shutdown_results = await asyncio.gather(
                *(asyncio.to_thread(item.shutdown, 10, True) for item in workers_to_shutdown),
                return_exceptions=True,
            )
        unreaped_workers = self._record_reap_results(workers_to_shutdown, shutdown_results)
        if unreaped_workers:
            self.unsafe_shutdown_reason = (
                "one or more CUDA contexts could not be confirmed reaped after hard-recovery failure"
            )
        logger.critical(f"[GPU {self.device_id}] QUARANTINED: {reason}")
        if unreaped_workers:
            raise UnsafeGPUContainmentError(
                self.unsafe_shutdown_reason,
                task_id=task_id,
                worker_id=unreaped_workers[0].worker_id,
            ) from error

    async def _begin_context_fault_validation(
        self,
        *,
        task_id: str,
        reason: str,
        worker: Optional[PersistentWorker] = None,
        allow_speculative_spare: bool = True,
    ) -> Optional[int]:
        """Enter the one-spare context-fault path and return its generation.

        A second context fault before a post-fault replacement reaches READY
        is escalated by the caller to device-level recovery.  A report from a
        context already retired by a completed hard recovery returns
        ``_STALE_HARD_RECOVERY_EPOCH`` and must not create another probe.
        """

        observed_hard_epoch = getattr(
            worker,
            "_pool_hard_recovery_epoch_at_checkout",
            self.hard_recovery_epoch,
        )
        async with self.lock:
            if observed_hard_epoch != self.hard_recovery_epoch:
                logger.info(
                    f"[GPU {self.device_id}] Ignoring stale context-fault report from hard epoch "
                    f"{observed_hard_epoch}; current={self.hard_recovery_epoch}, task={task_id}"
                )
                return _STALE_HARD_RECOVERY_EPOCH
            if self._closing or self.health_state != POOL_HEALTHY:
                return None
            self.pool_generation += 1
            self.pending_replacements = 0
            self._set_health_locked(
                POOL_DEGRADED_CHECK,
                reason=reason,
                task_id=task_id,
                fault_class=FAULT_CONTEXT,
            )
            self.speculative_dispatches_remaining = 1 if allow_speculative_spare else 0
            logger.warning(
                f"[GPU {self.device_id}] Context fault: allowing at most "
                f"{self.speculative_dispatches_remaining} pre-fault spare "
                f"while generation {self.pool_generation} creates a fresh CUDA context; task={task_id} reason={reason}"
            )
            return self.pool_generation

    async def _raise_shutdown_containment_result(
        self,
        *,
        task_id: str,
        worker_id: str,
    ) -> None:
        """Wait for the shutdown that closed admission, then transfer ownership.

        A plain return here would let the execution coroutine publish/ACK while
        process-group containment is still unresolved. The shared shutdown
        task is therefore the proof authority for every concurrent recovery.
        """

        shutdown_task = self._shutdown_task
        if shutdown_task is None:
            reason = self.unsafe_shutdown_reason or "pool entered closing state without a tracked shutdown proof"
            raise UnsafeGPUContainmentError(
                reason,
                task_id=task_id,
                worker_id=worker_id,
            )
        try:
            shutdown_safe, _ = await _complete_despite_cancellation(shutdown_task)
        except BaseException as shutdown_error:
            reason = self.unsafe_shutdown_reason or (
                f"worker-pool shutdown proof failed: {type(shutdown_error).__name__}: {shutdown_error}"
            )
            raise UnsafeGPUContainmentError(
                reason,
                task_id=task_id,
                worker_id=worker_id,
            ) from shutdown_error
        if shutdown_safe is not True or self.unsafe_shutdown_reason:
            reason = self.unsafe_shutdown_reason or "worker-pool shutdown did not prove every CUDA context exited"
            raise UnsafeGPUContainmentError(
                reason,
                task_id=task_id,
                worker_id=worker_id,
            )
        raise PoolShutdownContainmentError(f"worker-pool shutdown safely took ownership of task {task_id}")

    async def _recover_from_device_fault(self, worker: PersistentWorker, *, task_id: str, reason: str) -> None:
        """Serialize physical-GPU recovery so concurrent faults create one probe."""

        observed_hard_epoch = getattr(
            worker,
            "_pool_hard_recovery_epoch_at_checkout",
            self.hard_recovery_epoch,
        )

        async def _serialized_recovery() -> None:
            async with self._recovery_lock:
                if self._closing:
                    await self._raise_shutdown_containment_result(
                        task_id=task_id,
                        worker_id=worker.worker_id,
                    )
                if self.health_state == POOL_QUARANTINED:
                    if self.unsafe_shutdown_reason:
                        raise UnsafeGPUContainmentError(
                            self.unsafe_shutdown_reason,
                            task_id=task_id,
                            worker_id=worker.worker_id,
                        )
                    return
                if observed_hard_epoch != self.hard_recovery_epoch:
                    # A concurrent hard recovery already retired every worker
                    # checked out in the older hard-recovery epoch.
                    logger.info(
                        f"[GPU {self.device_id}] Ignoring stale device-fault report from hard epoch "
                        f"{observed_hard_epoch}; current={self.hard_recovery_epoch}, task={task_id}"
                    )
                    return
                try:
                    await self._recover_from_device_fault_once(worker, task_id=task_id, reason=reason)
                    # Pool state can change after the wrapper's first check but
                    # before (or during) the inner recovery.  Never translate a
                    # concurrently discovered unproven reap into an ordinary
                    # task error merely because the inner path became stale and
                    # returned without constructing another probe.
                    async with self.lock:
                        concurrent_unsafe_reason = self.unsafe_shutdown_reason
                        concurrent_shutdown = self._closing
                    if concurrent_unsafe_reason:
                        raise UnsafeGPUContainmentError(
                            concurrent_unsafe_reason,
                            task_id=task_id,
                            worker_id=worker.worker_id,
                        )
                    if concurrent_shutdown:
                        await self._raise_shutdown_containment_result(
                            task_id=task_id,
                            worker_id=worker.worker_id,
                        )
                except (PoolShutdownContainmentError, UnsafeGPUContainmentError):
                    raise
                except Exception as recovery_error:
                    # ``_recover_from_device_fault_once`` normally contains
                    # its own known failure modes.  This outer guard also
                    # covers lock/setup failures and tests/future refactors
                    # that raise before its local containment handler exists.
                    if self.health_state != POOL_QUARANTINED or self.health_scope != "gpu":
                        await self._contain_unexpected_hard_recovery_failure(
                            recovery_error,
                            task_id=task_id,
                            extra_workers=(worker,),
                        )
                    if self.unsafe_shutdown_reason:
                        raise UnsafeGPUContainmentError(
                            self.unsafe_shutdown_reason,
                            task_id=task_id,
                            worker_id=worker.worker_id,
                        ) from recovery_error
                    raise

        # Shield the serialized lock acquisition as well as the recovery.
        # ``asyncio.to_thread`` cannot stop a constructor once it has entered
        # cuInit; abandoning either phase could leak an untracked CUDA context.
        recovery_task = asyncio.create_task(_serialized_recovery())
        cancellation_requested = False
        while not recovery_task.done():
            try:
                await asyncio.shield(recovery_task)
            except asyncio.CancelledError:
                cancellation_requested = True
        recovery_task.result()
        if cancellation_requested:
            raise asyncio.CancelledError

    async def _recover_from_device_fault_once(
        self,
        worker: PersistentWorker,
        *,
        task_id: str,
        reason: str,
    ) -> None:
        """Gate the physical GPU, retire old contexts, and run one fresh probe."""

        shutdown_takeover = False
        async with self.lock:
            if self._closing:
                shutdown_takeover = True
            elif self.health_state == POOL_QUARANTINED:
                if self.unsafe_shutdown_reason:
                    raise UnsafeGPUContainmentError(
                        self.unsafe_shutdown_reason,
                        task_id=task_id,
                        worker_id=worker.worker_id,
                    )
                return
            else:
                self.pool_generation += 1
                self.hard_recovery_epoch += 1
                generation = self.pool_generation
                self.pending_replacements = 1
                self._set_health_locked(
                    POOL_SUSPECT,
                    reason=reason,
                    task_id=task_id,
                    fault_class=FAULT_DEVICE,
                )
                old_workers = self._tracked_workers_locked(worker)
                self.workers.clear()
                self.idle_workers.clear()
                self.busy_workers.clear()
                self._register_replenishment_ticket(generation)

        if shutdown_takeover:
            await self._raise_shutdown_containment_result(
                task_id=task_id,
                worker_id=worker.worker_id,
            )
            return

        validation_worker: Optional[PersistentWorker] = None
        try:
            logger.error(
                f"[GPU {self.device_id}] Device fault gate entered for task={task_id}; "
                f"retiring {len(old_workers)} pre-fault workers before fresh-context validation: {reason}"
            )
            shutdown_results: List[Any] = []
            if old_workers:
                shutdown_results = await asyncio.gather(
                    *(asyncio.to_thread(old_worker.shutdown, 10, True) for old_worker in old_workers),
                    return_exceptions=True,
                )

            unreaped_workers = self._record_reap_results(old_workers, shutdown_results)
            if unreaped_workers:
                self.unsafe_shutdown_reason = "pre-fault CUDA context could not be confirmed reaped"
                async with self.lock:
                    if generation == self.pool_generation and not self._closing:
                        self.pending_replacements = 0
                        self._set_health_locked(
                            POOL_QUARANTINED,
                            reason="pre-fault CUDA context could not be confirmed reaped",
                            task_id=task_id,
                            fault_class="pre_fault_reap_failure",
                            scope="gpu",
                        )
                logger.critical(
                    f"[GPU {self.device_id}] QUARANTINED: {len(unreaped_workers)} worker process(es) "
                    "could not be confirmed stopped; fresh probe suppressed"
                )
                raise UnsafeGPUContainmentError(
                    self.unsafe_shutdown_reason,
                    task_id=task_id,
                    worker_id=worker.worker_id,
                )

            old_replenishments_reaped = await asyncio.to_thread(
                self._wait_for_older_replenishments,
                generation,
            )

            validation_error = ""
            validation_scope = "gpu"
            validation_fault_class = "cuda_probe_failure"
            if not old_replenishments_reaped:
                unsafe_reason = "timed out waiting for pre-fault worker constructors to finish"
                await self._quarantine_unreaped_context(
                    unsafe_reason,
                    task_id=task_id,
                )
                raise UnsafeGPUContainmentError(
                    unsafe_reason,
                    task_id=task_id,
                    worker_id=worker.worker_id,
                )
            else:
                # A stale recovery must not open another CUDA context after a
                # shutdown or a newer fault epoch changed admission state.
                async with self.lock:
                    still_current = (
                        not self._closing and generation == self.pool_generation and self.health_state == POOL_SUSPECT
                    )
                    stale_unsafe_reason = self.unsafe_shutdown_reason if not still_current else ""
                if not still_current:
                    if stale_unsafe_reason:
                        raise UnsafeGPUContainmentError(
                            stale_unsafe_reason,
                            task_id=task_id,
                            worker_id=worker.worker_id,
                        )
                    return
                try:
                    validation_worker = await asyncio.to_thread(
                        PersistentWorker,
                        f"{self.worker_prefix}_{self.device_id}_validation_{generation}",
                        self.device_id,
                        f"(post-device-fault validation generation={generation})",
                        self.max_tasks_per_worker,
                    )
                except Exception as exc:
                    validation_error = str(exc)
                    if isinstance(exc, WorkerInitializationError) and not exc.reap_confirmed:
                        unsafe_reason = f"fresh validation CUDA context could not be confirmed reaped: {exc}"
                        await self._quarantine_unreaped_context(
                            unsafe_reason,
                            task_id=task_id,
                        )
                        raise UnsafeGPUContainmentError(
                            unsafe_reason,
                            task_id=task_id,
                            worker_id=f"{self.worker_prefix}_{self.device_id}_validation_{generation}",
                        ) from exc
                    if not (isinstance(exc, WorkerInitializationError) and exc.cuda_probe_failure):
                        validation_fault_class = "worker_bootstrap_failure"

            shutdown_stale = False
            async with self.lock:
                if self._closing or generation != self.pool_generation or self.health_state != POOL_SUSPECT:
                    shutdown_stale = validation_worker is not None
                elif validation_worker is None:
                    self.pending_replacements = 0
                    self._set_health_locked(
                        POOL_QUARANTINED,
                        reason=f"post-fault fresh-context validation failed: {validation_error}",
                        task_id=task_id,
                        fault_class=validation_fault_class,
                        scope=validation_scope,
                    )
                    logger.error(
                        f"[GPU {self.device_id}] QUARANTINED after fresh-context validation failure: "
                        f"{validation_error}"
                    )
                else:
                    self.pending_replacements = 0
                    self.workers.append(validation_worker)
                    self.idle_workers.append(validation_worker)
                    self.total_workers_restarted += 1
                    self.consecutive_replacement_failures = 0
                    self._set_health_locked(POOL_HEALTHY)
                    logger.info(f"[GPU {self.device_id}] Fresh-context validation passed; reopening task admission")
                    self._ensure_capacity_locked(asyncio.get_running_loop())

            if shutdown_stale and validation_worker is not None:
                try:
                    stopped = await asyncio.to_thread(validation_worker.shutdown, 10, True)
                except Exception as shutdown_error:
                    stopped = False
                    logger.exception(
                        f"[{validation_worker.worker_id}] Stale validation shutdown failed: {shutdown_error}"
                    )
                _record_worker_reap_result(self.device_id, validation_worker, stopped)
                if stopped is not True:
                    unsafe_reason = "stale validation CUDA context could not be reaped"
                    await self._quarantine_unreaped_context(
                        unsafe_reason,
                        task_id=task_id,
                        extra_worker=validation_worker,
                    )
                    raise UnsafeGPUContainmentError(
                        unsafe_reason,
                        task_id=task_id,
                        worker_id=validation_worker.worker_id,
                    )
        except (PoolShutdownContainmentError, UnsafeGPUContainmentError):
            raise
        except Exception as recovery_error:
            extras = [*old_workers]
            if validation_worker is not None:
                extras.append(validation_worker)
            await self._contain_unexpected_hard_recovery_failure(
                recovery_error,
                task_id=task_id,
                extra_workers=tuple(extras),
            )
            if self.unsafe_shutdown_reason:
                raise UnsafeGPUContainmentError(
                    self.unsafe_shutdown_reason,
                    task_id=task_id,
                    worker_id=worker.worker_id,
                ) from recovery_error
            raise
        finally:
            self._finish_replenishment_ticket(generation)

    def _next_top_up_worker_id(self) -> str:
        """Return a unique worker id for capacity top-up spares."""
        self._top_up_sequence += 1
        return f"{self.worker_prefix}_{self.device_id}_topup_{self._top_up_sequence}"

    def _ensure_capacity_locked(self, loop: asyncio.AbstractEventLoop) -> None:
        """Schedule spare creation until workers + pending reaches pool_size.

        The caller must hold ``self.lock``. This maintains the lower-bound
        side of the warm-pool invariant; the existing replacement accounting
        already prevents the pool from growing above ``pool_size``.
        """
        if self._closing or self.health_state != POOL_HEALTHY:
            return
        while (
            not self._closing
            and self.health_state == POOL_HEALTHY
            and len(self.workers) + self.pending_replacements + self.pending_retirements < self.pool_size
        ):
            worker_id = self._next_top_up_worker_id()
            self.pending_replacements += 1
            logger.warning(
                f"[GPU {self.device_id}] Pool below configured size; scheduling top-up spare "
                f"(workers={len(self.workers)}, idle={len(self.idle_workers)}, "
                f"busy={len(self.busy_workers)}, pending={self.pending_replacements}, "
                f"pool_size={self.pool_size}, new_worker={worker_id})"
            )
            self._start_replenishment_thread(
                old_worker=None,
                old_process=None,
                old_pid=None,
                worker_id=worker_id,
                loop=loop,
                reason="top-up",
                generation=self.pool_generation,
            )

    def _start_replenishment_thread(
        self,
        *,
        old_worker: Optional[PersistentWorker],
        old_process: Optional[mp.Process],
        old_pid: Optional[int],
        worker_id: str,
        loop: asyncio.AbstractEventLoop,
        reason: str,
        generation: Optional[int] = None,
        validates_context_fault: bool = False,
    ) -> None:
        """Start one background replacement/top-up worker creation thread.

        ``pending_replacements`` must already have been incremented by the
        caller before this is invoked.
        """

        old_wid = worker_id if old_worker is None else old_worker.worker_id
        if generation is None:
            generation = self.pool_generation
        self._register_replenishment_ticket(generation)
        ticket_finished = threading.Event()
        ticket_finish_lock = threading.Lock()
        detached_worker_reaped = threading.Event()

        def _finish_ticket_directly() -> None:
            with ticket_finish_lock:
                if ticket_finished.is_set():
                    return
                ticket_finished.set()
                self._finish_replenishment_ticket(generation)

        def _shutdown_retained_callback_worker(worker: PersistentWorker) -> None:
            """Finish a callback ticket only after its CUDA process is reaped."""

            try:
                stopped = worker.shutdown(10, True)
            except BaseException as shutdown_error:
                stopped = False
                logger.exception(
                    f"[{worker.worker_id}] Cancelled replenishment callback cleanup failed: {shutdown_error}"
                )
            _record_worker_reap_result(self.device_id, worker, stopped)
            if stopped is not True:
                logger.critical(f"[{worker.worker_id}] READY worker survived a cancelled/failed registration callback")
            _finish_ticket_directly()

        def _schedule_with_ticket(
            coro: Any,
            *,
            cleanup_worker: Optional[PersistentWorker] = None,
            registration_aborted: Optional[threading.Event] = None,
            registration_guard: Optional[threading.Lock] = None,
        ) -> None:
            """Keep the ticket through callback completion or proven cleanup."""

            try:
                future = asyncio.run_coroutine_threadsafe(coro, loop)
            except Exception:
                close = getattr(coro, "close", None)
                if close is not None:
                    close()
                if registration_aborted is not None:
                    registration_aborted.set()
                if cleanup_worker is not None and _worker_handle_is_retained(self.device_id, cleanup_worker):
                    _shutdown_retained_callback_worker(cleanup_worker)
                else:
                    _finish_ticket_directly()
                raise

            def _finish_callback(done_future: Any) -> None:
                callback_failed = False
                try:
                    done_future.result()
                except BaseException as exc:
                    callback_failed = True
                    logger.error(f"[{worker_id}] Replenishment callback failed: {exc}")
                guard = registration_guard or threading.Lock()
                with guard:
                    retained = cleanup_worker is not None and _worker_handle_is_retained(
                        self.device_id, cleanup_worker
                    )
                    if callback_failed and retained and registration_aborted is not None:
                        registration_aborted.set()
                if cleanup_worker is not None and retained:
                    # A Future can be cancelled before its coroutine is first
                    # polled (or while the loop is closing).  The constructor
                    # has nevertheless reached READY.  Retain the Process and
                    # reap it outside the event loop before ending the ticket.
                    try:
                        cleanup_thread = threading.Thread(
                            target=_shutdown_retained_callback_worker,
                            args=(cleanup_worker,),
                            daemon=True,
                            name=f"replenishment-callback-reaper-{worker_id}",
                        )
                        cleanup_thread.start()
                    except BaseException as thread_error:
                        logger.critical(
                            f"[{cleanup_worker.worker_id}] Could not start callback cleanup thread: "
                            f"{thread_error}; reaping synchronously"
                        )
                        _shutdown_retained_callback_worker(cleanup_worker)
                else:
                    _finish_ticket_directly()

            future.add_done_callback(_finish_callback)

        def _schedule_failure(
            error: Any,
            *,
            force_gpu_scope: bool = False,
            fault_class: Optional[str] = None,
        ) -> None:
            """Resolve pending capacity and fail closed after thread errors."""

            replacement_error = str(error)
            cuda_probe_failure = isinstance(error, WorkerInitializationError) and error.cuda_probe_failure
            # Once context validation has started, only a successful fresh
            # context may reopen this physical GPU.  Infrastructure/drain
            # failures still leave the device unproven and cannot be scoped to
            # a replaceable worker alias.
            physical_scope = force_gpu_scope or cuda_probe_failure or validates_context_fault
            resolved_fault_class = fault_class or (
                "cuda_probe_failure" if cuda_probe_failure else "worker_bootstrap_failure"
            )

            async def _mark_failed() -> None:
                workers_to_shutdown: List[PersistentWorker] = []
                retry_delay_s: Optional[float] = None
                if old_worker is not None and not detached_worker_reaped.is_set():
                    try:
                        detached_stopped = await asyncio.to_thread(old_worker.shutdown, 10, True)
                    except Exception as detached_error:
                        detached_stopped = False
                        logger.exception(f"[{old_worker.worker_id}] Detached worker shutdown failed: {detached_error}")
                    _record_worker_reap_result(self.device_id, old_worker, detached_stopped)
                    if detached_stopped is not True:
                        await self._quarantine_unreaped_context(
                            f"detached worker {old_worker.worker_id} could not be reaped after replenishment failure",
                            task_id=self.health_task_id,
                            extra_worker=old_worker,
                        )
                        return
                async with self.lock:
                    if self._closing or generation != self.pool_generation:
                        return
                    self.consecutive_replacement_failures += 1
                    must_quarantine = validates_context_fault or physical_scope
                    if must_quarantine:
                        self.pending_replacements = max(0, self.pending_replacements - 1)
                        failure_kind = "post-fault validation" if validates_context_fault else "worker replacement"
                        # The background thread has already synchronously
                        # reaped ``old_worker`` before attempting the failed
                        # constructor.  Do not submit an identical shutdown to
                        # the event-loop executor: besides being redundant, it
                        # can deadlock tests/embedders whose thread factory is
                        # intentionally unavailable.
                        extras = (
                            (old_worker,) if old_worker is not None and not detached_worker_reaped.is_set() else ()
                        )
                        workers_to_shutdown = self._tracked_workers_locked(*extras)
                        self.workers.clear()
                        self.idle_workers.clear()
                        self.busy_workers.clear()
                        self.pool_generation += 1
                        self.pending_replacements = 0
                        self._set_health_locked(
                            POOL_QUARANTINED,
                            reason=f"{failure_kind} failed: {replacement_error}",
                            task_id=self.health_task_id,
                            fault_class=resolved_fault_class,
                            scope="gpu" if physical_scope else "worker",
                        )
                        logger.error(
                            f"[GPU {self.device_id}] QUARANTINED after {failure_kind} failure: {replacement_error}"
                        )
                    elif self.consecutive_replacement_failures >= self.max_replacement_failures:
                        # The failed constructor was reaped and no CUDA/device
                        # fault was observed.  Close admission and ask the outer
                        # monitor to recycle this worker process; do not create a
                        # durable GPU/worker quarantine for recoverable bootstrap
                        # congestion.
                        self.pending_replacements = max(0, self.pending_replacements - 1)
                        self.pool_generation += 1
                        self.pending_replacements = 0
                        self._set_health_locked(
                            POOL_BOOTSTRAP_FAILED,
                            reason=(
                                f"worker replacement infrastructure failed "
                                f"{self.consecutive_replacement_failures} consecutive times: {replacement_error}"
                            ),
                            task_id=self.health_task_id,
                            fault_class="worker_bootstrap_failure",
                            scope="worker",
                        )
                        logger.error(
                            f"[GPU {self.device_id}] Worker bootstrap exhausted after "
                            f"{self.consecutive_replacement_failures} confirmed-reap failures; "
                            "outer worker restart required"
                        )
                    else:
                        # Keep this failed constructor's capacity reservation
                        # through the delay so another pool path cannot create a
                        # parallel retry storm.
                        retry_delay_s = min(
                            30.0,
                            _REPLACEMENT_RETRY_BASE_DELAY_S * (2 ** max(0, self.consecutive_replacement_failures - 1)),
                        )
                        logger.warning(
                            f"[GPU {self.device_id}] Confirmed-reap worker bootstrap failure "
                            f"{self.consecutive_replacement_failures}/{self.max_replacement_failures}; "
                            f"retrying capacity in {retry_delay_s:.1f}s: {replacement_error}"
                        )
                if workers_to_shutdown:
                    shutdown_results = await asyncio.gather(
                        *(asyncio.to_thread(item.shutdown, 10, True) for item in workers_to_shutdown),
                        return_exceptions=True,
                    )
                    unreaped_workers = self._record_reap_results(workers_to_shutdown, shutdown_results)
                    for item in unreaped_workers:
                        await self._quarantine_unreaped_context(
                            f"worker {item.worker_id} could not be reaped after replenishment failure",
                            task_id=self.health_task_id,
                            extra_worker=item,
                        )
                if retry_delay_s is not None:
                    await asyncio.sleep(retry_delay_s)
                    async with self.lock:
                        # A generation transition resets/reassigns capacity
                        # accounting.  A stale delayed retry must not consume a
                        # reservation belonging to the new epoch.
                        if self._closing or generation != self.pool_generation:
                            return
                        self.pending_replacements = max(0, self.pending_replacements - 1)
                        self._ensure_capacity_locked(loop)

            try:
                _schedule_with_ticket(_mark_failed())
            except Exception as schedule_error:
                logger.critical(
                    f"[{worker_id}] Could not deliver replenishment failure to event loop: "
                    f"{schedule_error}; pool shutdown is required"
                )

        async def _mark_unreaped_new_worker(
            reason_text: str,
            unreaped_worker: PersistentWorker,
        ) -> None:
            """A constructor completed but its CUDA process could not be reaped."""

            await self._quarantine_unreaped_context(
                reason_text,
                task_id=self.health_task_id,
                extra_worker=unreaped_worker,
            )

        def _background_replenish_inner() -> None:
            """Create one replacement/top-up worker and register it."""

            # Keep process lifecycle tied to the multiprocessing.Process
            # object.  Raw PID signalling risks killing an unrelated process
            # after PID reuse and cannot prove the CUDA context was reaped.
            if old_worker is not None:
                try:
                    old_stopped = old_worker.shutdown(10)
                except Exception as shutdown_error:
                    logger.exception(f"[{old_wid}] Old worker shutdown failed: {shutdown_error}")
                    old_stopped = False
                _record_worker_reap_result(self.device_id, old_worker, old_stopped)
                if old_stopped is not True:
                    _schedule_failure(
                        "old CUDA worker could not be confirmed stopped; fresh constructor suppressed",
                        force_gpu_scope=True,
                        fault_class="pre_fault_reap_failure",
                    )
                    return
                detached_worker_reaped.set()
                logger.info(f"[{old_wid}] Old process pid={old_pid} cleanup verified")
                time.sleep(2.0)
            elif old_process is not None:
                _schedule_failure(
                    "replacement received a raw process without its PersistentWorker owner",
                    force_gpu_scope=True,
                    fault_class="pre_fault_reap_failure",
                )
                return

            # A fault changes the generation while older capacity workers may
            # already be starting.  Cancel any constructor that has not begun;
            # if it has already completed, the registration callback below
            # will identify it as stale and synchronously reap it.
            if self._closing or generation != self.pool_generation:
                _finish_ticket_directly()
                return

            if validates_context_fault and not self._wait_for_older_replenishments(generation):
                _schedule_failure(
                    "timed out waiting for pre-fault worker constructors to finish",
                    force_gpu_scope=True,
                    fault_class="replenishment_drain_timeout",
                )
                return

            if self._closing or generation != self.pool_generation:
                _finish_ticket_directly()
                return
            if validates_context_fault and self.health_state != POOL_DEGRADED_CHECK:
                _finish_ticket_directly()
                return

            try:
                new_worker = PersistentWorker(
                    worker_id,
                    self.device_id,
                    f"(pool_size={self.pool_size}, max_tasks={self.max_tasks_per_worker}, {reason})",
                    max_tasks_per_worker=self.max_tasks_per_worker,
                )
            except Exception as e:
                logger.error(
                    f"[{worker_id}] Failed to create {reason} spare: {e}. Pool now has {len(self.workers)} workers"
                )
                _schedule_failure(e)
                return

            # A READY CUDA context exists before its event-loop registration
            # callback.  Retain the only Process handle immediately so loop
            # closure/cancellation and shutdown snapshots cannot omit it.
            _retain_unreaped_worker(self.device_id, new_worker)
            registration_aborted = threading.Event()
            registration_guard = threading.Lock()

            async def _register():
                shutdown_new_worker = False
                async with self.lock:
                    with registration_guard:
                        if registration_aborted.is_set():
                            shutdown_new_worker = True
                        elif self._closing or generation != self.pool_generation:
                            shutdown_new_worker = True
                        elif validates_context_fault and self.health_state != POOL_DEGRADED_CHECK:
                            shutdown_new_worker = True
                        elif not validates_context_fault and self.health_state != POOL_HEALTHY:
                            shutdown_new_worker = True

                        if shutdown_new_worker:
                            # Stale generations no longer contribute to current
                            # pending capacity accounting.
                            pass
                        else:
                            self.pending_replacements = max(0, self.pending_replacements - 1)
                            if len(self.workers) < self.pool_size:
                                self.workers.append(new_worker)
                                self.idle_workers.append(new_worker)
                                _release_reaped_worker(self.device_id, new_worker)
                                self.total_workers_restarted += 1
                                self.consecutive_replacement_failures = 0
                                if validates_context_fault:
                                    self._set_health_locked(POOL_HEALTHY)
                                    logger.info(
                                        f"[GPU {self.device_id}] Post-context-fault fresh worker READY; "
                                        "reopening normal warm-spare admission"
                                    )
                                logger.info(
                                    f"[{new_worker.worker_id}] Background spare ready — "
                                    f"pool: workers={len(self.workers)} idle={len(self.idle_workers)} "
                                    f"busy={len(self.busy_workers)} pending={self.pending_replacements} "
                                    f"(total restarts: {self.total_workers_restarted})"
                                )
                            else:
                                shutdown_new_worker = True
                                logger.warning(
                                    f"[{new_worker.worker_id}] Replacement no longer needed "
                                    f"(workers={len(self.workers)}, pool_size={self.pool_size}); shutting it down"
                                )
                            self._ensure_capacity_locked(loop)
                if shutdown_new_worker:
                    try:
                        stopped = await asyncio.to_thread(new_worker.shutdown, 10, True)
                    except Exception as shutdown_error:
                        stopped = False
                        logger.exception(
                            f"[{new_worker.worker_id}] Stale replacement shutdown failed: {shutdown_error}"
                        )
                    _record_worker_reap_result(self.device_id, new_worker, stopped)
                    if stopped is not True:
                        await _mark_unreaped_new_worker(
                            f"replacement worker {new_worker.worker_id} became stale but its "
                            "CUDA context could not be reaped",
                            new_worker,
                        )

            try:
                _schedule_with_ticket(
                    _register(),
                    cleanup_worker=new_worker,
                    registration_aborted=registration_aborted,
                    registration_guard=registration_guard,
                )
            except Exception:
                stopped = new_worker.shutdown(10)
                _record_worker_reap_result(self.device_id, new_worker, stopped)
                if stopped is not True:
                    logger.critical(
                        f"[{new_worker.worker_id}] Event-loop registration failed and the new CUDA "
                        "context could not be reaped"
                    )

        def _background_replenish():
            try:
                _background_replenish_inner()
            except Exception as exc:
                logger.exception(f"[{worker_id}] Unexpected replenishment thread failure: {exc}")
                _schedule_failure(exc, fault_class="replenishment_thread_failure")

        try:
            t = threading.Thread(target=_background_replenish, daemon=True)
            t.start()
        except Exception as start_error:
            # Schedule the same containment callback used by runtime thread
            # failures.  This method is normally invoked while self.lock is
            # held, so synchronous shutdown here could freeze the event loop.
            logger.critical(f"[{worker_id}] Replenishment thread could not start: {start_error}")
            _schedule_failure(start_error, fault_class="replenishment_thread_failure")

    def _init_workers(self):
        """初始化所有 workers"""
        pool_info = f"(pool_size={self.pool_size}, max_tasks={self.max_tasks_per_worker})"

        for i in range(self.pool_size):
            worker_id = f"{self.worker_prefix}_{self.device_id}_{i}"
            try:
                worker = PersistentWorker(
                    worker_id, self.device_id, pool_info, max_tasks_per_worker=self.max_tasks_per_worker
                )
                self.workers.append(worker)
                self.idle_workers.append(worker)
            except Exception as e:
                logger.error(f"[GPU {self.device_id}] Failed to start worker {worker_id}: {e}")
                # A fresh CUDA context failed.  Do not immediately hammer
                # cuInit with every remaining pool slot; reap any contexts that
                # did start and let GPUWorker persist the quarantine latch.
                reap_failed = False
                for started_worker in self._tracked_workers_locked():
                    try:
                        shutdown_result = started_worker.shutdown(timeout=10, force=True)
                        if not _record_worker_reap_result(
                            self.device_id,
                            started_worker,
                            shutdown_result,
                        ):
                            reap_failed = True
                    except Exception as shutdown_error:
                        _record_worker_reap_result(self.device_id, started_worker, shutdown_error)
                        reap_failed = True
                        logger.exception(
                            f"[{started_worker.worker_id}] Initial-pool rollback shutdown failed: {shutdown_error}"
                        )
                self.workers.clear()
                self.idle_workers.clear()
                self.busy_workers.clear()
                message = f"[GPU {self.device_id}] Worker pool start aborted: {e}"
                if reap_failed:
                    raise GPUProbeFailedError(
                        f"{message}; a partially initialized CUDA context could not be confirmed reaped"
                    ) from e
                if isinstance(e, WorkerInitializationError) and e.cuda_probe_failure:
                    raise GPUProbeFailedError(message) from e
                raise WorkerPoolInfrastructureError(message) from e

    async def execute_task(
        self,
        task_data: Dict[str, Any],
        timeout: int = 60,
        max_retries: int = 2,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """
        执行任务（自动选择空闲 worker）

        Args:
            task_data: 任务数据
            timeout: 超时时间
            max_retries: 最大重试次数（用于 worker 重启后重试）
                        注意：timeout错误不会重试，以避免阻塞队列
            cancel_event: 若被 set，则中止执行、回收 worker 并抛出 TaskCancelledError

        Returns:
            结果字典

        Raises:
            TaskCancelledError: 任务被取消（不会重试）
        """
        retry_count = 0
        last_error = None
        is_timeout_error = False  # Track if error was timeout
        terminal_device_fault = False
        task_id = task_data.get("task_id", "unknown")
        request_start = time.time()
        total_idle_wait_s = 0.0
        total_restart_s = 0.0
        last_execute_s = 0.0
        last_return_s = 0.0
        stage_dir = _require_fast_rw_dir(
            os.environ.get(_STAGE_METADATA_DIR_ENV, _STAGE_METADATA_DEFAULT_DIR),
            label=_STAGE_METADATA_DIR_ENV,
        )
        stage_metadata_path = os.path.join(
            stage_dir,
            f"kernelgym_stage_{os.getpid()}_{self.device_id}_{uuid.uuid4().hex}.json",
        )
        task_data["_stage_metadata_path"] = stage_metadata_path

        def _build_pool_timing(total_s: Optional[float] = None) -> Dict[str, Any]:
            final_total = time.time() - request_start if total_s is None else total_s
            return {
                "pool_idle_wait_s": total_idle_wait_s,
                "pool_execute_s": last_execute_s,
                "pool_restart_s": total_restart_s,
                "pool_return_s": last_return_s,
                "pool_total_s": final_total,
                "pool_retry_count": retry_count,
            }

        while retry_count <= max_retries:
            # 取消优先：在分配 worker 前就检查，避免对已取消任务还去抢占资源
            if cancel_event is not None and cancel_event.is_set():
                raise TaskCancelledError(f"[GPU {self.device_id}] Task {task_id} cancelled before dispatch")
            # 获取空闲 worker
            idle_wait_start = time.time()
            worker = await self._get_idle_worker(timeout=timeout, cancel_event=cancel_event)
            total_idle_wait_s += time.time() - idle_wait_start

            if worker is None:
                # 所有 workers 都忙，等待一下再试
                await asyncio.sleep(0.5)
                retry_count += 1
                continue

            try:
                # 执行任务（在线程池中执行，避免阻塞 asyncio）
                loop = asyncio.get_event_loop()
                execute_start = time.time()
                result = await loop.run_in_executor(None, worker.execute_task, task_data, timeout, cancel_event)
                last_execute_s = time.time() - execute_start

                # 任务完成
                self.total_tasks_processed += 1

                child_fault_severity = str(result.get("fault_severity") or FAULT_NONE)
                parent_fault_severity = _classify_cuda_fault(
                    str(result.get("error_type") or ""),
                    str(result.get("error_message") or ""),
                    final_sync_failed=result.get("final_sync_failed") is True,
                )
                fault_severity = _strongest_cuda_fault(child_fault_severity, parent_fault_severity)
                result["fault_severity"] = fault_severity
                result_payload = result.get("result")
                result_metadata = result_payload.get("metadata") if isinstance(result_payload, dict) else None
                has_deferred_sanitizer = bool(
                    isinstance(result_metadata, dict)
                    and isinstance(result_metadata.get("_runtime_sanitizer_request"), dict)
                )

                async def _complete_deferred_sanitizer(skip_reason: Optional[str] = None) -> None:
                    if has_deferred_sanitizer:
                        elapsed_s = time.time() - execute_start
                        remaining_s = max(0.0, float(timeout) - elapsed_s)
                        if not skip_reason and remaining_s <= 0:
                            skip_reason = "diagnostic skipped because the task timeout budget was exhausted"
                        await asyncio.to_thread(
                            _run_deferred_compute_sanitizer,
                            result,
                            skip_reason,
                            remaining_s,
                        )

                if fault_severity != child_fault_severity:
                    logger.warning(
                        f"[{worker.worker_id}] Parent strengthened child fault classification "
                        f"from {child_fault_severity!r} to {fault_severity!r}"
                    )
                if fault_severity == FAULT_CONTEXT:
                    restart_start = time.time()
                    validation_generation = await self._begin_context_fault_validation(
                        task_id=task_id,
                        reason=f"{result.get('error_type', 'CUDAError')}: {result.get('error_message', '')}",
                        worker=worker,
                        allow_speculative_spare=not has_deferred_sanitizer,
                    )
                    if validation_generation == _STALE_HARD_RECOVERY_EPOCH:
                        # The epoch changed because another task owns hard
                        # recovery. Serialize behind it before returning this
                        # old-context result; its reap may still fail unsafe.
                        await self._recover_from_device_fault(
                            worker,
                            task_id=task_id,
                            reason="stale context fault waiting for concurrent hard recovery",
                        )
                        await _complete_deferred_sanitizer("diagnostic skipped after stale hard-recovery epoch")
                        logger.info(
                            f"[{worker.worker_id}] Context fault belongs to an already retired hard epoch; "
                            "no additional fresh-context probe will be created"
                        )
                    elif validation_generation is None:
                        # A second CUDA-context fault while the post-fault
                        # context is still being validated is physical-device
                        # evidence: stop every pre-fault context immediately.
                        await self._recover_from_device_fault(
                            worker,
                            task_id=task_id,
                            reason="second CUDA context fault before validation completed",
                        )
                        await _complete_deferred_sanitizer(
                            "diagnostic skipped because concurrent context faults required hard recovery"
                        )
                    else:
                        await self._restart_worker(
                            worker,
                            task_id=task_id,
                            validation_generation=validation_generation,
                            after_reap_before_replenish=_complete_deferred_sanitizer,
                        )
                    total_restart_s += time.time() - restart_start
                elif fault_severity == FAULT_DEVICE:
                    restart_start = time.time()
                    await self._recover_from_device_fault(
                        worker,
                        task_id=task_id,
                        reason=f"{result.get('error_type', 'DeviceFault')}: {result.get('error_message', '')}",
                    )
                    await _complete_deferred_sanitizer(
                        "diagnostic skipped because the original failure was classified as a device fault"
                    )
                    total_restart_s += time.time() - restart_start
                elif not worker.is_alive():
                    # Normal max-tasks recycle, OOM/profiler dropout, and other
                    # non-device exits retain the existing warm-spare path.
                    logger.warning(f"[{worker.worker_id}] Worker needs restart after task")
                    restart_start = time.time()
                    await self._restart_worker(worker, task_id=task_id)
                    total_restart_s += time.time() - restart_start

                pool_timing = _build_pool_timing()
                result["pool_timing"] = pool_timing
                result["pool_health"] = self.get_health_snapshot()
                result_status = "success" if result.get("success", False) else "failed"
                logger.info(
                    f"[PoolTiming] device=cuda:{self.device_id} worker={worker.worker_id} "
                    f"task={task_id} status={result_status} idle_wait_s={pool_timing['pool_idle_wait_s']:.2f} "
                    f"execute_s={pool_timing['pool_execute_s']:.2f} restart_s={pool_timing['pool_restart_s']:.2f} "
                    f"return_s={pool_timing['pool_return_s']:.2f} total_s={pool_timing['pool_total_s']:.2f} "
                    f"retries={pool_timing['pool_retry_count']}"
                )

                return result

            except (PoolShutdownContainmentError, UnsafeGPUContainmentError):
                # This task's old CUDA context may still be alive.  Never turn
                # that state into an ordinary retry/result; GPUWorker must
                # freeze the exact inflight claim and stop admission.
                raise

            except TaskCancelledError as e:
                # The caller stopped observing the result while arbitrary
                # asynchronous GPU work may still be running.  Treat that as
                # a device-suspect boundary: gate admission, prove the old
                # context exited, then create exactly one fresh context.
                logger.warning(
                    f"[{worker.worker_id}] Task {task_id} cancelled mid-flight; "
                    "gating GPU until its in-flight CUDA context is reaped"
                )
                last_error = e
                restart_start = time.time()
                await self._recover_from_device_fault(
                    worker,
                    task_id=task_id,
                    reason="in-flight CUDA task cancelled before completion was observed",
                )
                total_restart_s += time.time() - restart_start
                raise

            except asyncio.CancelledError:
                # Cancelling the asyncio waiter does not cancel the executor
                # thread or the CUDA work already running in the child.  Mark
                # the worker unusable before the finally block and complete a
                # hard recovery before propagating cancellation.
                worker.is_alive_flag = False
                logger.warning(
                    f"[{worker.worker_id}] Async task {task_id} cancelled while CUDA work is in flight; "
                    "gating GPU until the context is reaped"
                )
                await self._recover_from_device_fault(
                    worker,
                    task_id=task_id,
                    reason="async waiter cancelled while CUDA work remained in flight",
                )
                raise

            except (RuntimeError, TimeoutError) as e:
                # Worker 可能已死亡或超时
                logger.error(f"[{worker.worker_id}] Task execution failed: {e}")
                last_error = e

                # Check if this is a timeout error
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "Task timeout after" in error_msg:
                    is_timeout_error = True
                    task_id = task_data.get("task_id", "unknown")
                    logger.warning(
                        f"[{worker.worker_id}] Task {task_id} timeout detected "
                        f"(timeout={timeout}s) - will NOT retry to avoid blocking queue"
                    )

                restart_start = time.time()
                exception_severity = _classify_cuda_fault(type(e).__name__, str(e))
                if isinstance(e, TimeoutError) or (exception_severity == FAULT_NONE and not worker.is_alive()):
                    exception_severity = FAULT_DEVICE

                if exception_severity == FAULT_DEVICE:
                    await self._recover_from_device_fault(
                        worker,
                        task_id=task_id,
                        reason=f"{type(e).__name__}: {e}",
                    )
                    terminal_device_fault = True
                elif exception_severity == FAULT_CONTEXT:
                    validation_generation = await self._begin_context_fault_validation(
                        task_id=task_id,
                        reason=f"{type(e).__name__}: {e}",
                        worker=worker,
                    )
                    if validation_generation == _STALE_HARD_RECOVERY_EPOCH:
                        await self._recover_from_device_fault(
                            worker,
                            task_id=task_id,
                            reason="stale context exception waiting for concurrent hard recovery",
                        )
                        logger.info(
                            f"[{worker.worker_id}] Context exception belongs to an already retired hard epoch; "
                            "the payload will not be replayed or reprobed"
                        )
                        terminal_device_fault = True
                    elif validation_generation is None:
                        await self._recover_from_device_fault(
                            worker,
                            task_id=task_id,
                            reason="second CUDA context fault before validation completed",
                        )
                        terminal_device_fault = True
                    else:
                        await self._restart_worker(
                            worker,
                            task_id=task_id,
                            validation_generation=validation_generation,
                        )
                else:
                    await self._restart_worker(worker, task_id=task_id)
                total_restart_s += time.time() - restart_start

                # Don't retry if timeout - exit immediately to free up worker queue
                if is_timeout_error:
                    pool_timing = _build_pool_timing()
                    logger.info(
                        f"[PoolTiming] device=cuda:{self.device_id} worker={worker.worker_id} "
                        f"task={task_id} status=timeout idle_wait_s={pool_timing['pool_idle_wait_s']:.2f} "
                        f"execute_s={pool_timing['pool_execute_s']:.2f} restart_s={pool_timing['pool_restart_s']:.2f} "
                        f"return_s={pool_timing['pool_return_s']:.2f} total_s={pool_timing['pool_total_s']:.2f} "
                        f"retries={pool_timing['pool_retry_count']}"
                    )
                    logger.error(
                        f"[{worker.worker_id}] Task failed due to timeout, not retrying to free up worker queue"
                    )
                    break  # Exit retry loop immediately

                if terminal_device_fault:
                    logger.error(
                        f"[{worker.worker_id}] Task hit a device-level fault; "
                        "the dangerous payload will not be replayed"
                    )
                    break

                # 重试（仅对非timeout错误）
                retry_count += 1

            except Exception as e:
                # A transport/runtime exception outside the known RuntimeError
                # family cannot prove whether the CUDA child is still active.
                # Fail closed as a device-suspect event instead of returning
                # the worker idle and publishing an ordinary retryable result.
                logger.exception(
                    f"[{worker.worker_id}] Unexpected worker execution exception; "
                    "gating GPU until process containment is proven"
                )
                last_error = e
                restart_start = time.time()
                await self._recover_from_device_fault(
                    worker,
                    task_id=task_id,
                    reason=f"unexpected worker execution exception {type(e).__name__}: {e}",
                )
                total_restart_s += time.time() - restart_start
                terminal_device_fault = True
                break

            finally:
                # 归还 worker 到 idle pool（如果还存活）
                primary_error = sys.exception()
                return_start = time.time()
                return_cancellation_requested = False
                try:
                    _, return_cancellation_requested = await _complete_despite_cancellation(
                        self._return_worker(worker)
                    )
                except BaseException as return_error:
                    if isinstance(
                        primary_error,
                        (PoolShutdownContainmentError, UnsafeGPUContainmentError),
                    ):
                        logger.critical(
                            f"[{worker.worker_id}] Worker-return cleanup raised "
                            f"{type(return_error).__name__} while unsafe containment was propagating; "
                            "preserving the containment signal"
                        )
                        raise primary_error
                    if isinstance(return_error, (PoolShutdownContainmentError, UnsafeGPUContainmentError)):
                        raise
                    if primary_error is not None:
                        logger.error(
                            f"[{worker.worker_id}] Worker-return cleanup raised "
                            f"{type(return_error).__name__}; preserving the primary "
                            f"{type(primary_error).__name__}"
                        )
                        raise primary_error
                    if not isinstance(return_error, Exception):
                        raise
                    # The child result is not published until its CUDA commit
                    # barrier and cleanup have completed.  A later pool-
                    # bookkeeping/top-up failure may reduce capacity, but it
                    # must not rewrite that observed result as a task failure.
                    logger.exception(
                        f"[{worker.worker_id}] Worker-return cleanup failed after the task result "
                        "was committed; preserving the completed result"
                    )
                last_return_s = time.time() - return_start
                if return_cancellation_requested:
                    if isinstance(
                        primary_error,
                        (PoolShutdownContainmentError, UnsafeGPUContainmentError),
                    ):
                        logger.critical(
                            f"[{worker.worker_id}] Cancellation arrived during worker-return cleanup; "
                            "preserving the unsafe containment signal"
                        )
                    else:
                        raise asyncio.CancelledError

        # Failed after all retries (or timeout)
        if is_timeout_error:
            task_id = task_data.get("task_id", "unknown")
            raise TimeoutError(
                f"[GPU {self.device_id}] Task {task_id} timeout after {timeout}s. "
                f"Not retried to avoid blocking worker queue."
            )
        else:
            pool_timing = _build_pool_timing()
            logger.info(
                f"[PoolTiming] device=cuda:{self.device_id} task={task_id} status=failed "
                f"idle_wait_s={pool_timing['pool_idle_wait_s']:.2f} "
                f"execute_s={pool_timing['pool_execute_s']:.2f} restart_s={pool_timing['pool_restart_s']:.2f} "
                f"return_s={pool_timing['pool_return_s']:.2f} total_s={pool_timing['pool_total_s']:.2f} "
                f"retries={pool_timing['pool_retry_count']}"
            )
            if terminal_device_fault:
                raise RuntimeError(
                    f"[GPU {self.device_id}] Task stopped after a device-level fault and was not retried. "
                    f"Last error: {last_error}"
                )
            raise RuntimeError(
                f"[GPU {self.device_id}] Task failed after {max_retries} retries. Last error: {last_error}"
            )

    async def _get_idle_worker(
        self, timeout: int = 60, cancel_event: Optional[threading.Event] = None
    ) -> Optional[PersistentWorker]:
        """
        获取一个空闲的 worker

        如果所有 workers 都忙，会等待直到有 worker 空闲。
        若在等待期间 ``cancel_event`` 被 set，立即返回 None，让上层中止。
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if cancel_event is not None and cancel_event.is_set():
                return None
            workers_to_reap: List[PersistentWorker] = []
            topology_corrupt = False
            reap_ticket_generation: Optional[int] = None
            async with self.lock:
                if self._closing:
                    raise GPUQuarantinedError(f"[GPU {self.device_id}] Worker pool is shutting down")
                if self.health_state in {POOL_SUSPECT, POOL_BOOTSTRAP_FAILED, POOL_QUARANTINED}:
                    raise GPUQuarantinedError(
                        f"[GPU {self.device_id}] Task admission blocked: "
                        f"state={self.health_state}, reason={self.health_reason}"
                    )

                tracked = self._tracked_workers_locked()
                tracked_ids = {id(item) for item in self.workers}
                retiring_ids = self._retiring_worker_ids
                live_orphans = [
                    item
                    for item in [*self.idle_workers, *self.busy_workers]
                    if id(item) not in tracked_ids and id(item) not in retiring_ids and item.is_alive()
                ]
                busy_ids = {id(item) for item in self.busy_workers}
                dead_busy_worker_present = any(
                    id(item) in busy_ids and id(item) not in retiring_ids and not item.is_alive() for item in tracked
                )
                dead_workers = [
                    item
                    for item in tracked
                    if id(item) not in busy_ids and id(item) not in retiring_ids and not item.is_alive()
                ]
                if live_orphans:
                    # Silently dropping a live, untracked Process loses the
                    # only safe handle capable of reaping its CUDA context.
                    topology_corrupt = True
                    workers_to_reap = tracked
                    reap_ticket_generation = self.pool_generation
                    self._register_replenishment_ticket(reap_ticket_generation)
                    self.pending_retirements += len(workers_to_reap)
                    self.workers.clear()
                    self.idle_workers.clear()
                    self.busy_workers.clear()
                    self.pool_generation += 1
                    self.hard_recovery_epoch += 1
                    self.pending_replacements = 0
                    self._set_health_locked(
                        POOL_QUARANTINED,
                        reason="live CUDA worker found outside canonical pool membership",
                        fault_class="worker_topology_corruption",
                        scope="gpu",
                    )
                elif dead_workers:
                    # Reap first; only a later loop iteration may start a
                    # replacement.  This prevents old/new context overlap.
                    workers_to_reap = dead_workers
                    reap_ticket_generation = self.pool_generation
                    self._register_replenishment_ticket(reap_ticket_generation)
                    self.pending_retirements += len(workers_to_reap)
                    dead_ids = {id(item) for item in dead_workers}
                    self.workers = [item for item in self.workers if id(item) not in dead_ids]
                    self.idle_workers = [item for item in self.idle_workers if id(item) not in dead_ids]
                    self.busy_workers = [item for item in self.busy_workers if id(item) not in dead_ids]
                else:
                    self._ensure_capacity_locked(asyncio.get_running_loop())

                # A busy owner is responsible for classifying/recovering its
                # just-finished worker.  Until it does, do not race another
                # dispatch or reap/replace the same object from this path.
                may_dispatch = not dead_busy_worker_present and (
                    self.health_state == POOL_HEALTHY
                    or (self.health_state == POOL_DEGRADED_CHECK and self.speculative_dispatches_remaining > 0)
                )
                if not workers_to_reap and self.idle_workers and may_dispatch:
                    worker = self.idle_workers.pop(0)
                    self.busy_workers.append(worker)
                    # Fault reports carry the generation in which this CUDA
                    # context was admitted.  This avoids confusing a stale
                    # concurrent report with a fault in the newly validated
                    # generation.
                    worker._pool_generation_at_checkout = self.pool_generation
                    worker._pool_hard_recovery_epoch_at_checkout = self.hard_recovery_epoch
                    if self.health_state == POOL_DEGRADED_CHECK:
                        self.speculative_dispatches_remaining -= 1
                        logger.warning(
                            f"[GPU {self.device_id}] Dispatching the single allowed pre-fault spare "
                            f"{worker.worker_id}; admission now waits for fresh-context validation"
                        )
                    self._ensure_capacity_locked(asyncio.get_running_loop())
                    return worker

            if workers_to_reap:

                async def _reap_detached_workers() -> List[PersistentWorker]:
                    failed_items: List[PersistentWorker] = []
                    try:
                        shutdown_results = await asyncio.gather(
                            *(asyncio.to_thread(item.shutdown, 10, True) for item in workers_to_reap),
                            return_exceptions=True,
                        )
                        failed_items = self._record_reap_results(workers_to_reap, shutdown_results)
                        if failed_items:
                            await self._quarantine_unreaped_context(
                                f"worker {failed_items[0].worker_id} could not be reaped during pool reconciliation",
                                extra_worker=failed_items[0],
                            )
                        return failed_items
                    finally:
                        async with self.lock:
                            self.pending_retirements = max(
                                0,
                                self.pending_retirements - len(workers_to_reap),
                            )
                            self._ensure_capacity_locked(asyncio.get_running_loop())
                        if reap_ticket_generation is not None:
                            self._finish_replenishment_ticket(reap_ticket_generation)

                failed_items, cancellation_requested = await _complete_despite_cancellation(_reap_detached_workers())
                if failed_items:
                    raise GPUQuarantinedError(f"[GPU {self.device_id}] Worker process could not be safely reaped")
                if topology_corrupt:
                    raise GPUQuarantinedError(
                        f"[GPU {self.device_id}] Live worker existed outside canonical pool membership"
                    )
                if cancellation_requested:
                    raise asyncio.CancelledError
                continue

            # 没有空闲 worker，等待一下
            await asyncio.sleep(0.1)

        # 超时
        logger.error(f"[GPU {self.device_id}] No idle worker available after {timeout}s")
        return None

    async def _return_worker(self, worker: PersistentWorker):
        """归还 worker 到 idle pool"""
        async with self.lock:
            if worker in self.busy_workers:
                self.busy_workers.remove(worker)

            # A recycled worker is removed from ``workers`` before its
            # asynchronous shutdown starts. It may still be alive when the
            # task's finally block gets here, but it must never be made idle
            # (and eligible for another task) again.
            if worker in self.workers and worker.is_alive():
                if worker not in self.idle_workers:
                    self.idle_workers.append(worker)
            self._ensure_capacity_locked(asyncio.get_running_loop())

    async def _restart_worker(
        self,
        worker: PersistentWorker,
        *,
        task_id: str = "",
        validation_generation: Optional[int] = None,
        after_reap_before_replenish: Optional[Callable[[], Any]] = None,
    ):
        """
        重启一个 worker (non-blocking, warm-spare promotion).

        Fast path (under lock):
        1. Remove the old worker from all tracking lists immediately.
        2. The existing idle spare(s) in the pool are already available
           for ``_get_idle_worker`` to hand out on the next task.

        Every retiring CUDA child is synchronously proven reaped before this
        task may return. Only construction of its replacement remains in the
        background, so an already-idle warm spare is still immediately
        dispatchable by a concurrent request.
        """
        async with self.lock:
            logger.info(
                f"[{worker.worker_id}] Recycling worker "
                f"(processed {worker.tasks_processed} tasks) — "
                f"spare replenishment will happen in background"
            )

            # --- fast: remove the dead worker from every list -----------
            if worker in self.workers:
                self.workers.remove(worker)
            if worker in self.idle_workers:
                self.idle_workers.remove(worker)
            if worker in self.busy_workers:
                self.busy_workers.remove(worker)

            spare_count = len(self.idle_workers)
            logger.info(
                f"[{worker.worker_id}] Pool state after removal: "
                f"workers={len(self.workers)} idle={spare_count} "
                f"busy={len(self.busy_workers)} pending={self.pending_replacements}"
            )

            validates_context_fault = validation_generation is not None
            validation_is_current = (
                validates_context_fault
                and validation_generation == self.pool_generation
                and self.health_state == POOL_DEGRADED_CHECK
            )
            should_replenish = validation_is_current or (
                not validates_context_fault
                and self.health_state == POOL_HEALTHY
                and len(self.workers) + self.pending_replacements + self.pending_retirements < self.pool_size
            )
            if should_replenish:
                self.pending_replacements += 1
                replacement_generation = self.pool_generation
            else:
                replacement_generation = self.pool_generation
                logger.info(
                    f"[{worker.worker_id}] Skipping replacement because pool already has capacity "
                    f"(workers={len(self.workers)}, pending={self.pending_replacements}, "
                    f"pool_size={self.pool_size})"
                )
            # Keep both a strong worker handle and a lifecycle ticket from the
            # instant canonical membership is detached until reap is proven.
            # Pool shutdown can therefore neither miss this process in its
            # snapshots nor finish its constructor drain across a ticket gap.
            _retain_unreaped_worker(self.device_id, worker)
            self._retiring_worker_ids.add(id(worker))
            self._register_replenishment_ticket(replacement_generation)
            self.pending_retirements += 1

        async def _reap_recycled_worker() -> bool:
            try:
                stopped = await asyncio.to_thread(worker.shutdown, 10, True)
            except Exception as shutdown_error:
                stopped = False
                logger.exception(f"[{worker.worker_id}] Recycled worker shutdown failed: {shutdown_error}")
            _record_worker_reap_result(self.device_id, worker, stopped)
            return stopped is True

        stopped, cancellation_requested = await _complete_despite_cancellation(_reap_recycled_worker())
        if stopped is not True:
            unsafe_reason = (
                f"recycled CUDA worker {worker.worker_id} could not be confirmed stopped before task return"
            )
            try:
                await self._quarantine_unreaped_context(
                    unsafe_reason,
                    task_id=task_id or self.health_task_id,
                    extra_worker=worker,
                )
            finally:
                async with self.lock:
                    self.pending_retirements = max(0, self.pending_retirements - 1)
                    self._retiring_worker_ids.discard(id(worker))
                self._finish_replenishment_ticket(replacement_generation)
            raise UnsafeGPUContainmentError(
                unsafe_reason,
                task_id=task_id or self.health_task_id,
                worker_id=worker.worker_id,
            )

        if after_reap_before_replenish is not None:
            _, action_cancelled = await _complete_despite_cancellation(after_reap_before_replenish())
            cancellation_requested = cancellation_requested or action_cancelled

        try:
            if should_replenish:
                # Register the constructor ticket before releasing the reap
                # ticket: shutdown always observes at least one owner for this
                # pending lifecycle. The old child is already gone, so only
                # fresh construction remains asynchronous.
                self._start_replenishment_thread(
                    old_worker=None,
                    old_process=None,
                    old_pid=None,
                    worker_id=worker.worker_id,
                    loop=asyncio.get_running_loop(),
                    reason="post-context-fault validation" if validates_context_fault else "restart",
                    generation=(
                        validation_generation if validation_generation is not None else replacement_generation
                    ),
                    validates_context_fault=validates_context_fault,
                )
        finally:
            async with self.lock:
                self.pending_retirements = max(0, self.pending_retirements - 1)
                self._retiring_worker_ids.discard(id(worker))
            self._finish_replenishment_ticket(replacement_generation)

        if not should_replenish:
            async with self.lock:
                self._ensure_capacity_locked(asyncio.get_running_loop())

        if cancellation_requested:
            raise asyncio.CancelledError

    def _zombie_reaper_loop(self):
        """Periodically reap zombie child processes.

        The CUDA driver keeps GPU memory allocated for a process until the
        parent joins it.  ``multiprocessing.active_children()`` reaps exited
        children tracked by multiprocessing without a blanket ``waitpid(-1)``
        that could steal exit status from unrelated child-process owners.
        """
        INTERVAL = 30  # seconds between sweeps
        while not self._reaper_stop.wait(timeout=INTERVAL):
            try:
                # active_children() calls waitpid(WNOHANG) for every
                # child Process that multiprocessing knows about.
                # This is the cheapest way to reap zombies.
                mp.active_children()
            except Exception as e:
                logger.warning(f"[GPU {self.device_id}] Zombie reaper error: {e}")

    async def shutdown(self, timeout: int = 30) -> bool:
        """Close the pool, sharing one containment proof with all waiters."""

        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown_once(timeout))
        shutdown_safe, cancellation_requested = await _complete_despite_cancellation(self._shutdown_task)
        if cancellation_requested:
            raise asyncio.CancelledError
        return bool(shutdown_safe)

    async def _shutdown_once(self, timeout: int) -> bool:
        """Perform the non-cancellable portion of pool shutdown."""

        logger.info(f"[GPU {self.device_id}] Shutting down worker pool...")

        async with self.lock:
            force_shutdown = self.health_state in {POOL_SUSPECT, POOL_BOOTSTRAP_FAILED, POOL_QUARANTINED}
            busy_worker_ids = {id(worker) for worker in self.busy_workers}
            self._closing = True
            self.pool_generation += 1
            self.pending_replacements = 0
            workers_to_shutdown = self._tracked_workers_locked()
            self.workers.clear()
            self.idle_workers.clear()
            self.busy_workers.clear()

        # Stop the zombie reaper thread
        if hasattr(self, "_reaper_stop"):
            self._reaper_stop.set()

        shutdown_results: List[Any] = []
        if workers_to_shutdown:
            per_worker_timeout = max(5, timeout // len(workers_to_shutdown))
            shutdown_results = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        worker.shutdown,
                        per_worker_timeout,
                        force_shutdown or id(worker) in busy_worker_ids,
                    )
                    for worker in workers_to_shutdown
                ),
                return_exceptions=True,
            )

        # Returning before a full constructor+shutdown lifecycle completes
        # would drop the only ticket tracking a possibly live CUDA context.
        replenishments_stopped = await asyncio.to_thread(
            self._wait_for_all_replenishments,
            max(_REPLENISHMENT_DRAIN_TIMEOUT_S, float(timeout)),
        )

        # A constructor may have crossed READY after the first snapshot but
        # before its ticket drained.  Snapshot retained/listed handles again
        # after the drain and force-reap every late arrival.  When the drain
        # times out this still catches all contexts constructed so far; the
        # unsafe return prevents any fresh probe/replacement from overlapping
        # a constructor that is still in flight.
        async with self.lock:
            already_seen = {id(worker) for worker in workers_to_shutdown}
            late_workers = [worker for worker in self._tracked_workers_locked() if id(worker) not in already_seen]
            self.workers.clear()
            self.idle_workers.clear()
            self.busy_workers.clear()
        if late_workers:
            late_results = await asyncio.gather(
                *(asyncio.to_thread(worker.shutdown, max(5, timeout), True) for worker in late_workers),
                return_exceptions=True,
            )
            workers_to_shutdown.extend(late_workers)
            shutdown_results.extend(late_results)

        unreaped_workers = self._record_reap_results(workers_to_shutdown, shutdown_results)
        workers_stopped = not unreaped_workers
        if not replenishments_stopped:
            self.unsafe_shutdown_reason = "timed out waiting for background CUDA worker constructors"
        elif not workers_stopped:
            self.unsafe_shutdown_reason = "one or more CUDA workers could not be confirmed stopped"

        shutdown_safe = replenishments_stopped and workers_stopped and not self.unsafe_shutdown_reason
        if not shutdown_safe:
            logger.critical(f"[GPU {self.device_id}] Worker pool shutdown is NOT SAFE: {self.unsafe_shutdown_reason}")
        else:
            logger.info(
                f"[GPU {self.device_id}] Worker pool shut down "
                f"(processed {self.total_tasks_processed} tasks, "
                f"restarted {self.total_workers_restarted} workers in "
                f"{time.time() - self.pool_start_time:.1f}s)"
            )
        return shutdown_safe

    def get_stats(self) -> Dict[str, Any]:
        """获取 pool 统计信息"""
        return {
            "device_id": self.device_id,
            "pool_size": self.pool_size,
            "workers_alive": len([w for w in self.workers if w.is_alive()]),
            "idle_workers": len(self.idle_workers),
            "busy_workers": len(self.busy_workers),
            "total_tasks_processed": self.total_tasks_processed,
            "total_workers_restarted": self.total_workers_restarted,
            "uptime": time.time() - self.pool_start_time,
            "workers": [w.get_stats() for w in self.workers],
            **self.get_health_snapshot(),
        }


# ============================================================================
# Worker Loop (在 subprocess 中运行)
# ============================================================================


def _run_deferred_compute_sanitizer(
    wrapper: Dict[str, Any],
    skip_reason: Optional[str] = None,
    total_timeout_s: Optional[float] = None,
) -> None:
    """Complete a child-requested diagnostic after its faulting context is reaped."""

    payload = wrapper.get("result")
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(metadata, dict):
        return
    request = metadata.pop("_runtime_sanitizer_request", None)
    if not isinstance(request, dict):
        return

    placeholder = payload.get("runtime_sanitizer")
    placeholder = placeholder if isinstance(placeholder, dict) else {}
    started = time.time()
    try:
        if skip_reason:
            raise RuntimeError(skip_reason)
        from kernelgym.toolkit.kernelbench.compute_sanitizer import run_compute_sanitizer

        diagnostic = run_compute_sanitizer(**request, total_timeout_s=total_timeout_s)
    except Exception as exc:
        diagnostic = {
            "status": "error",
            "passed": None,
            "measurement_complete": False,
            "requested_checks": list(placeholder.get("requested_checks") or []),
            "detected_issue_count": 0,
            "check_results": [],
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time_s": time.time() - started,
        }

    diagnostic.setdefault("selection_mode", placeholder.get("selection_mode"))
    diagnostic.setdefault("mode", placeholder.get("mode"))
    diagnostic.setdefault("error_classification", placeholder.get("error_classification"))
    payload["runtime_sanitizer"] = diagnostic
    metadata["runtime_sanitizer_status"] = diagnostic.get("status")
    metadata["runtime_sanitizer_issue_count"] = diagnostic.get("detected_issue_count", 0)
    metadata["kg_kernel_runtime_sanitizer_s"] = float(diagnostic.get("wall_time_s") or (time.time() - started))

    if diagnostic.get("status") == "issues_found":
        detail = None
        for check_result in diagnostic.get("check_results") or []:
            issues = check_result.get("issues") if isinstance(check_result, dict) else None
            if isinstance(issues, list) and issues:
                detail = issues[0].get("message")
                break
        message = "Runtime Sanitizer detected an unsafe CUDA kernel"
        if detail:
            message = f"{message}: {detail}"
        payload["status"] = "failed"
        payload["error_message"] = message
        payload["error_code"] = "RUNTIME_ERROR"


def _persistent_worker_loop(
    worker_id: str,
    device_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    max_tasks_per_worker: int = 100,
):
    """
    持久化 worker 的主循环

    这个函数在 subprocess 中运行：
    1. 启动时一次性初始化 torch 和 CUDA
    2. 循环处理任务
    3. **遇到 CUDA error 立即退出**
    4. 每次任务后清理 GPU 内存
    """
    init_start = time.time()
    init_stage = "bootstrap"

    # Capture the one-way sender and an un-set Event.wait before toolkit or
    # candidate modules can mutate process globals.  Production always uses the
    # raw JSON channel; the fallback keeps direct unit-test fakes ergonomic.
    if type(result_queue) is _ChildJSONResultChannel:
        send_result_message = result_queue.send
    else:
        fallback_put = result_queue.put

        def send_result_message(_kind: str, payload: Dict[str, Any]) -> None:
            fallback_put(payload)

    wait_for_parent_containment = threading.Event().wait
    parent_containment_acknowledged = False
    trusted_fault_none = FAULT_NONE
    try:
        prepare_core_dump_dir(os.environ.get(CORE_DUMP_DIR_ENV), os.environ.get(CORE_DUMP_KEEP_ENV), chdir=True)
    except Exception as exc:
        print(f"[{worker_id}] Failed to prepare core dump directory: {exc}", file=sys.stderr)

    try:
        _redirect_native_stderr_to_capture_file(worker_id)
    except Exception as exc:
        print(f"[{worker_id}] Failed to redirect native stderr for crash capture: {exc}", file=sys.stderr)

    try:
        # ====================================================================
        # 第1步：一次性初始化（只执行一次！）
        # ====================================================================

        # Use a dedicated process group while remaining inside the outer GPU
        # worker's session.  Normal fork/exec descendants inherit this PGID,
        # allowing the parent to freeze and reap the entire CUDA-owning group.
        # ``setsid`` is deliberately not used: the service/monitor owns the
        # outer session as its crash-containment boundary.
        init_stage = "process_containment"
        os.setpgid(0, 0)
        process_identity = _read_linux_process_identity(os.getpid())
        if (
            process_identity is None
            or process_identity.pgid != process_identity.pid
            or process_identity.sid != os.getsid(0)
        ):
            raise RuntimeError("worker failed to establish its dedicated process group")
        send_result_message(
            "contained",
            {
                "status": "CONTAINED",
                "pid": process_identity.pid,
                "start_ticks": process_identity.start_ticks,
                "pgid": process_identity.pgid,
                "sid": process_identity.sid,
            },
        )
        init_stage = "parent_containment_ack"
        try:
            containment_ack = task_queue.get(timeout=_PARENT_CONTAINMENT_ACK_TIMEOUT_S)
        except queue.Empty as exc:
            raise RuntimeError("parent containment acknowledgement timed out") from exc
        if containment_ack != _PARENT_CONTAINMENT_ACK:
            raise RuntimeError("worker received an invalid parent containment acknowledgement")
        parent_containment_acknowledged = True

        # Import 依赖
        init_stage = "imports"
        import torch
        import torch.cuda
        from kernelgym.backend import get_backend
        from kernelgym.toolkit import get_toolkit

        # 初始化 CUDA
        init_stage = "cuda_init"
        torch.cuda.init()
        device = torch.device(f"cuda:{device_id}")
        init_stage = "cuda_set_device"
        torch.cuda.set_device(device)

        # 预热（确保 CUDA 完全初始化）
        init_stage = "cuda_alloc"
        _ = torch.zeros(1, device=device)
        init_stage = "cuda_sync"
        synchronize_cuda = _capture_trusted_cuda_task_barrier(torch, device_id)
        synchronize_cuda()
        send_task_result = lambda payload: send_result_message("task_result", payload)
        trusted_task_ops = _capture_trusted_cuda_task_operations(
            synchronize_cuda,
            send_task_result,
            wait_for_parent_containment,
        )

        toolkit_cache: Dict[str, Any] = {}
        backend_cache: Dict[str, Any] = {}

        init_time = time.time() - init_start

        # 通知主进程：初始化成功
        send_result_message(
            "ready",
            {
                "status": "READY",
                "init_time": init_time,
                "device": str(device),
                "pid": process_identity.pid,
                "start_ticks": process_identity.start_ticks,
                "pgid": process_identity.pgid,
                "sid": process_identity.sid,
            },
        )

        # 日志
        print(f"[{worker_id}] Initialized successfully (device={device}, init_time={init_time:.2f}s)", file=sys.stderr)

        # ====================================================================
        # 第2步：任务处理循环
        # ====================================================================

        tasks_processed = 0
        unsafe_cuda_cleanup = False
        skip_final_cuda_cleanup = False

        while True:
            try:
                # 获取任务
                task_data = task_queue.get()

                # 检查是否是 shutdown 命令
                if isinstance(task_data, dict) and task_data.get("command") in ("SHUTDOWN", "GRACEFUL_SHUTDOWN"):
                    cmd = task_data.get("command")
                    print(f"[{worker_id}] Received {cmd} command", file=sys.stderr)
                    if cmd == "GRACEFUL_SHUTDOWN":
                        # Perform thorough GPU cleanup before exiting so
                        # that the CUDA context is released cleanly without
                        # needing SIGKILL.
                        print(
                            f"[{worker_id}] Graceful shutdown: cleaning up GPU...",
                            file=sys.stderr,
                        )
                        try:
                            _aggressive_gpu_cleanup(device_id)
                        except Exception as e:
                            print(
                                f"[{worker_id}] GPU cleanup during graceful shutdown failed: {e}",
                                file=sys.stderr,
                            )
                    break

                # 执行任务（先清空 stderr 捕获文件，让内容对应本次任务）
                _truncate_native_stderr_capture()
                result = _execute_task_in_worker(
                    task_data,
                    device,
                    toolkit_cache,
                    backend_cache,
                    get_toolkit,
                    get_backend,
                )

                # Preserve the deferred diagnostic descriptor without touching
                # the original, potentially poisoned CUDA context again. The
                # parent owns STOP/KILL/reap, sanitizer execution, and the
                # fresh-context validation transition.
                if str(result.get("fault_severity") or trusted_fault_none) != trusted_fault_none:
                    unsafe_cuda_cleanup = True
                    try:
                        trusted_task_ops.publish_and_wait(result)
                    except BaseException:
                        return
                    return

                # For long-lived contexts, do cache/GC maintenance before the
                # commit barrier.  No CUDA API is allowed after a successful
                # result is published.
                prepare_for_reuse = None
                if max_tasks_per_worker > 1:
                    prepare_for_reuse = lambda: _aggressive_gpu_cleanup(device_id, strict=True)

                # CUDA launches are asynchronous.  Treat this as the task's
                # commit barrier; the helper guarantees that neither the
                # reusable-worker cleanup nor any other CUDA API follows put().
                must_recycle = tasks_processed + 1 >= max_tasks_per_worker
                if must_recycle:
                    result["worker_exiting"] = True
                    trusted_task_ops.commit_and_wait(result, prepare_for_reuse)
                else:
                    trusted_task_ops.commit(result, prepare_for_reuse)

                tasks_processed += 1

            except Exception as task_error:
                # 任务执行失败
                error_traceback = traceback.format_exc()
                error_details = trusted_task_ops.classify_error(task_error)
                error_type = str(error_details["error_type"])
                error_message = str(error_details["error_message"])
                final_sync_failed = bool(error_details["final_sync_failed"])
                is_cuda_error = bool(error_details["is_cuda_error"])
                fault_severity = str(error_details["fault_severity"])
                is_profiler_error = bool(error_details["is_profiler_error"])

                if is_cuda_error or is_profiler_error or fault_severity != trusted_fault_none:
                    # CUDA error / profiler dropout！准备退出
                    print(
                        f"[{worker_id}] CUDA/profiler error detected! Worker will exit. "
                        f"Error: {error_type}: {error_message}",
                        file=sys.stderr,
                    )

                    error_result = {
                        "success": False,
                        "error_type": error_type,
                        "error_message": error_message,
                        "traceback": error_traceback,
                        "worker_exiting": True,
                        "cuda_error": is_cuda_error,
                        "profiling_error": is_profiler_error,
                        "final_sync_failed": final_sync_failed,
                        "fault_severity": fault_severity,
                        "device_suspect": fault_severity != trusted_fault_none,
                    }

                    # Sticky context/device faults make subsequent CUDA API
                    # calls unreliable.  Do not run empty_cache/stat resets or
                    # another synchronize; process teardown is the containment
                    # boundary.  OOM/profiler-only exits may still clean up.
                    unsafe_cuda_cleanup = fault_severity != trusted_fault_none

                    # Do not voluntarily exit and lose PPid ancestry after a
                    # sticky fault or a benign OOM/profiler recycle.  Benign
                    # failures already crossed a trusted sync; neither class
                    # makes another CUDA call after publication.  The parent
                    # owns the process group's STOP/KILL/reap transition.
                    try:
                        trusted_task_ops.publish_and_wait(error_result)
                    except BaseException:
                        return
                    return

                else:
                    # 非 CUDA error，返回错误但继续运行
                    print(f"[{worker_id}] Task error (non-CUDA): {error_type}: {error_message}", file=sys.stderr)

                    tasks_processed, must_recycle = trusted_task_ops.publish_non_cuda_failure(
                        {
                            "success": False,
                            "error_type": error_type,
                            "error_message": error_message,
                            "traceback": error_traceback,
                            "worker_exiting": False,
                            "cuda_error": False,
                        },
                        tasks_processed,
                        max_tasks_per_worker,
                    )
                    if must_recycle:
                        # Production blocks inside the trusted publication
                        # helper until parent STOP/KILL.  This branch is only a
                        # defensive fallback if that wait ever returns.
                        skip_final_cuda_cleanup = True
                        break

        # 正常退出 - 清理显存
        print(f"[{worker_id}] Worker exiting normally (processed {tasks_processed} tasks)", file=sys.stderr)

        if unsafe_cuda_cleanup or skip_final_cuda_cleanup:
            print(
                f"[{worker_id}] Skipping post-result CUDA cleanup; exiting context directly",
                file=sys.stderr,
            )
        else:
            print(f"[{worker_id}] Performing final GPU cleanup...", file=sys.stderr)
            try:
                _aggressive_gpu_cleanup(device_id)
                print(f"[{worker_id}] Final GPU cleanup completed", file=sys.stderr)
            except Exception as cleanup_err:
                print(f"[{worker_id}] Final GPU cleanup failed: {cleanup_err}", file=sys.stderr)

    except Exception as init_error:
        # 初始化失败
        print(f"[{worker_id}] Initialization failed: {init_error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        send_result_message(
            "init_failed",
            {
                "status": "INIT_FAILED",
                "init_stage": init_stage,
                "error": str(init_error),
                "traceback": traceback.format_exc(),
            },
        )
        if not parent_containment_acknowledged:
            # No CUDA-capable imports or contexts exist before the ACK.  A
            # bounded exit prevents an orphan from blocking forever on its own
            # duplicated multiprocessing.Queue writer when the parent dies.
            return
        # A partially initialized CUDA/import process may already own native
        # descendants.  Keep its attested leader alive so the constructor's
        # handshake failure path can freeze and reap the process tree.
        wait_for_parent_containment()
        raise _ProcessContainmentError("parent containment wait returned unexpectedly after init failure")


def _execute_task_in_worker(
    task_data: Dict[str, Any],
    device: Any,  # torch.device
    toolkit_cache: Dict[str, Any],
    backend_cache: Dict[str, Any],
    get_toolkit: Any,
    get_backend: Any,
) -> Dict[str, Any]:
    """
    在 worker 中执行单个任务

    Args:
        task_data: 任务数据字典
        device: torch.device
        toolkit: KernelBench integration 模块

    Returns:
        结果字典
    """

    def _has_no_cuda_events(result_obj: Any) -> bool:
        """Detect profiler dropouts where no CUDA events were captured."""
        try:
            metadata = None
            if isinstance(result_obj, dict):
                metadata = result_obj.get("metadata")
            else:
                metadata = getattr(result_obj, "metadata", None)
            if not isinstance(metadata, dict):
                return False
            profiling = metadata.get("profiling")
            if not isinstance(profiling, dict):
                return False
            return profiling.get("profiling_warning") == "no_cuda_events"
        except Exception:
            return False

    try:
        toolkit_name = task_data.get("toolkit")
        backend_adapter = task_data.get("backend_adapter")
        if not toolkit_name:
            raise ValueError("Task payload missing required 'toolkit'")
        if not backend_adapter:
            raise ValueError("Task payload missing required 'backend_adapter'")

        if toolkit_name not in toolkit_cache:
            toolkit_cache[toolkit_name] = get_toolkit(toolkit_name)
        if backend_adapter not in backend_cache:
            backend_cache[backend_adapter] = get_backend(backend_adapter)

        task_data["device"] = str(device)
        stage_metadata_path = task_data.get("_stage_metadata_path")
        previous_stage_metadata_path = os.environ.get(_STAGE_METADATA_PATH_ENV)
        if stage_metadata_path:
            os.environ[_STAGE_METADATA_PATH_ENV] = str(stage_metadata_path)

        toolkit = toolkit_cache[toolkit_name]
        backend = backend_cache[backend_adapter]
        try:
            result = toolkit.evaluate(task_data, backend=backend)
        finally:
            if previous_stage_metadata_path is None:
                os.environ.pop(_STAGE_METADATA_PATH_ENV, None)
            else:
                os.environ[_STAGE_METADATA_PATH_ENV] = previous_stage_metadata_path

        result_metadata = None
        if isinstance(result, dict):
            status = result.get("status")
            error_msg = result.get("error_message")
            result_metadata = result.get("metadata")
        else:
            status = getattr(result, "status", None)
            error_msg = getattr(result, "error_message", None)
            result_metadata = getattr(result, "metadata", None)

        candidate_runtime_context_fault = bool(
            isinstance(result_metadata, dict)
            and bool(result_metadata.get("runtime_error"))
            and (
                result_metadata.get("correctness_runtime_error_stage") == "custom_forward"
                or result_metadata.get("runtime_sanitizer_trigger") == "correctness_runtime_error"
            )
        )

        if status == "failed" and error_msg:
            if not candidate_runtime_context_fault:
                reported_cuda_error = (
                    "CUDA" in error_msg
                    or "cuda" in error_msg.lower()
                    or "illegal memory access" in error_msg.lower()
                    or "device-side assert" in error_msg.lower()
                )
                reported_fault = _classify_cuda_fault(
                    "RuntimeError",
                    str(error_msg),
                    is_cuda_error=reported_cuda_error,
                )
                if reported_cuda_error or reported_fault != FAULT_NONE:
                    raise RuntimeError(f"CUDA error detected: {error_msg}")

        if _has_no_cuda_events(result):
            raise RuntimeError("PROFILER_NO_CUDA_EVENTS")

        wrapped_result = {
            "success": True,
            "result": result,
            "worker_exiting": candidate_runtime_context_fault,
        }
        if candidate_runtime_context_fault:
            original_runtime_error = result_metadata.get("runtime_error")
            wrapped_result.update(
                {
                    "error_type": "CandidateRuntimeContextFault",
                    "error_message": str(original_runtime_error or error_msg or "correctness CUDA runtime failure"),
                    "cuda_error": True,
                    "fault_severity": FAULT_CONTEXT,
                    "device_suspect": True,
                }
            )
        return wrapped_result

    except Exception:
        # 这里的异常会被上层捕获并判断是否是 CUDA error
        raise
