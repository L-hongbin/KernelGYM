"""
Worker Monitor for KernelGym.
Monitors worker health and restarts crashed workers.

Enhancement: Persistent monitoring mode (opt-in via --persistent or env)
- When enabled, the monitor maintains target workers based on Redis keys that
  are populated by the service launcher:
    - f"{KEY_PREFIX}:expected_workers" (SET of worker_ids)
    - f"{KEY_PREFIX}:expected_worker:{worker_id}" (HASH: device, node_id, hostname)
    - f"{KEY_PREFIX}:worker_process:{worker_id}" (HASH: pid, start_time, device)
  The monitor will restart a worker if:
    - Its heartbeat hash key is missing (heartbeat key expired after crash), or
    - Its recorded PID is not alive, or
    - It meets original restart conditions (CUDA error shutdown / heartbeat timeout).
"""

import argparse
import asyncio
import errno
import json
import logging
import os
import redis.asyncio as redis
import re
import signal
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Set

from kernelgym.config import settings
from kernelgym.config.settings import PROJECT_ROOT
from kernelgym.utils.core_dumps import prepare_core_dump_dir
from kernelgym.utils.gpu_quarantine import (
    UNLATCHED_NOTIFICATION_PROVENANCE,
    gpu_quarantine_generation,
    read_gpu_quarantine,
    update_gpu_quarantine_notification,
    write_gpu_quarantine,
)
from kernelgym.utils.page_user_notifier import send_gpu_quarantine_page, send_gpu_worker_exclusion_page

KEY_PREFIX = settings.redis_key_prefix
from kernelgym.config import setup_logging
from redis.exceptions import (
    BusyLoadingError,
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
    ResponseError as RedisResponseError,
)

logger = logging.getLogger("kernelgym.worker_monitor")
_REPOSITORY_ROOT = PROJECT_ROOT.parent
_RESTART_LIMIT_PAGE_MAX_ATTEMPTS_PER_MONITOR = 2
_RESTART_LIMIT_PAGE_RETRY_BACKOFF_SECONDS = 60.0


@dataclass(frozen=True)
class ProcessIdentity:
    """Linux process identity that remains stable across PID reuse."""

    pid: int
    start_ticks: str
    state: str
    process_group: int
    session_id: int


class ProcessIdentityMismatch(RuntimeError):
    """The PID now names a different process generation."""


_PROCESS_MAP_DELETE_IF_CURRENT = """
local current_pid = redis.call('HGET', KEYS[1], 'pid')
if not current_pid or current_pid ~= ARGV[1] then
    return 0
end
local current_ticks = redis.call('HGET', KEYS[1], 'proc_start_ticks')
if ARGV[5] == '1' then
    if not current_ticks or current_ticks ~= ARGV[2] then
        return 0
    end
else
    if current_ticks and current_ticks ~= '' then
        return 0
    end
end
local current_group = redis.call('HGET', KEYS[1], 'process_group')
if ARGV[6] == '1' then
    if not current_group or current_group ~= ARGV[3] then
        return 0
    end
else
    if current_group and current_group ~= '' then
        return 0
    end
end
local current_session = redis.call('HGET', KEYS[1], 'session_id')
if ARGV[7] == '1' then
    if not current_session or current_session ~= ARGV[4] then
        return 0
    end
else
    if current_session and current_session ~= '' then
        return 0
    end
end
return redis.call('DEL', KEYS[1])
"""


_PROCESS_MAP_REGISTER_IF_EMPTY = """
local current_pid = redis.call('HGET', KEYS[1], 'pid')
if current_pid and current_pid ~= '' then
    return 0
end
redis.call(
    'HSET', KEYS[1],
    'pid', ARGV[1],
    'start_time', ARGV[2],
    'proc_start_ticks', ARGV[3],
    'process_group', ARGV[4],
    'session_id', ARGV[5],
    'device', ARGV[6]
)
return 1
"""


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _restart_budget_path(hostname: str, worker_id: str) -> Path:
    configured = os.environ.get("KERNELGYM_SAFETY_LATCH_DIR")
    root = Path(configured) if configured else _REPOSITORY_ROOT / "logs" / "safety_latches"
    return root / "restart_attempts" / _safe_path_component(hostname) / f"{_safe_path_component(worker_id)}.json"


def _read_restart_budget(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0
    if not isinstance(payload, dict) or not isinstance(payload.get("attempts"), int):
        raise RuntimeError(f"Invalid worker restart budget: {path}")
    return max(0, int(payload["attempts"]))


def _write_restart_budget(path: Path, *, hostname: str, worker_id: str, attempts: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "hostname": hostname,
                    "worker_id": worker_id,
                    "attempts": attempts,
                    "updated_at": datetime.now().isoformat(),
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _clear_restart_budget_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class WorkerMonitor:
    """Monitors worker health and manages restarts."""

    def __init__(self, redis_client: redis.Redis, persistent: bool = False):
        self.redis = redis_client
        self.running = False
        self.monitored_workers: Dict[str, Dict[str, Any]] = {}
        self.restart_queue: asyncio.Queue = asyncio.Queue()
        self.restart_in_progress: Set[str] = set()
        self.restart_attempts: Dict[str, int] = {}
        self._restart_budget_lock = asyncio.Lock()
        self._quarantine_page_attempts: Dict[tuple[str, str, str], int] = {}
        self._quarantine_page_retry_not_before: Dict[tuple[str, str, str], float] = {}
        self._quarantine_page_lock = asyncio.Lock()
        # Keep strong Popen references for children created by this monitor.
        # Without these, an exited child can remain a zombie whose empty
        # /proc/<pid>/cmdline makes it impossible to identify safely.
        self.spawned_processes: Dict[str, Any] = {}
        self.spawned_identities: Dict[str, ProcessIdentity] = {}
        self.persistent: bool = persistent
        # Restarts spawn processes on THIS host, so a monitor must only ever
        # enforce workers that belong to this host. In a multi-node cluster the
        # expected_workers set is shared via the primary's Redis.
        self.hostname: str = os.environ.get("HOSTNAME") or socket.gethostname() or "local"

        # Configuration
        self.heartbeat_timeout = max(5, settings.worker_monitor_heartbeat_timeout)
        self.startup_timeout = max(self.heartbeat_timeout, settings.worker_monitor_startup_timeout)
        self.monitor_interval = max(5, settings.worker_monitor_interval)
        self.max_restart_attempts = 3
        self.restart_cooldown = max(5, settings.worker_monitor_restart_cooldown)
        logger.info(
            "Worker monitor configured with heartbeat_timeout=%ss, startup_timeout=%ss, "
            "monitor_interval=%ss, restart_cooldown=%ss",
            self.heartbeat_timeout,
            self.startup_timeout,
            self.monitor_interval,
            self.restart_cooldown,
        )

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Worker monitor received signal {signum}")
        self.running = False

    async def start(self):
        """Start the worker monitor."""
        self.running = True
        logger.info("Starting worker monitor")

        try:
            # Start monitoring and restart tasks
            monitor_task = asyncio.create_task(self._monitor_loop())
            restart_task = asyncio.create_task(self._restart_loop())

            await asyncio.gather(monitor_task, restart_task)

        except Exception as e:
            logger.error(f"Error in worker monitor: {e}")
            raise

    async def stop(self):
        """Stop the worker monitor."""
        logger.info("Stopping worker monitor")
        self.running = False

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._check_workers()
                await asyncio.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(self.monitor_interval)

    async def _load_local_expected_ids(self) -> Set[str]:
        """Expected worker ids that belong to THIS host.

        Restarting a worker launches a local process, so ids registered by
        another node (hostname mismatch in the expected_worker hash) must never
        be enforced here. Ids with no recorded hostname are treated as local
        for backward compatibility with registrations that predate the field.
        """
        raw = await self.redis.smembers(f"{KEY_PREFIX}:expected_workers")
        all_ids = {wid.decode() if isinstance(wid, bytes) else wid for wid in raw} if raw else set()
        local_ids: Set[str] = set()
        for wid in all_ids:
            edata = await self.redis.hgetall(f"{KEY_PREFIX}:expected_worker:{wid}")
            owner = edata.get(b"hostname", b"").decode() if edata else ""
            if not owner or owner == self.hostname:
                local_ids.add(wid)
        return local_ids

    async def _is_worker_quarantined(
        self,
        worker_id: str,
        *,
        device: str = "",
        hostname: str = "",
    ) -> bool:
        record = await read_gpu_quarantine(
            self.redis,
            worker_id,
            device=device,
            hostname=hostname or self.hostname,
        )
        if record:
            notification_record = dict(record)
            notification_record.setdefault("worker_id", worker_id)
            notification_record["device"] = str(notification_record.get("device") or device)
            notification_record["hostname"] = str(notification_record.get("hostname") or hostname or self.hostname)
            if notification_record["device"].startswith("cuda:") and notification_record.get("scope") in {
                "physical_gpu",
                "worker_process",
            }:
                await self._ensure_quarantine_notification(notification_record)
        return bool(record)

    async def _ensure_quarantine_notification(self, record: Dict[str, str]) -> None:
        """Retry one CUDA quarantine page generation with bounded backoff.

        The notifier itself takes a durable per-device claim, so concurrent
        service/monitor processes cannot emit duplicate MCP pages. This method
        adds monitor-local retry pacing and a fresh budget after manual clear.
        """

        worker_id = str(record.get("worker_id") or "")
        device = str(record.get("device") or "")
        hostname = str(record.get("hostname") or self.hostname)
        scope = str(record.get("scope") or "")
        if not worker_id or not device.startswith("cuda:") or scope not in {"physical_gpu", "worker_process"}:
            return
        latch_generation = gpu_quarantine_generation(record)
        event_owner = f"{hostname}:{device}" if scope == "physical_gpu" else worker_id
        attempt_key = (scope, event_owner, latch_generation)
        # A manual clear followed by a new latch is a new exclusion event.
        # Keying the budget by ``created_at`` prevents an old event's attempts
        # from suppressing this generation.
        if record.get("page_user_state") == "sent":
            self._quarantine_page_attempts[attempt_key] = _RESTART_LIMIT_PAGE_MAX_ATTEMPTS_PER_MONITOR
            return

        attempts = self._quarantine_page_attempts.get(attempt_key, 0)
        if record.get("page_user_state") == "failed" and attempts == 0:
            try:
                failed_at = datetime.fromisoformat(str(record.get("page_user_updated_at") or ""))
                now = datetime.now(tz=failed_at.tzinfo)
                remaining = _RESTART_LIMIT_PAGE_RETRY_BACKOFF_SECONDS - max(
                    0.0,
                    (now - failed_at).total_seconds(),
                )
                if remaining > 0:
                    self._quarantine_page_retry_not_before[attempt_key] = max(
                        self._quarantine_page_retry_not_before.get(attempt_key, 0.0),
                        time.monotonic() + remaining,
                    )
            except (TypeError, ValueError):
                pass

        if attempts >= _RESTART_LIMIT_PAGE_MAX_ATTEMPTS_PER_MONITOR:
            return
        if time.monotonic() < self._quarantine_page_retry_not_before.get(attempt_key, 0.0):
            return

        async with self._quarantine_page_lock:
            attempts = self._quarantine_page_attempts.get(attempt_key, 0)
            if attempts >= _RESTART_LIMIT_PAGE_MAX_ATTEMPTS_PER_MONITOR:
                return
            if time.monotonic() < self._quarantine_page_retry_not_before.get(attempt_key, 0.0):
                return
            self._quarantine_page_attempts[attempt_key] = attempts + 1

            async def _deliver_and_record() -> None:
                try:
                    sender = send_gpu_quarantine_page if scope == "physical_gpu" else send_gpu_worker_exclusion_page
                    outcome = await sender(record)
                    success = outcome.success
                    protocol_version = outcome.protocol_version or "unknown"
                    superseded = protocol_version == "superseded"
                    error = "" if success else f"{outcome.error_kind or 'unknown'}: {outcome.error or ''}"
                except Exception as exc:  # pragma: no cover - notifier promises not to raise
                    success = False
                    protocol_version = "unknown"
                    superseded = False
                    error = f"unexpected_error: {type(exc).__name__}"
                state = "sent" if success else "failed"
                unlatched = record.get("notification_provenance") == UNLATCHED_NOTIFICATION_PROVENANCE
                if not superseded and not unlatched:
                    try:
                        await update_gpu_quarantine_notification(
                            self.redis,
                            worker_id,
                            device=device,
                            hostname=hostname,
                            scope=scope,
                            expected_generation=latch_generation,
                            state=state,
                            error=error,
                        )
                    except Exception as exc:
                        logger.critical(f"Failed to persist page-user state for GPU quarantine {worker_id}: {exc}")
                if success:
                    self._quarantine_page_attempts[attempt_key] = _RESTART_LIMIT_PAGE_MAX_ATTEMPTS_PER_MONITOR
                    logger.warning(
                        f"Sent page-user notification for GPU quarantine {worker_id} via MCP {protocol_version}"
                    )
                else:
                    self._quarantine_page_retry_not_before[attempt_key] = (
                        time.monotonic() + _RESTART_LIMIT_PAGE_RETRY_BACKOFF_SECONDS
                    )
                    logger.critical(f"Failed to send page-user notification for GPU quarantine {worker_id}: {error}")

            notification_task = asyncio.create_task(_deliver_and_record())
            cancellation_requested = False
            while not notification_task.done():
                try:
                    await asyncio.shield(notification_task)
                except asyncio.CancelledError:
                    cancellation_requested = True
            notification_task.result()
            if cancellation_requested:
                raise asyncio.CancelledError

    async def _ensure_restart_limit_notification(self, record: Dict[str, str]) -> None:
        """Backward-compatible wrapper for worker restart-limit call sites."""

        await self._ensure_quarantine_notification(record)

    @staticmethod
    def _decode_hash(data: Dict[Any, Any]) -> Dict[str, str]:
        return {
            (key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)): (
                value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
            )
            for key, value in data.items()
        }

    @staticmethod
    def _read_process_identity(pid: int) -> Optional[ProcessIdentity]:
        """Read state, PGID, SID, and Linux start ticks from /proc."""

        try:
            raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        closing_paren = raw_stat.rfind(")")
        if closing_paren < 0:
            raise RuntimeError(f"Malformed /proc/{pid}/stat")
        fields = raw_stat[closing_paren + 2 :].split()
        if len(fields) < 20:
            raise RuntimeError(f"Incomplete /proc/{pid}/stat")
        return ProcessIdentity(
            pid=pid,
            state=fields[0],
            process_group=int(fields[2]),
            session_id=int(fields[3]),
            start_ticks=fields[19],
        )

    def _live_process_group_members(self, process_group: int) -> list[ProcessIdentity]:
        """Return every non-zombie member of one Linux process group.

        Reaping only the worker's group leader is not a CUDA containment proof:
        a multiprocessing child can outlive that leader while retaining its GPU
        context.  A failed ``/proc`` enumeration is therefore an error, not an
        empty group.
        """

        try:
            proc_entries = list(Path("/proc").iterdir())
        except OSError as exc:
            raise RuntimeError(f"Could not enumerate /proc for process group {process_group}") from exc

        members: list[ProcessIdentity] = []
        for entry in proc_entries:
            if not entry.name.isdigit():
                continue
            try:
                identity = self._read_process_identity(int(entry.name))
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Could not inspect PID {entry.name} while proving process group {process_group} empty"
                ) from exc
            if identity is not None and identity.process_group == process_group and identity.state != "Z":
                members.append(identity)
        return members

    def _process_group_is_drained(self, process_group: int) -> bool:
        """Atomically prove that the Linux process group no longer exists.

        A ``/proc`` snapshot alone has a fork/exit TOCTOU. ``killpg(..., 0)``
        asks the kernel about the group as one operation: only ESRCH is proof of
        absence. Success and EPERM both mean the group may still hold a CUDA
        context. ``/proc`` snapshots are intentionally excluded from this
        polling predicate: they are useful diagnostics, but cannot strengthen
        or replace the kernel's atomic ESRCH proof.
        """

        if process_group <= 0:
            raise RuntimeError(f"Invalid process group for drain proof: {process_group}")
        if not hasattr(os, "killpg"):
            raise RuntimeError("Atomic process-group drain proof requires os.killpg")
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return True
            raise RuntimeError(f"Could not prove process group {process_group} absent") from exc
        return False

    def _live_session_members(self, session_id: int) -> list[ProcessIdentity]:
        """Return a fail-closed snapshot of every member of one Linux SID."""

        if session_id <= 1:
            raise RuntimeError(f"Invalid session for containment proof: {session_id}")
        try:
            proc_entries = list(Path("/proc").iterdir())
        except OSError as exc:
            raise RuntimeError(f"Could not enumerate /proc for session {session_id}") from exc

        members: list[ProcessIdentity] = []
        for entry in proc_entries:
            if not entry.name.isdigit():
                continue
            try:
                identity = self._read_process_identity(int(entry.name))
            except (OSError, RuntimeError) as exc:
                # Silently omitting one unreadable stat record could omit the
                # surviving CUDA owner.  Deployments must expose /proc stat for
                # the container PID namespace; otherwise restart fails closed.
                raise RuntimeError(
                    f"Could not inspect PID {entry.name} while proving session {session_id} empty"
                ) from exc
            if identity is not None and identity.session_id == session_id:
                members.append(identity)
        return sorted(members, key=lambda item: (item.pid, item.start_ticks))

    def _session_is_drained(self, session_id: int, observed_process_groups: set[int]) -> bool:
        """Require an empty SID plus kernel ESRCH for every observed PGID."""

        members = self._live_session_members(session_id)
        observed_process_groups.update(member.process_group for member in members)
        if members:
            return False
        return all(self._process_group_is_drained(group) for group in observed_process_groups)

    async def _wait_for_session_drain(
        self,
        session_id: int,
        observed_process_groups: set[int],
        timeout: float,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        last_scan_error: Exception | None = None
        while True:
            try:
                if self._session_is_drained(session_id, observed_process_groups):
                    return True
                last_scan_error = None
            except (OSError, RuntimeError, ValueError) as exc:
                # A numeric /proc entry can disappear between enumeration and
                # stat while the session is exiting.  Retry that incomplete
                # snapshot; it proves neither containment nor its failure.
                last_scan_error = exc
            if time.monotonic() >= deadline:
                if last_scan_error is not None:
                    raise RuntimeError(f"Could not complete session {session_id} drain scan") from last_scan_error
                return False
            await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    async def _freeze_worker_session(
        self,
        session_id: int,
        *,
        expected_leader_start_ticks: str,
        timeout: float,
    ) -> tuple[bool, set[int], str]:
        """Stop all SID process groups to a stable fixed point."""

        deadline = time.monotonic() + max(0.0, timeout)
        observed_process_groups: set[int] = set()
        previous_signature: tuple[tuple[int, str, int], ...] | None = None
        stable_passes = 0
        while True:
            try:
                members = self._live_session_members(session_id)
            except Exception as exc:
                return False, observed_process_groups, f"session /proc scan failed: {type(exc).__name__}"
            for member in members:
                if (
                    member.pid == session_id
                    and expected_leader_start_ticks
                    and member.start_ticks != expected_leader_start_ticks
                ):
                    return False, observed_process_groups, f"session leader PID {session_id} generation changed"
                if member.process_group <= 1:
                    return (
                        False,
                        observed_process_groups,
                        f"invalid PGID {member.process_group} in session {session_id}",
                    )
                observed_process_groups.add(member.process_group)
            if not members:
                return True, observed_process_groups, ""

            for process_group in sorted({member.process_group for member in members}):
                try:
                    os.killpg(process_group, signal.SIGSTOP)
                except ProcessLookupError:
                    continue
                except OSError as exc:
                    if exc.errno == errno.ESRCH:
                        continue
                    return (
                        False,
                        observed_process_groups,
                        f"SIGSTOP failed for session {session_id} PGID {process_group}: {type(exc).__name__}",
                    )

            try:
                confirmation = self._live_session_members(session_id)
            except Exception as exc:
                return False, observed_process_groups, f"session confirmation scan failed: {type(exc).__name__}"
            observed_process_groups.update(member.process_group for member in confirmation)
            for member in confirmation:
                if (
                    member.pid == session_id
                    and expected_leader_start_ticks
                    and member.start_ticks != expected_leader_start_ticks
                ):
                    return False, observed_process_groups, f"session leader PID {session_id} generation changed"

            signature = tuple((member.pid, member.start_ticks, member.process_group) for member in confirmation)
            all_frozen = all(member.state in {"T", "t", "Z"} for member in confirmation)
            if all_frozen and signature == previous_signature:
                stable_passes += 1
            elif all_frozen:
                stable_passes = 1
            else:
                stable_passes = 0
            previous_signature = signature
            if stable_passes >= 2:
                return True, observed_process_groups, ""
            if time.monotonic() >= deadline:
                states = ",".join(f"{member.pid}:{member.state}" for member in confirmation)
                return (
                    False,
                    observed_process_groups,
                    f"session {session_id} did not freeze to a stable snapshot ({states})",
                )
            await asyncio.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    async def _force_kill_worker_session(
        self,
        worker_id: str,
        *,
        pid: int,
        expected_leader_start_ticks: str,
        session_id: int,
        observed_process_groups: set[int] | None = None,
    ) -> tuple[bool, str]:
        """Freeze, kill, reap, and prove an authenticated worker SID absent."""

        if pid <= 1 or session_id != pid:
            return False, f"invalid authenticated session leader identity: pid={pid}, sid={session_id}"
        if not expected_leader_start_ticks:
            return False, f"session {session_id} has no authenticated leader start_ticks"
        known_groups = set(observed_process_groups or ())
        frozen, frozen_groups, reason = await self._freeze_worker_session(
            session_id,
            expected_leader_start_ticks=expected_leader_start_ticks,
            timeout=10,
        )
        known_groups.update(frozen_groups)
        if not frozen:
            return False, reason

        # Signal only groups authenticated by the frozen SID snapshot.  Older
        # mapped PGID numbers remain final ESRCH gates but may have been reused
        # outside this session and therefore must not be signalled bare.
        for process_group in sorted(frozen_groups):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    continue
                return False, f"SIGKILL failed for session {session_id} PGID {process_group}: {type(exc).__name__}"

        stopped = await self._wait_for_process_exit(
            worker_id,
            pid,
            expected_leader_start_ticks,
            10,
            expected_process_group=pid,
            expected_session_id=session_id,
            observed_process_groups=known_groups,
        )
        if stopped:
            return True, ""
        return False, f"session {session_id} or one of its observed process groups survived SIGKILL"

    async def _quarantine_unsafe_process_group(
        self,
        worker_id: str,
        device: str,
        reason: str,
    ) -> None:
        """Persist a physical latch and page when process containment is unproven."""

        if not device.startswith("cuda:"):
            logger.critical(f"Worker {worker_id} process containment is unproven: {reason}")
            return

        async def _persist_page_and_record() -> None:
            record: Dict[str, str]
            try:
                record = await write_gpu_quarantine(
                    self.redis,
                    worker_id,
                    device=device,
                    reason=reason,
                    fault_class="unsafe_process_group_shutdown",
                    node_id=self.hostname,
                    hostname=self.hostname,
                    physical_scope=True,
                )
            except Exception as exc:
                # Admission still fails closed because the caller refuses to
                # delete the old PID map or spawn. Paging must not depend on
                # Redis/latch availability.
                logger.critical(f"Failed to persist physical quarantine for unsafe process group {worker_id}: {exc}")
                record = {
                    "state": "quarantined",
                    "scope": "physical_gpu",
                    "worker_id": worker_id,
                    "device": device,
                    "reason": reason,
                    "fault_class": "unsafe_process_group_shutdown",
                    "node_id": self.hostname,
                    "hostname": self.hostname,
                    "page_user_state": "pending",
                    "notification_provenance": UNLATCHED_NOTIFICATION_PROVENANCE,
                }

            try:
                outcome = await send_gpu_quarantine_page(record)
                state = "sent" if outcome.success else "failed"
                superseded = outcome.protocol_version == "superseded"
                error = "" if outcome.success else f"{outcome.error_kind or 'unknown'}: {outcome.error or ''}"
            except Exception as exc:  # pragma: no cover - notifier promises not to raise
                state = "failed"
                superseded = False
                error = f"unexpected_error: {type(exc).__name__}"

            unlatched = record.get("notification_provenance") == UNLATCHED_NOTIFICATION_PROVENANCE
            if not superseded and not unlatched:
                try:
                    await update_gpu_quarantine_notification(
                        self.redis,
                        worker_id,
                        device=device,
                        hostname=self.hostname,
                        expected_generation=gpu_quarantine_generation(record),
                        state=state,
                        error=error,
                    )
                except Exception as exc:
                    logger.critical(f"Failed to persist page-user state for unsafe process group {worker_id}: {exc}")

            logger.critical(
                f"Worker {worker_id} physical GPU {device} QUARANTINED because old process-group "
                f"containment was not proven: {reason}"
            )

        # Once the unsafe group is detected, caller cancellation must not leave
        # a durable physical latch without its required page/delivery marker.
        notification_task = asyncio.create_task(_persist_page_and_record())
        cancellation_requested = False
        while not notification_task.done():
            try:
                await asyncio.shield(notification_task)
            except asyncio.CancelledError:
                cancellation_requested = True
        notification_task.result()
        if cancellation_requested:
            raise asyncio.CancelledError

    @staticmethod
    def _cmdline_matches_worker(pid: int, worker_id: str) -> bool:
        try:
            raw_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except FileNotFoundError:
            return False
        argv = [part.decode("utf-8", errors="replace") for part in raw_cmdline.split(b"\0") if part]
        is_kernelgym_worker = any(
            part in {"kernelgym.worker.single_worker", "kernelgym.worker.cpu_worker"} for part in argv
        )
        return is_kernelgym_worker and worker_id in argv

    def _verified_process_identity(
        self,
        pid: int,
        worker_id: str,
        expected_start_ticks: str,
        *,
        allow_zombie: bool = True,
    ) -> Optional[ProcessIdentity]:
        identity = self._read_process_identity(pid)
        if identity is None:
            return None
        if expected_start_ticks and identity.start_ticks != expected_start_ticks:
            raise ProcessIdentityMismatch(
                f"PID {pid} generation changed for {worker_id}: "
                f"expected start_ticks={expected_start_ticks}, found={identity.start_ticks}"
            )
        if identity.state == "Z":
            if allow_zombie and expected_start_ticks:
                return identity
            raise ProcessIdentityMismatch(
                f"Cannot authenticate zombie PID {pid} for {worker_id} without stored start ticks"
            )
        if not self._cmdline_matches_worker(pid, worker_id):
            raise ProcessIdentityMismatch(f"PID {pid} command line no longer belongs to {worker_id}")
        return identity

    def _reap_exact_zombie(
        self,
        worker_id: str,
        pid: int,
        expected_start_ticks: str,
    ) -> bool:
        """Reap only a zombie whose immutable Linux generation still matches."""

        retained = self.spawned_processes.get(worker_id)
        if retained is not None and int(getattr(retained, "pid", 0)) == pid:
            if retained.poll() is not None:
                retained.wait()
                self.spawned_processes.pop(worker_id, None)
                self.spawned_identities.pop(worker_id, None)
                return True

        identity = self._verified_process_identity(pid, worker_id, expected_start_ticks)
        if identity is None:
            return True
        if identity.state != "Z":
            return False
        try:
            waited_pid, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return False
        if waited_pid == pid:
            self.spawned_processes.pop(worker_id, None)
            self.spawned_identities.pop(worker_id, None)
            return True
        return False

    async def _compare_and_delete_process_map(
        self,
        worker_id: str,
        *,
        pid: int,
        map_start_ticks: Optional[str],
        map_process_group: Optional[str] = None,
        map_session_id: Optional[str] = None,
    ) -> bool:
        key = f"{KEY_PREFIX}:worker_process:{worker_id}"
        deleted = await self.redis.eval(
            _PROCESS_MAP_DELETE_IF_CURRENT,
            1,
            key,
            str(pid),
            map_start_ticks or "",
            map_process_group or "",
            map_session_id or "",
            "1" if map_start_ticks is not None else "0",
            "1" if map_process_group is not None else "0",
            "1" if map_session_id is not None else "0",
        )
        return bool(deleted)

    async def _register_spawned_process(
        self,
        worker_id: str,
        device: str,
        identity: ProcessIdentity,
    ) -> bool:
        if identity.pid <= 1 or identity.process_group != identity.pid or identity.session_id != identity.pid:
            raise ProcessIdentityMismatch(f"Worker {worker_id} PID {identity.pid} did not establish its own session")
        key = f"{KEY_PREFIX}:worker_process:{worker_id}"
        registered = await self.redis.eval(
            _PROCESS_MAP_REGISTER_IF_EMPTY,
            1,
            key,
            str(identity.pid),
            datetime.now().isoformat(),
            identity.start_ticks,
            str(identity.process_group),
            str(identity.session_id),
            device,
        )
        return bool(registered)

    async def _make_restart_request(self, worker_id: str, device: str, reason: str) -> Dict[str, Any]:
        """Bind queued work to the process generation observed by the detector."""

        process_info = self._decode_hash(await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{worker_id}"))
        try:
            observed_pid = int(process_info.get("pid") or 0)
            observed_process_group = int(process_info.get("process_group") or observed_pid)
            observed_session_id = int(process_info.get("session_id") or observed_pid)
        except ValueError:
            observed_pid = 0
            observed_process_group = 0
            observed_session_id = 0
        observed_ticks = process_info.get("proc_start_ticks", "")
        if observed_pid and not observed_ticks:
            identity = self._read_process_identity(observed_pid)
            if (
                identity is not None
                and identity.state != "Z"
                and self._cmdline_matches_worker(observed_pid, worker_id)
            ):
                observed_ticks = identity.start_ticks
                observed_process_group = identity.process_group
                observed_session_id = identity.session_id
        return {
            "worker_id": worker_id,
            "device": device,
            "reason": reason,
            "timestamp": datetime.now(),
            "observed_pid": observed_pid,
            "observed_start_ticks": observed_ticks,
            "observed_process_group": observed_process_group,
            "observed_session_id": observed_session_id,
            "replacement_required": reason
            in {
                "Process dead",
                "Missing process info & heartbeat",
            },
        }

    async def _restart_request_generation_is_current(self, restart_info: Dict[str, Any]) -> bool:
        if "observed_pid" not in restart_info:
            return True
        worker_id = str(restart_info["worker_id"])
        process_info = self._decode_hash(await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{worker_id}"))
        try:
            current_pid = int(process_info.get("pid") or 0)
            observed_pid = int(restart_info.get("observed_pid") or 0)
        except (TypeError, ValueError):
            return False
        if current_pid != observed_pid:
            return False
        observed_ticks = str(restart_info.get("observed_start_ticks") or "")
        current_ticks = process_info.get("proc_start_ticks", "")
        if observed_ticks and current_ticks and observed_ticks != current_ticks:
            return False
        if current_pid and observed_ticks:
            identity = self._read_process_identity(current_pid)
            if identity is not None and identity.start_ticks != observed_ticks:
                return False
        try:
            current_group = int(process_info.get("process_group") or current_pid)
            current_session = int(process_info.get("session_id") or current_pid)
            observed_group = int(restart_info.get("observed_process_group") or observed_pid)
            observed_session = int(restart_info.get("observed_session_id") or observed_pid)
        except (TypeError, ValueError):
            return False
        if current_group != observed_group or current_session != observed_session:
            return False
        return True

    async def _restart_budget_value(self, worker_id: str) -> int:
        path = _restart_budget_path(self.hostname, worker_id)
        durable_attempts = await asyncio.to_thread(_read_restart_budget, path)
        attempts = max(self.restart_attempts.get(worker_id, 0), durable_attempts)
        self.restart_attempts[worker_id] = attempts
        return attempts

    async def _set_restart_budget(self, worker_id: str, attempts: int) -> None:
        path = _restart_budget_path(self.hostname, worker_id)
        if attempts <= 0:
            self.restart_attempts.pop(worker_id, None)
            await asyncio.to_thread(_clear_restart_budget_file, path)
            return
        await asyncio.to_thread(
            _write_restart_budget,
            path,
            hostname=self.hostname,
            worker_id=worker_id,
            attempts=attempts,
        )
        self.restart_attempts[worker_id] = attempts

    async def _reserve_gpu_restart_attempt(self, worker_id: str, device: str, reason: str) -> bool:
        """Reserve one bounded restart, or persist quarantine before attempt N+1."""

        if not device.startswith("cuda:"):
            return True
        exclusion_record: Dict[str, str]
        async with self._restart_budget_lock:
            attempts = await self._restart_budget_value(worker_id)
            if attempts < self.max_restart_attempts:
                await self._set_restart_budget(worker_id, attempts + 1)
                return True

            reason_text = (
                f"worker failed to become healthy after {attempts} automatic restart attempts; last reason: {reason}"
            )
            persistence_succeeded = False
            try:
                exclusion_record = await write_gpu_quarantine(
                    self.redis,
                    worker_id,
                    device=device,
                    reason=reason_text,
                    fault_class="restart_limit",
                    hostname=self.hostname,
                    physical_scope=False,
                )
                persistence_succeeded = True
            except Exception as exc:
                # The restart budget stays exhausted, so this worker remains
                # excluded even when both Redis and the shared durable latch
                # are unavailable.  Paging must still proceed without a
                # durable notification claim.
                logger.critical(f"Failed to persist restart-limit quarantine for {worker_id}: {exc}")
                exclusion_record = {
                    "state": "quarantined",
                    "scope": "worker_process",
                    "worker_id": worker_id,
                    "device": device,
                    "reason": reason_text,
                    "fault_class": "restart_limit",
                    "node_id": self.hostname,
                    "hostname": self.hostname,
                    "manual_clear_required": "true",
                    "page_user_state": "pending",
                    "notification_provenance": UNLATCHED_NOTIFICATION_PROVENANCE,
                }
            if persistence_succeeded:
                # The durable quarantine is now the stronger guard. Removing
                # the counter lets an explicit clear start a fresh budget, but
                # a counter-cleanup error must not suppress the required page.
                try:
                    await self._set_restart_budget(worker_id, 0)
                except Exception as exc:
                    logger.error(f"Failed to clear restart budget after quarantining {worker_id}: {exc}")
            logger.error(f"Worker {worker_id} QUARANTINED: {reason_text}")
        # Reaching the restart limit proves this worker is unusable, but not
        # that the physical card is defective. Keep worker_process scope while
        # still notifying the user that scheduling has been closed.
        if exclusion_record.get("scope") == "worker_process":
            await self._ensure_restart_limit_notification(exclusion_record)
        return False

    async def _release_gpu_restart_attempt(self, worker_id: str, device: str) -> None:
        """Undo a reservation when a queued restart becomes obsolete before kill."""

        if not device.startswith("cuda:"):
            return
        async with self._restart_budget_lock:
            attempts = await self._restart_budget_value(worker_id)
            await self._set_restart_budget(worker_id, max(0, attempts - 1))

    async def _clear_gpu_restart_attempts(self, worker_id: str, device: str) -> None:
        if not device.startswith("cuda:"):
            return
        async with self._restart_budget_lock:
            await self._set_restart_budget(worker_id, 0)

    async def _cuda_shutdown_flag_is_current(
        self,
        worker_id: str,
        worker_info: Dict[str, str],
    ) -> bool:
        """Bind a shutdown flag to the process generation that created it."""

        if worker_info.get("cuda_error_shutdown", "false").lower() != "true":
            return False
        process_info = self._decode_hash(await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{worker_id}"))
        shutdown_text = worker_info.get("shutdown_time", "")
        process_start_text = process_info.get("start_time", "")
        if not shutdown_text or not process_start_text:
            return True
        try:
            shutdown_at = datetime.fromisoformat(shutdown_text)
            process_started_at = datetime.fromisoformat(process_start_text)
            if shutdown_at.tzinfo != process_started_at.tzinfo:
                return True
        except (TypeError, ValueError):
            return True
        if process_started_at <= shutdown_at:
            return True

        # This flag belongs to the old process.  Clearing it is best effort;
        # generation comparison remains authoritative if Redis is still
        # unavailable, so the confirmed replacement is never killed for it.
        try:
            await self.redis.hdel(
                f"{KEY_PREFIX}:worker:{worker_id}",
                "cuda_error_shutdown",
                "shutdown_time",
            )
        except Exception as exc:
            logger.error(f"Could not clear stale shutdown flag for replacement {worker_id}: {exc}")
        return False

    async def _worker_admission_is_currently_open(self, worker_id: str, device: str) -> bool:
        """Return whether a worker has demonstrably recovered since it was queued.

        A stale Redis hash is not recovery evidence by itself.  Require a fresh
        heartbeat in addition to the online/admission fields so a dead worker
        whose key has not expired cannot suppress a needed restart.
        """

        worker_data = await self.redis.hgetall(f"{KEY_PREFIX}:worker:{worker_id}")
        if not worker_data:
            return False
        worker_info = {
            (key.decode() if isinstance(key, bytes) else str(key)): (
                value.decode() if isinstance(value, bytes) else str(value)
            )
            for key, value in worker_data.items()
        }
        if await self._cuda_shutdown_flag_is_current(worker_id, worker_info):
            return False
        if worker_info.get("online", "false").lower() != "true":
            return False

        if device.startswith("cuda:"):
            if worker_info.get("health_state", "").lower() not in {"healthy", "degraded_check"}:
                return False
            if worker_info.get("accepting_tasks", "false").lower() != "true":
                return False

        heartbeat_text = worker_info.get("last_heartbeat", "")
        if not heartbeat_text:
            return False
        try:
            heartbeat = datetime.fromisoformat(heartbeat_text)
            now = datetime.now(tz=heartbeat.tzinfo)
        except (TypeError, ValueError):
            return False
        return now - heartbeat <= timedelta(seconds=self.heartbeat_timeout)

    async def _registered_process_is_live(self, worker_id: str) -> bool:
        process_info = self._decode_hash(await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{worker_id}"))
        try:
            pid = int(process_info.get("pid") or 0)
        except ValueError:
            return False
        if not pid:
            return False

        retained = self.spawned_processes.get(worker_id)
        if retained is not None and int(getattr(retained, "pid", 0)) == pid and retained.poll() is not None:
            retained.wait()
            self.spawned_processes.pop(worker_id, None)
            self.spawned_identities.pop(worker_id, None)
            return False

        expected_ticks = process_info.get("proc_start_ticks", "")
        try:
            identity = self._verified_process_identity(pid, worker_id, expected_ticks, allow_zombie=False)
        except (OSError, ProcessIdentityMismatch, RuntimeError):
            return False
        return identity is not None

    async def _registered_process_is_within_startup_grace(self, worker_id: str) -> bool:
        """Keep a live, newly launched worker out of the restart queue.

        The service records the authenticated process generation before the
        worker finishes importing and writes its first heartbeat.  A monitor
        that scans during that interval must not kill the healthy generation
        merely because the heartbeat key is not visible yet.  The normal
        heartbeat timeout bounds this grace, so a live-but-stuck startup is
        still recycled after the configured deadline.
        """

        process_info = self._decode_hash(await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{worker_id}"))
        started_text = process_info.get("start_time", "")
        if not started_text:
            return False
        try:
            started_at = datetime.fromisoformat(started_text)
            now = datetime.now(tz=started_at.tzinfo)
        except (TypeError, ValueError):
            return False
        if now - started_at > timedelta(seconds=self.startup_timeout):
            return False
        return await self._registered_process_is_live(worker_id)

    async def _check_workers(self):
        """Check health of all workers."""
        try:
            # Get all worker keys
            worker_keys = [key async for key in self.redis.scan_iter(f"{KEY_PREFIX}:worker:*", count=500)]
            # In persistent mode, load expected workers set once per cycle
            expected_ids: Set[str] = set()
            if self.persistent:
                try:
                    expected_ids = await self._load_local_expected_ids()
                except Exception:
                    expected_ids = set()

            for key in worker_keys:
                worker_id = key.decode().split(":")[-1]
                try:
                    worker_data = await self.redis.hgetall(key)
                    if not worker_data:
                        continue
                    worker_info = self._decode_hash(worker_data)
                    device = worker_info.get("device", "") or f"cuda:{worker_id.split('_')[-1]}"

                    if await self._is_worker_quarantined(
                        worker_id,
                        device=worker_info.get("device", ""),
                        hostname=worker_info.get("hostname", ""),
                    ):
                        logger.warning(f"Worker {worker_id} is GPU-quarantined; automatic restart is disabled")
                        self.restart_in_progress.discard(worker_id)
                        continue

                    needs_restart = False
                    restart_reason = ""
                    heartbeat_fresh = False
                    heartbeat_text = worker_info.get("last_heartbeat", "")
                    if heartbeat_text:
                        try:
                            heartbeat = datetime.fromisoformat(heartbeat_text)
                            now = datetime.now(tz=heartbeat.tzinfo)
                            heartbeat_fresh = now - heartbeat <= timedelta(seconds=self.heartbeat_timeout)
                        except (TypeError, ValueError):
                            restart_reason = "Invalid heartbeat"
                            needs_restart = True
                    else:
                        restart_reason = "Missing heartbeat"
                        needs_restart = True

                    if await self._cuda_shutdown_flag_is_current(worker_id, worker_info):
                        needs_restart = True
                        restart_reason = "CUDA error shutdown"
                    elif not needs_restart and not heartbeat_fresh:
                        needs_restart = True
                        restart_reason = "Heartbeat timeout"
                    elif not needs_restart and (
                        worker_info.get("online") == "false" or worker_info.get("status") == "offline"
                    ):
                        offline_since = self.monitored_workers.get(worker_id, {}).get("offline_since")
                        if offline_since is None:
                            offline_since = datetime.now()
                        if datetime.now() - offline_since > timedelta(seconds=60):
                            needs_restart = True
                            restart_reason = "Worker offline"
                        self.monitored_workers.setdefault(worker_id, {})["offline_since"] = offline_since

                    if self.persistent and worker_id not in expected_ids:
                        needs_restart = False

                    if (
                        needs_restart
                        and restart_reason in {"Missing heartbeat", "Invalid heartbeat", "Heartbeat timeout"}
                        and self.persistent
                        and await self._registered_process_is_within_startup_grace(worker_id)
                    ):
                        logger.info(
                            f"Worker {worker_id} reports {restart_reason!r} but its authenticated process "
                            "is live inside startup grace; deferring restart"
                        )
                        needs_restart = False

                    if needs_restart and worker_id not in self.restart_in_progress:
                        logger.warning(f"Worker {worker_id} needs restart: {restart_reason}")
                        await self.restart_queue.put(
                            await self._make_restart_request(worker_id, device, restart_reason)
                        )
                        self.restart_in_progress.add(worker_id)

                    if (
                        not needs_restart
                        and heartbeat_fresh
                        and worker_id not in self.restart_in_progress
                        and worker_info.get("health_state") == "healthy"
                        and worker_info.get("accepting_tasks") == "true"
                        and await self._registered_process_is_live(worker_id)
                    ):
                        await self._clear_gpu_restart_attempts(worker_id, device)

                    self.monitored_workers[worker_id] = {
                        **self.monitored_workers.get(worker_id, {}),
                        "last_check": datetime.now(),
                        "status": worker_info.get("online", worker_info.get("status", "unknown")),
                        "device": worker_info.get("device", "unknown"),
                    }
                except Exception as exc:
                    # One malformed/partially-written worker hash must not stop
                    # health and PID checks for every other GPU on this host.
                    logger.error(f"Error checking worker {worker_id}; failing it closed for this cycle: {exc}")

            # Persistent mode: also ensure expected workers are running even if
            # their heartbeat keys are missing or their PIDs are dead.
            if self.persistent:
                await self._check_persistent_expectations()

        except Exception as e:
            logger.error(f"Error checking workers: {e}")

    async def _check_persistent_expectations(self) -> None:
        """In persistent mode, restart workers missing from heartbeat keys
        or with dead PIDs according to expected worker list and process map."""
        try:
            # Load expected workers set (only ids owned by this host)
            expected_ids = await self._load_local_expected_ids()

            # Build set of existing heartbeat worker ids
            existing_ids = {
                k.decode().split(":")[-1] async for k in self.redis.scan_iter(f"{KEY_PREFIX}:worker:*", count=500)
            }

            # Restart missing-heartbeat workers
            missing_ids = expected_ids - existing_ids
            for wid in missing_ids:
                if wid in self.restart_in_progress:
                    continue
                # Determine device from expected worker hash if available
                edata = await self.redis.hgetall(f"{KEY_PREFIX}:expected_worker:{wid}")
                device = edata.get(b"device", b"").decode() if edata else f"cuda:{wid.split('_')[-1]}"
                owner = edata.get(b"hostname", b"").decode() if edata else ""
                if await self._is_worker_quarantined(wid, device=device, hostname=owner):
                    logger.warning(f"Worker {wid} is quarantined; ignoring missing heartbeat")
                    continue
                if await self._registered_process_is_within_startup_grace(wid):
                    logger.info(f"Worker {wid} is live inside startup grace; deferring missing-heartbeat restart")
                    continue
                logger.warning(f"Worker {wid} missing heartbeat key; scheduling restart (persistent mode)")
                await self.restart_queue.put(await self._make_restart_request(wid, device, "Missing heartbeat key"))
                self.restart_in_progress.add(wid)

            # Check PID liveness for all expected workers
            for wid in expected_ids:
                # Skip ones already queued
                if wid in self.restart_in_progress:
                    continue
                proc_info_raw = await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{wid}")
                proc_info = self._decode_hash(proc_info_raw)
                proc_device = proc_info.get("device", "")
                if await self._is_worker_quarantined(wid, device=proc_device):
                    continue
                if not proc_info:
                    # No process info recorded; if also no heartbeat key, schedule restart
                    if wid not in existing_ids:
                        edata = await self.redis.hgetall(f"{KEY_PREFIX}:expected_worker:{wid}")
                        device = edata.get(b"device", b"").decode() if edata else f"cuda:{wid.split('_')[-1]}"
                        logger.warning(
                            f"Worker {wid} has no process info and no heartbeat; restarting (persistent mode)"
                        )
                        await self.restart_queue.put(
                            await self._make_restart_request(wid, device, "Missing process info & heartbeat")
                        )
                        self.restart_in_progress.add(wid)
                    continue
                map_ticks: Optional[str] = (
                    proc_info.get("proc_start_ticks") if "proc_start_ticks" in proc_info else None
                )
                map_group: Optional[str] = proc_info.get("process_group") if "process_group" in proc_info else None
                map_session: Optional[str] = proc_info.get("session_id") if "session_id" in proc_info else None
                device = proc_info.get("device", "") or f"cuda:{wid.split('_')[-1]}"
                try:
                    pid = int(proc_info.get("pid") or 0)
                    process_group = int(map_group or pid)
                    session_id = int(map_session or pid)
                except (TypeError, ValueError):
                    await self._quarantine_unsafe_process_group(
                        wid,
                        device,
                        "worker process map contains a malformed PID/PGID/SID identity",
                    )
                    continue
                if pid <= 1 or process_group <= 1 or session_id <= 1 or process_group != pid or session_id != pid:
                    await self._quarantine_unsafe_process_group(
                        wid,
                        device,
                        f"worker process map has invalid new-session identity: "
                        f"pid={pid}, pgid={process_group}, sid={session_id}",
                    )
                    continue

                process_dead = False
                if pid:
                    retained = self.spawned_processes.get(wid)
                    if retained is not None and int(getattr(retained, "pid", 0)) == pid:
                        if retained.poll() is not None:
                            retained.wait()
                            self.spawned_processes.pop(wid, None)
                            self.spawned_identities.pop(wid, None)
                            process_dead = True
                    if not process_dead:
                        try:
                            identity = self._read_process_identity(pid)
                            command_matches = bool(
                                identity is not None
                                and (identity.state == "Z" or self._cmdline_matches_worker(pid, wid))
                            )
                        except (OSError, RuntimeError) as exc:
                            await self._quarantine_unsafe_process_group(
                                wid,
                                device,
                                f"could not inspect worker PID {pid} while validating session {session_id}: "
                                f"{type(exc).__name__}",
                            )
                            continue
                        process_dead = identity is None or identity.state == "Z"
                        identity_mismatch = bool(
                            identity is not None
                            and (
                                (map_ticks and identity.start_ticks != map_ticks)
                                or identity.process_group != process_group
                                or identity.session_id != session_id
                                or not command_matches
                            )
                        )
                        if identity_mismatch:
                            observed_groups = {process_group}
                            try:
                                session_drained = self._session_is_drained(session_id, observed_groups)
                            except (OSError, RuntimeError) as exc:
                                await self._quarantine_unsafe_process_group(
                                    wid,
                                    device,
                                    f"could not prove recorded session {session_id} drained after PID identity "
                                    f"mismatch: {type(exc).__name__}",
                                )
                                continue
                            if session_drained:
                                logger.critical(
                                    f"Worker {wid} PID map points to a different process generation; "
                                    "the recorded SID is empty, deleting only the stale Redis mapping"
                                )
                                deleted = await self._compare_and_delete_process_map(
                                    wid,
                                    pid=pid,
                                    map_start_ticks=map_ticks,
                                    map_process_group=map_group,
                                    map_session_id=map_session,
                                )
                                process_dead = deleted
                            else:
                                await self._quarantine_unsafe_process_group(
                                    wid,
                                    device,
                                    f"PID {pid} generation changed while recorded session {session_id} "
                                    "was not proven drained",
                                )
                                process_dead = False

                if process_dead:
                    logger.warning(f"Worker {wid} PID {pid or '-'} not alive; scheduling restart (persistent mode)")
                    await self.restart_queue.put(await self._make_restart_request(wid, device, "Process dead"))
                    self.restart_in_progress.add(wid)
        except Exception as e:
            logger.error(f"Error in persistent expectation check: {e}")

    async def _restart_loop(self):
        """Process worker restart requests."""
        while self.running:
            restart_info: Optional[Dict[str, Any]] = None
            try:
                # Get restart request with timeout
                try:
                    restart_info = await asyncio.wait_for(self.restart_queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    continue

                worker_id = restart_info["worker_id"]
                device = restart_info["device"]
                reason = restart_info["reason"]

                # Re-check after dequeue: quarantine can be written while an
                # older restart request is already waiting in this queue.
                if await self._is_worker_quarantined(worker_id, device=device):
                    logger.warning(f"Dropping queued restart for quarantined worker {worker_id}")
                    self.restart_in_progress.discard(worker_id)
                    continue

                if not await self._restart_request_generation_is_current(restart_info):
                    logger.info(f"Dropping stale process-generation restart request for {worker_id}")
                    self.restart_in_progress.discard(worker_id)
                    continue

                if not await self._reserve_gpu_restart_attempt(worker_id, device, reason):
                    self.restart_in_progress.discard(worker_id)
                    continue

                logger.info(f"Attempting to restart worker {worker_id} on {device}: {reason}")

                # Restart worker
                success = await self._restart_worker(worker_id, device, restart_info=restart_info)

                if success is None:
                    # The queued request became stale while waiting: no
                    # process was killed, so it must not consume the bounded
                    # automatic-restart budget or be requeued.
                    await self._release_gpu_restart_attempt(worker_id, device)
                    logger.info(f"Dropping stale restart request for recovered worker {worker_id}")
                elif success:
                    logger.info(f"Successfully restarted worker {worker_id}")
                    # The replacement exists now; consume the request before
                    # any best-effort bookkeeping.  A Redis failure below must
                    # never requeue old work that could kill this new process.
                    restart_info = None
                    self.restart_in_progress.discard(worker_id)
                    try:
                        await self.redis.hdel(
                            f"{KEY_PREFIX}:worker:{worker_id}",
                            "cuda_error_shutdown",
                            "shutdown_time",
                        )
                    except Exception as exc:
                        logger.error(
                            f"Restarted {worker_id}, but could not clear old shutdown flags; "
                            f"the successful restart will not be replayed: {exc}"
                        )
                    # Give worker time to initialize (API registration + GPU init can take 30-60s)
                    # This prevents the monitor from immediately detecting the worker as "missing"
                    # before it has a chance to send its first real heartbeat
                    logger.info(f"Waiting 45s for worker {worker_id} to complete initialization...")
                    await asyncio.sleep(45)
                    continue
                else:
                    logger.error(f"Failed to restart worker {worker_id}")
                    # Retry later
                    await asyncio.sleep(self.restart_cooldown)
                    await self.restart_queue.put(restart_info)
                    restart_info = None  # queued item retains restart_in_progress ownership

                # Remove from in-progress set
                if restart_info is not None:
                    self.restart_in_progress.discard(worker_id)

            except Exception as e:
                logger.error(f"Error in restart loop: {e}")
                if restart_info is not None:
                    # Fail closed on Redis/monitor errors without losing the
                    # dequeued request or permanently wedging in_progress.
                    await self.restart_queue.put(restart_info)
                await asyncio.sleep(5)

    async def _reset_gpu_device(self, device: str):
        """Deliberately disabled: GPU recovery requires an operator action."""

        logger.warning(f"Automatic GPU reset is disabled for {device}; leaving device state unchanged")

    async def _restart_worker(
        self,
        worker_id: str,
        device: str,
        *,
        restart_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[bool]:
        """Restart a worker, or return ``None`` when the request is obsolete."""
        process = None
        process_identity: Optional[ProcessIdentity] = None
        process_registered = False
        try:
            if await self._is_worker_quarantined(worker_id, device=device):
                logger.warning(f"Refusing to restart quarantined worker {worker_id}")
                return False

            if restart_info is not None and not await self._restart_request_generation_is_current(restart_info):
                logger.info(f"Worker {worker_id} process generation changed; skipping stale restart")
                return None

            # The request may have waited behind another restart.  Check the
            # live heartbeat/admission state immediately before any signal so
            # an already-recovered process is never killed by stale work.
            replacement_required = bool(restart_info and restart_info.get("replacement_required"))
            if not replacement_required and await self._worker_admission_is_currently_open(worker_id, device):
                logger.info(f"Worker {worker_id} has recovered; skipping queued restart")
                return None

            # Kill existing worker process if any
            expected_pid: Optional[int] = None
            expected_start_ticks = ""
            if restart_info is not None and "observed_pid" in restart_info:
                expected_pid = int(restart_info.get("observed_pid") or 0)
                expected_start_ticks = str(restart_info.get("observed_start_ticks") or "")
            if not await self._kill_worker_process(
                worker_id,
                expected_pid=expected_pid,
                expected_start_ticks=expected_start_ticks,
                device_hint=device,
            ):
                logger.critical(f"Refusing to spawn {worker_id}: the previous process was not confirmed stopped")
                return False
            if restart_info is not None:
                restart_info["observed_pid"] = 0
                restart_info["observed_start_ticks"] = ""
                restart_info["observed_process_group"] = 0
                restart_info["observed_session_id"] = 0
                restart_info["replacement_required"] = True

            # A monitor-level GPU reset can disrupt unrelated containers and
            # can mask a driver-level fault.  Recovery is now explicit: normal
            # process crashes get a clean process, while quarantined GPUs are
            # never restarted here.
            await asyncio.sleep(5)

            # A latch can be written while the old process is draining.
            if await self._is_worker_quarantined(worker_id, device=device):
                logger.warning(f"Refusing to restart newly quarantined worker {worker_id}")
                return False

            # Start new worker process
            import subprocess

            if device == "cpu":
                cmd = [
                    sys.executable,
                    "-m",
                    "kernelgym.worker.cpu_worker",
                    "--worker-id",
                    worker_id,
                ]
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "kernelgym.worker.single_worker",
                    "--worker-id",
                    worker_id,
                    "--device",
                    device,
                    "--persistent",
                ]

            # Ensure logs directory exists and append logs to the same pattern as manual start.
            # Honor the configured (host-nested) LOG_DIR so monitor-restarted workers land
            # in the same per-host subdirectory as everything else.
            logs_dir = Path(settings.log_dir)
            if not logs_dir.is_absolute():
                logs_dir = PROJECT_ROOT / settings.log_dir
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = logs_dir / f"{worker_id}.log"
            # Start worker as subprocess with stdout/stderr redirected to log file
            log_fh = open(log_file_path, "a", buffering=1)
            try:
                prepare_core_dump_dir(settings.core_dump_dir, settings.core_dump_keep)
            except Exception as exc:
                logger.warning("Failed to prepare core dump directory before restarting %s: %s", worker_id, exc)
            # ``start_new_session`` asks Popen to create the worker SID without
            # running Python code between fork and exec. ``preexec_fn=os.setsid``
            # can deadlock when this async monitor has other live threads.
            start_new_session = False
            creationflags = 0
            if hasattr(os, "setsid"):
                start_new_session = True
            elif os.name == "nt":
                # On Windows, use CREATE_NEW_PROCESS_GROUP if available
                try:
                    import subprocess as sp

                    creationflags = getattr(sp, "CREATE_NEW_PROCESS_GROUP", 0)
                except Exception:
                    creationflags = 0
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env={**os.environ},
                    start_new_session=start_new_session,
                    creationflags=creationflags,
                )
                # This is deliberately the first operation after Popen.  A log
                # close or /proc failure must still leave a retained handle for
                # cancellation-safe whole-group cleanup.
                self.spawned_processes[worker_id] = process
            finally:
                # Popen duplicates the descriptor for the child.  The monitor
                # must not retain one descriptor per restart forever.
                log_fh.close()

            # Register the immutable Linux identity immediately.  Waiting five
            # seconds first leaves an untracked, new-session process if this
            # monitor is cancelled or crashes during the startup probe.
            process_identity = self._read_process_identity(process.pid)
            if process_identity is None:
                logger.error(f"Worker {worker_id} exited before its PID identity could be registered")
                await self._finish_spawn_cleanup(
                    process,
                    worker_id,
                    None,
                    device,
                )
                process = None
                return False
            self.spawned_identities[worker_id] = process_identity
            try:
                process_registered = await self._register_spawned_process(worker_id, device, process_identity)
            except BaseException:
                logger.exception(
                    f"Could not register PID {process.pid} for {worker_id}; stopping the untracked process"
                )
                await self._finish_spawn_cleanup(
                    process,
                    worker_id,
                    process_identity,
                    device,
                )
                process = None
                raise
            if not process_registered:
                logger.critical(f"Refusing duplicate worker {worker_id}: another process generation owns its PID map")
                await self._finish_spawn_cleanup(
                    process,
                    worker_id,
                    process_identity,
                    device,
                )
                process = None
                return False

            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                await self._finish_spawn_cleanup(
                    process,
                    worker_id,
                    process_identity,
                    device,
                )
                process = None
                raise

            if process.poll() is None:
                logger.info(
                    f"Worker {worker_id} process started with PID {process.pid} "
                    f"(start_ticks={process_identity.start_ticks})"
                )
                return True

            logger.error(f"Worker {worker_id} process exited immediately")
            await self._finish_spawn_cleanup(
                process,
                worker_id,
                process_identity,
                device,
            )
            process = None
            return False

        except Exception as e:
            logger.error(f"Error restarting worker {worker_id}: {e}")
            if process is not None:
                await self._finish_spawn_cleanup(
                    process,
                    worker_id,
                    process_identity,
                    device,
                )
            return False

    def _pid_matches_worker(self, pid: int, worker_id: str, expected_start_ticks: str = "") -> bool:
        """Protect against signalling a reused PID owned by another process."""

        try:
            return (
                self._verified_process_identity(
                    pid,
                    worker_id,
                    expected_start_ticks,
                    allow_zombie=False,
                )
                is not None
            )
        except (OSError, ProcessIdentityMismatch, RuntimeError):
            return False

    def _send_verified_signal(
        self,
        pid: int,
        worker_id: str,
        expected_start_ticks: str,
        signum: int,
    ) -> bool:
        """Revalidate immutable identity immediately before every signal."""

        identity = self._verified_process_identity(pid, worker_id, expected_start_ticks)
        if identity is None:
            return False
        try:
            if hasattr(os, "killpg"):
                if identity.process_group != pid:
                    raise ProcessIdentityMismatch(
                        f"Worker {worker_id} PID {pid} is not its process-group leader; refusing an incomplete cleanup"
                    )
                if identity.session_id != pid:
                    raise ProcessIdentityMismatch(
                        f"Worker {worker_id} PID {pid} is not its session leader; refusing an incomplete cleanup"
                    )
                os.killpg(identity.process_group, signum)
            else:
                os.kill(pid, signum)
        except ProcessLookupError:
            return False
        logger.info(f"Sent {signal.Signals(signum).name} to worker {worker_id} generation PID {pid}")
        return True

    async def _wait_for_process_exit(
        self,
        worker_id: str,
        pid: int,
        expected_start_ticks: str,
        timeout: float,
        *,
        expected_process_group: Optional[int] = None,
        expected_session_id: Optional[int] = None,
        observed_process_groups: Optional[set[int]] = None,
    ) -> bool:
        process_group = expected_process_group or pid
        session_id = expected_session_id or pid
        known_groups = observed_process_groups if observed_process_groups is not None else {process_group}
        known_groups.add(process_group)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            leader_gone = False
            retained = self.spawned_processes.get(worker_id)
            if retained is not None and int(getattr(retained, "pid", 0)) == pid:
                if retained.poll() is not None:
                    retained.wait()
                    self.spawned_processes.pop(worker_id, None)
                    self.spawned_identities.pop(worker_id, None)
                    leader_gone = True
            if not leader_gone:
                try:
                    identity = self._verified_process_identity(pid, worker_id, expected_start_ticks)
                except ProcessIdentityMismatch:
                    # During exit Linux can make the leader's command line
                    # unreadable before /proc/<pid> disappears.  The immutable
                    # safety property is absence of the old SID, not continued
                    # readability of the dying leader's argv.  Never signal a
                    # mismatched/reused PID; use the remainder of the existing
                    # exit deadline to prove the recorded SID fully drained.
                    if await self._wait_for_session_drain(
                        session_id,
                        known_groups,
                        max(0.0, deadline - time.monotonic()),
                    ):
                        return True
                    raise
                if identity is None:
                    leader_gone = True
                elif identity.process_group != process_group:
                    raise ProcessIdentityMismatch(
                        f"Worker {worker_id} PID {pid} moved from process group "
                        f"{process_group} to {identity.process_group}"
                    )
                elif identity.session_id != session_id:
                    raise ProcessIdentityMismatch(
                        f"Worker {worker_id} PID {pid} moved from session {session_id} to {identity.session_id}"
                    )
                elif identity.state == "Z" and self._reap_exact_zombie(
                    worker_id,
                    pid,
                    expected_start_ticks,
                ):
                    leader_gone = True

            # A dead/reaped leader is necessary but not sufficient. Inner warm
            # workers may lead independent PGIDs while retaining this SID.
            if leader_gone:
                return await self._wait_for_session_drain(
                    session_id,
                    known_groups,
                    max(0.0, deadline - time.monotonic()),
                )
            await asyncio.sleep(0.25)
        return False

    async def _stop_spawned_process(
        self,
        process: Any,
        worker_id: str,
        identity: Optional[ProcessIdentity] = None,
        device: str = "",
        *,
        discover_identity: bool = True,
    ) -> bool:
        """Terminate, reap, and prove the Popen child's complete SID absent."""

        identity_error: Optional[Exception] = None
        if identity is None and discover_identity:
            try:
                identity = self.spawned_identities.get(worker_id) or self._read_process_identity(process.pid)
            except (OSError, ProcessIdentityMismatch, RuntimeError) as exc:
                identity_error = exc
        process_group = identity.process_group if identity is not None else process.pid
        session_id = identity.session_id if identity is not None else process.pid
        observed_process_groups = {process_group}
        if process.poll() is not None:
            await asyncio.to_thread(process.wait)
            try:
                if self._session_is_drained(session_id, observed_process_groups):
                    self.spawned_processes.pop(worker_id, None)
                    self.spawned_identities.pop(worker_id, None)
                    return True
            except (OSError, RuntimeError) as exc:
                identity_error = exc
            if identity is not None:
                stopped, reason = await self._force_kill_worker_session(
                    worker_id,
                    pid=process.pid,
                    expected_leader_start_ticks=identity.start_ticks,
                    session_id=session_id,
                    observed_process_groups=observed_process_groups,
                )
                if stopped:
                    self.spawned_processes.pop(worker_id, None)
                    self.spawned_identities.pop(worker_id, None)
                    return True
            else:
                reason = (
                    f"session inspection failed: {type(identity_error).__name__}"
                    if identity_error is not None
                    else "leader generation was never authenticated"
                )
            await self._quarantine_unsafe_process_group(
                worker_id,
                device,
                f"worker leader PID {process.pid} exited but session {session_id} was not drained: {reason}",
            )
            return False
        if identity is None:
            # The leader can exit between the first poll and the /proc read.
            # That race is still not a containment proof: its CUDA children may
            # remain in the process group after the Popen leader is reapable.
            if process.poll() is not None:
                await asyncio.to_thread(process.wait)
                try:
                    if self._session_is_drained(session_id, observed_process_groups):
                        self.spawned_processes.pop(worker_id, None)
                        self.spawned_identities.pop(worker_id, None)
                        return True
                except (OSError, RuntimeError) as exc:
                    identity_error = exc
                await self._quarantine_unsafe_process_group(
                    worker_id,
                    device,
                    f"worker leader PID {process.pid} exited during cleanup but unauthenticated session "
                    f"{session_id} was not proven drained"
                    + (f": {type(identity_error).__name__}" if identity_error is not None else ""),
                )
                return False
            await self._quarantine_unsafe_process_group(
                worker_id,
                device,
                f"could not authenticate live worker leader PID {process.pid} before session cleanup"
                + (f": {type(identity_error).__name__}" if identity_error is not None else ""),
            )
            return False
        if identity.process_group != identity.pid or identity.session_id != identity.pid:
            await self._quarantine_unsafe_process_group(
                worker_id,
                device,
                f"worker PID {identity.pid} is not an authenticated new-session leader "
                f"(pgid={identity.process_group}, sid={identity.session_id})",
            )
            return False
        try:
            self._send_verified_signal(process.pid, worker_id, identity.start_ticks, signal.SIGTERM)
            stopped = await self._wait_for_process_exit(
                worker_id,
                process.pid,
                identity.start_ticks,
                10,
                expected_process_group=process_group,
                expected_session_id=session_id,
                observed_process_groups=observed_process_groups,
            )
            reason = ""
            if not stopped:
                leader_after_term = self._read_process_identity(process.pid)
                if leader_after_term is not None and leader_after_term.start_ticks != identity.start_ticks:
                    raise ProcessIdentityMismatch(
                        f"Worker {worker_id} PID {process.pid} was reused before session {session_id} drained"
                    )
                stopped, reason = await self._force_kill_worker_session(
                    worker_id,
                    pid=process.pid,
                    expected_leader_start_ticks=identity.start_ticks,
                    session_id=session_id,
                    observed_process_groups=observed_process_groups,
                )
            if not stopped:
                logger.critical(f"Failed to stop/reap untracked worker {worker_id} PID {process.pid}")
                await self._quarantine_unsafe_process_group(
                    worker_id,
                    device,
                    reason or f"session {session_id} remained live after SIGKILL escalation",
                )
                return False
            # Popen.poll() normally reaps, but wait() is idempotent and makes
            # the exact-child reap guarantee explicit for alternate Popen
            # implementations and tests.
            await asyncio.to_thread(process.wait)
            self.spawned_processes.pop(worker_id, None)
            self.spawned_identities.pop(worker_id, None)
            return True
        except (OSError, ProcessIdentityMismatch, RuntimeError) as exc:
            logger.critical(f"Failed to stop/reap untracked worker {worker_id} PID {process.pid}: {exc}")
            await self._quarantine_unsafe_process_group(
                worker_id,
                device,
                f"could not prove session {session_id} was drained during spawn rollback: {type(exc).__name__}",
            )
            return False

    async def _finish_spawn_cleanup(
        self,
        process: Any,
        worker_id: str,
        identity: Optional[ProcessIdentity],
        device: str,
    ) -> bool:
        """Finish spawn rollback even when the caller is being cancelled."""

        async def _cleanup() -> bool:
            stopped = await self._stop_spawned_process(
                process,
                worker_id,
                identity,
                device,
                # A failed first identity read is immutable evidence that this
                # launch generation was never authenticated. A later /proc PID
                # could be a reused generation and must not be signalled.
                discover_identity=identity is not None,
            )
            if stopped and identity is not None:
                await self._compare_and_delete_process_map(
                    worker_id,
                    pid=identity.pid,
                    map_start_ticks=identity.start_ticks,
                    map_process_group=str(identity.process_group),
                    map_session_id=str(identity.session_id),
                )
            return stopped

        cleanup_task = asyncio.create_task(_cleanup())
        cancellation_requested = False
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                cancellation_requested = True
        result = cleanup_task.result()
        if cancellation_requested:
            raise asyncio.CancelledError
        return result

    async def _delete_stopped_process_map(
        self,
        worker_id: str,
        *,
        pid: int,
        map_start_ticks: Optional[str],
        map_process_group: Optional[str],
        map_session_id: Optional[str],
    ) -> bool:
        deleted = await self._compare_and_delete_process_map(
            worker_id,
            pid=pid,
            map_start_ticks=map_start_ticks,
            map_process_group=map_process_group,
            map_session_id=map_session_id,
        )
        if deleted:
            return True
        current = await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{worker_id}")
        if current:
            logger.critical(
                f"Worker {worker_id} process map changed while PID {pid} was being reaped; "
                "refusing to spawn over the newer generation"
            )
            return False
        return True

    async def _kill_worker_process(
        self,
        worker_id: str,
        *,
        expected_pid: Optional[int] = None,
        expected_start_ticks: str = "",
        device_hint: str = "",
    ) -> bool:
        """Kill one authenticated process generation and confirm exact exit."""

        device = device_hint
        if not device and worker_id.startswith("worker_gpu_"):
            device = f"cuda:{worker_id.rsplit('_', 1)[-1]}"
        process_group = 0
        session_id = 0
        containment_proven = False
        try:
            raw_process_info = await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{worker_id}")
            process_info = self._decode_hash(raw_process_info)
            if not process_info:
                retained = self.spawned_processes.get(worker_id)
                retained_identity = self.spawned_identities.get(worker_id)
                if retained is not None and retained_identity is None:
                    logger.critical(
                        f"Worker {worker_id} has an unauthenticated locally tracked PID {retained.pid}; "
                        "proving its new-session process group absent before any replacement"
                    )
                    return await self._finish_spawn_cleanup(
                        retained,
                        worker_id,
                        None,
                        device,
                    )
                if retained is not None and retained_identity is not None:
                    if retained.poll() is not None:
                        retained.wait()
                        observed_groups = {retained_identity.process_group}
                        if self._session_is_drained(retained_identity.session_id, observed_groups):
                            self.spawned_processes.pop(worker_id, None)
                            self.spawned_identities.pop(worker_id, None)
                            return True
                        stopped, reason = await self._force_kill_worker_session(
                            worker_id,
                            pid=retained_identity.pid,
                            expected_leader_start_ticks=retained_identity.start_ticks,
                            session_id=retained_identity.session_id,
                            observed_process_groups=observed_groups,
                        )
                        if stopped:
                            self.spawned_processes.pop(worker_id, None)
                            self.spawned_identities.pop(worker_id, None)
                            return True
                        await self._quarantine_unsafe_process_group(
                            worker_id,
                            device,
                            f"worker leader PID {retained_identity.pid} exited but session "
                            f"{retained_identity.session_id} was not drained: {reason}",
                        )
                        return False
                    logger.critical(
                        f"Worker {worker_id} PID {retained_identity.pid} is locally tracked but absent from Redis; "
                        "reaping it before any replacement"
                    )
                    return await self._stop_spawned_process(retained, worker_id, retained_identity, device)
                return expected_pid in {None, 0}
            device = device or process_info.get("device", "")
            try:
                pid = int(process_info.get("pid") or 0)
            except (TypeError, ValueError):
                await self._quarantine_unsafe_process_group(
                    worker_id,
                    device,
                    "worker process map contains a malformed PID",
                )
                return False
            if pid <= 1:
                await self._quarantine_unsafe_process_group(
                    worker_id,
                    device,
                    f"worker process map contains an unsafe PID: {pid}",
                )
                return False
            if expected_pid is not None and pid != expected_pid:
                logger.info(
                    f"Worker {worker_id} PID generation changed from request {expected_pid} to {pid}; "
                    "refusing stale restart"
                )
                return False

            map_start_ticks: Optional[str] = (
                process_info.get("proc_start_ticks") if "proc_start_ticks" in process_info else None
            )
            map_process_group: Optional[str] = (
                process_info.get("process_group") if "process_group" in process_info else None
            )
            map_session_id: Optional[str] = process_info.get("session_id") if "session_id" in process_info else None
            try:
                process_group = int(process_info.get("process_group") or pid)
                session_id = int(process_info.get("session_id") or pid)
            except ValueError:
                await self._quarantine_unsafe_process_group(
                    worker_id,
                    device,
                    f"invalid process-group/session identity for worker PID {pid}",
                )
                return False
            if process_group <= 1 or session_id <= 1 or process_group != pid or session_id != pid:
                await self._quarantine_unsafe_process_group(
                    worker_id,
                    device,
                    f"invalid new-session identity for worker PID {pid}: pgid={process_group}, sid={session_id}",
                )
                return False
            if expected_start_ticks and map_start_ticks and expected_start_ticks != map_start_ticks:
                return False
            start_ticks = expected_start_ticks or map_start_ticks or ""
            retained_identity = self.spawned_identities.get(worker_id)
            if not start_ticks and retained_identity is not None and retained_identity.pid == pid:
                start_ticks = retained_identity.start_ticks

            identity = self._read_process_identity(pid)
            if identity is None:
                observed_groups = {process_group}
                if not self._session_is_drained(session_id, observed_groups):
                    if start_ticks:
                        stopped, reason = await self._force_kill_worker_session(
                            worker_id,
                            pid=pid,
                            expected_leader_start_ticks=start_ticks,
                            session_id=session_id,
                            observed_process_groups=observed_groups,
                        )
                    else:
                        stopped = False
                        reason = "leader exited and no authenticated start_ticks are available"
                    if stopped:
                        containment_proven = True
                        return await self._delete_stopped_process_map(
                            worker_id,
                            pid=pid,
                            map_start_ticks=map_start_ticks,
                            map_process_group=map_process_group,
                            map_session_id=map_session_id,
                        )
                    await self._quarantine_unsafe_process_group(
                        worker_id,
                        device,
                        f"worker leader PID {pid} exited but session {session_id} was not drained: {reason}",
                    )
                    return False
                containment_proven = True
                return await self._delete_stopped_process_map(
                    worker_id,
                    pid=pid,
                    map_start_ticks=map_start_ticks,
                    map_process_group=map_process_group,
                    map_session_id=map_session_id,
                )
            if not start_ticks:
                if identity.state == "Z":
                    logger.critical(
                        f"Cannot authenticate legacy zombie PID {pid} for {worker_id}; start ticks were not recorded"
                    )
                    await self._quarantine_unsafe_process_group(
                        worker_id,
                        device,
                        f"cannot authenticate legacy zombie PID {pid}; start ticks were not recorded",
                    )
                    return False
                if not self._cmdline_matches_worker(pid, worker_id):
                    logger.critical(f"PID ownership mismatch for {worker_id}: refusing reused PID {pid}")
                    await self._quarantine_unsafe_process_group(
                        worker_id,
                        device,
                        f"PID ownership mismatch for mapped worker PID {pid}; session cleanup is unproven",
                    )
                    return False
                start_ticks = identity.start_ticks

            try:
                identity = self._verified_process_identity(pid, worker_id, start_ticks)
            except ProcessIdentityMismatch:
                # The process can exit between the initial /proc identity read
                # and command-line verification.  Do not signal anything in
                # that ambiguous state; allow natural exit within the normal
                # cleanup grace, prove the recorded SID empty, and then remove
                # only the exact Redis generation.
                observed_groups = {process_group}
                if await self._wait_for_session_drain(session_id, observed_groups, 10):
                    containment_proven = True
                    return await self._delete_stopped_process_map(
                        worker_id,
                        pid=pid,
                        map_start_ticks=map_start_ticks,
                        map_process_group=map_process_group,
                        map_session_id=map_session_id,
                    )
                raise
            if identity is None:
                observed_groups = {process_group}
                if not self._session_is_drained(session_id, observed_groups):
                    stopped, reason = await self._force_kill_worker_session(
                        worker_id,
                        pid=pid,
                        expected_leader_start_ticks=start_ticks,
                        session_id=session_id,
                        observed_process_groups=observed_groups,
                    )
                    if stopped:
                        containment_proven = True
                        return await self._delete_stopped_process_map(
                            worker_id,
                            pid=pid,
                            map_start_ticks=map_start_ticks,
                            map_process_group=map_process_group,
                            map_session_id=map_session_id,
                        )
                    await self._quarantine_unsafe_process_group(
                        worker_id,
                        device,
                        f"worker leader PID {pid} exited but session {session_id} was not drained: {reason}",
                    )
                    return False
                containment_proven = True
                return await self._delete_stopped_process_map(
                    worker_id,
                    pid=pid,
                    map_start_ticks=map_start_ticks,
                    map_process_group=map_process_group,
                    map_session_id=map_session_id,
                )
            if identity.process_group != process_group:
                raise ProcessIdentityMismatch(
                    f"Worker {worker_id} PID {pid} process group changed from "
                    f"mapped {process_group} to {identity.process_group}"
                )
            if identity.session_id != session_id:
                raise ProcessIdentityMismatch(
                    f"Worker {worker_id} PID {pid} session changed from mapped {session_id} to {identity.session_id}"
                )
            if identity.state == "Z" and self._reap_exact_zombie(worker_id, pid, start_ticks):
                observed_groups = {process_group}
                if not self._session_is_drained(session_id, observed_groups):
                    stopped, reason = await self._force_kill_worker_session(
                        worker_id,
                        pid=pid,
                        expected_leader_start_ticks=start_ticks,
                        session_id=session_id,
                        observed_process_groups=observed_groups,
                    )
                    if stopped:
                        containment_proven = True
                        return await self._delete_stopped_process_map(
                            worker_id,
                            pid=pid,
                            map_start_ticks=map_start_ticks,
                            map_process_group=map_process_group,
                            map_session_id=map_session_id,
                        )
                    await self._quarantine_unsafe_process_group(
                        worker_id,
                        device,
                        f"worker leader PID {pid} was reaped but session {session_id} was not drained: {reason}",
                    )
                    return False
                containment_proven = True
                return await self._delete_stopped_process_map(
                    worker_id,
                    pid=pid,
                    map_start_ticks=map_start_ticks,
                    map_process_group=map_process_group,
                    map_session_id=map_session_id,
                )

            self._send_verified_signal(pid, worker_id, start_ticks, signal.SIGTERM)
            observed_groups = {process_group}
            stopped = await self._wait_for_process_exit(
                worker_id,
                pid,
                start_ticks,
                10,
                expected_process_group=process_group,
                expected_session_id=session_id,
                observed_process_groups=observed_groups,
            )
            if not stopped:
                leader_after_term = self._read_process_identity(pid)
                if leader_after_term is not None and leader_after_term.start_ticks != start_ticks:
                    raise ProcessIdentityMismatch(
                        f"Worker {worker_id} PID {pid} was reused before session {session_id} drained"
                    )
                stopped, reason = await self._force_kill_worker_session(
                    worker_id,
                    pid=pid,
                    expected_leader_start_ticks=start_ticks,
                    session_id=session_id,
                    observed_process_groups=observed_groups,
                )
            if not stopped:
                reason = reason or f"session {session_id} remained live after SIGKILL escalation"
                logger.critical(f"Worker {worker_id} {reason}")
                await self._quarantine_unsafe_process_group(worker_id, device, reason)
                return False

            containment_proven = True
            self.spawned_processes.pop(worker_id, None)
            self.spawned_identities.pop(worker_id, None)
            return await self._delete_stopped_process_map(
                worker_id,
                pid=pid,
                map_start_ticks=map_start_ticks,
                map_process_group=map_process_group,
                map_session_id=map_session_id,
            )
        except (OSError, ProcessIdentityMismatch, RuntimeError) as exc:
            logger.error(f"Error killing worker process {worker_id}: {exc}")
            if not containment_proven:
                await self._quarantine_unsafe_process_group(
                    worker_id,
                    device,
                    f"could not prove session {session_id or '-'} was drained: {type(exc).__name__}",
                )
            return False
        except Exception as exc:
            logger.error(f"Error killing worker process {worker_id}: {exc}")
            if not containment_proven:
                await self._quarantine_unsafe_process_group(
                    worker_id,
                    device,
                    f"unexpected error before session {session_id or '-'} drain proof: {type(exc).__name__}",
                )
            return False


async def main():
    """Main entry point for worker monitor."""
    # Parse CLI args
    parser = argparse.ArgumentParser(description="KernelGym Worker Monitor")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Enable persistent monitoring (restart workers even if heartbeat keys disappear)",
    )
    args = parser.parse_args()

    # Configure logging
    logger = setup_logging("worker_monitor")

    # Initialize Redis connection with readiness wait
    async def _wait_for_redis_ready(url: str, timeout_sec: float = 60.0, interval_sec: float = 0.5):
        start = asyncio.get_event_loop().time()
        client = redis.from_url(url)
        last_err = None
        while True:
            try:
                await client.ping()
                return client
            except (BusyLoadingError, RedisResponseError) as e:
                last_err = e
                logger.warning(f"[monitor] Redis not ready (loading data): {e}. Retrying...")
            except (RedisConnectionError, RedisTimeoutError) as e:
                last_err = e
                logger.warning(f"[monitor] Redis connection not ready: {e}. Retrying...")
            except Exception as e:
                last_err = e
                logger.warning(f"[monitor] Redis ping error: {e}. Retrying...")
            if (asyncio.get_event_loop().time() - start) > timeout_sec:
                raise RuntimeError(f"Redis not ready within {timeout_sec}s: {last_err}")
            await asyncio.sleep(interval_sec)

    redis_client = await _wait_for_redis_ready(settings.redis_url)
    logger.info("Redis connection established for worker monitor")

    # Create and start monitor
    monitor = WorkerMonitor(redis_client, persistent=bool(args.persistent))

    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker monitor error: {e}")
        sys.exit(1)
    finally:
        await monitor.stop()
        await redis_client.close()


if __name__ == "__main__":
    import os

    asyncio.run(main())
