import asyncio
import importlib.util
from pathlib import Path
import time


SUBPROCESS_POOL_PATH = Path(__file__).resolve().parents[1] / "kernelgym" / "worker" / "subprocess_pool.py"
spec = importlib.util.spec_from_file_location("subprocess_pool_under_test", SUBPROCESS_POOL_PATH)
assert spec is not None and spec.loader is not None
subprocess_pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subprocess_pool)
SubprocessWorkerPool = subprocess_pool.SubprocessWorkerPool


class FakeWorker:
    def __init__(self, worker_id: str, alive: bool = True) -> None:
        self.worker_id = worker_id
        self.tasks_processed = 0
        self.shutdown_called = False
        self._alive = alive
        self.process = None

    def is_alive(self) -> bool:
        return self._alive

    def shutdown(self, timeout: int = 10) -> None:
        self.shutdown_called = True
        self._alive = False


class _NoOpThread:
    """Captures the target without actually starting a thread."""

    def __init__(self, target=None, daemon: bool = False, **_: object) -> None:
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True


async def _drain_loop() -> None:
    for _ in range(5):
        await asyncio.sleep(0.01)


def _pool_without_processes(*, pool_size: int = 1) -> SubprocessWorkerPool:
    pool = SubprocessWorkerPool.__new__(SubprocessWorkerPool)
    pool.device_id = 0
    pool.pool_size = pool_size
    pool.worker_prefix = "test_worker"
    pool.max_tasks_per_worker = 1
    pool.workers = []
    pool.idle_workers = []
    pool.busy_workers = []
    pool.pending_replacements = 0
    pool._top_up_sequence = 0
    pool.total_tasks_processed = 0
    pool.total_workers_restarted = 0
    pool.pool_start_time = time.time()
    pool.lock = asyncio.Lock()
    return pool


def test_get_idle_worker_waits_for_pending_replacement(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes()
        pool.pending_replacements = 1

        def fail_if_emergency_worker_starts(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("emergency worker should not start while replacement is pending")

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fail_if_emergency_worker_starts)

        assert await pool._get_idle_worker(timeout=0.2) is None
        assert pool.pending_replacements == 1

    asyncio.run(scenario())


def test_restart_worker_does_not_grow_pool_past_configured_size() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        old_worker = FakeWorker("old", alive=False)
        existing_worker = FakeWorker("existing", alive=True)
        pool.workers = [existing_worker, old_worker]
        pool.idle_workers = [existing_worker]
        pool.busy_workers = [old_worker]

        await pool._restart_worker(old_worker)  # type: ignore[arg-type]

        assert pool.workers == [existing_worker]
        assert pool.idle_workers == [existing_worker]
        assert pool.busy_workers == []
        assert pool.pending_replacements == 0
        assert old_worker.shutdown_called is True

    asyncio.run(scenario())


def test_pool_size_2_recycle_leaves_spare_idle(monkeypatch) -> None:
    """When one of two workers recycles, the other stays idle and the
    replacement is scheduled in the background."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        spare = FakeWorker("spare", alive=True)
        recycling = FakeWorker("recycling", alive=False)
        pool.workers = [spare, recycling]
        pool.idle_workers = [spare]
        pool.busy_workers = [recycling]

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)

        await pool._restart_worker(recycling)  # type: ignore[arg-type]

        # The live spare is still tracked and immediately available.
        assert pool.workers == [spare]
        assert pool.idle_workers == [spare]
        assert pool.busy_workers == []
        # One replacement is pending and a background thread was queued.
        assert pool.pending_replacements == 1
        assert len(threads) == 1 and threads[0].started is True
        # Invariant: workers + pending must never exceed pool_size.
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size

    asyncio.run(scenario())


def test_pool_size_2_two_concurrent_recycles_respect_capacity(monkeypatch) -> None:
    """Even when both workers recycle in a tight window, workers+pending
    stays bounded by pool_size."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        a = FakeWorker("a", alive=False)
        b = FakeWorker("b", alive=False)
        pool.workers = [a, b]
        pool.idle_workers = []
        pool.busy_workers = [a, b]

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)

        await pool._restart_worker(a)  # type: ignore[arg-type]
        await pool._restart_worker(b)  # type: ignore[arg-type]

        # Both workers were removed; two replacements are queued.
        assert pool.workers == []
        assert pool.busy_workers == []
        assert pool.pending_replacements == 2
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size
        # Each recycle queued exactly one background thread.
        assert len(threads) == 2
        assert all(t.started for t in threads)

    asyncio.run(scenario())


def test_pool_size_2_recycle_at_capacity_shuts_down_extra() -> None:
    """If a worker recycles while a replacement is already pending and the
    remaining workers already meet pool_size, no extra replacement is
    scheduled and the recycled worker is shut down synchronously.

    No threading monkeypatch here — the synchronous-shutdown path uses
    asyncio.to_thread, which would deadlock if threading.Thread were stubbed.
    Instead we infer "no background thread spawned" from the fact that the
    `should_replenish=False` branch is mutually exclusive with the thread
    branch.
    """

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        live = FakeWorker("live", alive=True)
        recycling = FakeWorker("recycling", alive=False)
        pool.workers = [live, recycling]
        pool.idle_workers = []
        pool.busy_workers = [live, recycling]
        # Simulate an in-flight replacement from a previous recycle.
        pool.pending_replacements = 1

        await pool._restart_worker(recycling)  # type: ignore[arg-type]

        # `live` stays tracked; recycling was removed.
        assert pool.workers == [live]
        # Pending unchanged: 1 (live) + 1 (in-flight) == pool_size=2 already.
        assert pool.pending_replacements == 1
        # The recycled worker was shut down synchronously instead.
        assert recycling.shutdown_called is True
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size

    asyncio.run(scenario())


def test_pool_size_2_get_idle_returns_spare_during_recycle(monkeypatch) -> None:
    """While one worker is being replaced in the background, the warm spare
    is handed out immediately to the next request — no emergency path."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        spare = FakeWorker("spare", alive=True)
        pool.workers = [spare]
        pool.idle_workers = [spare]
        pool.busy_workers = []
        # Mark one replacement as in-flight (the previously-recycled worker).
        pool.pending_replacements = 1

        def fail_if_emergency_worker_starts(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("emergency worker must not start; spare is idle")

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fail_if_emergency_worker_starts)

        worker = await pool._get_idle_worker(timeout=1.0)

        assert worker is spare
        assert pool.busy_workers == [spare]
        assert pool.idle_workers == []
        # pending unchanged — emergency must not have fired.
        assert pool.pending_replacements == 1

    asyncio.run(scenario())


def test_pool_size_2_get_idle_tops_up_when_only_one_worker_remains(monkeypatch) -> None:
    """Checking out the only idle worker must start a top-up immediately.

    Otherwise a busy worker can sit alone until it returns, which defeats the
    warm-spare invariant under sustained load.
    """

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        only_worker = FakeWorker("only", alive=True)
        pool.workers = [only_worker]
        pool.idle_workers = [only_worker]

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)

        worker = await pool._get_idle_worker(timeout=1.0)

        assert worker is only_worker
        assert pool.busy_workers == [only_worker]
        assert pool.idle_workers == []
        assert pool.pending_replacements == 1
        assert len(threads) == 1
        assert threads[0].started is True
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size

    asyncio.run(scenario())


def test_pool_size_2_no_emergency_when_pending_already_in_flight(monkeypatch) -> None:
    """With pool_size=2 and pending_replacements=2 (both replacements in
    flight), _get_idle_worker waits without spawning an emergency."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        pool.pending_replacements = 2

        def fail_if_emergency_worker_starts(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("emergency worker should not start while replacements are pending")

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fail_if_emergency_worker_starts)

        assert await pool._get_idle_worker(timeout=0.2) is None
        assert pool.pending_replacements == 2

    asyncio.run(scenario())


def test_pool_size_2_degraded_recycle_schedules_top_up(monkeypatch) -> None:
    """If the pool has already degraded to one worker, a recycle should
    schedule the direct replacement plus another top-up spare."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        recycling = FakeWorker("recycling", alive=False)
        pool.workers = [recycling]
        pool.busy_workers = [recycling]

        created: list[FakeWorker] = []

        def fake_persistent_worker(worker_id, *_args, **_kwargs):  # noqa: ANN001
            worker = FakeWorker(worker_id, alive=True)
            created.append(worker)
            return worker

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fake_persistent_worker)
        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)

        await pool._restart_worker(recycling)  # type: ignore[arg-type]
        assert pool.pending_replacements == 1
        assert len(threads) == 1

        threads[0].target()
        await _drain_loop()

        assert pool.workers == [created[0]]
        assert pool.idle_workers == [created[0]]
        assert pool.pending_replacements == 1
        assert len(threads) == 2
        assert threads[1].started is True
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size

    asyncio.run(scenario())


def test_pool_size_2_failed_replacement_keeps_top_up_pending(monkeypatch) -> None:
    """A failed replacement must not leave the pool permanently undersized."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        recycling = FakeWorker("recycling", alive=False)
        pool.workers = [recycling]
        pool.busy_workers = [recycling]

        def fake_persistent_worker(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("spawn failed")

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fake_persistent_worker)
        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)

        await pool._restart_worker(recycling)  # type: ignore[arg-type]
        assert pool.pending_replacements == 1
        assert len(threads) == 1

        threads[0].target()
        await _drain_loop()

        assert pool.workers == []
        assert pool.idle_workers == []
        assert pool.pending_replacements == 2
        assert len(threads) == 3
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size

    asyncio.run(scenario())


def test_return_worker_does_not_revive_recycling_worker(monkeypatch) -> None:
    """The task finally block must not return an untracked recycled worker."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        recycling = FakeWorker("recycling", alive=True)
        pool.workers = [recycling]
        pool.busy_workers = [recycling]

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            thread = _NoOpThread(*args, **kwargs)
            threads.append(thread)
            return thread

        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)

        # _restart_worker removes the worker immediately, while its process is
        # still alive until the captured background shutdown thread runs.
        await pool._restart_worker(recycling)  # type: ignore[arg-type]
        assert recycling.is_alive() is True
        assert recycling not in pool.workers
        assert len(threads) == 1 and threads[0].started is True

        # This is the execute_task finally path that used to resurrect it.
        await pool._return_worker(recycling)  # type: ignore[arg-type]

        assert recycling not in pool.workers
        assert recycling not in pool.idle_workers
        assert recycling not in pool.busy_workers
        assert pool.pending_replacements == 1

    asyncio.run(scenario())


def test_get_idle_worker_reconciles_stale_pool_state_and_tops_up(monkeypatch) -> None:
    """Dead tracked and live untracked references cannot block pool recovery."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        dead_idle = FakeWorker("dead-idle", alive=False)
        dead_busy = FakeWorker("dead-busy", alive=False)
        untracked_idle = FakeWorker("untracked-idle", alive=True)
        untracked_busy = FakeWorker("untracked-busy", alive=True)
        pool.workers = [dead_idle, dead_busy]
        pool.idle_workers = [dead_idle, untracked_idle]
        pool.busy_workers = [dead_busy, untracked_busy]

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            thread = _NoOpThread(*args, **kwargs)
            threads.append(thread)
            return thread

        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)

        assert await pool._get_idle_worker(timeout=0.2) is None
        assert pool.workers == []
        assert pool.idle_workers == []
        assert pool.busy_workers == []
        assert pool.pending_replacements == pool.pool_size
        assert len(threads) == pool.pool_size
        assert all(thread.started for thread in threads)

    asyncio.run(scenario())
