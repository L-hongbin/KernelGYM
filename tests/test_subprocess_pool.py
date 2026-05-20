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
