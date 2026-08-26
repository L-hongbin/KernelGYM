"""Bug #2 regression: a crashed worker subprocess must be reported promptly as a
crash, not waited out and misreported as a TIMEOUT.

Covers ``PersistentWorker.execute_task``'s process-death detection. No GPU or real
subprocess is needed -- the worker is built with ``__new__`` and driven with a
scripted fake process/queue, exactly along the changed control-flow path.
"""

import importlib.util
import queue
import time
from pathlib import Path


SUBPROCESS_POOL_PATH = Path(__file__).resolve().parents[2] / "kernelgym" / "worker" / "subprocess_pool.py"
spec = importlib.util.spec_from_file_location("subprocess_pool_crash_under_test", SUBPROCESS_POOL_PATH)
assert spec is not None and spec.loader is not None
subprocess_pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subprocess_pool)
PersistentWorker = subprocess_pool.PersistentWorker


class _ScriptedProc:
    """``is_alive()`` yields the scripted values, then repeats the last one.

    A ``[True, False]`` sequence models "alive at the initial guard check, dead
    by the time the poll loop re-checks".
    """

    def __init__(self, alive_seq, exitcode=127, pid=4242):
        self._seq = list(alive_seq)
        self.exitcode = exitcode
        self.pid = pid

    def is_alive(self):
        return self._seq.pop(0) if len(self._seq) > 1 else self._seq[0]


class _ScriptedQueue:
    def __init__(self, get_seq):
        self._seq = list(get_seq)
        self.put_calls = 0

    def put(self, *args, **kwargs):
        self.put_calls += 1

    def get(self, timeout=None):
        if not self._seq:
            raise queue.Empty
        item = self._seq.pop(0)
        if item is queue.Empty:
            raise queue.Empty
        return item


def _make_worker(proc, result_queue):
    worker = PersistentWorker.__new__(PersistentWorker)
    worker.worker_id = "test_worker"
    worker.is_alive_flag = True
    worker.process = proc
    worker.task_queue = _ScriptedQueue([])
    worker.result_queue = result_queue
    worker.tasks_processed = 0
    worker.max_tasks_per_worker = 100
    return worker


def test_crash_reported_promptly_not_as_timeout() -> None:
    worker = _make_worker(_ScriptedProc(alive_seq=[True, False]), _ScriptedQueue([]))
    start = time.monotonic()
    result = worker.execute_task({"task_id": "t1"}, timeout=30, poll_interval=0.01)
    elapsed = time.monotonic() - start

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["error_type"] == "WorkerProcessCrashed"
    assert result["crashed"] is True
    assert result["worker_exiting"] is True
    assert worker.is_alive_flag is False
    # Must fail fast instead of waiting out the 30s deadline.
    assert elapsed < 5.0


def test_alive_but_unresponsive_worker_still_times_out() -> None:
    worker = _make_worker(_ScriptedProc(alive_seq=[True]), _ScriptedQueue([]))
    start = time.monotonic()
    raised = None
    try:
        worker.execute_task({"task_id": "t2"}, timeout=0.6, poll_interval=0.05)
    except BaseException as exc:  # noqa: BLE001
        raised = exc
    elapsed = time.monotonic() - start

    # No false crash: a live-but-silent worker is still a genuine timeout.
    assert isinstance(raised, TimeoutError)
    assert elapsed >= 0.5


def test_final_result_enqueued_just_before_death_is_drained() -> None:
    real_result = {"success": True, "result": {"ok": 1}}
    worker = _make_worker(
        _ScriptedProc(alive_seq=[True, False]),
        _ScriptedQueue([queue.Empty, real_result]),
    )
    result = worker.execute_task({"task_id": "t3"}, timeout=30, poll_interval=0.01)
    # The result the worker published right before exiting wins over a crash dict.
    assert result["success"] is True
    assert result["result"] == {"ok": 1}
