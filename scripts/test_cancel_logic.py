"""Offline deterministic test of TaskManager cancellation/queue logic.

Stubs the torch-heavy backend/toolkit imports and drives the *real* TaskManager
against an in-memory async-redis fake. Validates Problem #1 (pending cancel must
leave the queue + never be dispatched) and the cancellation-marker plumbing used
by the in-flight interrupt (Problem #2).
"""

import asyncio
import sys
import types

# --- stub the only torch-pulling imports task_manager needs at import time ---
for name, attrs in {
    "kernelgym.backend": {"list_backends": lambda: ["kernelbench"], "get_backend": lambda n: object()},
    "kernelgym.toolkit": {"list_toolkits": lambda: ["kernelbench"], "get_toolkit": lambda n: object()},
}.items():
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


class FakeAsyncRedis:
    """Minimal in-memory async Redis supporting the ops TaskManager uses."""

    def __init__(self):
        self.hashes = {}
        self.lists = {}
        self.kv = {}

    @staticmethod
    def _enc(v):
        if isinstance(v, bytes):
            return v
        if isinstance(v, str):
            return v.encode()
        return str(v).encode()

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping=None, **kw):
        m = mapping or kw
        h = self.hashes.setdefault(key, {})
        for k, v in m.items():
            h[self._enc(k)] = self._enc(v)
        return len(m)

    async def lpush(self, key, *vals):
        lst = self.lists.setdefault(key, [])
        for v in vals:
            lst.insert(0, self._enc(v))
        return len(lst)

    async def rpush(self, key, *vals):
        lst = self.lists.setdefault(key, [])
        for v in vals:
            lst.append(self._enc(v))
        return len(lst)

    async def rpop(self, key):
        lst = self.lists.get(key)
        if not lst:
            return None
        return lst.pop()

    async def lrem(self, key, count, value):
        lst = self.lists.get(key)
        if not lst:
            return 0
        target = self._enc(value)
        before = len(lst)
        self.lists[key] = [x for x in lst if x != target]
        return before - len(self.lists[key])

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def exists(self, *keys):
        n = 0
        for k in keys:
            if self.hashes.get(k) or self.lists.get(k) or k in self.kv:
                n += 1
        return n

    async def set(self, key, value, ex=None):
        self.kv[key] = self._enc(value)
        return True

    async def delete(self, *keys):
        for k in keys:
            self.hashes.pop(k, None)
            self.lists.pop(k, None)
            self.kv.pop(k, None)
        return len(keys)


from kernelgym.server.task_manager import TaskManager  # noqa: E402
from kernelgym.server.scheduler import TaskManagerScheduler  # noqa: E402
from kernelgym.common import TaskStatus  # noqa: E402


def _passed(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    assert cond, msg


async def seed_pending(tm, r, task_id, resource="gpu", assigned_worker=""):
    await r.hset(
        f"{tm.key_prefix}:task:{task_id}",
        mapping={
            "data": '{"task_id": "%s", "toolkit": "kernelbench", "backend_adapter": "kernelbench"}' % task_id,
            "status": TaskStatus.PENDING.value,
            "assigned_worker": assigned_worker,
        },
    )
    if assigned_worker:
        await r.lpush(f"{tm.key_prefix}:queue:worker:{assigned_worker}", task_id)
    else:
        await r.lpush(tm.resource_queues[resource], task_id)


async def main():
    r = FakeAsyncRedis()
    tm = TaskManager(r)
    gpu_q = tm.resource_queues["gpu"]

    # Scenario 1: cancel a pending task -> removed from queue, terminal, marked.
    await seed_pending(tm, r, "T1")
    ok = await tm.cancel_task("T1")
    _passed(ok is True, "cancel_task(pending) returns True")
    _passed(await r.llen(gpu_q) == 0, "S1: cancelled task removed from gpu queue")
    th = await r.hgetall(f"{tm.key_prefix}:task:T1")
    _passed(th.get(b"status") == b"failed", "S1: task status set to failed/cancelled")
    _passed(b"cancelled_at" in th, "S1: cancelled_at recorded on task hash")
    res = await r.hgetall(f"{tm.key_prefix}:result:T1")
    _passed(res.get(b"error") == b"Task cancelled", "S1: result error == 'Task cancelled'")
    _passed(await tm.is_task_cancelled("T1") is True, "S1: is_task_cancelled(T1) True")
    _passed(await tm.is_task_cancelled("nope") is False, "S1: is_task_cancelled(unknown) False")

    # Scenario 2: defense-in-depth — a terminal task still sitting in the queue
    # must be dropped on dequeue, and a following healthy task is returned.
    await seed_pending(tm, r, "T2")
    await r.hset(f"{tm.key_prefix}:task:T2", mapping={"status": TaskStatus.FAILED.value})  # stale enqueued
    await seed_pending(tm, r, "T3")  # healthy, enqueued after T2
    got = await tm.get_next_task("worker_gpu_0", resources=["gpu"])
    _passed(got is not None and got.get("task_id") == "T3", "S2: get_next_task skips terminal T2, returns T3")
    _passed(await r.llen(gpu_q) == 0, "S2: terminal T2 dropped from queue (not re-deferred)")
    t2 = await r.hgetall(f"{tm.key_prefix}:task:T2")
    _passed(t2.get(b"status") == b"failed", "S2: T2 never moved to processing")
    t3 = await r.hgetall(f"{tm.key_prefix}:task:T3")
    _passed(t3.get(b"status") == b"processing", "S2: T3 moved to processing")

    # Scenario 3: cancelling an already-terminal task returns False.
    await r.hset(f"{tm.key_prefix}:task:T4", mapping={"status": TaskStatus.COMPLETED.value, "data": "{}"})
    _passed(await tm.cancel_task("T4") is False, "S3: cancel of completed task returns False")

    # Scenario 4: worker-queue path removal.
    await seed_pending(tm, r, "T5", assigned_worker="worker_gpu_3")
    wq = f"{tm.key_prefix}:queue:worker:worker_gpu_3"
    _passed(await r.llen(wq) == 1, "S4: T5 enqueued on worker queue")
    await tm.cancel_task("T5")
    _passed(await r.llen(wq) == 0, "S4: cancelled T5 removed from worker queue")

    # Scenario 5: unknown task -> False
    _passed(await tm.cancel_task("does_not_exist") is False, "S5: cancel of unknown task returns False")

    # Scenario 6: workflow parent id (no task hash) is cancellable while active.
    _passed(await tm.cancel_task("WF_parent") is False, "S6: cancel of unregistered parent returns False")
    await tm.register_workflow("WF_parent")
    _passed(await tm._is_workflow_active("WF_parent") is True, "S6: workflow registered active")
    ok = await tm.cancel_task("WF_parent")
    _passed(ok is True, "S6: cancel of active workflow parent returns True")
    _passed(
        await tm.is_task_cancelled("WF_parent") is True,
        "S6: base-scope cancel marker set (sub-tasks poll this via base_task_id)",
    )
    await tm.unregister_workflow("WF_parent")
    _passed(await tm._is_workflow_active("WF_parent") is False, "S6: workflow unregistered")

    # Scenario 7: a queued SUB-task whose workflow PARENT was cancelled is
    # dropped on dequeue (parent cancel marker set, sub-task carries base_task_id).
    await r.hset(
        f"{tm.key_prefix}:task:PARENT2_kernel",
        mapping={
            "data": '{"task_id": "PARENT2_kernel", "base_task_id": "PARENT2", "toolkit": "kernelbench", "backend_adapter": "kernelbench"}',
            "status": TaskStatus.PENDING.value,
            "assigned_worker": "",
        },
    )
    await r.lpush(gpu_q, "PARENT2_kernel")
    await tm.register_workflow("PARENT2")
    await tm.cancel_task("PARENT2")  # sets base cancel marker for PARENT2
    await seed_pending(tm, r, "HEALTHY1")  # a healthy task queued behind it
    got = await tm.get_next_task("worker_gpu_9", resources=["gpu"])
    _passed(
        got is not None and got.get("task_id") == "HEALTHY1",
        "S7: queued sub-task of cancelled parent dropped; healthy task returned",
    )
    _passed(await r.llen(gpu_q) == 0, "S7: cancelled sub-task removed from queue (not re-deferred)")

    # Scenario 8: orphan-close — wait_unless_cancelled pulls the specific queued
    # child out of its queue on parent cancel (no reliance on marker TTL).
    sched = TaskManagerScheduler(tm)
    await r.hset(
        f"{tm.key_prefix}:task:PARENT3_kernel",
        mapping={
            "data": '{"task_id": "PARENT3_kernel", "base_task_id": "PARENT3", "toolkit": "kernelbench", "backend_adapter": "kernelbench"}',
            "status": TaskStatus.PENDING.value,
            "assigned_worker": "",
        },
    )
    await r.lpush(gpu_q, "PARENT3_kernel")
    await tm.register_workflow("PARENT3")
    await tm.cancel_task("PARENT3")  # parent cancel -> base marker only
    ret = await sched.wait_unless_cancelled("PARENT3_kernel", "PARENT3")
    _passed(ret is None, "S8: wait_unless_cancelled returns None on parent cancel")
    _passed(await r.llen(gpu_q) == 0, "S8: queued child pulled from queue (orphan-close, no TTL dependency)")
    ch = await r.hgetall(f"{tm.key_prefix}:task:PARENT3_kernel")
    _passed(ch.get(b"status") == b"failed", "S8: queued child marked terminal")

    print("\nALL OFFLINE CANCELLATION-LOGIC CHECKS PASSED")


asyncio.run(main())
