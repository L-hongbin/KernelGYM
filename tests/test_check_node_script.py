import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECK_NODE = ROOT / "scripts" / "check_node.py"


def load_check_node():
    spec = importlib.util.spec_from_file_location("check_node_script", CHECK_NODE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_summary_fails_stale_gpu_heartbeat(capsys) -> None:
    check_node = load_check_node()
    health = {
        "status": "healthy",
        "timestamp": "2026-05-27T12:00:00Z",
        "gpu_status": {"cuda:0": {"available": True}},
        "queue_status": {},
    }
    workers = {
        "worker_gpu_0": {
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:55:00",
        }
    }

    exit_code = check_node.render_summary("http://node:20111", health, workers, max_heartbeat_age_s=180)

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "status:" in out
    assert "WARN" in out
    assert "gpu_workers_fresh" in out
    assert "0/1" in out


def test_render_summary_passes_fresh_gpu_heartbeat(capsys) -> None:
    check_node = load_check_node()
    health = {
        "status": "healthy",
        "timestamp": "2026-05-27T12:00:00Z",
        "gpu_status": {"cuda:0": {"available": True}},
        "queue_status": {},
    }
    workers = {
        "worker_gpu_0": {
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
        }
    }

    exit_code = check_node.render_summary("http://node:20111", health, workers, max_heartbeat_age_s=180)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "UP" in out
    assert "gpu_workers_fresh" in out
    assert "1/1" in out


def test_render_summary_reports_queue_and_busy_counts(capsys) -> None:
    check_node = load_check_node()
    health = {
        "status": "healthy",
        "timestamp": "2026-05-27T12:00:00Z",
        "gpu_status": {"cuda:0": {"available": True}},
        "queue_status": {"pending": 99},
    }
    workers = {
        "worker_gpu_0": {
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
            "current_task": "gpu-task",
        },
        "worker_cpu_0": {
            "device": "cpu",
            "status": "processing",
            "last_heartbeat": "2026-05-27T11:59:30",
        },
        "worker_cpu_1": {
            "device": "cpu",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
        },
    }

    exit_code = check_node.render_summary(
        "http://node:20111",
        health,
        workers,
        max_heartbeat_age_s=180,
        queue_status={"pending": 3, "worker_queues": {"worker_gpu_0": 2}},
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "queue_count" in out
    assert "5" in out
    assert "gpu_busy" in out
    assert "cpu_busy" in out
    assert out.count("1") >= 2


def test_render_summary_marks_busy_unknown_without_busy_signals(capsys) -> None:
    check_node = load_check_node()
    health = {
        "status": "healthy",
        "timestamp": "2026-05-27T12:00:00Z",
        "gpu_status": {"cuda:0": {"available": True}},
        "queue_status": {"pending": 3},
    }
    workers = {
        "worker_gpu_0": {
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
        }
    }

    exit_code = check_node.render_summary("http://node:20111", health, workers, max_heartbeat_age_s=180)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "gpu_busy" in out
    assert "unknown" in out


def test_merge_workers_uses_redis_current_task(capsys) -> None:
    check_node = load_check_node()
    health = {
        "status": "healthy",
        "timestamp": "2026-05-27T12:00:00Z",
        "gpu_status": {"cuda:0": {"available": True}},
    }
    api_workers = {
        "worker_gpu_0": {
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
        }
    }
    redis_workers = {
        "worker_gpu_0": {
            "current_task": "task-from-redis",
            "tasks_processed": "12",
        }
    }

    workers = check_node._merge_workers(api_workers, redis_workers)
    exit_code = check_node.render_summary("http://node:20111", health, workers, max_heartbeat_age_s=180)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "gpu_busy" in out
    assert "1" in out


def test_queue_count_prefers_explicit_redis_count() -> None:
    check_node = load_check_node()

    count = check_node._queue_count(
        {
            "queue_count": 9,
            "pending": 1,
            "worker_queues": {"worker_gpu_0": 100},
            "resource_queues": {"gpu": 7, "cpu": 2},
        }
    )

    assert count == 9


def test_render_summary_uses_processing_fallback_for_unknown_busy(capsys) -> None:
    check_node = load_check_node()
    health = {
        "status": "healthy",
        "timestamp": "2026-05-27T12:00:00Z",
        "gpu_status": {"cuda:0": {"available": True}},
    }
    workers = {
        "worker_gpu_0": {
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
        },
        "worker_cpu_0": {
            "device": "cpu",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
        },
    }

    exit_code = check_node.render_summary(
        "http://node:20111",
        health,
        workers,
        max_heartbeat_age_s=180,
        queue_status={"processing_by_resource": {"gpu": 1, "cpu": 0}},
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "gpu_busy" in out
    assert "cpu_busy" in out
    assert "unknown" not in out
