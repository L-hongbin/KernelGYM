import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
    assert "queue_count" not in out
    assert "queue_pending" not in out
    assert "active_tasks" not in out
    assert "uptime_s" not in out
    assert "gpu_busy" in out
    assert "cpu_busy" not in out
    assert "cpu_workers_busy" in out
    assert "1/2" in out
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
    assert "cpu_busy" not in out
    assert "cpu_workers_busy" in out
    assert "unknown" not in out


def test_nvidia_smi_gpu_snapshot_reports_whole_device_memory(monkeypatch) -> None:
    check_node = load_check_node()
    monkeypatch.setattr(
        check_node.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )

    class Completed:
        returncode = 0
        stdout = "2, NVIDIA GeForce RTX 4090, 19343, 24564, 99\n"

    monkeypatch.setattr(check_node.subprocess, "run", lambda *_args, **_kwargs: Completed())

    snapshot = check_node._nvidia_smi_gpu_snapshot()

    assert snapshot == {
        "cuda:2": {
            "name": "NVIDIA GeForce RTX 4090",
            "memory_total": "24.0GB",
            "memory_used": "18.9GB",
            "memory_used_percent": "78.7%",
            "utilization_gpu_percent": "99.0%",
            "available": True,
            "source": "nvidia-smi",
        }
    }


def test_short_gpu_name_removes_vendor_prefix() -> None:
    check_node = load_check_node()

    assert check_node._short_gpu_name("NVIDIA GeForce RTX 4090") == "RTX 4090"
    assert check_node._short_gpu_name("NVIDIA H100 80GB HBM3") == "H100 80GB HBM3"


def test_short_hostname_removes_ai_prefix() -> None:
    check_node = load_check_node()

    assert check_node._short_hostname("ai-16-39") == "16-39"
    assert check_node._short_hostname("worker-1") == "worker-1"


def test_short_task_id_removes_parallel_task_prefix() -> None:
    check_node = load_check_node()

    assert check_node._short_task_id("parallel_task_001434_2700a1c8_kernel") == "001434_2700a1c8_kernel"
    assert check_node._short_task_id("manual_task") == "manual_task"


def test_render_verbose_omits_fresh_column(capsys) -> None:
    check_node = load_check_node()
    health = {
        "timestamp": "2026-05-27T12:00:00Z",
        "gpu_status": {
            "cuda:0": {
                "name": "NVIDIA GeForce RTX 4090",
                "memory_used": "0.4GB",
                "memory_used_percent": "1.9%",
                "available": True,
            }
        },
    }
    workers = {
        "worker_gpu_0": {
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
            "current_task": "parallel_task_001434_2700a1c8_kernel",
            "node_id": "v1",
            "hostname": "ai-16-39",
        },
        "worker_cpu_0": {
            "device": "cpu",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
        },
        "worker_cpu_1": {
            "device": "cpu",
            "status": "processing",
            "last_heartbeat": "2026-05-27T11:59:30",
            "current_task": "cpu-task",
        },
    }

    check_node.render_verbose(health, workers, max_heartbeat_age_s=180)

    out = capsys.readouterr().out
    assert "fresh" not in out
    assert "available" not in out
    assert "age_s" in out
    assert "RTX 4090" in out
    assert "0.4GB" in out
    assert "1.9%" not in out
    assert "16-39" in out
    assert "ai-16-39" not in out
    assert "001434_2700a1c8_kernel" in out
    assert "parallel_task_001434_2700a1c8_kernel" not in out
    assert "/workers/status" not in out
    assert "cpu_workers_busy:" not in out
    assert "worker_id" not in out


def test_render_verbose_keeps_multi_node_workers_with_same_cuda_device(capsys) -> None:
    check_node = load_check_node()
    health = {
        "timestamp": "2026-05-27T12:00:00Z",
        "gpu_status": {
            "cuda:0": {
                "name": "NVIDIA GeForce RTX 4090",
                "memory_used": "0.4GB",
                "memory_used_percent": "1.9%",
                "available": True,
            }
        },
    }
    workers = {
        "worker_gpu_0": {
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
        },
        "v1-worker-1_gpu_0": {
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": "2026-05-27T11:59:30",
            "node_id": "v1-worker-1",
            "hostname": "ai-16-39",
        },
    }

    check_node.render_verbose(health, workers, max_heartbeat_age_s=180)

    out = capsys.readouterr().out
    assert "worker_gpu_0" not in out
    assert "v1-worker-1_gpu_0" not in out
    assert out.count("cuda:0") == 2
    assert "16-39" in out
    assert "ai-16-39" not in out
    assert "/workers/status" not in out


def test_merge_gpu_status_prefers_local_nvidia_smi_values() -> None:
    check_node = load_check_node()
    health = {
        "gpu_status": {
            "cuda:2": {
                "name": "NVIDIA GeForce RTX 4090",
                "memory_total": "23.5GB",
                "memory_used_percent": "0.0%",
                "available": True,
            }
        }
    }
    local_gpus = {
        "cuda:2": {
            "memory_total": "24.0GB",
            "memory_used": "18.9GB",
            "memory_used_percent": "78.7%",
            "source": "nvidia-smi",
        }
    }

    merged = check_node._merge_gpu_status(health, local_gpus)

    assert merged["gpu_status"]["cuda:2"]["memory_total"] == "24.0GB"
    assert merged["gpu_status"]["cuda:2"]["memory_used"] == "18.9GB"
    assert merged["gpu_status"]["cuda:2"]["memory_used_percent"] == "78.7%"
    assert merged["gpu_status"]["cuda:2"]["source"] == "nvidia-smi"
