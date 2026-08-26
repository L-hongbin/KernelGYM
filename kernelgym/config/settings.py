"""KernelGym configuration settings."""

import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from pydantic import Field, validator
from pydantic_settings import BaseSettings

from kernelgym.deployment_profiles import (
    API_PORT,
    API_RELOAD,
    API_WORKERS,
    METRICS_PORT,
    REDIS_DB,
    REDIS_KEY_PREFIX,
    REDIS_KEY_PREFIX_LEGACY,
    REDIS_PASSWORD,
    REDIS_PORT,
)

PROJECT_ROOT = Path(__file__).parent.parent
KERNELBENCH_ROOT = PROJECT_ROOT.parent


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: ClassVar[int] = API_PORT
    api_workers: ClassVar[int] = API_WORKERS
    api_reload: ClassVar[bool] = API_RELOAD
    api_worker_healthcheck_timeout_sec: int = Field(
        default=60,
        env="API_WORKER_HEALTHCHECK_TIMEOUT_SEC",
        description=(
            "uvicorn kills a child that misses a keep-alive ping for this long. The 5s default "
            "murders children mid torch-import (GIL held during NFS dlopen), causing endless recycling."
        ),
    )

    gpu_devices: List[int] = Field(default_factory=lambda: list(range(8)), env="GPU_DEVICES")
    node_id: str = Field(default="", env="NODE_ID")
    worker_name_prefix: str = Field(default="", env="WORKER_NAME_PREFIX")
    worker_only_mode: bool = Field(default=False, env="WORKER_ONLY_MODE")

    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: ClassVar[int] = REDIS_PORT
    redis_db: ClassVar[int] = REDIS_DB
    redis_password: ClassVar[str] = REDIS_PASSWORD
    redis_key_prefix: ClassVar[str] = REDIS_KEY_PREFIX
    redis_key_prefix_legacy: ClassVar[str] = REDIS_KEY_PREFIX_LEGACY

    celery_task_serializer: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
    celery_accept_content: List[str] = Field(default_factory=lambda: ["json"], env="CELERY_ACCEPT_CONTENT")
    celery_timezone: str = Field(default="UTC", env="CELERY_TIMEZONE")

    default_num_trials: int = Field(default=100, env="DEFAULT_NUM_TRIALS")
    default_timeout: int = Field(default=180, env="DEFAULT_TIMEOUT")
    default_backend: str = Field(default="auto", env="DEFAULT_BACKEND")
    default_toolkit: str = Field(default="kernelbench", env="DEFAULT_TOOLKIT")
    default_backend_adapter: str = Field(default="kernelbench", env="DEFAULT_BACKEND_ADAPTER")
    max_concurrent_tasks: int = Field(default=4, env="MAX_CONCURRENT_TASKS")

    verbose_error_traceback: bool = Field(
        default=True,
        env="VERBOSE_ERROR_TRACEBACK",
        description="Return full error traceback in metadata. Set to False for production to reduce response size",
    )

    enable_profiling: bool = Field(
        default=True,
        env="ENABLE_PROFILING",
        description="Enable torch.profiler for performance diagnostics. Default False to minimize overhead.",
    )
    profiling_activities: List[str] = Field(
        default_factory=lambda: ["cuda"],
        env="PROFILING_ACTIVITIES",
        description="Profiling activities. Defaults to CUDA/GPU only; use ['cpu', 'cuda'] for full profiling.",
    )
    profiling_record_shapes: bool = Field(
        default=True,
        env="PROFILING_RECORD_SHAPES",
        description="Record tensor shapes in profiler. Useful for debugging shape mismatches.",
    )
    profiling_profile_memory: bool = Field(
        default=True,
        env="PROFILING_PROFILE_MEMORY",
        description="Profile memory allocations. Adds ~5% overhead but provides memory insights.",
    )
    profiling_with_stack: bool = Field(
        default=False,
        env="PROFILING_WITH_STACK",
        description="Record stack traces in profiler. Adds significant overhead (~15-20%), use for deep debugging only.",
    )
    profiling_retry_count: int = Field(
        default=1,
        env="PROFILING_RETRY_COUNT",
        description="Retry count when profiler returns empty results (0 to disable).",
    )
    num_profiling_trials: int = Field(
        default=-1,
        env="NUM_PROFILING_TRIALS",
        description=(
            "Extra candidate forwards per profiler context. Values < 1 mean auto: 1 forward when the "
            "CUPTI TSC timestamp bug is absent (CUDA >= 13.1 or KINETO_TSC_FIXED=true), otherwise the "
            "legacy min(10, num_trials) workaround that masks empty captures on CUDA 12.6u2-13.0."
        ),
    )
    kineto_tsc_fixed: bool = Field(
        default=False,
        env="KINETO_TSC_FIXED",
        description=(
            "Declare that the deployed Kineto build already version-gates the CUPTI TSC timestamp "
            "callback, so auto profiling-trial resolution may drop to 1 forward on affected CUPTI versions."
        ),
    )
    enable_ncu: bool = Field(
        default=True,
        env="ENABLE_NCU",
        description="Collect a compact Nsight Compute metric set for correct kernels.",
    )
    ncu_path: str = Field(default="/usr/local/cuda-12.9/bin/ncu", env="NCU_PATH")
    ncu_timeout_s: int = Field(default=90, env="NCU_TIMEOUT_S")
    ncu_max_kernels: int = Field(default=8, env="NCU_MAX_KERNELS")
    ncu_warmup: int = Field(default=2, env="NCU_WARMUP")
    ncu_profile_version: str = Field(default="v1", env="NCU_PROFILE_VERSION")
    ncu_metrics: List[str] = Field(
        default_factory=lambda: [
            "gpu__time_duration.sum",
            "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__throughput.avg.pct_of_peak_sustained_active",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "launch__occupancy_per_block_size",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block",
        ],
        env="NCU_METRICS",
        description="Compact NCU metric set returned in evaluation metadata.",
    )
    enable_compute_sanitizer: bool = Field(
        default=True,
        env="ENABLE_COMPUTE_SANITIZER",
        description="Run isolated Compute Sanitizer trials after a correctness runtime failure.",
    )
    compute_sanitizer_path: str = Field(
        default="/usr/local/cuda-12.9/bin/compute-sanitizer",
        env="COMPUTE_SANITIZER_PATH",
    )
    compute_sanitizer_timeout_s: int = Field(default=60, env="COMPUTE_SANITIZER_TIMEOUT_S")
    compute_sanitizer_max_kernels: int = Field(default=16, env="COMPUTE_SANITIZER_MAX_KERNELS")
    compute_sanitizer_max_issues: int = Field(default=4, env="COMPUTE_SANITIZER_MAX_ISSUES")
    compute_sanitizer_profile_version: str = Field(default="v1", env="COMPUTE_SANITIZER_PROFILE_VERSION")

    adaptive_perf_trials: bool = Field(
        default=True,
        env="ADAPTIVE_PERF_TRIALS",
        description="Adaptively size the kernel perf trial count: run at least perf_min_trials, "
        "then continue only while timing CV exceeds perf_cv_threshold, up to num_perf_trials.",
    )
    perf_min_trials: int = Field(
        default=20,
        env="PERF_MIN_TRIALS",
        description="Minimum kernel perf trials before adaptive CV-based early stop is considered.",
    )
    perf_cv_threshold: float = Field(
        default=0.05,
        env="PERF_CV_THRESHOLD",
        description="Coefficient-of-variation (std/mean) below which kernel timing is deemed stable "
        "and adaptive measurement stops early.",
    )
    correctness_timeout_enabled: bool = Field(
        default=True,
        env="CORRECTNESS_TIMEOUT_ENABLED",
        description="Enforce a shorter wall-clock timeout on the correctness stage (where hung / "
        "pathologically-slow kernels stall) so they are killed fast; the performance loop keeps the "
        "full per-task timeout.",
    )
    correctness_timeout_floor_s: float = Field(
        default=150.0,
        env="CORRECTNESS_TIMEOUT_FLOOR_S",
        description="Minimum correctness-stage timeout in seconds.",
    )
    correctness_timeout_ref_multiplier: float = Field(
        default=50.0,
        env="CORRECTNESS_TIMEOUT_REF_MULTIPLIER",
        description="Correctness-stage timeout scales as multiplier * reference_runtime_seconds when the "
        "reference runtime is known in the task payload, bounded to [floor, per-task timeout].",
    )

    reference_cache_dataset_path: str = Field(default="", env="REFERENCE_CACHE_DATASET_PATH")
    val_data_cache_dataset_path: str = Field(default="", env="VAL_DATA_CACHE_DATASET_PATH")
    enable_reference_cache: bool = Field(default=False, env="ENABLE_REFERENCE_CACHE")

    enable_sandbox: bool = Field(default=True, env="ENABLE_SANDBOX")
    docker_image: str = Field(default="kernelserver:latest", env="DOCKER_IMAGE")
    max_memory_per_task: str = Field(default="4GB", env="MAX_MEMORY_PER_TASK")
    max_gpu_time_per_task: int = Field(default=600, env="MAX_GPU_TIME_PER_TASK")

    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    health_stats_refresh_interval: float = Field(
        default=5.0,
        env="HEALTH_STATS_REFRESH_INTERVAL",
        description="Seconds between background refreshes of the cached GPU/system stats served by /health and /metrics.",
    )

    save_eval_results: bool = Field(
        default=False,
        env="SAVE_EVAL_RESULTS",
        description="Persist evaluation results to local JSONL file.",
    )
    eval_results_path: str = Field(
        default="logs/eval_results.jsonl",
        env="EVAL_RESULTS_PATH",
        description="JSONL file path for persisted evaluation results.",
    )
    metrics_port: ClassVar[int] = METRICS_PORT
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    worker_monitor_interval: int = Field(default=30, env="WORKER_MONITOR_INTERVAL")
    worker_monitor_heartbeat_timeout: int = Field(default=120, env="WORKER_MONITOR_HEARTBEAT_TIMEOUT")
    worker_monitor_restart_cooldown: int = Field(default=60, env="WORKER_MONITOR_RESTART_COOLDOWN")
    worker_queue_wait_timeout_sec: int = Field(default=180, env="WORKER_QUEUE_WAIT_TIMEOUT_SEC")
    worker_queue_wait_monitor_interval: int = Field(default=20, env="WORKER_QUEUE_WAIT_MONITOR_INTERVAL")
    worker_queue_wait_scan_limit: int = Field(default=200, env="WORKER_QUEUE_WAIT_SCAN_LIMIT")
    # Dead-worker reaper: recover tasks stranded in the queues of dead/unregistered
    # workers, which the queue-wait monitor (live workers only) cannot see.
    dead_worker_reaper_enabled: bool = Field(default=True, env="DEAD_WORKER_REAPER_ENABLED")
    dead_worker_timeout_s: int = Field(
        default=90,
        env="DEAD_WORKER_TIMEOUT_S",
        description="Heartbeat age (s) beyond which a worker is treated as dead by the reaper. "
        "Keep well above the worker heartbeat interval so briefly-laggy live workers are not reaped.",
    )
    dead_worker_reaper_interval_s: int = Field(default=20, env="DEAD_WORKER_REAPER_INTERVAL_S")
    max_requeue_attempts: int = Field(
        default=3,
        env="MAX_REQUEUE_ATTEMPTS",
        description="Max times a stuck task may be requeued before it is failed instead of looping (0 disables the cap).",
    )
    worker_execution_timeout_grace_sec: int = Field(default=60, env="WORKER_EXECUTION_TIMEOUT_GRACE_SEC")
    worker_shutdown_drain_sec: int = Field(
        default=120,
        env="KERNELGYM_WORKER_SHUTDOWN_DRAIN_SEC",
        description=(
            "On SIGTERM, wait up to this many seconds for the in-flight task to finish "
            "before failing it with 'Worker shutdown'. 0 disables draining."
        ),
    )
    worker_execution_timeout_monitor_interval: int = Field(default=30, env="WORKER_EXECUTION_TIMEOUT_MONITOR_INTERVAL")
    worker_pool_size: int = Field(
        default=2,
        env="WORKER_POOL_SIZE",
        description="Number of persistent subprocess workers per GPU. 2 = 1 active + 1 warm spare for zero-downtime recycling.",
    )
    max_tasks_per_worker: int = Field(
        default=1,
        env="MAX_TASKS_PER_WORKER",
        description="Max tasks a subprocess worker handles before restart. Set to 1 for per-task isolation.",
    )
    cpu_compile_workers: int = Field(
        default=2,
        env="CPU_COMPILE_WORKERS",
        description="Number of CPU workers that consume compile-stage tasks.",
    )
    split_compile_and_execute: bool = Field(
        default=False,
        env="SPLIT_COMPILE_AND_EXECUTE",
        description="Force KernelBench requests through CPU compile + GPU execute split mode.",
    )

    log_dir: str = Field(default="logs", env="LOG_DIR")
    log_to_file: bool = Field(default=True, env="LOG_TO_FILE")
    log_max_size: str = Field(default="100MB", env="LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")

    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    enable_result_cache: bool = Field(default=True, env="ENABLE_RESULT_CACHE")
    terminal_task_ttl_sec: int = Field(
        default=12 * 3600,
        env="TERMINAL_TASK_TTL_SEC",
        description="TTL for completed/failed task status hashes. Set <=0 to keep terminal task records indefinitely.",
    )
    terminal_result_ttl_sec: int = Field(
        default=12 * 3600,
        env="TERMINAL_RESULT_TTL_SEC",
        description="TTL for completed/failed result cache hashes. Set <=0 to keep terminal result records indefinitely.",
    )
    core_dump_dir: str = Field(
        default="logs/core_dumps",
        env="KERNELGYM_CORE_DUMP_DIR",
        description="Directory where GPU subprocesses chdir so Linux core dumps land outside the repo root.",
    )
    core_dump_keep: int = Field(
        default=10,
        env="KERNELGYM_CORE_DUMP_KEEP",
        description="Maximum core dump files to retain per core dump directory.",
    )

    kernelbench_path: str = Field(default=str(KERNELBENCH_ROOT), env="KERNELBENCH_PATH")
    gpu_arch: List[str] = Field(default_factory=lambda: ["Hopper"], env="GPU_ARCH")

    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")

    @validator("gpu_devices", pre=True)
    def validate_gpu_devices(cls, v):
        if isinstance(v, str):
            try:
                import json

                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [int(x) for x in parsed]
                return [int(parsed)]
            except Exception:
                try:
                    return [int(x.strip()) for x in v.split(",")]
                except Exception:
                    return list(range(8))
        if isinstance(v, list):
            return [int(x) for x in v]
        return list(range(8))

    @validator("gpu_arch", pre=True)
    def validate_gpu_arch(cls, v):
        if isinstance(v, str):
            try:
                import json

                return json.loads(v)
            except Exception:
                return [v]
        if isinstance(v, list):
            return v
        return ["Hopper"]

    def setup_log_directory(self) -> None:
        log_path = Path(self.log_dir)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / self.log_dir
        os.makedirs(log_path, exist_ok=True)

    def get_redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def redis_url(self) -> str:
        return self.get_redis_url()

    @property
    def celery_broker_url(self) -> str:
        return self.get_redis_url()

    @property
    def celery_result_backend(self) -> str:
        return self.get_redis_url()

    def get_celery_config(self) -> Dict[str, Any]:
        return {
            "broker_url": self.celery_broker_url,
            "result_backend": self.celery_result_backend,
            "task_serializer": self.celery_task_serializer,
            "accept_content": self.celery_accept_content,
            "timezone": self.celery_timezone,
            "task_routes": {
                "worker.tasks.evaluate_kernel": {"queue": "gpu_evaluation"},
                "worker.tasks.compile_kernel": {"queue": "compilation"},
            },
            "task_annotations": {"worker.tasks.evaluate_kernel": {"rate_limit": f"{self.max_concurrent_tasks}/h"}},
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

        @classmethod
        def prepare_field_value(cls, field_name: str, field, field_value, value_is_complex: bool):
            if field_name == "gpu_devices" and isinstance(field_value, str):
                try:
                    import json

                    parsed = json.loads(field_value)
                    if isinstance(parsed, list):
                        return [int(x) for x in parsed]
                    return [int(parsed)]
                except Exception:
                    try:
                        return [int(x.strip()) for x in field_value.split(",")]
                    except Exception:
                        return list(range(8))
            if field_name == "gpu_arch" and isinstance(field_value, str):
                try:
                    import json

                    return json.loads(field_value)
                except Exception:
                    return [field_value]
            return field_value


settings = Settings()

GPU_DEVICE_MAP = {
    f"cuda:{i}": {
        "device_id": i,
        "worker_queue": f"gpu_{i}",
    }
    for i in settings.gpu_devices
}

TASK_CONFIGS = {
    "quick": {
        "num_correct_trials": 3,
        "num_perf_trials": 10,
        "timeout": 60,
        "priority": "high",
    },
    "standard": {
        "num_correct_trials": 5,
        "num_perf_trials": 100,
        "timeout": 300,
        "priority": "normal",
    },
    "thorough": {
        "num_correct_trials": 10,
        "num_perf_trials": 1000,
        "timeout": 600,
        "priority": "low",
    },
}


def get_logging_config() -> Dict[str, Any]:
    settings.setup_log_directory()

    log_path = Path(settings.log_dir)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / settings.log_dir

    handlers = {
        "console": {
            "level": settings.log_level,
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    }

    if settings.log_to_file:
        handlers.update(
            {
                "file_server": {
                    "level": settings.log_level,
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_path / "kernelgym.log"),
                    "maxBytes": 104857600,
                    "backupCount": settings.log_backup_count,
                    "encoding": "utf8",
                },
                "file_worker": {
                    "level": settings.log_level,
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_path / "workers.log"),
                    "maxBytes": 104857600,
                    "backupCount": settings.log_backup_count,
                    "encoding": "utf8",
                },
                "file_api": {
                    "level": settings.log_level,
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_path / "api.log"),
                    "maxBytes": 104857600,
                    "backupCount": settings.log_backup_count,
                    "encoding": "utf8",
                },
            }
        )

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
            "detailed": {"format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d] - %(message)s"},
        },
        "handlers": handlers,
        "loggers": {
            "": {
                "handlers": ["console"] + (["file_server"] if settings.log_to_file else []),
                "level": settings.log_level,
                "propagate": False,
            },
            "kernelgym.api": {
                "handlers": ["console"] + (["file_api"] if settings.log_to_file else []),
                "level": settings.log_level,
                "propagate": False,
            },
            "kernelgym.worker": {
                "handlers": ["console"] + (["file_worker"] if settings.log_to_file else []),
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console"] + (["file_server"] if settings.log_to_file else []),
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console"] + (["file_api"] if settings.log_to_file else []),
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    return config


_QUEUE_LISTENERS: List[Any] = []
_LOGGING_CONFIGURED_PID: int | None = None


def _install_queue_handlers(logger_names: List[str]) -> None:
    """Re-home each configured logger's handlers behind a QueueHandler.

    Emission then only enqueues in-memory; the actual console/file writes
    happen on a QueueListener thread, so slow sinks (NFS-backed log files)
    can never stall the caller — in the API server that caller is the
    asyncio event loop serving every request and worker heartbeat.
    """
    import atexit
    import logging
    import logging.handlers
    import queue

    for name in logger_names:
        target = logging.getLogger(name)
        sinks = list(target.handlers)
        if not sinks or any(isinstance(h, logging.handlers.QueueHandler) for h in sinks):
            continue
        q: queue.Queue = queue.Queue()
        listener = logging.handlers.QueueListener(q, *sinks, respect_handler_level=True)
        listener.start()
        _QUEUE_LISTENERS.append(listener)
        atexit.register(listener.stop)
        target.handlers = [logging.handlers.QueueHandler(q)]


def setup_logging(component_name: str = "server"):
    import logging.config

    global _LOGGING_CONFIGURED_PID

    if component_name == "api":
        logger_name = "kernelgym.api"
    elif component_name == "worker":
        logger_name = "kernelgym.worker"
    else:
        logger_name = ""

    # Configure at most once per process: a second dictConfig would close the
    # handlers the queue listeners are still writing to, which is how the
    # "I/O operation on closed file" logging-error floods started.
    if _LOGGING_CONFIGURED_PID == os.getpid():
        return logging.getLogger(logger_name)

    config = get_logging_config()
    logging.config.dictConfig(config)
    _install_queue_handlers(list(config["loggers"].keys()))
    # A broken sink must never inject "--- Logging error ---" tracebacks into
    # stderr (they land in the service's redirected stdout log and flood it).
    logging.raiseExceptions = False
    _LOGGING_CONFIGURED_PID = os.getpid()

    logger = logging.getLogger(logger_name)
    logger.info(f"Logging configured for {component_name} - File logging: {settings.log_to_file}")

    if settings.log_to_file:
        log_path = Path(settings.log_dir)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / settings.log_dir
        logger.info(f"Log files will be written to: {log_path}")

    return logger
