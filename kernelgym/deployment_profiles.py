"""Functional deployment profiles and shared runtime constants."""

from __future__ import annotations

from dataclasses import dataclass


API_PORT = 20111
API_WORKERS = 4
API_RELOAD = False
REDIS_PORT = 20110
REDIS_DB = 0
REDIS_PASSWORD = ""
REDIS_KEY_PREFIX = "kernelgym"
REDIS_KEY_PREFIX_LEGACY = "kernelserver"
METRICS_PORT = 20112
DEFAULT_PROFILE_NAME = "v1"


def bool_env(value: bool) -> str:
    return "true" if value else "false"


@dataclass(frozen=True)
class RewardProfile:
    name: str
    api_host: str = "0.0.0.0"
    redis_host: str = "localhost"
    gpu_devices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

    def env(self) -> dict[str, str]:
        return {
            "KERNELGYM_DEPLOYMENT_PROFILE": self.name,
            "KERNELGYM_CONTAINER_REQUIRED": "true",
            "API_HOST": self.api_host,
            "API_PORT": str(API_PORT),
            "API_WORKERS": str(API_WORKERS),
            "API_RELOAD": bool_env(API_RELOAD),
            "GPU_DEVICES": "[" + ",".join(str(device) for device in self.gpu_devices) + "]",
            "NODE_ID": self.name,
            "REDIS_HOST": self.redis_host,
            "REDIS_PORT": str(REDIS_PORT),
            "REDIS_DB": str(REDIS_DB),
            "REDIS_PASSWORD": REDIS_PASSWORD,
            "REDIS_KEY_PREFIX": REDIS_KEY_PREFIX,
            "WORKER_POOL_SIZE": "2",
            "MAX_TASKS_PER_WORKER": "1",
            "CPU_COMPILE_WORKERS": "2",
            "DEFAULT_TIMEOUT": "180",
            "DEFAULT_TOOLKIT": "kernelbench",
            "DEFAULT_BACKEND_ADAPTER": "kernelbench",
            "DEFAULT_BACKEND": "auto",
            "LOG_LEVEL": "INFO",
            "LOG_DIR": f"logs/{self.name}",
            "PY_LOG_DIR": f"py_logs/{self.name}",
            "ENABLE_METRICS": "true",
            "METRICS_PORT": str(METRICS_PORT),
            "ENABLE_PROFILING": "true",
            "VERBOSE_ERROR_TRACEBACK": "true",
            "SAVE_EVAL_RESULTS": "false",
            "EVAL_RESULTS_PATH": f"logs/{self.name}/eval_results.jsonl",
            "KERNELGYM_NVCC_THREADS": "4",
            "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE": "true",
            "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE_INDEX": "redis",
            "KERNELGYM_COMPILE_ARTIFACT_CACHE": "true",
            "KERNELGYM_CORRECTNESS_GPU_INPUTS": "true",
        }


PROFILE_REGISTRY: dict[str, RewardProfile] = {
    DEFAULT_PROFILE_NAME: RewardProfile(name=DEFAULT_PROFILE_NAME),
}


def profile_names() -> tuple[str, ...]:
    return tuple(PROFILE_REGISTRY)


def get_profile(name: str) -> RewardProfile:
    profile_name = DEFAULT_PROFILE_NAME if name == "auto" else name
    try:
        return PROFILE_REGISTRY[profile_name]
    except KeyError as exc:
        choices = ", ".join(("auto", *profile_names()))
        raise ValueError(f"Unknown reward profile: {name}. Choices: {choices}") from exc
