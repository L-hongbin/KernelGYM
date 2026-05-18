"""Python deployment profiles and shared runtime constants."""

from __future__ import annotations

from typing import ClassVar


API_PORT = 20111
API_WORKERS = 4
API_RELOAD = False
REDIS_PORT = 20110
REDIS_DB = 0
REDIS_PASSWORD = ""
REDIS_KEY_PREFIX = "kernelgym"
REDIS_KEY_PREFIX_LEGACY = "kernelserver"
METRICS_PORT = 20112


def bool_env(value: bool) -> str:
    return "true" if value else "false"


class BaseRewardProfile:
    host: ClassVar[str]
    ssh_runtime: ClassVar[str] = "physical_host"
    container_required: ClassVar[bool] = True
    lock_gpu_clocks: ClassVar[bool] = True
    gpu_devices: ClassVar[tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 6, 7)

    @classmethod
    def profile_id(cls) -> str:
        return f"reward-{cls.host.rsplit('.', 1)[-1]}"

    @classmethod
    def env(cls) -> dict[str, str]:
        profile_id = cls.profile_id()
        return {
            "KERNELGYM_DEPLOYMENT_PROFILE": profile_id,
            "KERNELGYM_SSH_RUNTIME": cls.ssh_runtime,
            "KERNELGYM_CONTAINER_REQUIRED": bool_env(cls.container_required),
            "KERNELGYM_LOCK_GPU_CLOCKS": bool_env(cls.lock_gpu_clocks),
            "API_HOST": cls.host,
            "API_PORT": str(API_PORT),
            "API_WORKERS": str(API_WORKERS),
            "API_RELOAD": bool_env(API_RELOAD),
            "GPU_DEVICES": "[" + ",".join(str(device) for device in cls.gpu_devices) + "]",
            "NODE_ID": profile_id,
            "REDIS_HOST": "localhost",
            "REDIS_PORT": str(REDIS_PORT),
            "REDIS_DB": str(REDIS_DB),
            "REDIS_PASSWORD": REDIS_PASSWORD,
            "REDIS_KEY_PREFIX": REDIS_KEY_PREFIX,
            "WORKER_POOL_SIZE": "1",
            "MAX_TASKS_PER_WORKER": "1",
            "DEFAULT_TOOLKIT": "kernelbench",
            "DEFAULT_BACKEND_ADAPTER": "kernelbench",
            "DEFAULT_BACKEND": "triton",
            "LOG_LEVEL": "INFO",
            "LOG_DIR": f"logs/{profile_id}",
            "PY_LOG_DIR": f"py_logs/{profile_id}",
            "ENABLE_METRICS": "true",
            "METRICS_PORT": str(METRICS_PORT),
            "ENABLE_PROFILING": "true",
            "VERBOSE_ERROR_TRACEBACK": "true",
            "SAVE_EVAL_RESULTS": "false",
            "EVAL_RESULTS_PATH": f"logs/{profile_id}/eval_results.jsonl",
            "KERNELGYM_NVCC_THREADS": "4",
        }


class Reward39Profile(BaseRewardProfile):
    host = "192.168.16.39"


class Reward40Profile(BaseRewardProfile):
    host = "192.168.16.40"


PROFILE_CLASSES: dict[str, type[BaseRewardProfile]] = {
    Reward39Profile.profile_id(): Reward39Profile,
    Reward40Profile.profile_id(): Reward40Profile,
}


def profile_names() -> tuple[str, ...]:
    return tuple(PROFILE_CLASSES)


def get_profile(name: str) -> type[BaseRewardProfile]:
    try:
        return PROFILE_CLASSES[name]
    except KeyError as exc:
        choices = ", ".join(profile_names())
        raise ValueError(f"Unknown reward profile: {name}. Choices: {choices}") from exc
