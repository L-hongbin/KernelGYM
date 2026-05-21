from pathlib import Path

from kernelgym import deployment_profiles as profiles


ROOT = Path(__file__).resolve().parents[1]


def test_functional_reward_profile_matches_runtime_constants() -> None:
    profile = profiles.get_profile("v1")
    values = profile.env()

    assert values["KERNELGYM_DEPLOYMENT_PROFILE"] == "v1"
    assert values["NODE_ID"] == "v1"
    assert values["API_HOST"] == "0.0.0.0"
    assert values["API_PORT"] == str(profiles.API_PORT)
    assert values["API_WORKERS"] == str(profiles.API_WORKERS)
    assert values["API_RELOAD"] == profiles.bool_env(profiles.API_RELOAD)
    assert values["REDIS_HOST"] == "localhost"
    assert values["REDIS_PORT"] == str(profiles.REDIS_PORT)
    assert values["REDIS_DB"] == str(profiles.REDIS_DB)
    assert values["REDIS_PASSWORD"] == profiles.REDIS_PASSWORD
    assert values["REDIS_KEY_PREFIX"] == profiles.REDIS_KEY_PREFIX
    assert values["METRICS_PORT"] == str(profiles.METRICS_PORT)
    assert values["GPU_DEVICES"] == "[0,1,2,3,4,5,6,7]"
    assert values["DEFAULT_BACKEND"] == "auto"
    assert values["KERNELGYM_CORRECTNESS_GPU_INPUTS"] == "true"
    assert values["LOG_DIR"] == "logs/v1"
    assert values["PY_LOG_DIR"] == "py_logs/v1"
    assert "GPU_MEMORY_LIMIT" not in values
    assert "CUDA_HOME" not in values
    assert "KERNELGYM_CUDA_AGENT_NVCC_THREADS" not in values
    # The 2-trial / 20s wall-clock budget pass-on-success mechanism is
    # intentionally disabled in the v1 profile: stop_on_first_failure remains
    # the only correctness short-circuit; all configured trials run otherwise.
    assert "KERNELGYM_CORRECTNESS_MAX_WALL_S" not in values
    assert "KERNELGYM_CORRECTNESS_PASS_ON_BUDGET" not in values
    assert "KERNELGYM_CORRECTNESS_BUDGET_MIN_PASS_TRIALS" not in values


def test_auto_is_alias_for_default_functional_profile() -> None:
    assert profiles.get_profile("auto").env() == profiles.get_profile("v1").env()


def test_profiles_do_not_accept_host_based_names() -> None:
    for invalid in ("192.168.16.40", "reward-40"):
        try:
            profiles.get_profile(invalid)
        except ValueError as exc:
            assert "v1" in str(exc)
        else:
            raise AssertionError(f"{invalid} should not be a valid functional profile")
