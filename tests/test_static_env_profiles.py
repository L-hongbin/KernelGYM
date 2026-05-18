from pathlib import Path

from kernelgym import deployment_profiles as profiles


ROOT = Path(__file__).resolve().parents[1]


def test_python_reward_profiles_inherit_base_env_and_match_runtime_constants() -> None:
    expected = {
        "reward-39": "192.168.16.39",
        "reward-40": "192.168.16.40",
    }

    for profile_name, host in expected.items():
        profile = profiles.get_profile(profile_name)
        values = profile.env()
        assert values["API_HOST"] == host
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
        assert values["LOG_DIR"] == f"logs/{values['NODE_ID']}"
        assert values["PY_LOG_DIR"] == f"py_logs/{values['NODE_ID']}"
        assert "GPU_MEMORY_LIMIT" not in values
        assert "CUDA_HOME" not in values
        assert "KERNELGYM_CUDA_AGENT_NVCC_THREADS" not in values


def test_reward_host_profiles_only_define_non_derivable_fields() -> None:
    for profile in (profiles.Reward39Profile, profiles.Reward40Profile):
        assert set(profile.__dict__) & {"name", "node_id", "deployment_profile"} == set()
        assert profile.env()["NODE_ID"] == profile.profile_id()
        assert profile.env()["KERNELGYM_DEPLOYMENT_PROFILE"] == profile.profile_id()
