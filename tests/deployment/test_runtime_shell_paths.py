import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_VENV = "/root/kernelgym-reward-only/.venv"
OFFLINE_WHEELS = "/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only/wheels"
MS_WHEELS = "/ms/FM/lihongbin/code/Code-Agent/KernelENV/env_wheel"


def test_runtime_paths_pin_local_venv_and_absolute_offline_wheelhouse() -> None:
    paths = (ROOT / "scripts" / "runtime_paths.sh").read_text(encoding="utf-8")

    assert LOCAL_VENV in paths
    assert OFFLINE_WHEELS in paths
    assert MS_WHEELS in paths
    assert "export WHELL_PATH" in paths
    assert 'KERNELGYM_OFFLINE_WHEEL_DIR:-${WHELL_PATH}' in paths
    assert 'KERNELGYM_OFFLINE_REDIS_DIR:-${WHELL_PATH}/redis/ubuntu-24.04-amd64' in paths
    assert "must be an absolute path" in paths
    assert "findmnt" in paths
    assert "findmnt is required to prove the venv path is node-local" in paths
    assert "nfs|nfs4|cifs|smb3|lustre|ceph|gpfs|9p|virtiofs|fuse.*" in paths


def test_runtime_paths_honors_explicit_whell_path(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    env = os.environ.copy()
    env["WHELL_PATH"] = str(wheelhouse)
    env.pop("KERNELGYM_OFFLINE_WHEEL_DIR", None)
    env.pop("KERNELGYM_OFFLINE_REDIS_DIR", None)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && printf "%s\n%s\n%s" "${WHELL_PATH}" '
            '"${KERNELGYM_OFFLINE_WHEEL_DIR}" "${KERNELGYM_OFFLINE_REDIS_DIR}"',
            "bash",
            str(ROOT / "scripts/runtime_paths.sh"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        str(wheelhouse),
        str(wheelhouse),
        str(wheelhouse / "redis" / "ubuntu-24.04-amd64"),
    ]


def test_redis_bootstrap_prefers_offline_bundle_and_has_online_fallback() -> None:
    redis = (ROOT / "scripts" / "ensure_redis.sh").read_text(encoding="utf-8")

    assert 'source "${ROOT_DIR}/scripts/runtime_paths.sh"' in redis
    assert "sha256sum --check --strict --quiet SHA256SUMS" in redis
    assert "dpkg-deb -f" in redis
    assert "--no-download" in redis
    assert 'Dir::Etc::sourcelist="${APT_SANDBOX}/sources.list"' in redis
    assert "Dir::State::status" in redis
    assert "/usr/sbin/policy-rc.d" in redis
    assert "offline_bundle_available" in redis
    assert "redis_is_installed" in redis
    assert "apt-get update" in redis
    assert "install redis-server redis-tools" in redis
    assert 'REDIS_DIR="${KERNELGYM_OFFLINE_REDIS_DIR}"' in redis
    assert "KERNELGYM_PROXY" not in redis
    assert "HTTP_PROXY" not in redis


def test_redis_bootstrap_reuses_installed_redis_when_bundle_is_missing(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command_name in ("redis-server", "redis-cli"):
        executable = fake_bin / command_name
        executable.write_text("#!/usr/bin/env bash\necho 'Redis server v=7.0.15'\n", encoding="utf-8")
        executable.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["KERNELGYM_OFFLINE_REDIS_DIR"] = str(tmp_path / "missing-bundle")
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "ensure_redis.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "using the existing system Redis installation" in completed.stdout
    assert "installing from configured apt repositories" not in completed.stderr


def test_verify_redis_bundle_does_not_fall_back_online(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["KERNELGYM_OFFLINE_REDIS_DIR"] = str(tmp_path / "missing-bundle")
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "ensure_redis.sh"), "--verify-bundle"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert "offline bundle is unavailable or incomplete" in completed.stderr


def test_environment_bootstrap_is_offline_and_ignores_shared_venv() -> None:
    ensure = (ROOT / "ensure_venv.sh").read_text(encoding="utf-8")
    deploy = (ROOT / "deploy_node.sh").read_text(encoding="utf-8")

    assert 'source "${KERNELGYM_LOCAL_VENV_DIR}/bin/activate"' in ensure
    assert 'source "${KERNELGYM_LOCAL_VENV_DIR}/bin/activate"' in deploy
    assert "--offline --no-cache --no-index" in ensure
    assert '--find-links "${KERNELGYM_OFFLINE_WHEEL_DIR}"' in ensure
    assert '-r "${ROOT_DIR}/requirements-offline.txt"' in ensure
    assert '--no-deps -e "${ROOT_DIR}"' in ensure
    assert "source .venv/bin/activate" not in ensure
    assert "source .venv/bin/activate" not in deploy


def test_offline_lock_pins_cuda_and_tvm_runtime() -> None:
    locked = (ROOT / "requirements-offline.txt").read_text(encoding="utf-8").splitlines()

    assert "apache-tvm-ffi==0.1.11" in locked
    assert "torch==2.11.0+cu129" in locked
    assert "torchvision==0.26.0+cu129" in locked
    assert "nvidia-cuda-runtime-cu12==12.9.79" in locked
