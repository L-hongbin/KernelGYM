from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_VENV = "/root/kernelgym-reward-only/.venv"
OFFLINE_WHEELS = "/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only/wheels"
OFFLINE_REDIS = f"{OFFLINE_WHEELS}/redis/ubuntu-24.04-amd64"


def test_runtime_paths_pin_local_venv_and_absolute_offline_wheelhouse() -> None:
    paths = (ROOT / "scripts" / "runtime_paths.sh").read_text(encoding="utf-8")

    assert LOCAL_VENV in paths
    assert OFFLINE_WHEELS in paths
    assert OFFLINE_REDIS in paths
    assert "must be an absolute path" in paths
    assert "findmnt" in paths
    assert "findmnt is required to prove the venv path is node-local" in paths
    assert "nfs|nfs4|cifs|smb3|lustre|ceph|gpfs|9p|virtiofs|fuse.*" in paths


def test_redis_bootstrap_uses_only_the_pinned_offline_bundle() -> None:
    redis = (ROOT / "scripts" / "ensure_redis.sh").read_text(encoding="utf-8")

    assert 'source "${ROOT_DIR}/scripts/runtime_paths.sh"' in redis
    assert "sha256sum --check --strict --quiet SHA256SUMS" in redis
    assert "dpkg-deb -f" in redis
    assert "--no-download" in redis
    assert 'Dir::Etc::sourcelist="${APT_SANDBOX}/sources.list"' in redis
    assert "Dir::State::status" in redis
    assert "/usr/sbin/policy-rc.d" in redis
    assert "apt-get update" not in redis
    assert "KERNELGYM_PROXY" not in redis
    assert "HTTP_PROXY" not in redis


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
