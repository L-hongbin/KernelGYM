#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

# Assume the venv has been bootstrapped already (bash ensure_venv.sh). Here we
# only: make sure redis-server is installed (cheap; no-op when already there),
# activate the venv, scrub the env so the host's torch tree doesn't shadow us,
# sanity-check the runtime, then hand off to the Python deploy driver.
bash "${ROOT_DIR}/scripts/ensure_redis.sh"

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/scrub_venv_env.sh"

python "${ROOT_DIR}/scripts/validate_runtime.py"

exec python "${ROOT_DIR}/scripts/deploy_node.py" "$@"
