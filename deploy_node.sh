#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

# Assume ensure_venv.sh was already run separately. Just activate the existing
# .venv, scrub the env, sanity-check the CUDA/torch install, then hand off.
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/scrub_venv_env.sh"

python "${ROOT_DIR}/scripts/validate_cuda129.py"

exec python "${ROOT_DIR}/scripts/deploy_node.py" "$@"
