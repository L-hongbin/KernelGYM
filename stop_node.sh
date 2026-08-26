#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/runtime_paths.sh"

# Keep shell responsibility minimal: activate the existing venv, then hand off to the service CLI.
# shellcheck disable=SC1091
source "${KERNELGYM_LOCAL_VENV_DIR}/bin/activate"
exec python -m kernelgym.cli.service stop --profile v1 "$@"
