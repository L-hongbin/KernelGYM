#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
bash "${ROOT_DIR}/ensure_venv.sh"

# Keep shell responsibility minimal: activate the existing venv, then hand off to the service CLI.
# shellcheck disable=SC1091
source .venv/bin/activate
exec python -m kernelgym.cli.service stop --profile v1 "$@"
