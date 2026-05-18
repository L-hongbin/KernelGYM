#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

python -m pip install -r "${ROOT_DIR}/requirements.txt"

if command -v apt-get >/dev/null 2>&1; then
    echo "Optional system packages for local service management: iproute2 redis-server"
    echo "Install them manually if redis-server, ss, or redis-cli are unavailable."
fi
