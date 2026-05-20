#!/usr/bin/env bash
# Send a tiny KernelBench evaluation (vector add reference + Triton kernel)
# to the local reward server and print compiled/correctness/speedup.
#
# Usage:
#   bash test_reward.sh [--host HOST] [--port PORT] [--timeout SEC] [--verbose]
#
# Defaults to 127.0.0.1:20111. Stdlib only — no .venv activation needed,
# but the script does enforce no_proxy=* so http_proxy doesn't poison
# the LAN probe.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${ROOT_DIR}/scripts/test_reward.py" "$@"
