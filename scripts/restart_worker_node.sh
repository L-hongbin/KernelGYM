#!/bin/bash
# Safe restart of a KernelGYM worker-only node: drain in-flight tasks, stop, re-join.
#
# Run ON the worker node. The primary (API/Redis) node is untouched: `service stop`
# never contacts a remote Redis, and the graceful window lets every worker finish
# the task it is running (workers stop pulling new tasks immediately on SIGTERM),
# so a restart produces zero spurious task failures.
#
# Usage: restart_worker_node.sh PRIMARY_ADDR [extra deploy_node.py args, e.g. --clear-cache]
set -euo pipefail
PRIMARY=${1:?usage: restart_worker_node.sh PRIMARY_ADDR [--clear-cache]}
shift || true
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="$ROOT/.venv/bin/python"
DRAIN=${KERNELGYM_WORKER_SHUTDOWN_DRAIN_SEC:-120}
GRACE=$((DRAIN + 30))

echo "=== Draining and stopping local workers (graceful window: ${GRACE}s) ==="
"$PY" -m kernelgym.cli.service stop --profile v1 --graceful-seconds "$GRACE"

echo "=== Re-joining cluster at $PRIMARY ==="
exec "$PY" "$ROOT/scripts/deploy_node.py" --join "$PRIMARY" "$@"
