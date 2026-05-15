#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${ROOT_DIR}/start_all_with_monitor.sh" \
  --log-dir logs/debug \
  --eval-results-path logs/debug/eval_results.jsonl \
  "$@"
