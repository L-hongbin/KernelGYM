#!/usr/bin/env bash
set -euo pipefail

MARKER_PATH="${1:-/ms}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: scripts/detect_profile.sh [marker-path]"
    exit 0
fi

if [[ $# -gt 1 ]]; then
    echo "Usage: scripts/detect_profile.sh [marker-path]" >&2
    exit 2
fi

if [[ -L "${MARKER_PATH}" || ! -e "${MARKER_PATH}" ]]; then
    echo "external"
else
    echo "internal"
fi
