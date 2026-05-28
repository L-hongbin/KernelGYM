#!/usr/bin/env bash
# Probe the reward API and report whether the node is deployed and healthy.
#
# Usage:
#   bash check_node.sh [--host HOST] [--port PORT] [--redis-port PORT] [--max-heartbeat-age SEC] [--verbose]
#
# Defaults to the local machine running this script (127.0.0.1:20111). Bypasses
# http_proxy so LAN probes don't get routed through a corporate proxy. Exits 0
# iff /health returns status=healthy and all registered GPU workers have fresh
# heartbeats.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults: probe the local machine unless the caller overrides via flag or
# env var. Useful for in-container checks; pass --host for remote nodes.
# ---------------------------------------------------------------------------
API_HOST="${KERNELGYM_REWARD_HOST:-127.0.0.1}"
API_PORT="${KERNELGYM_REWARD_PORT:-20111}"
VERBOSE_FLAG=()
MAX_HEARTBEAT_AGE="${KERNELGYM_MAX_HEARTBEAT_AGE:-180}"
REDIS_HOST="${REDIS_HOST:-${KERNELGYM_REDIS_HOST:-127.0.0.1}}"
REDIS_PORT="${REDIS_PORT:-${KERNELGYM_REDIS_PORT:-20110}}"
REDIS_KEY_PREFIX="${REDIS_KEY_PREFIX:-${KERNELGYM_REDIS_KEY_PREFIX:-kernelgym}}"

# ---------------------------------------------------------------------------
# Argument parsing. Keeps the flag surface deliberately small; anything more
# elaborate belongs in the Python companion.
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            API_HOST="$2"
            shift 2
            ;;
        --port)
            API_PORT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE_FLAG=(--verbose)
            shift
            ;;
        --max-heartbeat-age)
            MAX_HEARTBEAT_AGE="$2"
            shift 2
            ;;
        --redis-host)
            REDIS_HOST="$2"
            shift 2
            ;;
        --redis-port)
            REDIS_PORT="$2"
            shift 2
            ;;
        --redis-prefix)
            REDIS_KEY_PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

BASE="http://${API_HOST}:${API_PORT}"
# --noproxy '*' guards against http_proxy routing LAN probes through a proxy.
CURL=(curl -sf -m 5 --noproxy '*')

# ---------------------------------------------------------------------------
# Hit /health first. If the API is unreachable, fail fast with a vertical
# block matching the Python summary layout so callers can grep uniformly.
# ---------------------------------------------------------------------------
if ! HEALTH=$("${CURL[@]}" "${BASE}/health"); then
    cat <<EOF
status:     DOWN
url:        ${BASE}/health
reason:     unreachable (timeout 5s)
EOF
    exit 1
fi

# /queue/status and /workers/status are best-effort; fall back to empty JSON so
# the renderer still has something to chew on.
if ! QUEUE=$("${CURL[@]}" "${BASE}/queue/status" 2>/dev/null); then
    QUEUE='{}'
fi

if ! WORKERS=$("${CURL[@]}" "${BASE}/workers/status" 2>/dev/null); then
    WORKERS='{}'
fi

# ---------------------------------------------------------------------------
# Hand the two JSON blobs off to check_node.py for formatting. printf '%s'
# avoids any shell expansion of literal `$` characters that JSON values
# could contain.
# ---------------------------------------------------------------------------
PAYLOAD=$(printf '{"health": %s, "queue": %s, "workers": %s}' "${HEALTH}" "${QUEUE}" "${WORKERS}")
printf '%s' "${PAYLOAD}" | python3 "${SCRIPT_DIR}/scripts/check_node.py" \
    --base "${BASE}" \
    --max-heartbeat-age "${MAX_HEARTBEAT_AGE}" \
    --redis-host "${REDIS_HOST}" \
    --redis-port "${REDIS_PORT}" \
    --redis-prefix "${REDIS_KEY_PREFIX}" \
    "${VERBOSE_FLAG[@]}"
