#!/usr/bin/env bash
# Ensure the redis-server binary is installed on the host.
#
# - If redis-server is already on PATH, print its version and exit 0.
# - If missing and apt-get is available, install redis-server (with sudo when
#   not running as root). On external profiles (no /ms link) the install is
#   retried via KERNELGYM_PROXY when the direct fetch fails.
# - This script does NOT start the daemon. `kernelgym-service start-local`
#   owns the actual redis lifecycle.
#
# Both ensure_venv.sh and deploy_node.sh invoke this so the binary is in place
# regardless of which path the operator used to bootstrap the host.

set -euo pipefail

PROXY="${KERNELGYM_PROXY:-${KERNELGYM_FALLBACK_PROXY:-http://192.168.28.186:7897}}"

PROFILE="internal"
if [[ -L /ms || ! -e /ms ]]; then
    PROFILE="external"
fi

# Same retry-with-proxy helper as ensure_venv.sh, scoped to apt.
network() {
    if "$@"; then
        return 0
    fi
    if [[ "${PROFILE}" != "external" || -z "${PROXY}" ]]; then
        return 1
    fi
    echo "Retrying with proxy ${PROXY}"
    env HTTP_PROXY="${PROXY}" HTTPS_PROXY="${PROXY}" ALL_PROXY="${PROXY}" \
        http_proxy="${PROXY}" https_proxy="${PROXY}" all_proxy="${PROXY}" "$@"
}

run_apt_get() {
    if [[ "$(id -u)" == "0" ]]; then
        network env DEBIAN_FRONTEND=noninteractive apt-get "$@"
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        echo "redis-server is missing and sudo is not available for apt-get install" >&2
        exit 1
    fi
    network sudo -E env DEBIAN_FRONTEND=noninteractive apt-get "$@"
}

if command -v redis-server >/dev/null 2>&1; then
    redis-server --version
    exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "redis-server is missing and apt-get is not available" >&2
    exit 1
fi

echo "Installing redis-server"
run_apt_get update
run_apt_get install -y --no-install-recommends redis-server

if ! command -v redis-server >/dev/null 2>&1; then
    echo "redis-server is still not available after installation" >&2
    exit 1
fi
redis-server --version
