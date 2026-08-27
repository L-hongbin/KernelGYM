#!/usr/bin/env bash
# Prefer the pinned offline Redis bundle, falling back to the configured apt
# repositories only when the bundle is unavailable and Redis is not installed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/runtime_paths.sh"

VERIFY_ONLY=0
if [[ "${1:-}" == "--verify-bundle" ]]; then
    VERIFY_ONLY=1
elif [[ -n "${1:-}" ]]; then
    echo "Usage: scripts/ensure_redis.sh [--verify-bundle]" >&2
    exit 2
fi

fail() {
    echo "Redis bootstrap failed: $*" >&2
    exit 1
}

REDIS_DIR="${KERNELGYM_OFFLINE_REDIS_DIR}"

offline_bundle_available() {
    [[ -d "${REDIS_DIR}" ]] &&
        [[ -f "${REDIS_DIR}/platform.txt" ]] &&
        [[ -f "${REDIS_DIR}/packages.txt" ]] &&
        [[ -f "${REDIS_DIR}/SHA256SUMS" ]] &&
        compgen -G "${REDIS_DIR}/*.deb" >/dev/null
}

redis_is_installed() {
    command -v redis-server >/dev/null 2>&1 && command -v redis-cli >/dev/null 2>&1
}

require_service_start_policy() {
    [[ -x /usr/sbin/policy-rc.d ]] || fail "/usr/sbin/policy-rc.d must deny service starts during package installation"
    set +e
    /usr/sbin/policy-rc.d redis-server start >/dev/null 2>&1
    POLICY_RC=$?
    set -e
    [[ "${POLICY_RC}" == "101" ]] || fail "/usr/sbin/policy-rc.d returned ${POLICY_RC}, expected deny code 101"
}

run_apt() {
    if [[ "$(id -u)" == "0" ]]; then
        env DEBIAN_FRONTEND=noninteractive "$@"
    else
        command -v sudo >/dev/null 2>&1 || fail "root or sudo is required to install Redis"
        sudo env DEBIAN_FRONTEND=noninteractive "$@"
    fi
}

install_redis_online() {
    command -v apt-get >/dev/null 2>&1 || fail "offline bundle is unavailable and apt-get is not installed"
    require_service_start_policy
    echo "WARNING: offline Redis bundle unavailable at ${REDIS_DIR}; installing from configured apt repositories" >&2
    run_apt apt-get update
    run_apt apt-get -y --no-install-recommends install redis-server redis-tools
    redis_is_installed || fail "redis-server or redis-cli is unavailable after online installation"
    redis-server --version
    echo "Redis installed from configured apt repositories"
}

if ! offline_bundle_available; then
    if [[ "${VERIFY_ONLY}" == "1" ]]; then
        fail "offline bundle is unavailable or incomplete: ${REDIS_DIR}"
    fi
    if redis_is_installed; then
        redis-server --version
        echo "Offline Redis bundle unavailable; using the existing system Redis installation"
        exit 0
    fi
    install_redis_online
    exit 0
fi

for command_name in dpkg dpkg-deb dpkg-query sha256sum; do
    command -v "${command_name}" >/dev/null 2>&1 || fail "${command_name} is unavailable"
done

EXPECTED_OS_ID=""
EXPECTED_OS_VERSION_ID=""
EXPECTED_ARCH=""
while IFS='=' read -r key value; do
    case "${key}" in
        os_id) EXPECTED_OS_ID="${value}" ;;
        os_version_id) EXPECTED_OS_VERSION_ID="${value}" ;;
        architecture) EXPECTED_ARCH="${value}" ;;
        ""|'#'*) ;;
        *) fail "unknown platform key: ${key}" ;;
    esac
done < "${REDIS_DIR}/platform.txt"
[[ -n "${EXPECTED_OS_ID}" && -n "${EXPECTED_OS_VERSION_ID}" && -n "${EXPECTED_ARCH}" ]] || \
    fail "platform.txt is incomplete"

# shellcheck disable=SC1091
source /etc/os-release
ACTUAL_ARCH="$(dpkg --print-architecture)"
[[ "${ID:-}" == "${EXPECTED_OS_ID}" ]] || fail "requires ${EXPECTED_OS_ID}, found ${ID:-unknown}"
[[ "${VERSION_ID:-}" == "${EXPECTED_OS_VERSION_ID}" ]] || \
    fail "requires OS ${EXPECTED_OS_VERSION_ID}, found ${VERSION_ID:-unknown}"
[[ "${ACTUAL_ARCH}" == "${EXPECTED_ARCH}" ]] || fail "requires ${EXPECTED_ARCH}, found ${ACTUAL_ARCH}"

declare -A EXPECTED_VERSIONS=()
declare -a PACKAGE_NAMES=()
while IFS='=' read -r package_name package_version; do
    [[ -z "${package_name}" || "${package_name}" == \#* ]] && continue
    [[ "${package_name}" =~ ^[a-z0-9][a-z0-9+.-]*$ ]] || fail "invalid package name: ${package_name}"
    [[ -n "${package_version}" ]] || fail "missing version for ${package_name}"
    [[ -z "${EXPECTED_VERSIONS[${package_name}]+x}" ]] || fail "duplicate package: ${package_name}"
    EXPECTED_VERSIONS["${package_name}"]="${package_version}"
    PACKAGE_NAMES+=("${package_name}")
done < "${REDIS_DIR}/packages.txt"
[[ -n "${EXPECTED_VERSIONS[redis-server]+x}" && -n "${EXPECTED_VERSIONS[redis-tools]+x}" ]] || \
    fail "packages.txt must pin redis-server and redis-tools"

(
    cd "${REDIS_DIR}"
    sha256sum --check --strict --quiet SHA256SUMS
) || fail "bundle checksum verification failed"

shopt -s nullglob
DEB_FILES=("${REDIS_DIR}"/*.deb)
shopt -u nullglob
[[ "${#DEB_FILES[@]}" -eq "${#PACKAGE_NAMES[@]}" ]] || \
    fail "expected ${#PACKAGE_NAMES[@]} debs, found ${#DEB_FILES[@]}"

declare -A SEEN_PACKAGES=()
for deb_file in "${DEB_FILES[@]}"; do
    package_name="$(dpkg-deb -f "${deb_file}" Package)"
    package_version="$(dpkg-deb -f "${deb_file}" Version)"
    package_arch="$(dpkg-deb -f "${deb_file}" Architecture)"
    [[ -n "${EXPECTED_VERSIONS[${package_name}]+x}" ]] || fail "unexpected deb package: ${package_name}"
    [[ "${package_version}" == "${EXPECTED_VERSIONS[${package_name}]}" ]] || \
        fail "${package_name} deb is ${package_version}, expected ${EXPECTED_VERSIONS[${package_name}]}"
    [[ "${package_arch}" == "${EXPECTED_ARCH}" || "${package_arch}" == "all" ]] || \
        fail "${package_name} deb is for ${package_arch}, expected ${EXPECTED_ARCH}"
    [[ -z "${SEEN_PACKAGES[${package_name}]+x}" ]] || fail "duplicate deb for ${package_name}"
    SEEN_PACKAGES["${package_name}"]=1
done
for package_name in "${PACKAGE_NAMES[@]}"; do
    [[ -n "${SEEN_PACKAGES[${package_name}]+x}" ]] || fail "missing deb for ${package_name}"
done

prepare_offline_apt() {
    APT_SANDBOX="$(mktemp -d /tmp/kernelgym-apt.XXXXXX)"
    : > "${APT_SANDBOX}/sources.list"
    mkdir "${APT_SANDBOX}/sources.list.d"
    APT_ARGS=(
        -o Dir::Etc::sourcelist="${APT_SANDBOX}/sources.list"
        -o Dir::Etc::sourceparts="${APT_SANDBOX}/sources.list.d"
        -o APT::Get::List-Cleanup=0
        -o Acquire::Retries=0
        --no-download
        --no-install-recommends
    )
    trap 'rm -rf -- "${APT_SANDBOX}"; if [[ -n "${TEMP_STATUS:-}" ]]; then rm -f -- "${TEMP_STATUS}"; fi' EXIT
}

installed_package_version() {
    local package_name="$1"
    local record
    record="$(dpkg-query -W -f='${db:Status-Status}\t${Version}' "${package_name}" 2>/dev/null || true)"
    if [[ "${record}" == $'installed\t'* ]]; then
        printf '%s\n' "${record#*$'\t'}"
    fi
}

if [[ "${VERIFY_ONLY}" == "1" ]]; then
    command -v apt-get >/dev/null 2>&1 || fail "apt-get is unavailable"
    prepare_offline_apt
    TEMP_STATUS="$(mktemp /tmp/kernelgym-dpkg-status.XXXXXX)"
    PACKAGE_CSV="$(IFS=,; echo "${PACKAGE_NAMES[*]}")"
    awk -v names="${PACKAGE_CSV}" '
        BEGIN {
            RS=""; ORS="\n\n"
            split(names, wanted, ",")
            for (i in wanted) skip[wanted[i]]=1
        }
        {
            package=""
            count=split($0, lines, "\n")
            for (i=1; i<=count; i++) {
                if (lines[i] ~ /^Package: /) {
                    package=substr(lines[i], 10)
                    break
                }
            }
            if (!skip[package]) print
        }
    ' /var/lib/dpkg/status > "${TEMP_STATUS}"
    apt-get -o Dir::State::status="${TEMP_STATUS}" "${APT_ARGS[@]}" \
        --simulate install "${DEB_FILES[@]}"
    rm -f -- "${TEMP_STATUS}"
    TEMP_STATUS=""
    echo "Offline Redis bundle verified: ${REDIS_DIR}"
    exit 0
fi

ALL_INSTALLED=1
for package_name in "${PACKAGE_NAMES[@]}"; do
    installed_version="$(installed_package_version "${package_name}")"
    if [[ "${installed_version}" != "${EXPECTED_VERSIONS[${package_name}]}" ]]; then
        ALL_INSTALLED=0
        break
    fi
done
if [[ "${ALL_INSTALLED}" == "1" ]]; then
    redis-server --version
    echo "Pinned offline Redis packages are already installed"
    exit 0
fi

command -v apt-get >/dev/null 2>&1 || fail "apt-get is unavailable"
require_service_start_policy

prepare_offline_apt
APT_COMMAND=(apt-get "${APT_ARGS[@]}" -y install "${DEB_FILES[@]}")
if [[ "$(id -u)" == "0" ]]; then
    env DEBIAN_FRONTEND=noninteractive "${APT_COMMAND[@]}"
else
    command -v sudo >/dev/null 2>&1 || fail "root or sudo is required to install Redis"
    sudo env DEBIAN_FRONTEND=noninteractive "${APT_COMMAND[@]}"
fi

for package_name in "${PACKAGE_NAMES[@]}"; do
    installed_version="$(installed_package_version "${package_name}")"
    [[ "${installed_version}" == "${EXPECTED_VERSIONS[${package_name}]}" ]] || \
        fail "${package_name} is ${installed_version:-missing} after install"
done
redis-server --version
