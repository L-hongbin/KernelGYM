#!/usr/bin/env bash
set -euo pipefail

# Newer runtime images place Python 3.12 under /usr/bin, while the shared
# repo-local .venv was created with /usr/local/bin/python3 as its interpreter.
# Keep the old interpreter path valid so the existing venv remains usable.
PYTHON_TARGET="${KERNELGYM_PYTHON_TARGET:-/usr/bin/python3.12}"
PYTHON_LINK="${KERNELGYM_PYTHON_LINK:-/usr/local/bin/python3}"

if [[ ! -x "${PYTHON_TARGET}" ]]; then
    echo "python target not executable: ${PYTHON_TARGET}" >&2
    exit 1
fi

mkdir -p "$(dirname "${PYTHON_LINK}")"
ln -sfn "${PYTHON_TARGET}" "${PYTHON_LINK}"

if [[ -e .venv/bin/python && ! -x .venv/bin/python ]]; then
    echo ".venv/bin/python still is not executable after linking ${PYTHON_LINK}" >&2
    exit 1
fi

echo "python_link=${PYTHON_LINK} -> ${PYTHON_TARGET}"
