#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
bash "${ROOT_DIR}/ensure_venv.sh"

# Keep shell responsibility minimal: activate the existing venv, then hand off to Python.
# shellcheck disable=SC1091
source .venv/bin/activate

# Re-apply the same LD_LIBRARY_PATH / PYTHONPATH scrub that ensure_venv.sh does
# inside its own subshell, so the services we exec below inherit a clean env.
# See scrub_env_for_venv() in ensure_venv.sh for the rationale.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    _clean=""
    IFS=":" read -ra _PARTS <<< "${LD_LIBRARY_PATH}"
    for _part in "${_PARTS[@]}"; do
        case "${_part}" in
            ""|*/dist-packages/torch|*/dist-packages/torch/*|*/dist-packages/torch_tensorrt|*/dist-packages/torch_tensorrt/*|*/site-packages/torch|*/site-packages/torch/*)
                continue
                ;;
        esac
        _clean="${_clean:+${_clean}:}${_part}"
    done
    export LD_LIBRARY_PATH="${_clean}"
fi
unset PYTHONPATH

exec python "${ROOT_DIR}/scripts/deploy_node.py" "$@"
