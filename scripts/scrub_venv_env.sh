# Scrub LD_LIBRARY_PATH and PYTHONPATH so the local .venv's torch (and the
# wheel-vendored CUDA runtime libs under site-packages/nvidia/*/lib) win over
# any host-side Python tree.
#
# NGC-style images set LD_LIBRARY_PATH to point at the system Python's torch
# tree (.../dist-packages/torch/lib and .../torch_tensorrt/lib) and a
# PYTHONPATH that imports torch from outside the venv. Once the local .venv is
# active, the dynamic linker still resolves transitive deps (libnvJitLink,
# libnccl, libcusparse) against the system torch's vendored copies — producing
# `undefined symbol: __nvJitLinkGetErrorLogSize_12_9` /
# `ncclDevCommCreate` — and sys.path can pick up a completely different torch
# entirely.
#
# Drop every LD_LIBRARY_PATH entry that points at .../dist-packages/torch* or
# .../site-packages/torch*, and unset PYTHONPATH. Driver-side entries
# (/usr/local/nvidia/lib*, /usr/local/cuda/compat/*) stay intact.
#
# This file is meant to be SOURCED, not executed:
#   source "${ROOT_DIR}/scripts/scrub_venv_env.sh"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    _scrub_clean=""
    IFS=":" read -ra _scrub_parts <<< "${LD_LIBRARY_PATH}"
    for _scrub_part in "${_scrub_parts[@]}"; do
        case "${_scrub_part}" in
            ""|*/dist-packages/torch|*/dist-packages/torch/*|*/dist-packages/torch_tensorrt|*/dist-packages/torch_tensorrt/*|*/site-packages/torch|*/site-packages/torch/*)
                continue
                ;;
        esac
        _scrub_clean="${_scrub_clean:+${_scrub_clean}:}${_scrub_part}"
    done
    export LD_LIBRARY_PATH="${_scrub_clean}"
    unset _scrub_clean _scrub_part _scrub_parts
fi
unset PYTHONPATH
