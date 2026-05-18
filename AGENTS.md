# Repository Notes

This repository is reward-only KernelGYM. Keep `drkernel`, training launchers, rollout code, checkpoint
utilities, and run-specific experiment state out of this repo.

Use `ruff format` for formatting and `ruff check` for linting. Do not add Black.

Source lineage and implementation differences must stay documented in:

- `docs/SOURCE_LINEAGE.md`
- `docs/IMPLEMENTATION_DIFFERENCES.md`

When changing reward behavior, update those docs if the relationship to
`KernelGYM-vllm018-cuda-agent` or `KernelGYM-lhb` changes.
