# Agents Notes — KernelGYM Reward-Only

## Harness Files

| File | Purpose | When to update |
|---|---|---|
| `AGENTS.md` | Collaboration rules | User explicitly requests a policy change |
| `RUNTIME.md` | Run-specific facts: environments, endpoints, paths, hyperparams | Tracked facts materially change |
| `INDEX.md` | Index of docs, scripts, artifacts, evidence | Tracked references materially change |
| `CONFIRMATION_GATES.md` | Confirmation gates recorded by user request | User explicitly asks to add a gate |

`RUNTIME.md` and `INDEX.md` are updated proactively and kept under 100 lines each.
Do not maintain a separate running-status log unless explicitly asked.

## Working Style

- Inspect representative real examples (prompts, logs, outputs, scored samples), not only automated checks.
- Dump reviewable evidence to files so the user can inspect the same data.
- Treat `/nfs` and `/ms` as shared filesystems.
- Keep bash files under 200 lines — use them only for parameter config and simple glue; put real logic in Python.
- In Markdown prose, do not hard-wrap ordinary paragraphs at a fixed character count; keep natural paragraph lines unless structure such as lists, tables, or code blocks requires line breaks.

## Execution

- Ask for explicit user approval before restarting services.
- Complete each simplify milestone only after unit tests and a read-only current-machine `kimip` review pass (default `kimi-code/k3`, high thinking), then notify the user through the page-user MCP with the milestone outcome.
- Adapt pragmatically when instructions need mid-run adjustment; report what changed and why at the end (what happened → why → what changed → current status → remaining gaps).
- On repeated errors, stop retrying — research 3–5 fixes, pick the best, implement it.
