# Agents Notes

## Top-Level Harness Files

These files are the stable operational references for work in this repo:

- `AGENTS.md`: repository-level principles and stable collaboration rules. Do not use it as a routine scratchpad; update it only when the user explicitly asks for an instruction or policy change.
- `SPEC.md`: run-specific facts such as environments, service endpoints, model paths, dataset paths, hyperparameters, and other experiment configuration.
- `INDEX.md`: an index of important documents, logs, scripts, code entry points, generated artifacts, and reviewable evidence files.
- `CONFIRMATION_GATES.md`: confirmation gates the user explicitly asked to record. It is not exhaustive; still ask for confirmation when judgment, risk, or ambiguity requires it. After confirmation, delete the item or mark the confirmed result.

## Documentation Policy

1. Update `SPEC.md` proactively when run-specific facts materially change.
2. Update `INDEX.md` proactively when important references materially change.
3. Do not create or maintain a separate running-status log unless the user explicitly asks for one.
4. Add items to `CONFIRMATION_GATES.md` only when the user explicitly asks to record an additional confirmation gate.
5. Keep `SPEC.md` and `INDEX.md` concise and under 100 lines each.

## Working Conventions

1. For any artifact, behavior, or result that can be practically reviewed, inspect representative real examples in addition to automated checks.
2. When useful, dump reviewable prompts, data samples, logs, model outputs, scored examples, or other concrete artifacts to text files so the user can inspect the same evidence.
3. Treat `/nfs` and `/ms` as shared filesystems.

## Execution Policy

1. If user instructions need adjustment during execution, adapt pragmatically instead of stalling.
2. Report any such adjustment at the end, including why the original instruction could not be executed as-is, what changed, the result, current status, and remaining gaps.
3. If execution encounters a situation that the relevant skill did not anticipate, report what the skill did not cover, what workaround or judgment call was used, and whether the gap remains.
4. In final summaries, prefer a combined problem-and-resolution structure instead of separating problems and adjustments.
5. When the same error occurs twice, stop repeating the same action. Research 3-5 plausible fixes, choose the most efficient one, and implement it.
