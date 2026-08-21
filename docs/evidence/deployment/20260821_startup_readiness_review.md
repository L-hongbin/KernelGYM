# Startup Readiness Merge and Repair Evidence

Date: 2026-08-21

The simplification branch merged `dev_csl` startup-readiness commit `86c6aba` as merge commit `64eab6c`, then repaired review findings before accepting the milestone.

The final implementation keeps expected-worker hostname writes aligned with worker registration via `socket.gethostname()`. Drain and monitor lookup also recognize the current `HOSTNAME` value as a legacy local alias so upgrades can retire records written by the previous identity rule without touching other hosts. A confirmed-reaped READY handshake timeout is classified as infrastructure failure rather than a physical CUDA probe failure. Node readiness fails closed when Redis contract reads fail, accepts supported GPU-only worker contracts, and preserves curl error diagnostics.

Validation:

- Current machine: all pre-commit hooks passed with an isolated Ruff cache.
- Debug host `.17`: related deployment/worker tests reached 100% with no failures.
- Debug host `.17`: 566 tests collected; the full suite reached 100% with exit code 0 using an isolated temporary Redis instance.
- Current machine: the first `cursor-grok-4.6-xhigh` review identified legacy hostname cleanup as P1; after the compatibility fix and regression tests, the final XHigh review reported **No findings**.

The debug host was used only to execute tests. Source edits, Git operations, pre-commit, and agentp review ran on the current machine.
