# Startup Readiness Merge and Repair Evidence

Date: 2026-08-21

The simplification branch incorporated the `dev_csl` startup-readiness change and then repaired review findings before accepting the milestone.

The final implementation keeps expected-worker hostname writes, drain ownership, and monitor ownership aligned on `socket.gethostname()`. The inherited `HOSTNAME` environment variable is never accepted as a node-ownership alias because it can be stale or name another cluster node. A confirmed-reaped READY handshake timeout is classified as infrastructure failure rather than a physical CUDA probe failure. Node readiness fails closed when Redis contract reads fail, accepts supported GPU-only worker contracts, and preserves curl error diagnostics.

Validation:

- Current machine: all pre-commit hooks passed with an isolated Ruff cache.
- Debug host `.17`: 107 related deployment/worker tests passed.
- Debug host `.17`: all 566 tests passed with the TVM-FFI integration test enabled and an isolated temporary Redis instance.
- A later adversarial review found that trusting `HOSTNAME` as a legacy alias could cross node boundaries. The follow-up regression simulates host B inheriting `HOSTNAME=host-a` and verifies that B neither drains nor monitors A's workers.
- Current machine: the final `cursor-grok-4.6-xhigh` review reported **No findings** and confirmed that no worker-ownership path reads `HOSTNAME`.

The debug host was used only to execute tests. Source edits, Git operations, pre-commit, and agentp review ran on the current machine.
