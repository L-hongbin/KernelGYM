# Agentp Connectivity Failure and Reviewer Switch

Date: 2026-08-21

## Outcome

Milestone review now uses `kimip` on the current development machine, with its configured default `kimi-code/k3` model and high thinking effort. Debug host `.17` remains test-only.

## Observed Failure

Multiple read-only `agentp --model cursor-grok-4.6-xhigh` review attempts lost their long-lived connection to `https://agentn.global.api5.cursor.sh`. New sessions and `--continue` both entered repeated reconnect loops. One resumed attempt ended with `RetriableError: [resource_exhausted] Error`. The attempts were stopped after repeated failures instead of retried indefinitely.

The final narrow review session remained connected for several minutes but was intentionally stopped when the user selected `kimip` as the replacement; it produced no review conclusion.

## Diagnostic Evidence

- DNS resolution for `agentn.global.api5.cursor.sh` succeeded.
- Short HTTPS requests through `http://192.168.28.186:17897` returned HTTP 200.
- `agentp status` reported a valid login, and `cursor-grok-4.6-xhigh` remained listed as available.
- Direct HTTPS without the configured proxy was reset during TLS setup, so bypassing the proxy was not viable.
- The local `agentp` wrapper always configures the shared HTTP proxy before launching Cursor Agent.

These checks rule out a missing model, expired local login, and DNS failure. They do not distinguish conclusively between instability in the required proxy's long-lived streaming path and an upstream Cursor service/resource issue. The document therefore records the failure boundary rather than claiming a single proven network root cause.

## Replacement

`kimip` is installed on the current machine, passes `kimi doctor`, and uses the managed Kimi coding endpoint through the same configured proxy. Review remains read-only by prompt and is run only after tests and static checks pass.
