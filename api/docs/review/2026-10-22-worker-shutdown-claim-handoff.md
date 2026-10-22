# Worker shutdown and claimed-start recovery — 2026-10-22

Worker SIGTERM now checkpoints native attempt authority and preserves its
containers for restart adoption. Newly started and already adopted work follow
the same rule. Shutdown no longer stops containers and sends unsupported
`queued` reports. The runner's `finally` block preserves mounts, encrypted
volumes/workspaces and local bookkeeping while the same attempt is journaled
for shutdown handoff. Ordinary terminal cleanup remains unchanged. Shutdown
also refuses host deregistration while work is unsettled; decommissioning
requires draining that work first.

A serverless grant from an older API can still omit its callback credential.
The upgraded agent records that execution never started, reports `failed`
through the exact attempt/fence, and then NACKs the command. Native settlement
releases the lease and GPU allocation and closes serverless capacity. Transport
failure retains the abort in the local journal and arms a retry loop; restart
retries it directly without Docker discovery, lease renewal or container start.
An old response cannot erase replacement authority. Journal/command-result
failures before execution use the same claimed-start settlement path.

The API-side guard from the previous repair remains in place: a serverless
credential mint miss rolls the entire claim transaction back. The agent repair
covers mixed-version API/agent handoff as well as the current API behavior.

The broader gate passes **2,398 tests** with no failures, errors or skips and
the same three explicit baseline exclusions. The focused gate passes
**113 tests**, including:

- A tokenless old-API grant followed by immediate native settlement against
  disposable PostgreSQL and the real FastAPI status route.
- An unavailable status route, persisted abort, simulated process restart and
  recovery that releases allocation without waiting for lease expiry.
- SIGTERM followed by API-verified renewal/adoption of the same running native
  attempt, including work that was already adopted.
- Managed/shared NFS mounts, encrypted volumes/workspaces and GPU cleanup
  preserved at shutdown; normal cleanup still executes outside handoff.
- In-process retries and stale success/fence/error replies preserving newer
  attempt authority; active work preventing host deregistration.

Docker execution is isolated in these tests. They prove worker decisions and
real database/API transitions, not a live GPU/container restart journey.
The [machine-readable receipt](2026-10-22-worker-shutdown-claim-handoff.json)
records the broader gate, static checks, source/signature hashes and cleanup.
Detailed local evidence is in
`artifacts/worker-shutdown-claim-handoff-20261022/`.

This is a source candidate. No production worker handoff, migration or repair
enforcement was performed. Repairs remain `report_only`. Canonical specs,
remaining JSONB paths, canonical column/trigger contraction, historical
disposition, all-table semantic review and durable fairness/preemption remain
open in B16; these two worker repairs do not complete the broader cutover.
