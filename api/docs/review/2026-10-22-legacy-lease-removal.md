# Legacy lease authority removal — 2026-10-22

The candidate physically removes the old `events.Lease` model and all five
legacy `EventStore` lease operations: grant, renew, release, lookup and expiry.
The scheduler's separate old-lease expiry/requeue loop is removed. Native
`control_plane.leases` still owns fenced claim, renewal, release and expiry.
Historical lease event types and the event hash-chain reader remain intact.

Authenticated `/agent/lease/claim`, `/renew` and `/release` now return permanent
HTTP 410 responses in paused, canary and active modes. They cannot rewrite old
lease rows, change a current job, append lifecycle events, or announce work.
The plural `/agent/v2/leases/*` protocol remains operational and separately
tested. No new alias routes are added.

Both lease diagnostics use one PostgreSQL view function over `placement_leases`.
The selected lease must match the job's current attempt and host and the
attempt's job, host, fence and active state. Stale pointers, mismatched hosts
or fences, and closed attempts are excluded. The response uses a host alias
and does not expose the physical host ID or fencing token. An overdue current
lease stays visible so operators can diagnose missed renewals. Reads neither
renew authority nor silently fall back to the old table.

The compatibility event diagnostic retains `events:read` and ownership checks;
the v1 diagnostic explicitly enforces `instances:read` for scoped credentials
and its existing tenant boundary. The generated inventory now classifies the
event diagnostic as redundant with the existing MCP active-lease capability.
All 568 operations remain classified.

Runtime tests replace the old lease implementation tests and inline source
redaction check: real HTTP responses prove retirement without historical
mutation, native offered/claimed lease parity, tenant/scope separation, stale
binding rejection, overdue visibility, and response redaction. Existing native
lease and reconciliation suites retain claim, renewal, settlement and expiry
coverage. The focused gate passed 175 tests, including inventory regeneration;
the complete gate and source identities are recorded in the
[machine-readable receipt](2026-10-22-legacy-lease-removal.json).

Production remains schema 125 on application PostgreSQL port 5434. A read-only
probe found 302 historical leases (225 expired, 77 released), no active jobs,
and all eight repairs in `report_only`. The old table is retained for validated
retention/export/restore work. No migration, historical deletion, application
deployment or worker restart was performed.

Remaining source work includes worker-side v1 compatibility helpers, legacy
queue and status paths, SSH lifecycle/serverless/checkpoint callers, JSONB
compatibility and decomposition. Dedicated-role deployment and contraction of
schema/triggers remain separate gates; the historical spec and identity gaps
from the [contraction audit](2026-10-21-legacy-control-plane-removal.md) remain
unresolved. Durable fairness/preemption and repair qualification remain open.

The final combined gate passed **1,536 tests**, with no failures, errors or
skips. Three named exclusions remain unchanged: optional Compose smoke, future
B16.2 contraction, and the missing companion-prose source check. Repository
Ruff, all twelve changed Python files under Black, Pyright (zero errors),
generated artifacts and twelve worker signatures passed. Two existing
duplicate operation-ID warnings remain; whole-tree formatting is outside
this candidate. All changed Python hashes match the successful gate manifest.

The final read-only probe matched the complete 302-row legacy lease digest
and all twelve protected stopped-job row hashes to their baselines. Repairs
still resolve to `report_only`; no production jobs are active. Six protected
unrelated workspace file hashes also remain unchanged.

Worker-side polling, lease clients, raw status mutation and unfenced execution
are removed in the later [native-only worker follow-up](2026-10-22-native-only-worker.md).
This lease receipt remains the evidence snapshot for its own source commit.
