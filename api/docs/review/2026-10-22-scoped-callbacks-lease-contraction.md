# Scoped callbacks and retired lease rows/columns — 2026-10-22

This candidate adds migration 126 for hash-only serverless callback credentials
and migration 127 to archive inactive v1 lease rows and drop the unused `leases`
table with its eight columns. It does not drop canonical job/host columns or
the transitional projection triggers that remaining writers still require.

The native lease claim transaction issues a random credential only for the
exact bound serverless worker/job/attempt/host/fence. Only its SHA-256 digest is
stored, alongside the endpoint's tenant identity; callbacks also require that
tenant binding to remain current. The plaintext travels in that claim response
to the host and then to the worker container, without entering its recovery
journal or persisted job spec. The host's broad API credential is no longer injected into serverless
containers, and the SDK no longer falls back to that credential.

Callback routes verify the digest, worker ID, current attempt, endpoint/job/host
binding, fence and live lease. Middleware recognizes this token only on the
specific callback methods/paths; it creates no API/host principal. Shared API
tokens, customer sessions and unauthenticated test mode cannot bypass callback
authentication. The credential follows lease renewal, loses authority on lease
expiry/release or terminal settlement, and is revoked in the terminal worker
projection. An agent refuses serverless startup if its grant lacks the callback
credential. SDK callbacks use the public API URL; host mTLS credentials are not
forwarded into containers.

The bounded daily `serverless_callback_credential_retention` task deletes at
most 500 expired credential digests belonging to terminal attempts, after a
24-hour retention window. It skips locked rows and retains live attempts,
native attempt history and all billing evidence.

Migration 127 locks the obsolete table, refuses active/unknown lease states and
unexpected columns, archives complete inactive rows with checked SHA-256
digests, verifies row equality/counts, then drops the table without CASCADE.
An unreviewed dependency aborts the transaction. Downgrade validates the archive
and restores the exact original rows, column types, keys and surviving indexes.
Runtime role grants allow only reads of the archive. The transitional role
catalog still recognizes `leases` while a database is at 126; after 127 its
missing relation is filtered from grants. This is no runtime lease fallback.

The read-only production export found 258 lease rows whose jobs have a recorded
tenant and **44 without one**. Archive metadata copies a known tenant as-is;
unresolved rows carry `tenant_attributed=false` in the restricted
`system:unattributed-legacy-lease-archive` scope. That scope records an ownership
gap rather than inventing a customer. The original eight-field lease records
remain unchanged. Both new tables carry non-null tenant scope and an index
beginning with it. No customer-facing archive route is added.

The isolated drill used an explicit read-only export of **all 302 production
lease rows**, with all eight original columns. Up/down/up tests verified exact
row restoration and an unchanged protected stopped-job record. Tests separately
check exact preservation of the 258 known and 44 unresolved tenant attributions.
They also prove refusal of active/unknown leases, extra columns and dependent
views, with no partial archive or version advance. The broader production-shaped
053-to-head rehearsal now demonstrates that contract removal refuses an active
v1 lease until the native handoff has explicitly closed that old authority.
It still checks native backfill, tenancy, job states and unchanged money totals.

The previous CUDA acceptance script still contained a dead v1 lease/start/stop
rehearsal calling removed worker functions. That block is removed; its native
CUDA and billing journey remains. No CUDA acceptance was run for this candidate.
The rebuilt SDK image was tested nonroot with a read-only filesystem and no
external network. Its real HTTP client sent the scoped credential to a local
callback server, served healthy readiness and drained without forwarding the
host token. This qualifies SDK packaging/HTTP behavior, not GPU execution.

Rollout has two stages. Back up and apply **126** first, then hand off to the
new API/background/scheduler runtime, signed agents and updated SDK images.
The new runtime's readiness floor is 126, so it can run before contraction.
Verify old writers are retired and no active v1 authority remains before
applying **127**. Do not run a one-shot upgrade through 127 while old processes
can still query the removed table. Re-provision dedicated role grants against
the resulting schema when deploying that separate role candidate.

Production remains schema **125** on application PostgreSQL **5434**. No live
migration, row/column deletion, application deployment, host restart or repair
enforcement occurred. All eight repair modes remain `report_only`. The real
snapshot drill and committed migrations make the pruning reviewable; they are
not a claim that production has already been pruned.

Final verification passed **2,269 tests**, with zero failures/errors/skips and
the same three baseline exclusions (optional Compose smoke, future B16.2
contraction, unavailable companion prose). Repository Ruff, Black on all 31
changed Python files, Pyright (243 files; zero errors/warnings), generated
inventory/public OpenAPI checks and all 12 worker distribution signatures pass.
All changed Python and worker-signature hashes match the passing manifest.
The disposable PostgreSQL cluster and Redis container were removed afterward.

Final results, image identity, input/source hashes and production preservation
checks are in the [JSON receipt](2026-10-22-scoped-callbacks-lease-contraction.json).
Remaining B16 work includes other SSH/backend compatibility callers, worker and
JSONB decomposition, canonical spec/identity resolution, remaining column/trigger
contraction, historical event/state disposition, dedicated-role deployment,
durable fairness/preemption and coordinated production/GPU acceptance.
