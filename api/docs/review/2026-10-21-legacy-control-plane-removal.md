# Legacy control-plane removal candidate — 2026-10-21

This candidate removes the process-memory preemption module, scheduler SSH
startup, and scheduler JSON-file adoption and mirrors. It does not complete
B16.2 or B16.3. No application rollout, worker restart, schema migration,
historical deletion, or repair-policy change was performed.

`core/agent_preempt.py`, scheduler `process_assigned` and `run_job`, and the
worker's old preemption poll and handler are physically removed. Authenticated
preemption URLs remain permanent HTTP 410 tombstones, including their v2
alias. They cannot enqueue, consume, checkpoint, or announce preemption.

Host-pinned API launches persist the pin in the initial submission transaction.
They remain queued until the transactional scheduler creates the attempt,
allocation, placement lease and durable start command. Filter policy
`filters/v3` enforces the pin, and reservation rechecks it under the job lock.
An unavailable pinned host does not authorize fallback to another host.

Cancellation uses a fenced stop/remove command for an active attempt. Only a
queued job with no host or active attempt may settle immediately, with both
status and host checked under the mutation lock. Historical running, assigned,
stopped, and host-bound queued rows without fenced authority return HTTP 409
without being rewritten or having their volumes detached.

The old immediate victim requeue and volume-detach algorithm is removed. Both
host drain routes stop new placement while leaving running workloads intact.
Eviction execution and non-preview spot preemption return HTTP 501 until the
B16.4 durable controller can prove worker stop before releasing capacity.
An approved eviction plan stays approved and retryable when execution is
unavailable. Capacity preview reads existing prices and does not update price
history. The price monitor continues to refresh prices independently.

Empty canonical job or host tables no longer import old JSON files or `state`
namespaces. Scheduler list state is read and written only in PostgreSQL;
existing namespace keys are preserved to avoid losing billing, marketplace,
pool, or compute-score metadata. Removing the file fallback exposed a decoder
bug: psycopg's already-decoded JSONB lists were discarded. The shared decoder
now accepts those lists, and database roundtrips work without a file mirror.
SQL JSONB payload and namespace compatibility still remains.

The read-only production audit selected database `xcelsior` on **5434**, schema
**125**, and found no active jobs. All eight repair policies remained
`report_only`; twelve protected stopped-job row hashes matched across probes.
The audit ran in a read-only, repeatable-read transaction with a statement
timeout and did not export payloads or credentials.

| Contraction prerequisite | Current production evidence |
| --- | --- |
| Replace status-derived projection writers | Both job and host projection triggers remain. `DatabaseOps.upsert_job`, fenced lifecycle changes and reservation still depend on the job trigger. Dropping it would break phase/desired-state consistency. |
| Populate and validate canonical job specs | All 364 historical jobs have null `spec`; 360 have null `spec_hash`. No spec or hash was invented. |
| Resolve identity and tenancy | 72 of 364 jobs lack normalized owner and tenant IDs. Four of six hosts lack normalized owner/provider IDs; all six lack normalized tenant IDs. |
| Validate required-column contraction | Job phase, desired state and queue timestamp have zero nulls but remain nullable. Generation, version and effective priority are already non-null. Existing values alone do not prove writers are ready. |
| Archive/drop legacy lease history | 302 old leases remain: 225 expired and 77 released, none active. Retention/export and restore validation are still required before deletion. |
| Historical event/state disposition | 9,216 legacy events and 26 `state` namespaces remain. Some still serve runtime callers; they cannot be dropped wholesale. |

Other B16.3 work remains: old lease protocol/store code, legacy queue walkers,
remaining SSH lifecycle/serverless and checkpoint/migration helpers, legacy
resume/control paths, JSONB compatibility, and scheduler/worker decomposition.
Dedicated database-role deployment is a separate candidate. Durable fairness
and preemption, repair enforcement qualification, and historical disposition
remain open; none is counted complete by this removal.

Validation and source identities are recorded in the
[machine-readable evidence](2026-10-21-legacy-control-plane-removal.json).
The isolated runtime gate uses `api/scripts/run_security_gates.py`; the schema
inventory script and full logs are retained under
`artifacts/legacy-control-plane-removal-20261021/` in the operator workspace.

The combined gate passed **1,259 tests**, with zero failures, errors or skips.
Three named exclusions remain explicit: optional Compose smoke, the future
B16.2 contract-migration placement check, and the companion-prose check whose
source document is absent. Repository Ruff, all 24 changed Python files under
Black, Pyright (zero errors), generated inventory/public OpenAPI checks, and
all twelve worker distribution signatures passed. Two pre-existing duplicate
HEAD-operation-ID warnings remain in the public OpenAPI generator. The
whole-tree Black backlog is outside this candidate. Utility formatting was
verified to preserve its parsed ASTs.

The expanded run also repairs test isolation: older suites preserve the
harness's synthetic worker token, and the login-redirect fixture registers its
own OAuth client and clears its session before checking anonymous behavior.
No production authentication behavior changed. Six previously protected
unrelated workspace file hashes still match; all unrelated main-worktree
edits were left outside this branch.
