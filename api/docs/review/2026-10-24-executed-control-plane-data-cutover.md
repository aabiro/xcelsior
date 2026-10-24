# Executed control-plane and data cutover, 2026-10-24

The staged production handoff completed through schema 139. The API, scheduler,
maintenance and SSH gateway serve runtime `30b3daa6`; both GPU workers serve its
signed worker bytes. MCP serves an independently verified image whose complete
MCP source matches the runtime checkout. The next population/metadata candidate
is being qualified separately and has not been deployed.

The application database is PostgreSQL on port **5434**. Seven dedicated logins
are enabled only for their runtime or one-shot responsibility. Seven runtime
checks pass; live application sessions use neither the old owner nor migrator.
Ownership transfer applied 189 scoped statements with inverse SQL retained.
The old login remains restricted to offline backups without DDL or application
DML. Unused retrieval/read-only logins remain disabled. Secret environments and
release configurations remain private; the original environment files were not
rewritten.

## Executed data changes

The ordered 126–139 migration and native writer boundaries retain all 364
historical jobs: four exact recoveries and 360 explicit retained-unavailable
histories. The old 302 leases and 364 placement projections were archived. Both
job/host projection triggers, four old placement columns and obsolete stop-task
execution are gone. Durable scheduling records exist; preemption and deadline
force remain disabled. Every one of the eight repair policies remains
`report_only` in API, scheduler and maintenance.

The independently qualified historical duplicate-volume correction committed
two immutable credit receipts for **300 micro-CAD**. Replaying the operator
command issued no new credit. Original cycles, ledger entries and evidence were
retained. Later live billing deltas were reconciled rather than waived as hash
exceptions.

## Deployed acceptance

Three scoped operations-account jobs ran actual CUDA on the explicitly pinned
TUF GPU. Their matrix result appeared through the deployed HTTP logs. One job
was adopted across a live worker restart without changing container identity or
PID, then stopped through a fenced command and restarted under a fresh attempt.
Old reports were fenced with 409. All attempts, leases and allocations released.
Four native usage meters, five charged cycles and five wallet debits agree at
**54,200 micro-CAD**, with zero wallet-ledger mismatches. Read-only draft invoice
preview and CSV download agree with the usage; no external provider payment or
transfer is claimed.

The final short job used signed worker SHA-256
`f5111cc837a94ef9ee65a139345d8fc3fd220007765557c3980159650ea50f57`.
Eight repeated HTTP log reads and a log download contained the CUDA marker;
there was no implicit shared-NFS setup. All twelve public worker distributions
passed the pinned Ed25519 trust-root and SHA header checks.

The four API-side services and MCP are healthy with zero restarts. Native
backup, direct-migrator restore and partition jobs executed successfully; all
three timers remain enabled and active. Their freshness is ingested by the
actual Prometheus server, alongside four distinct control-plane heartbeat
identities and metrics availability equal to one. All three partition alerts
are healthy. No restore clones remain on application 5434 or the separate 5432
cluster. The final API startup audit retained two support-doc permission
warnings; the population candidate replaces its unsupported TRUNCATE operation
with transactional DML and separately qualifies the restricted role.

## Restore and source qualification

A fresh schema-139 offline backup restored with all **154 logical-table hashes**
matching, exact historical disposition/credit receipts and zero ledger mismatch.
The native application restore command also passed under its direct migrator.
All fifteen prior backups remain retained. The restored 54,305,156-byte dump,
captured at 2026-10-24T16:51:02.413809 UTC, has SHA-256
`d44137638e5562c9cfd6dce5484f1c06e03a1206a4f99ed72e4683bb77237597`.

The full gate passed **3,543 tests** on `f1e225f5` with no failures/errors/skips or
exclusions. Final `30b3daa6` differs in six recorded files, qualified by 158
boundary tests, twelve generated-contract tests and an actual final-image boot
at schema 139. Database, billing and migration sources are identical to the full
gate. The final image also verified readiness, metrics and twelve signatures.
The three previously excluded broad gates were fixed and included. Scoped
Black/Ruff, Pyright and distribution verification passed.

## Current population review and remaining boundaries

The subsequent read-only scan covers **176 tables and 2,078 columns**, including
partitions, with no scan errors; 92 tables are populated. A complete count and
runtime-reference inventory is retained. Missing values require their actual
semantics: historical identities, payment dates that never occurred, absent
active attempts and unused feature tables must not be filled with invented
facts. The population follow-up fixes confirmed writer/protocol gaps and
canonical owner reads; source qualification is separate from deployed proof.

The live trigger catalog contains the retained Lightning compatibility trigger
in addition to required reconciliation and immutability triggers. Its removal
and remaining Lightning column contraction are tracked in the follow-up.
Existing old draft invoices include 201,301 empty zero-total drafts; they are
retained financial history pending a separately qualified disposition, not
silently deleted.

Production continuous WAL archiving remains disabled. Local physical
base-backup plus WAL restore-point recovery is qualified, but independent
encrypted offsite storage, production PITR/failover and object recovery require
their own executed proof. There is no inferred destination or unbounded WAL
spool. Module decomposition, remaining JSONB observation/adaptor cleanup and
historical evidence disposition remain explicit source work. No blanket
control-plane completion claim follows from the executed handoff.
