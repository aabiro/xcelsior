# Executed control-plane and data cutover, 2026-10-24

Production is at schema **141**, runtime **fdaab814**, with both signed GPU
workers activated. Seven dedicated runtime/one-shot database identities pass
their runtime-context audits. All eight repairs remain `report_only`;
preemption and deadline force remain disabled. The persistent Compose override
pins the imported immutable image and actual build/service versions. The
original environment files and worker environments are unchanged.

The [sanitized executed receipt](2026-10-24-native-population-production-receipt.json)
records the source, image, migration, acceptance, backup and restore evidence.
This supersedes earlier schema-125/prepared-only and schema-139 runtime status.
It does not assert that every broader control-plane plan is complete.

## Executed data changes

The ordered 126–141 migrations crossed the reviewed native writer boundaries.
All 364 historical jobs remain: four exact recoveries and 360 explicit
retained-unavailable histories. The old 302 leases and 364 placement projections
were archived. Both job/host projection triggers, four old placement columns,
obsolete stop-task execution, the unused command result, unimplemented journal
cursor and Lightning compatibility trigger are gone. Lightning's six float and
epoch columns were archived with exact digest verification before removal;
canonical monetary values and typed dates were retained.

The 140/141 handoff compared every pre-existing logical table while writers
were quiesced. All **153** tables other than the schema stamp retained their
normalized contents exactly, accounting for the reviewed removed columns and
recorded command-time recovery. Seven old command timestamps were qualified in
the earlier restore and eight in the fresh pre-change snapshot. No missing
owner, trace, payment date or exit code was invented.

Ownership transfer previously applied 189 scoped statements with inverse SQL
retained privately. Application PostgreSQL is on port **5434**. Old-owner and
migrator logins do not serve runtime sessions. The old login is restricted to
offline backups without application DML or DDL; unused read-only/retrieval
logins remain disabled. New archive privileges and gateway native reads were
converged without changing existing credentials or login states.

The qualified duplicate-volume correction retains two immutable credit receipts
for **300 micro-CAD**. Replaying it created no additional credit. Original
cycles, ledger entries and evidence were retained.

## Deployed native acceptance

Four scoped operations-account jobs ran actual CUDA on the pinned TUF GPU.
The earlier lifecycle journey preserved container identity/PID across worker
restart adoption, then stopped through a fence and restarted under a fresh
attempt. Old reports were rejected with 409. The final metadata job
`5f1c3bed` completed as attempt `5f69964d-2afc-422b-a661-a28484c43912`, fence 9,
with measured exit code **0**, a recorded command timestamp and authenticated
request provenance. Its matrix marker was visible in durable HTTP logs and
downloads. Its attempt, lease and GPU allocation released completely.

Five usage meters, six charged cycles and six wallet debits agree at
**55,800 micro-CAD**, with zero wallet-ledger mismatches. The final debit arrived
through the normal scheduled billing pass. Draft invoice preview and CSV agree
with usage and create no stored invoice. No external provider payment/transfer
is claimed.

Both workers have fresh native observation sessions and capability snapshots.
Their physical GPU UUID, PCI identity, model, capacity and observation time are
stored in native device rows; old synthetic identities remain retired history.
All twelve public distributions pass the pinned Ed25519 trust root, signatures
and SHA headers. The deployed worker SHA-256 is
`65c6aeb142ba2fc2996b375871e1a4115d6fa5f2998f185743950c40af08fda1`.

The four API-side containers and MCP are healthy with zero restarts and no
new error patterns in the acceptance log scan. The old support-doc permission
warning is fixed by transactional DML. Prometheus ingests metrics availability,
four distinct heartbeat identities and native recovery-job freshness. All
three partition-maintenance alerts are healthy. Public status and MCP readiness
pass. Tower/VTT services were preserved.

## Qualification and recovery

The frozen candidate passed **3,889 tests**, zero failures/errors/skips, on an
isolated Unix-only PostgreSQL 16 and owned networkless Redis. All 1,127 captured
source hashes remained unchanged. Eight inherited dependency/OpenAPI warnings
are retained. The actual non-root image passed readiness, metrics and all
signatures. Real image-based ordered migrations passed with the strict direct
migrator. Generated API inventory, OpenAPI/types, scoped Black/Ruff and Pyright
passed.

A fresh pre-change full backup restored with all 154 logical-table hashes
matching, then passed 139→140→141 with exact reviewed deltas and zero ledger
mismatch. The post-deployment backup captured at **2026-10-24 21:28:11.877963 UTC**
restored with all **155 logical-table hashes** matching and the real current
Alembic entrypoint. It is 54,716,852 bytes, mode 0600, SHA-256
`5b73fae4b978314cd8096374fc87ce3e8445d3ab4f2bf0a154de0702408bf69a`.

Native backup, direct-migrator restore and partition maintenance all executed
successfully against the new runtime. All three timers remain enabled/active,
and all fifteen retained prior backups remain present. Isolated restore cleanup
passed. An earlier operator hash query failed before migration because a JSONB
subtraction needed parentheses before conversion to text; the corrected query
and full comparison passed before contraction.

## Remaining boundaries

The post-cutover read-only scan covers **177 physical tables and 2,077 columns**,
including partitions, with 93 nonempty tables and zero scan errors. Nullable
values require their actual lifecycle/feature contract. Historical identities
and absent events remain unknown; empty feature tables are not repaired by
inventing facts. A source column/reference inventory supports continued semantic
review.

Remaining source work includes native fleet observation reads, compatibility
execution handlers and module decomposition. A separate qualified follow-up
persists explicit provider IDs, projects further canonical host fields and
prevents environment overrides from lowering a runtime's schema floor; it is
not included in this deployed source. Historical empty zero-value draft invoices
remain retained pending a qualified disposition.

Production continuous WAL archiving remains disabled. Local physical base-backup
plus WAL recovery is qualified, but independent encrypted offsite storage,
production PITR/failover and object recovery still require an executed destination
and their own proof. No application bucket was repurposed and no unbounded WAL
spool was enabled. These boundaries remain open; the executed migration is not
a blanket completion of the wider plans.
