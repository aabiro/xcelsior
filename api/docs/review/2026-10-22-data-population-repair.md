# Data population audit and recorded metadata repair — 2026-10-22

The read-only production audit scanned **164 application tables and 1,984
columns**, including partitions, with no scan errors. The
[complete column inventory](2026-10-22-data-population-inventory.csv) records
row/null/blank/empty-JSON/zero counts, types, nullability and whether a default
exists. It contains no row values, default expressions, credentials or DSNs.
The source audit is repeatable with `api/scripts/audit_data_population.py
--expected-database xcelsior --expected-port 5434` under the intended service
environment. It verifies the target, runs a read-only repeatable snapshot,
bounds lock/statement waits and explicitly reports incomplete scans.

An empty field is not sufficient evidence of a defect. Optional lifecycle
timestamps, unused payment rails, absent active attempts and empty tables
need their semantic/runtime contracts. The inventory explicitly leaves
unreviewed fields pending; this is complete scan coverage, not a claim that
every field's meaning or every control-plane requirement has been completed.

The audit identified reproducible writer defects. The job upsert never
maintained normalized owner/tenant/team identity or update time. The host
upsert omitted normalized identity/location/security capabilities. Both now
write the recorded fields in the same statement as the existing record and
preserve known identity/capabilities when a partial update omits them. No
provider identity is inferred from a user ID. The heartbeat writer now records
the actual Alembic revision read from its database. Build identity still
requires truthful deployment configuration; no fake version is substituted.

The outbox's newer envelope columns were empty on all **66,195 rows** in the
audit snapshot. All four runtime insert sites now use the existing single
`append_event` authority, which persists occurrence time, canonical payload
SHA-256, registered classification and available trusted context. Native
job/host identity is read from normalized columns. Existing headers supply
trace/correlation/causation context when present. Idempotent replays preserve
the original event, and each intent remains in its state-change transaction.

Migration **128** repairs recorded metadata in separate resumable table units:

- Job owner, tenant and team from already-recorded identity evidence, preserving
  known normalized values. Invalid/non-string or absent evidence stays unknown.
- Host owner/tenant and nonempty recorded capability objects.
- Outbox occurrence time from original insertion time, canonical digest from
  the stored payload, registered classification and original header context.
  Historical attribution never comes from an aggregate's current owner.
- Lease-archive attribution where recovered job evidence now resolves it.

Selection is bounded to 500 rows with `SKIP LOCKED`, statement/lock waits are
bounded, and a remaining locked row prevents a successful migration stamp.
The row repair also compares every field outside its explicit metadata
allowlist before/after update. A transitional trigger changing priority,
admission or other state aborts that table transaction; a real PostgreSQL
regression proves refusal preserves the original priority and missing owner.
Each table unit commits independently and can be rerun. Downgrade retains
recovered facts because they are also valid under 127; it does not turn known
owners back into NULL. Neither original event payloads/headers nor sealed audit
events/hash chains are rewritten. This is a data repair, not trigger removal.

An additional isolated rehearsal used a restricted read-only export of the
actual **364 jobs' identity evidence, six hosts' identity/capability evidence
and all 302 old leases**. Migration 128 recovered **52 job owner/tenant pairs**,
all six host tenants and four missing host owners. It resolved **36** of the
archive's 44 unknown tenant attributions, leaving **20 jobs and eight archived
leases unresolved**. All original lease records and all other tested job
fields stayed identical. Downgrade/upgrade repeated without erasing recovered
facts. The restricted input is not committed; its digest and drill evidence
are recorded in the [JSON receipt](2026-10-22-data-population-repair.json).

Production remains schema **125** on application PostgreSQL **5434**. These are
source and disposable-database changes, not a deployed backfill. The existing
126 runtime handoff and 127 lease-table contraction must precede 128. Repairs
remain `report_only`; protected stopped-job history and production service
identities remain unchanged.

Final verification passed **2,352 tests** with no failures/errors/skips and
the same three baseline exclusions. A separate final production-record drill
passed with the exact migration source. Black checked all 13 changed Python
files; repository Ruff, Pyright (244 files; zero errors/warnings), generated
inventory/public OpenAPI and all 12 distribution signatures pass. Changed
Python hashes match both passing manifests. Disposable PostgreSQL clusters
and Redis were removed.

The full objective remains open. Next work includes canonical spec persistence
and submission context, remaining JSONB readers/writers, all-table semantic
review, real worker inventory/session fields, remaining canonical-column and
projection-trigger contraction, historical unknown disposition, dedicated-role
deployment, durable fairness/preemption and coordinated production acceptance.
