# Xcelsior production data architecture companion

Implementation reference, reconstructed from the checked-in schema and runtime
on 2026-10-24. This is not a recovered copy of the original design document.
The [control-plane blueprint](xcelsior-production-control-plane-mcp-blueprint.md)
and [Track B checklist](track-b-implementation-checklist.md) retain the broader
requirements and unfinished work. A description of candidate behavior here
does not establish that production has been migrated or accepted.

## 1. Authority

PostgreSQL owns admission, placement, attempts, leases, device allocations,
wallets, charges and durable delivery intent. Redis, worker journals, metrics,
search indexes and API response fields cannot grant execution or money rights.
Every mutation must identify its authoritative row and transaction boundary.

## 2. Execution records

An admitted job has an immutable, hash-bound execution specification. Reservation
validates that spec, host preferences and current capacity. A current attempt,
lease, host and fencing token authorize callbacks and settlement. Rejected or
expired authority cannot update another attempt. Native lifecycle intent and
desired state live in normalized columns.

## 3. Observations and projections

JSONB remains appropriate for immutable execution specs, versioned event bodies
and non-authoritative observations. A mutable job payload is never a replacement
launch spec. Compatibility response fields do not create lifecycle authority.
Historical observations may be retained when exact admission evidence is absent.

## 4. PostgreSQL domains and schema discipline

### 4.1 Domains

`control_plane/db_roles.py` assigns relations to control, identity, billing,
marketplace, audit, serverless, retrieval, storage or analytics. The catalog
drift gate requires every application relation to have a domain.

### 4.2 Service roles

API, scheduler, reconciler, billing, projector, retrieval, read-only and SSH
gateway roles have explicit privileges. Runtime roles have no schema DDL.
Billing has ledger and meter rights, with no host command rights. Scheduler
cannot mutate wallets or user credentials. A separate one-shot migrator owns
DDL. Maintenance task identity is selected explicitly; cross-domain tasks still
assigned to API are visible debt in `control_plane/task_identities.py`.

### 4.3 Tenant and retention boundaries

New governed tables require typed timestamps, tenant attribution and indexes.
Append-only provenance archives have explicit runtime write denials. A missing
historical tenant is recovered only from recorded evidence; it is not guessed.
Retention must preserve ledger, billing, audit and legal obligations.

### 4.4 Conformance state as implemented

No float money columns remain in the candidate public schema: the gate checks
that no `_cad` column uses `real` or `double precision`. Authoritative money
uses integer micro/minor units or exact numeric values. API display floats are
not ledger authority. Governed timestamp columns use `TIMESTAMPTZ`; inherited
older epoch columns remain separately tracked. The money gate does not assert
that every inherited timestamp or column has already been contracted.

`tests/test_companion_schema_discipline.py` derives governed tables from upgrade
migrations, follows renames and checks the live catalog. Its conformance prose
check fails if this file disappears or a float money column returns.

## 5. Redis

Redis supplies bounded ephemeral coordination and caches. Correctness-sensitive
admission and payment state survives cache loss in PostgreSQL. Cache outages
must not turn authorization checks into permissive local fallbacks. Recovery
must include cache/projection rebuilds, not just database readiness.

## 6. Object storage

The database owns object/catalog identity and lifecycle metadata; an object
store owns bytes. A successful metadata restore does not prove bytes exist.
Recovery acceptance compares catalog entries, object versions/checksums and
access policy. Independent encrypted offsite copies need their own credentials,
retention and tested recovery path.

## 7. Wallets and charges

Wallet mutation, signed ledger rows, usage cycles and customer notification
intent commit together. Replay must not duplicate a debit. A paid cursor moves
only after successful settlement. Insufficient funds do not turn an unpaid
interval into a paid one. Existing balances reconcile with signed transactions.

## 8. Attempt-scoped metering

Native server timestamps and lease evidence bound GPU usage. A validated
admitted quote supplies customer prices. Missing claims, mismatched attempts,
nonfinite intervals or observation-only allocations cannot authorize charging.
Serverless cycles enter invoices once; provider usage meters do not duplicate
the customer's serverless cycle charge.

## 9. Delivery

Durable outbox rows are committed with business changes. Consumers use bounded
claims, idempotency and retry evidence. Best-effort projections run after the
transaction; they cannot be the only record of required business intent.

## 10. Scheduling

PostgreSQL owns team scheduling policy, weighted service credit and preemption
plans. Default policy does not enable eviction. Stopping a victim does not free
capacity until the fenced attempt settles. Fairness tests and synthetic
preemption tests do not authorize production policy changes.

## 11. Historical recovery

Migration 129 qualifies exact launch evidence using the saved hash, command and
attempt. Migration 132 records retained disposition for recovered and unavailable
history. Unavailable evidence remains unavailable; it is not reconstructed from
the latest observation. Financial links and original rows remain available.

## 12. Audit and projections

Outbox, audit delivery, checkpoints and immutable provenance have independent
retention and verification. Restore must preserve event ordering and allow
projections to rebuild. Hashes and counts prove the stated snapshot only.

## 13. Time and identity

Server time bounds authoritative intervals. Worker-provided timestamps are
observations. Runtime credentials are scoped by service; callback credentials
are scoped by attempt and are not persistent plaintext launch metadata.

## 14. Migration order

Alembic is the production DDL authority. The real chain and numbering live in
[the migration ledger](../migrations/README.md). Contractions are staged at
127, 130, 131, 134, 135 and 136, with explicit compatible writers and quiescence.
A future final global cleanup marker must remain the head. Later native-schema
migrations do not restore removed legacy authority.

## 15. Availability and capacity

Readiness includes the runtime schema floor and required dependencies. Database
connection, lock, disk, WAL, replication and transaction-age signals require
actual ingestion and alert delivery. A green HTTP health check alone proves
neither available capacity nor a successful workload.

## 16. Backup and recovery

Logical backups need explicit cluster identity, private files, checksums,
catalog verification and isolated restore. Continuous WAL archiving, PITR,
independent offsite/object recovery and failover need separate acceptance.
An archive copied or checksummed is not a restore or a demonstrated RPO/RTO.

## 17. Privacy

Deletion tracks per-sink state, retention exceptions and retries. A projection
cannot claim deletion while authoritative retained data still exists. Recovery
must avoid resurrecting content beyond its retention and privacy policy.

## 18. Operational evidence

Retain schema/image/source identities, effective roles, repair policies,
protected-table hashes and cleanup results with each handoff. Production
repairs remain `report_only` unless separately authorized. The latest detailed
candidate audit is [the control-plane/data-layer review](review/2026-10-24-control-plane-data-layer-audit.md).

## 19. Qualification

Run Python gates through `api/scripts/run_security_gates.py` with disposable
PostgreSQL and stripped credentials. Relevant gates include schema discipline,
service roles, migration ledger/lock discipline, historical qualification,
native admission, transactional billing and full-schema restore. The Compose
smoke builds tracked source and mounts only the disposable PostgreSQL socket.

## 20. Release boundaries

Schema, writer and worker versions must agree at every stage. A direct upgrade
from production 125 to the candidate head while old writers run is unsafe.
Follow the staged handoff; retain compatible images and inverse evidence.

## 21. Recovery acceptance

Exercise database restore, object integrity, projection rebuild, native worker
adoption, wallet/invoice reconciliation and ingress. Record achieved recovery
point and time separately for each scenario. Local tests do not close deployed
failover or region-loss requirements.

## 22. Remaining work

Track B remains the completion authority. Source qualification, production
handoff, historical financial disposition, service-role isolation and disaster
recovery must each have their own evidence. Do not infer completion from this
reference or turn an unperformed check into a passed checkbox.
