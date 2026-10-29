# Projection-delivery audit — 2026-07-27

## Verdict

The pasted report was not wholly hallucinated, but it was not a reliable
codebase audit. It correctly noticed that the B4.4 delivery primitives had no
production caller. It then mixed that real gap with unsupported optimization
claims, work deliberately scheduled for later Track B phases, and sink
recommendations that conflict with Xcelsior's implementation order.

The most important defect was deeper than the report stated:
`audit_events_v2` had no non-test writer, and the event contracts were never
registered outside tests. B4.4 was therefore marked complete while its runtime
and its primary current sink did not exist.

## Claim-by-claim result

| Claim | Verdict at audit time | Resolution / correct phase |
|---|---|---|
| No projection dispatcher | **True** | Added durable stage-one and stage-two scheduled runtime in `control_plane/projection_runtime.py` and `bg_worker.py`. |
| Nothing schedules `prepare_fanout` | **True** | Added `projection_fanout_prepare` every 5 seconds with bounded backlog draining. |
| No warehouse/SSE/webhook delivery tasks | **Misleading** | The missing runtime was real, but those are not three interchangeable current sinks. `audit_log` is active now. SSE remains on Track A's outbox/`NOTIFY` path. Warehouse is B11 and must not activate before its governed landing/IAM/load implementation. No second generic webhook system was invented. |
| MCP is not integrated with projection delivery | **Partly true** | The in-progress B5 implementation writes `mcp_tool_audit` and an outbox event atomically, but no runtime could project it. The event now uses the existing `default` destination class and independently fans out to `audit_log`. Legacy `destination_class='audit'` rows are safely acknowledged for upgrade compatibility. |
| Outbox and projection boundaries are unclear | **Docs were clear; runtime was incomplete** | Track A destination classes settle the original side effect. B4.4 materializes independent per-sink receipts from the same row. There is still one outbox authority. The runtime module now states and enforces that boundary. |
| Missing error handling | **Overstated, with real gaps** | Per-event exception isolation, leases, retries, and dead-letter state already existed. Added permanent-error classification, bounded full-jitter retry, settlement savepoints, stale-owner protection, empty-external-id rejection, orphan detection, and structured warnings. |
| No backpressure | **False as stated** | Batch limits, leases, due times, and retry backoff already bounded work. The runtime now also caps batches per scheduled run. Adaptive sink-specific flow control is not justified without measured pressure. |
| No projection performance monitoring | **True** | Added database-derived queue depth, dead-letter, orphan, oldest-pending, and p95 delivery-latency health data and Prometheus exposition. DB-derived metrics remain truthful across the separate API and bg-worker processes. |
| Batch processing is inefficient | **Unsupported** | No measurement or failing load gate supported the accusation. The documented bounded `SKIP LOCKED` design is retained. Load/chaos evidence belongs to B14.5. |
| Active sinks need a cache | **Not a defect** | The sink set is tiny, correctness-sensitive configuration read inside the fan-out transaction. A cache would add stale activation/deactivation behavior with no measured benefit. |
| Audit trail has gaps | **True** | `audit_events_v2` had no production writer and contracts were test-seeded only. Added contract bootstrap plus redacted, hash-chained, per-stream serialized, idempotent WORM audit delivery. |
| No projection retention policy | **True** | Successful receipts now have configurable 30-day retention. Pending and dead-letter evidence is retained. Outbox pruning now refuses to remove a source that an active sink has not settled. |
| Projection access controls are missing | **False for the current surface** | Projection operations are internal worker/database-role functions, not public routes. `projection_deliveries` and checkpoints already belong to the projector/audit database domain. A future operator retry UI/API must be separately scoped in B6/B15. |
| No full-flow integration test | **True** | Added a real PostgreSQL outbox → fan-out → audit WORM row → delivery receipt test, including redaction and replay idempotency. |
| Failure tests are insufficient | **Partly true** | Existing tests covered retry/dead-letter and replay. Added unknown-contract permanent failure, source-retention, safe late-sink activation, invalid backfill, audit redaction, and runtime registration gates. Broader network/database fault injection remains B14.5. |
| No chaos engineering | **True but planned, not a B4.4 regression** | B14.5 explicitly owns repeated PostgreSQL, object-sink, network, restart, and clock-skew game days. It remains open and must not be represented as complete. |
| No projection health check | **True** | Added projection health to `/api/v1/control-plane/health`; dead letters, orphaned receipts, unavailable metrics, and failed scheduled tasks degrade the result. |
| Documentation is incomplete | **True** | This audit and the B4.4 checklist correction document the shipped boundary, tasks, sink activation, failure behavior, retention, metrics, and deferred work. |
| Migration validation is missing | **False** | Migration 074 already had up/down/from-empty/ledger and real PostgreSQL gates. This correction requires no schema change. Migration validation remains part of the existing repository gates. |
| Alerting is inadequate | **True but explicitly B7.5** | Health and scrapeable metrics now expose the signal. Alertmanager rules, SLOs, and paging drills remain open B7 work; this audit does not falsely mark them complete. |
| CLI tools are missing | **True but explicitly later operations work** | Added a bounded internal `retry_dead_letters` primitive. A separately authorized operator API/UI/runbook remains B6/B15 work; exposing an unscoped repair CLI now would be a security regression. |
| Dead-letter handling is incomplete | **Partly true** | State and retry thresholds already existed. Added permanent-versus-retryable handling, logging, health/metrics, orphan quarantine, and a bounded retry primitive. Alert routing and operator UX remain B7/B15. |

## Additional defects found by the real audit

1. A newly registered sink stored a NULL lower bound, which the code interpreted
   as the beginning of history. It could consume an arbitrary old unprepared
   backlog despite the explicit-backfill contract. New sinks now default to
   registration time; historical delivery requires a validated `[from, to)`
   range.
2. The outbox retention worker could delete a published source row before
   projection fan-out or sink settlement. It now waits for fan-out and all
   materialized receipts.
3. Missing source rows left delivery receipts in an endless claim/lease cycle.
   They now dead-letter as an invariant breach and appear as orphans in health.
4. Backfilling an old outbox event into its original audit timestamp could alter
   an interval already sealed by a signed checkpoint. Audit rows now use append
   time while preserving source occurrence/outbox time in projection metadata.
5. `validate_contract` rejected secret fields but accidentally allowed an entire
   event classified `credential_secret`. Event-level secret classifications are
   now rejected too.
6. `projection_checkpoints.last_prepared_at` was never advanced by preparation.
   It is now maintained.
7. The control-plane health route searched for scheduled-task status `error`,
   but the task schema/runtime records `failed`. Failed tasks were therefore
   omitted from degradation. The status comparison is corrected.
8. The in-progress MCP implementation introduced an `audit` Track A destination
   with no handler. That backlog could starve other outbox work. New events use
   `default`; legacy rows have an upgrade-compatible acknowledgement while
   their independent audit projection remains protected by retention.
9. The retry expression claimed a 600-second full-jitter cap but multiplied
   the cap by up to 1.5, permitting a 900-second delay. It now applies true
   full jitter inside the stated 600-second bound.
10. Outbox integration tests assumed no older valid work existed in the shared
    test database and could fail after wider suites populated a backlog. They
    now drain bounded batches until their own event settles, without deleting
    or skipping unrelated work.
11. The current B5 reconciliation endpoint interpolated allowlisted SQL
    fragments into its query, which tripped the repository's injection gate.
    Each supported status now selects a fully static statement. The immutable
    migration 068 identifier interpolation is separately documented in the
    guard as safe because every identifier comes from its hard-coded
    `WALLET_MONEY` tuple.
12. The projection metrics availability gauge was emitted only on failure,
    making healthy scrapes ambiguous, and the per-sink series lacked
    exposition metadata. Healthy scrapes now emit availability `1` and every
    projection metric has a declared gauge type.
13. FastAPI lifespan started a PostgreSQL `LISTEN` thread on every context but
    never stopped it. Repeated integration clients leaked listeners and
    eventually deadlocked requests. The listener is now lifecycle-owned,
    promptly stoppable, joined during shutdown, and covered by low-level plus
    real-lifespan regression tests.
14. Successful delivery receipts are retained longer than their source outbox
    rows, but health initially classified every missing source as an orphan.
    Normal settled retention would therefore degrade production for weeks.
    Orphan health now counts only unresolved receipts whose source is missing.
15. The first runtime draft used only the process ID as claim owner. Container
    replicas commonly share PID 1, which defeats stale-owner fencing across
    replicas. Audit delivery claims now include hostname, PID, and a per-run
    random identity.

## Broader repository test-gate residual

The repository's single-process full pytest run is still not an honest green
gate. Pytest imports every module before execution, while multiple legacy test
modules overwrite process-global environment variables, scheduler file paths,
`db.AUTH_DB_FILE`, and logger handlers at import time. Consequently endpoint
modules that pass in isolation and in a 505-test ordered prefix fail when all
4,393 tests are collected; later, the global PostgreSQL pool can be exhausted
and request tests time out. The leaked `pg-listen` lifecycle defect found
during this audit is fixed, but the remaining import-time test isolation work
is cross-suite infrastructure and is not falsely represented as resolved by
the projection runtime changes.

## Deliberately not implemented here

- BigQuery/GCS warehouse delivery (B11).
- A second SSE path or generic webhook path duplicating shipped systems.
- Sink caching without measured database pressure.
- Alertmanager/Grafana/SLO policy and chaos drills (B7/B14).
- An unscoped public repair endpoint or CLI (B6/B15).

Those are real roadmap items where noted, not evidence that the current B4
correctness runtime should fabricate their dependencies early.
