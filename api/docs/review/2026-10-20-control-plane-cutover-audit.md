# Cutover audit and follow-up work — 2026-10-20

Audited baseline: `a7f139f6`, including runtime changes `742c63b1` and
`e388434f`. Apple Developer Program purchase, signing and notarization are
explicitly excluded. CI cockpit and unrelated portfolio projects are excluded.

The read-only live verifier passed: three healthy active services, twelve
matching source hashes per service, schema 125, twelve unchanged historical
stopped jobs, all repairs `report_only`, two ready hosts, and the same native
Grafana process/dashboard hash. This proves those operational checks; it does
not prove crash durability, absence of late command races, or all Track B work.

## Findings reproduced before implementation

| Priority | Finding | Reproduction and consequence |
| --- | --- | --- |
| P1 | Journal persistence is not durable or serialized | Concurrent saves use the same `.tmp` outside the attempt lock; neither file nor parent directory is fsynced before ACK. The concurrency test detects overlapping writers, and simulated sync failure leaves a success replayable. A crash can lose a result the server considers acknowledged. |
| P1 | Command ACK/NACK ignores session and delivery generation | After claim expiry and redelivery to a new session, the former session can still ACK with HTTP 200 or NACK the row back to pending. Host identity alone does not identify the current delivery. |
| P2 | Host stop/remove reports failed Docker calls as success | Permission/daemon errors on inspect count as absence; failed stop/remove calls still increment dispatch success and discard local ownership. Three fault-injection cases reproduce false ACK. |
| P1 | Restore migration can target the live database | A read-only Compose render with a disposable database override still resolves the migration DSN to `xcelsior`, because the production override pins that URI. The restore drill must rewrite and verify the connection before invoking Alembic. |

All **nine audit regression cases failed against the unchanged implementation**
on a disposable PostgreSQL database. Evidence is retained in
`artifacts/control-plane-audit-followups-20261020/before/` with source hashes,
JUnit output and the migration/test exit codes. No production fault injection,
customer workload or GPU job was used to reproduce these findings.

## Follow-up order and limits

1. Correct these reproduced defects, prove that the old session cannot change
   a newer claim, sign the worker distribution, run the cutover regression gate,
   and deploy with readiness and source-identity checks.
2. Configure and exercise an operational alert receiver. The operator selected
   `hello@xcelsior.ca`, then permitted a webhook primary and requested a Telegram
   bot as a second sink. Discover existing configuration without printing secrets;
   verify actual acceptance by each transport and distinguish transport acceptance
   from a person reading the message.
3. Verify backup/restore isolation, recovery evidence and dedicated runtime-role
   readiness. Preserve historical records and missing-volume metadata.
4. Contract migration and durable fairness/preemption remain separate engineering
   work. The projection trigger still supplies normalized fields and overwrites
   effective priority on every job update; deleting it before replacing its
   callers is unsafe. An active scheduler is not schema contraction or B16.4.

The prior 420-test gate passed, but omitted the concurrency/durability and stale
claim cases above. Its passing result is retained as historical evidence and is
not used to dismiss these findings. The remaining Track B checkboxes and external
rollout gates must be assessed from implementation and runnable evidence, rather
than marked complete from this runtime cutover.

## Corrective implementation

Journal snapshots and writes now share a reentrant lock, use a private unique
temporary file, sync its contents before rename, and sync the parent directory
before exposing success. Failed saves restore the previous cached result and
cannot produce an ACK. Attempt commands also require a durable result before
ACK; a start cannot reach container execution without its authority journal.

Active ACK/NACK requests require the worker session and delivery attempt count.
Their SQL compare-and-set matches both fields. Each execution thread retains
its own delivery identity, including when the same command is redelivered while
an earlier handler is still running. Current duplicate ACKs replay their stored
result; the former session or former delivery cannot change the new claim.
The claim-expiry controller and ACK race through row updates: completion before
revocation can succeed, while completion after redelivery cannot borrow the new
delivery. Long host operations do not need a fabricated claim refresh.

Docker inspect failures count as absence only for an explicit missing-container
response. Failed stops do not proceed to forced removal; failed removals do not
discard ownership or report success. Such failures receive retryable NACKs.

The expanded selected gate passed **432 tests**, with zero failures, errors or
skips and the optional Compose smoke class deselected. It includes the original
cutover suites plus twelve audit cases. All twelve distribution signatures and
targeted Ruff checks passed. Production deployment is recorded separately;
these test results alone do not prove deployment.

The restore helper now rejects names outside the generated restore namespace,
rewrites every inherited DSN, removes URI database-query overrides, disables
dotenv loading and verifies `current_database()` before Alembic can run. Restore
objects are owned by the designated application role. The active-attempt check
uses the current state vocabulary. A fresh real PostgreSQL regression migrated
an empty disposable database while the inherited DSNs pointed at the source;
the source schema remained unchanged. Seven restore/backup tests passed.
The real test also exposed ConfigParser interpreting percent-encoded URI bytes;
the Alembic environment now escapes them before setting its configuration URL.
