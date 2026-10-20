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
| P1 | Fenced status bypasses normal billing settlement | Removing the compatibility PATCH also removes the old caller of normal metering and hold release. Three real HTTP/PostgreSQL cases return running/success while creating no usage meter. Repeated running status also lacks the legacy server-start projection needed by wallet billing. |

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

The normal fenced lifecycle now opens and closes one attempt meter within the
same transaction. Reservation captures owner, explicit rate and GPU count from
locked server rows; a later host price cannot reprice the meter or wallet
period. Server attempt timestamps populate compatibility fields, terminal
settlement releases linked launch holds, and wallet periods cannot include
stopped time before the current attempt's server start. Invalid or absent rates
for owned work refuse placement rather than fabricate a price. These are normal
forward lifecycle actions, not an enablement of reconciliation enforcement.
Historical attempts without captured evidence are not retrospectively charged.

The alert route surfaced two such older proof attempts (`track-a-v2-proof`,
`tower-exclusive-smoke-20261017`). Their missing usage evidence stays visible as
a warning; open meters after terminal attempts remain a separate critical alert.
The old critical notification was not evidence of new customer traffic.

## Deployed follow-up and final gates

The final application runtime is `cb71106b`, image
`xcelsior-api:cutover-audit-cb71106b`. API, maintenance and scheduler handoffs
passed readiness and source checks: all three are healthy and active, with
23 verified source fingerprints each. Imported image Config, RootFS, creation
and platform match the frozen local image. All eight repair types remain
`report_only`; schema remains 125 and the twelve protected stopped-job row
hashes are unchanged. Selected protected volume metadata, zero attachments
and zero snapshots remain unchanged.

Both workers run the newly signed source SHA-256
`296bdd6fe71317abb04f7fc0112fa9dcae66da384f13e6726de5835d09da8e17`.
Twelve package signatures passed for each installation. Real GPU-free health
commands reached durable ACK with the corresponding session identity. Tower
serving container identity/PID and both VTT process identities were preserved.
The old worker mutation routes remain closed in active mode.

The final combined gate at `cb71106b` passed **623 tests**, with zero failures,
errors or skips. One optional Docker Compose smoke class was deliberately
excluded. Tests used a fresh local Unix-socket-only PostgreSQL cluster, without
live dotenv credentials or Tower access. The subsequent native monitoring
change separately passed production `promtool` validation for **16 rules and
12 fixture scenarios**. These gates do not constitute a production customer
purchase or GPU workload journey.

The native Alertmanager receiver has independent webhook and Telegram
integrations. Both accepted firing and resolved delivery. The webhook adapter
requires an upstream receipt before returning success; three receipt IDs are
retained in the evidence. Telegram transport reported acceptance with no
failures. Native Alertmanager 0.26 counters lack receiver labels: an initial
probe used an unsupported selector and timed out, then resolved its test.
The later verification used integration counters and dedicated adapter
receipts without resending. Historical failures on the other webhook receiver
are not attributed to this test. Human reading of the notification is not
claimed. The operator selected webhook delivery, so no email was sent to
`hello@xcelsior.ca`.

The real restore first failed closed before Alembic because host tools chose
5432 while the application URI selected 5434. Investigation also found stale
native unit paths after the script relocation. These are corrected, including
backup/manual-restore dotenv lookup and safe data-only configuration parsing.
The [port audit](2026-10-20-postgres-port-audit.md) distinguishes application,
SPIRE, test and example references.

A new checksummed backup of the application cluster on 5434 then restored into
a disposable database, validated migration head 125 and passed exclusive GPU
allocation, active-attempt reference, one-active-attempt and exact-wallet
projection checks. Restore duration was **23 seconds**. Both clusters were
checked for disposable-database cleanup, all fifteen earlier backups retained
their hashes, and both native timers are active and enabled. No live database
was restored over. The full regional/PITR/object recovery contract remains open.

A final monitoring check found that native node exporter had no textfile
collector directory. Its owned drop-in now enables the scripts' existing
metrics directory, preserving its existing arguments. Only the production
host exporter was restarted. Fresh backup/restore timestamps now reach native
Prometheus, with missing/stale/failed-since-success alerts routed through the
same webhook and Telegram receiver. Newer success clears historical failure;
missing metrics do not become a successful zero. Grafana and Tower monitoring
configuration were preserved.

Machine-readable deployment, delivery, test and recovery evidence is in
[the follow-up evidence](2026-10-20-control-plane-cutover-audit-evidence.json).
Raw test failures and frozen manifests remain under the local artifact
root; production drill journals and credential-bearing release/configuration
backups remain private on the production host. The six unrelated workspace files and four existing live
source files were preserved byte-for-byte.

## Remaining engineering work

- Dedicated runtime-role readiness still fails because the service-specific
  DSNs are absent. Shared-role production operation is not role isolation.
- B16.2 schema contraction, B16.3 physical removal of compatibility code and
  B16.4 durable fair-share/preemption remain open. Active-mode endpoint closure
  does not remove the projection trigger or build a durable preemption plan.
- Repair enforcement qualification remains open; no repair policy was enabled.
- Historical missing-volume recovery/disposition and proof-attempt usage
  evidence remain explicitly unresolved. The separate `xcelsior` database on
  5432 is preserved; no current application sessions were observed there. No data or monetary history was
  fabricated to clear an alert.
- Broader UI, trace/backend, HA/PITR, retrieval/warehouse and live product
  journeys retain their individual checklist gates. Apple Developer Program
  work and CI cockpit remain outside this request.
