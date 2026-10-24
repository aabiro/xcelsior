# Control-plane and data-layer audit, 2026-10-24

Candidate runtime `e41f180dc26b287d2f922485bc4d9ff53b8bd17d` contains the
control-plane review fixes in `2e175630`, atomic billing and encrypted execution
delivery in `7e51a2d0`, and serverless interval fencing in `e41f180d`.
Production remains on schema 125 and `xcelsior-api:cutover-audit-cb71106b`.
This receipt qualifies a source/image candidate; the ordered schema 126–137,
service-role and worker handoff is still pending. Repairs remain `report_only`.

## Findings fixed in this candidate

| Defect and impact | Resulting behavior and regression proof |
|---|---|
| Submission could survive a durable-outbox failure and fall back to a process-local event. | Admission and its outbox commit together; exact replay repairs the event from the immutable admission. `test_control_plane_production_readiness.py`. |
| Stale host updates could reopen administrative state; preemption cleanup could cancel a plan from a stale observation. | Host upserts preserve native administration, and preemption rereads the locked state. `test_control_plane_production_readiness.py`, `test_durable_preemption.py`. |
| Optional GPU coordinator errors could be mistaken for absence. | Only an explicit absence response or proven connection refusal permits the absent-coordinator path. Uncertain outcomes retain the journal/flock and require compensation before execution or release. `test_worker_optional_coordination.py`. |
| Runtime reports and proxy routing could rely on mutable payload identity/ports; auto-launch credentials were predictable. | Reports bind host, attempt, fence and live lease to admitted ports/services. Proxy routes use current native authority. Auto-launch secrets are random. `test_runtime_callback_fencing.py`, `test_auto_launch.py`, `test_port_proxy.py`. |
| Endpoint credentials were decrypted before persisting admission and durable commands; decryption errors became empty strings. | Admission and commands keep ciphertext. The authenticated, non-cacheable lease claim decrypts the hash-validated admitted environment in the same transaction. Failed decryption rolls back the claim; incomplete grants abort and settle before a container starts. Endpoint plaintext never enters the worker journal. `test_native_environment_delivery.py`. |
| GPU or volume debits could commit while the cycle transaction rolled back. | Debit, ledger, notice and cycle use one connection/transaction. Delivery and optional Stripe projection happen after commit. `test_data_layer_readiness.py`, `test_control_plane_review_regressions.py`. |
| Generic compute billing accepted incomplete native authority, mutable observations/live prices, and missed completion without an API stop. | Exact admitted quotes and native start/end/lease bounds authorize slices. A final native settlement tick bills once, survives overlapping ticks and retries after funding changes. Failed cycles do not advance the paid cursor. `test_data_layer_readiness.py`, `test_gpu_billing_writes_its_cycle_row.py`. |
| Serverless billing could reconstruct customer rates, accept a corrupt allocation or future slice, and duplicate provider usage in invoices. | Positive customer rates are frozen at admission; exact worker/job/spec/attempt/lease evidence bounds the allocated interval. Allocation includes startup but must fall between native reservation and start. Future ends are capped at server time; nonfinite intervals, expired periodic leases and another resource type cannot authorize a debit. Customer invoices count serverless cycles once. `test_serverless_billing_authority.py`, `test_serverless_chaos_billing.py`. |
| Backup lock cleanup could remove another run's lock; database tools could default to the wrong cluster; manual restore could drop before checking its archive or quiescing every configured writer. | Backups use an owned file lock and private partial dump. An explicit valid port is required. Restore checks checksum/catalog, stops API/API-blue/scheduler/maintenance, verifies quiescence and leaves the writers stopped for the staged handoff. `test_data_layer_readiness.py`. |
| Worker callback aliases and generated classification had drifted; old tests used unpriced or unclaimed capacity. | Gateway aliases and inventory identify the actual origin capability. Billing fixtures now reserve, claim, start and retain native timing/quote evidence. `test_agent_v2_aliases_do_not_double_count.py`, public/MCP OpenAPI checks and billing harnesses. |

The test-only follow-up `c1cd32f0` supplies explicit calculator interval
evidence and a real paid cursor. Its two changed files are under `tests/`;
every runtime file remains identical to the frozen image source.

## Verification

| Gate | Result and scope |
|---|---|
| `clean-control-data-expanded` | **3,430 passed**, zero failures/errors/skips, three explicit baseline exclusions. Frozen test commit `c1cd32f0`; every runtime file matches `e41f180d`. Owned PostgreSQL 16 and network-disabled Redis cleanup passed. |
| Financial, secret and role regressions | `financial-race-followup`: 17 passed; `native-secret-delivery`: 85 passed; `serverless-bounds-and-roles-final`: 58 passed; calculator repair: 29 passed. Includes actual restricted maintenance login, rollback, concurrent settlement and secret-failure cases. |
| Generated contracts | 12 passed. Endpoint inventory, public/MCP OpenAPI, aliases and generated TypeScript agree. |
| Static/distribution checks | Black/Ruff pass on all 50 changed Python files; targeted runtime Pyright has zero errors/warnings; three shell scripts pass `bash -n`; twelve worker distribution signatures verify. |
| MCP client/server | API generation, TypeScript check and build pass; source-only suite **293 passed, two skipped**. |
| Frozen images | Three actual Gunicorn images booted on PostgreSQL 16 at schema/floor 126/126, 136/134 and 137/137. Each verified twelve signatures and exported metrics availability equal to one. UID 10001, read-only root, no network/capabilities; cleanup passed. |
| CUDA worker protocol | Five real, bounded CUDA journeys passed: successful/failed container exit, failed terminal report before/after commit and journal restart recovery. Proves protocol/observation/meter/invoice behavior; excludes public launch, `run_job` provisioning, wallet debit and provider payment acceptance. |
| Fresh all-schema restore | One complete snapshot restore passed, followed by ordered targets 126-137. All 142 logical-table fingerprints match and 38 protected tables remain unchanged at every target; disposable database/cluster cleanup passed. |

Selections overlap. The broad run retains three explicit baseline exclusions:
the Docker Compose smoke class, the historical contract-cleanup migration-order
test, and the conformance-prose test whose companion document is absent. These
are not passing evidence. The MCP skips are the opt-in real-stack end-to-end
case and the knowledge-URL-dependent case. Existing duplicate GET/HEAD operation
ID warnings on static installer/file routes remain visible in raw schema
generation; those routes are outside the published public/MCP API contract.

Failed reproduction runs, fixture/probe invocation errors and the interrupted
superseded broad run remain retained as diagnostics. They are not counted as
qualification. The final broad gate uses the frozen runtime above. Test-created
model binaries are kept outside Git, and the user's nine protected root edits
retain their recorded hashes.

## Read-only production data review

The configured application target was verified as database `xcelsior`, port
**5434**. The population scan covered **164 relations, 1,984 columns and 22
partitions**, with no scan errors. Null, blank, zero and empty-JSON counts are
observations; they do not establish that a row/column should be deleted.

The integrity pass checked **35 foreign keys** with zero orphans. It found no
unvalidated constraints, invalid indexes, disabled triggers, unlogged application
tables, active lease/attempt mismatch or duplicate exclusive device allocation.
Wallet balances exactly reconcile with the signed ledger. Both negative wallets
are suspended. There was no live native attempt/lease/device allocation in the
baseline, all nine dedicated service roles were `NOLOGIN`, and all eight repair
policies were `report_only`.

Three historical overlapping charged **volume** intervals remain on one job,
dated 2026-04-07. The affected cycle amounts total **300 micro-CAD**; overlap
duration is about 60–63 seconds. These rows are retained for financial
disposition. This aggregate does not by itself establish a refund amount.

A sensitive-key aggregate across endpoint environments, job specs/observations
and command specs found zero nonempty matching values in the live baseline.
This limits the observed exposure; the execution-secret fix is still required
for future provisioning.

The latest nightly backup was independently checksummed on the source host:
`xcelsior_20261024_030023.dump`, **52,393,435 bytes**, mode **0600**, SHA-256
`79cbf7cc449910bec137138f4e5c88aed7747deb1b37552e03bb7b7cb60824f6`.
Fifteen backups were present. This checksum check is separate from the restored
snapshot below.

A fresh **read-only exported snapshot at 2026-10-24 06:52:17 UTC** was
shared with an all-schema `pg_dump`. The private 53,435,995-byte
archive and snapshot metadata remain mode 0600 outside Git. Dump SHA-256:
`b05da641ffc4a8638d871488845f43ecbca17e2379e7f0e5c5b9c3f93023e4f6`.

An isolated PostgreSQL 16 restore with matching `C.UTF-8` collation verified
full-row fingerprints and counts for all **142 logical tables**: 138 public
and four storage tables. Partitioned parents include their 22 leaves. The
restored logical table inventory also matches exactly. Ordered migrations
126-137 preserved **38 protected tables at every target**: 30 financial/
marketplace tables, four execution/storage-history tables and four tables in
the storage schema. All 364 jobs remain: four exact recorded specs recovered,
360 explicitly retained as unavailable. The qualification audit and archive
checks passed. No unrecorded execution specification was invented.

All 140 migration Python files match the earlier qualified source as well as
the current runtime. The fresh restore runs against the current migration
source and leaves the runner's separate database at its original head.

Production durability settings have `fsync`, `full_page_writes` and
`synchronous_commit` enabled. `archive_mode` and page checksums are off.
The restore proof establishes logical backup recovery; point-in-time recovery,
off-site/object recovery and failover acceptance remain open. No production
database, policy, service role, application or worker handoff was performed.

## Frozen release and remaining acceptance

The final image is `xcelsior-api:cutover-audited-e41f180d`, ID
`sha256:e50da92fb4698b9a60fc747ff0aaaea383a2f5b7d5028558db28c8bc2b987f04`. It is built from 3,078 tracked files with runtime
environment files and private signing keys excluded. Its runtime floor is 137.

`audited-handoff-images.tar.gz` contains this image plus the required transition
and intermediate images: **896,788,408 bytes**, SHA-256
`e6960bf144edee46c11165392d682c52fdfc4b504c6d58c521ce7270471f1cb3`. Independent inspection hashes every
archived blob and matches each image's Config, RootFS layers and platform to the frozen
local image. The archive has not been transferred to production.

The [staged handoff](2026-10-23-staged-control-plane-handoff.md) names the
required transition image at 126, intermediate image at 134–136, and final image
at 137. Normal deploy cannot skip these writer/schema boundaries. Source and
image gates do not establish production ingress, current-worker provisioning,
customer wallet/provider settlement or deployed fairness/preemption acceptance.

B16 remains open for that handoff and acceptance, the three baseline test gaps,
historical financial disposition and operational recovery gates. Observation/API
projection adapters, the old spot-preemption 501 surface and scheduler/worker
module decomposition remain explicit source residuals. The maintenance billing
task still uses the API role because it crosses billing, control and notification
domains; dedicated logins alone do not remove that domain-coupling debt.

The machine-readable [receipt](2026-10-24-control-plane-data-layer-audit.json)
records aggregate results and artifact hashes. Raw logs, private dump/snapshot
metadata, build contexts and the image archive remain under
`artifacts/control-plane-production-readiness-20261023/`, outside Git.
