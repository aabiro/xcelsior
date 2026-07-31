# Xcelsior MVP 2.0 Production Cutover Handoff

Last updated: 2026-07-30 (America/Toronto)

This is the authoritative restart document for the next agent. The user wants
the started Track B/control-plane/data-architecture work finished as one
production-ready MVP 2.0 checkpoint, then deployed and cut over without leaving
indefinite legacy/shadow modes.

## Progress since this handoff (2026-07-31)

Completed by the follow-on agent. Sections below are kept as written by the
original author; this block records what has since changed.

- **Item 1 — privacy sinks type error: FIXED.** The five `users` flags are
  INTEGER; the SQL booleans aborted anonymization so no identity was erased.
  Also found and fixed a second defect it was hiding: `casl_consent` and
  `user_encryption_keys` were created at runtime by `privacy.py`, outside the
  chain, so any migration-built database raised `UndefinedTable`. Migration
  `084` now owns them. Privacy sinks + workflow + endpoint coverage: 17 passed.
- **Item 2 — `082` draft: FINISHED.** It could never have run: the
  grandfathering backfill built a digest from `':migration-082'`, and
  SQLAlchemy `text()` read `:migration` as an unbound bind parameter. Fixed;
  rehearsed `081 -> 082 -> 081 -> 082` on a disposable database. Added
  `routes/host_admission.py` (7 endpoints, provider/operator split enforced at
  the routing layer), `db_roles` ownership, a production-only startup check for
  `XCELSIOR_COMPAT_SESSION_SECRET`, and 23 tests.
- **Migration head is now `084`** (082 host admission, 083 agent API keys,
  084 privacy tables).
- **Agent credentials replaced.** Quick Connect no longer mints ~1 KB JWTs held
  in the Redis auth cache; it issues `xcel_ai_` keys (51 chars) stored in
  Postgres as a SHA-256 digest, non-expiring and revocable, with `last_used_at`
  so the dashboard can tell a live key from an unused one.
- **`.env.test` was pinning `XCELSIOR_AGENT_PUBLIC_INGRESS=deny`**, running the
  whole suite against the retired public worker ingress and collecting 410s.
  The deny posture is production's and is already covered by
  `tests/test_hardened_runtime_and_ingress.py`, which monkeypatches both modes.

Still open from the original plan: the remaining Track B gaps in "Major
original-goal gaps that remain", personal-data export, warehouse deletion sink,
and observability deployment.

## Remaining work, with evidence (2026-07-31)

Suite is at **114 failed / 4411 passed** (was 235 failed when this pass began).
The remaining failures are almost entirely tests that encode contracts the
Phase 10 and 082 work deliberately retired — they describe the old system, not
defects in the new one. In priority order:

1. **Agent routes now require host authentication (~40 failures).**
   `tests/test_instance_flow.py`, `tests/test_api.py` and `tests/test_agent_v2.py`
   still post to `/agent/*` with no credentials and assert 200. The correct fix
   is the pattern already used in `tests/test_host_agent_tokens.py:409` — issue a
   host token via `control_plane.agent_tokens` and send
   `Authorization: Bearer <secret>`. Do not relax the auth to make these pass.
2. **`XCELSIOR_AGENT_HOST_TOKENS=require` with hosts that have no token.**
   Startup validation reports this against whichever database `.env` points at.
   In development that is one leftover fixture host (`h-skep-leg-750fcb`,
   registered 2026-07-19, one job attached), not a production condition.
3. **Money precision is half-migrated.** `wallets` carries both
   `auto_topup_amount_cad` / `auto_topup_threshold_cad` (float, what the code
   reads) and `auto_topup_amount_micros` / `auto_topup_threshold_micros`
   (integer, populated on 147 rows, read by nothing). Float is the wrong
   representation for money. Finish the cutover to `_micros` and drop the
   `_cad` pair; do not drop `_micros`, which is what migration `085`
   deliberately left alone.
4. **Still open from the original plan:** personal-data export (the settings
   button still downloads billing CSV only), the governed warehouse deletion
   sink, observability stack deployment, and the "Major original-goal gaps"
   section below.

Verified along the way and safe to rely on: migrations rehearse up/down/up
(`081->082->081->082` and `084->085->084->085`), the migration ledger head is
`085` and enforced by `tests/test_migration_ledger.py`, and production startup
validation now fails closed on both a missing compatibility-session secret and
a defaulted audit signing key.

## Read this first

**Do not deploy the current worktree.** It is a large, shared, uncommitted
checkpoint with two known incomplete areas:

1. the concrete privacy sinks still have a reproducible PostgreSQL type error;
2. migration `082` and `host_admission.py` are untested drafts and have no API
   routes.

Nothing in this branch has been committed, staged, pushed, or deployed. Preserve
all current files and review them in place; do not reset, checkout, or discard
changes.

The three governing documents are:

- `docs/track-b-implementation-checklist.md`
- `docs/xcelsior-production-control-plane-mcp-blueprint.md`
- `docs/xcelsior-production-data-architecture-companion.md`

The current branch is `codex/mvp2-production-cutover`. Its base and current
commit are both:

```text
417c16c5fe720c69aa9f55c8242fb7f6307799c3
```

`main` and `origin/main` also pointed to that commit when this handoff was
written. All MVP 2 work is therefore only in the dirty worktree.

## Current production truth

Production is the remote VPS `linuxuser@149.28.121.61`, deployed under
`/opt/xcelsior`. That directory is not a Git checkout; the deployed revision is
stored in `/opt/xcelsior/.deploy_hash`.

Verified immediately before this handoff:

```text
deploy hash: df03eb6f0d45989dc0cd0a2a30af1f65df5ab190
database revision: 079
running: api, bg-worker, frontend, jaeger, mcp-blue, scheduler-worker, ssh-gateway
XCELSIOR_SCHEDULER_MODE=shadow
XCELSIOR_AGENT_HOST_TOKENS=allow
XCELSIOR_TRUSTED_AGENT_GATEWAY is unset
XCELSIOR_AGENT_SHARED_BEARER_MIGRATION is unset
```

Reconciler action flags were not present in the scheduler container, so the
current defaults/report-only behavior must be treated as active until explicitly
verified.

Public probe state:

```text
https://xcelsior.ca/healthz   200
https://xcelsior.ca/livez     404
https://xcelsior.ca/readyz    404
https://xcelsior.ca/startupz  404
```

Internal API state on `127.0.0.1:9500`:

```text
/healthz   200
/livez     401
/readyz    503  (NFS volume storage degraded)
/startupz  401
```

The branch contains fixes for the auth/edge routing problem and for the NFS
container-boundary health check, but they are not deployed. The remaining
production NFS requirement is to expose a dedicated host SSH key to the
unprivileged API identity safely. Do not make the user's general home SSH key
world-readable. Prefer a dedicated `/opt/xcelsior/secrets/...` key, owned by
root and readable by the container's exact group, mounted read-only.

### Production backup state

The original cron backup had silently stopped after 2026-07-18 because
`scripts/backup-db.sh` sourced `.env`; a value containing spaces was executed as
a shell command.

The production safety repair is already installed and active:

- protected pre-change dump:
  `/var/backups/xcelsior/xcelsior_pre_mvp2_20260730_232146.dump`
- fresh hardened backup:
  `/var/backups/xcelsior/xcelsior_20260730_233412.dump`
- successful restore-drill evidence:
  `/var/backups/xcelsior/restore-evidence/20260730_233427.json`
- `xcelsior-db-backup.timer`: active, next daily run at 03:00 UTC
- `xcelsior-db-restore-drill.timer`: active, monthly first-Sunday drill
- both services last reported `success`
- the duplicate root/user cron jobs were removed only after the services passed
- previous crontabs were preserved in `/var/backups/xcelsior`
- backup/restore Prometheus textfile metrics exist under
  `/var/lib/node_exporter/textfile_collector`

The local branch contains the corresponding scripts/tests/systemd units:

- `scripts/backup-db.sh`
- `scripts/restore-db-drill.sh`
- `infra/systemd/`
- `tests/test_backup_workflow.py`

Still missing: encrypted off-host backups and true PostgreSQL PITR/WAL archiving.
Do not call local daily dumps a complete disaster-recovery design.

## Production data snapshot (re-query before acting)

This was the audit snapshot, not permission to delete:

```text
users: 108
wallets: 113
wallet transactions: about 40,372 and changing
jobs: 327
hosts: 4 (1 admitted, 3 pending)
job attempts: 1
usage meters: 5 (0 open at the snapshot)
billing cycles: 110,799
invoices: 200,702, mostly draft/zero historical rows
payout splits: 0
open reconciliation findings: 2
artifacts: 0
OAuth clients: 18
```

Other important facts:

- 86 of 108 users match synthetic/test patterns.
- There is effectively one paying customer.
- Wallet ledger total was exactly 10,167.1743 CAD at the snapshot.
- 22 payment intents existed: roughly $180 created and one $10 success.
- Four wallets had paid deposits.
- Ongoing volume charges were about 60-64 per hour, mostly synthetic/test
  volumes.
- One legacy job was running on `aaryn-tuf-rtx2060`.
- Open findings were an unmanaged `xcl-criu-demo3` container and a
  `billing_missing_meter` finding.
- Stripe meter delivery repeatedly logged a missing `gb_months` payload.

No customer/test-data rollover or deletion has been performed. Before any
cleanup:

1. take and verify another protected backup;
2. inventory exact user/payment/provider/customer identifiers;
3. produce a keep/delete mapping and financial control totals;
4. preserve the real paying account and all legally required payment records;
5. run the import/reconciliation against a disposable database;
6. only then perform the production rollover.

## Implemented work in the current worktree

### 1. Readiness, public probes, backup and restore

Implemented:

- `/livez` and `/startupz` added to public auth bypasses;
- Nginx routes all four probes and blocks public metrics;
- the NFS readiness check crosses the Docker host boundary instead of testing
  `/exports/volumes` inside the API container;
- backup parsing treats `.env` as data and never evaluates it;
- backup checksum/catalog verification, secure permissions, rotation, and
  metrics;
- disposable restore drill with Alembic-to-head and data invariants;
- hardened systemd services/timers.

Verification already run:

```text
tests/test_volumes.py + tests/test_startup_validation.py: 91 passed
tests/test_backup_workflow.py + startup tests: 28 passed
shell syntax, ShellCheck and systemd calendar checks: passed
manual production backup and restore services: passed
```

These application/Nginx fixes are not deployed.

### 2. Authoritative provider settlement (`080`)

Files include:

- `migrations/versions/080_authoritative_provider_settlement.py`
- `provider_settlement.py`
- `billing.py`
- `stripe_connect.py`
- `paypal_connect.py`
- `routes/billing.py`
- `routes/providers.py`
- related billing/PayPal/provider tests

Implemented:

- exact micro-CAD settlement math;
- one cross-rail settlement identity;
- PostgreSQL-derived job/customer/provider/currency/amount authority;
- durable `SKIP LOCKED` leases with expired-claim recovery and fenced
  completion;
- deterministic cent rounding with zero residual;
- Stripe/PayPal persistent idempotency;
- province tax tuple fix;
- no fallback payout inserts;
- claim/idempotency credentials removed from public responses;
- legacy rows and rail IDs preserved.

Verification from the settlement agent:

```text
110 targeted tests passed
080 -> 079 -> 080 rehearsal passed on an isolated test database
all 71 legacy payout rows preserved
all 20 existing PayPal capture IDs preserved
Ruff, compile and diff checks passed
```

Legacy frontend-supplied amount fields are ignored by the backend but still need
to be removed from the clients/contracts during the contract cleanup.

### 3. Provider registration and wizard hardening

Implemented:

- `/agent/versions` is compatibility-only and cannot admit/list hosts;
- manual and self-reported hosts enter `pending` atomically;
- both marketplace listing backends exclude unknown/non-admitted hosts;
- existing admitted hosts remain admitted across advisory self-reports;
- skipped/failed server verification cannot show success;
- installer fails closed without working NVIDIA detection and an advertised
  NVIDIA Docker runtime;
- wizard invocation is pinned to `@xcelsior-gpu/wizard@0.1.0`, not `@latest`;
- provider-facing copy says pending/unlisted.

Verification:

```text
backend provider/API/spot suite: 182 passed
integration suite: 27 passed
wizard suite: 563 passed
installer hardening: 7 passed
wizard build and changed frontend ESLint: passed
Python lint, shell syntax and diff checks: passed
```

This intentionally exposed the next blocker: without the unfinished `082`
workflow, no production path can admit a new host. Existing admitted capacity
continues to work; all new providers remain pending indefinitely.

### 4. Private observability stack

Implemented:

- OTel Collector, Prometheus, Alertmanager, Grafana, Loki, Tempo,
  postgres-exporter, node-exporter, and transitional Jaeger in
  `docker-compose.yml`;
- loopback-only published ports and private bridge networking;
- persistence, retention, resource/PID bounds, read-only roots, dropped
  capabilities, health checks;
- durable scheduler/reconciler/outbox/maintenance freshness heartbeats;
- queue age/depth, missing/open meters, stale leases/fences/observations,
  reconciliation, outbox, projection and task metrics;
- 13 recording rules and 32 alert rules covering the required failure modes;
- Grafana datasources/dashboard;
- public `/metrics` and `/metrics/*` return 404;
- the node exporter reads the exact backup/restore textfile directory;
- privacy environment variable pass-through is wired without values.

Files:

- `control_plane/operational_metrics.py`
- `infra/observability/**`
- `tests/test_observability_stack.py`
- changes in `routes/health.py`, worker loops, compose and Nginx

Verification from the observability agent:

```text
observability structural/signal tests: 16 passed
related metrics endpoint tests: 3 passed
bg-worker tests: 4 passed
promtool: config + 13 recording + 32 alert rules passed
amtool: default and routing example passed
OTel, Loki and Tempo native validation passed
Grafana 12.1 ephemeral provisioning/boot passed before the final small cleanup
docker compose config -q, Ruff and git diff --check passed
```

Not deployed and not started as one complete stack. Production prerequisites:

- a secret-managed real Alertmanager routing file; the checked-in default sends
  nowhere;
- Grafana admin/secret values;
- PostgreSQL exporter credentials;
- the node-exporter textfile directory;
- rerun the final Grafana ephemeral boot after the last config cleanup;
- exercise a synthetic alert end-to-end.

### 5. Durable privacy deletion (`081`) — incomplete

Files:

- `migrations/versions/081_privacy_deletion_workflow.py`
- `privacy_deletion.py`
- `privacy_sinks.py`
- `routes/privacy.py`
- `routes/auth.py`
- `privacy.py`
- settings/API frontend changes
- privacy workflow/sink tests

Implemented in the core:

- one active/idempotent request per pseudonymous subject;
- keyed subject references and one-time status tokens;
- per-sink status/evidence/deadline;
- expiring worker leases, `SKIP LOCKED`, fencing and retry;
- deadline failure cannot be reported as success;
- artifact legal hold is a visible terminal result, not silent deletion;
- account deletion returns `202` with a tracking receipt;
- failed retention purges remain due and retryable;
- the incorrect `usage_meters.customer_id` purge was fixed to `owner`;
- request/completion audit-outbox contracts;
- the UI stores the tracking receipt before logout.

Passing tests:

```text
privacy, endpoint, startup, event-contract and workflow groups: 107 passed
workflow-only concurrency/idempotency/deadline group: 7 passed
```

Known reproducible failure at handoff:

```text
.venv/bin/python -m pytest tests/test_privacy_sinks.py -q --tb=short
2 failed

privacy_sinks.py:
users.notifications_enabled is INTEGER, but the anonymization UPDATE assigns
the SQL boolean false.
```

The same `users` table uses integer flags for:

- `notifications_enabled`
- `canada_only_routing`
- `mfa_enabled`
- `email_verified`
- `is_admin`

Change those assignments to `0`/`1` as appropriate, rerun the concrete sink
tests, and continue fixing any subsequent schema errors rather than weakening
the assertions. `wallets.auto_topup_enabled` and `casl_consent.active` are real
booleans.

Other privacy work still required:

- exercise every concrete sink, including artifact/volume retry and PostHog
  asynchronous verification;
- implement the governed warehouse deletion sink before enabling a warehouse;
- complete a real personal-data export; the current settings button still
  downloads only billing CSV despite saying “Export Data”;
- add a polished public tracking-status view using the stored request/token;
- define financial/legal retention evidence and operator escalation;
- add deadline/backlog metrics and alerts;
- rehearse `081` up/down/up on a disposable production-shaped snapshot;
- scrub any failed-test fixtures with the `privacy-sink-` prefix from
  `xcelsior_pytest` only. Never run that cleanup against production.

Both the normal local database and `xcelsior_pytest` were at revision `081`
when this handoff was written.

### 6. Authoritative host admission (`082`) — untested draft

The provider-admission agent was stopped immediately for this handoff.

Only these files belong to the draft:

- `migrations/versions/082_authoritative_host_admission.py`
- `host_admission.py`

Draft intent:

- normalized host admission state/version;
- grandfather existing explicitly admitted hosts;
- expiring compatibility sessions with Ed25519 proof-of-possession;
- advisory provider evidence vs authoritative verifier/operator evidence;
- durable evidence and decisions;
- row-locked idempotent admit/reject/revoke operations;
- atomic host state/marketplace/outbox updates;
- redacted and hashed hardware evidence.

Current status:

- Ruff happens to pass, but no runtime or migration test was performed;
- `082` was not applied to any database;
- no router/API endpoints call the service;
- no wizard/desktop integration;
- legacy verification endpoints are not integrated with this authority;
- scheduler health/pending-state compatibility is not finished;
- new tables are not assigned in `control_plane/db_roles.py`;
- migration ledger/head/bootstrap files still declare `079`;
- `XCELSIOR_COMPAT_SESSION_SECRET` is not in startup validation or
  `.env.example`;
- no tests or documentation/checklist updates.

**Do not apply or deploy `082` until all of the above is implemented and its
migration is rehearsed.**

## Migration ledger state

Files currently form this draft chain:

```text
079 (production)
  -> 080 authoritative provider settlement
  -> 081 privacy deletion workflow
  -> 082 authoritative host admission draft (repository head only)
```

However:

```text
tests/test_migration_ledger.py EXPECTED_HEAD = "079"
migrations/README.md says repository head = 079
local databases are at 081
production is at 079
082 has never been applied
```

Finish/review `082` first, then update every head/bootstrap/role registry in one
coherent change. Required migration gates include:

- `alembic heads` returns exactly one head;
- empty-database upgrade to head;
- production-shaped `079 -> head`;
- downgrade/re-upgrade on a disposable clone only;
- row/control-total fingerprint before and after;
- table-domain/role registry coverage;
- no production downgrade.

## Major original-goal gaps that remain

Even after the code above is stabilized, the full user request is not complete:

- scheduler is still shadow in production;
- reconciler destructive actions are not enforced;
- per-host tokens are `allow`, not `require`;
- trusted private agent ingress and the complete SPIRE mesh are not live;
- dedicated least-privilege database roles are not cut over per service;
- hardened worker/service identity rollout is incomplete;
- B6 operator/customer UI is not complete;
- secure desktop/helper compatibility UX is not connected to admission;
- B9.2 artifact deletion still lacks full per-replica generation
  preconditions/robust `SKIP LOCKED` claiming/tombstone proof;
- B11 warehouse/governed analytics is not implemented;
- B12.1 versioned cache consistency remains open;
- B13 infrastructure-as-code/Canadian managed data services remain open;
- production PostgreSQL/Redis are single-host, not HA;
- off-host encrypted backup and PITR are missing;
- legacy columns/contracts and zero-use evidence are incomplete;
- test/synthetic data rollover has not happened;
- the full UI, E2E, chaos, restore/replay and post-cutover proof has not run.

## Ordered continuation plan

### Step 1 — Stabilize this worktree

1. Read this document and all three source-of-truth documents.
2. Run `git status --short --branch` and `git diff --check`.
3. Review changes by subsystem; do not mix fixes blindly across the shared
   diff.
4. Fix `tests/test_privacy_sinks.py` first and make the concrete authority and
   retrieval tests pass.
5. Clean only synthetic failed-test rows from `xcelsior_pytest`.
6. Run the full focused privacy suite.

### Step 2 — Finish or replace the `082` draft

1. Review every SQL statement and trust-boundary decision.
2. Add role/domain registry ownership.
3. Add authenticated operator and compatibility-session routes.
4. Keep browser/helper reports advisory; only an authoritative verifier or
   authorized operator may admit.
5. Integrate existing verification endpoints so no second admission authority
   remains.
6. Add session expiry, replay, signature, concurrency, tenant isolation,
   reject/revoke and grandfathering tests.
7. Integrate the wizard/desktop helper without allowing arbitrary command
   execution.
8. Rehearse `081 -> 082 -> 081 -> 082` on a disposable clone.
9. Update migration ledger/head/bootstrap/docs only after it is green.

### Step 3 — Close the remaining implementation gaps

1. Finish personal-data export and deletion status UI.
2. Harden the artifact deletion worker and prove hold/retention/replica
   semantics.
3. Remove inert frontend settlement inputs and legacy backend contracts.
4. Complete provider onboarding/admission UX.
5. Finish service roles, SPIRE/private ingress, host tokens and worker
   identity.
6. Finish Track B UI and data-source-of-truth items, updating the checklist with
   evidence rather than optimistic checkmarks.

### Step 4 — Run comprehensive local/staging gates

At minimum:

```bash
git diff --check
.venv/bin/ruff check .
.venv/bin/alembic heads
.venv/bin/dotenv -f .env.test run -- .venv/bin/alembic current

.venv/bin/python -m pytest \
  tests/test_privacy_deletion_workflow.py \
  tests/test_privacy_sinks.py \
  tests/test_provider_settlement.py \
  tests/test_observability_stack.py \
  tests/test_backup_workflow.py \
  tests/test_startup_validation.py \
  tests/test_volumes.py -q --tb=short

cd frontend
npm run lint
npm run test
npm run build

cd ../wizard
npm run test
npm run build

cd ..
docker compose config -q
```

Then run the repository-wide Python suite serially and the migration-from-empty
and production-snapshot gates. Do not call the branch ready based only on
targeted green tests.

### Step 5 — Production preflight

Before changing production:

1. re-audit deploy hash, migration, services, flags, queue, running jobs,
   reconciliation, outbox, wallet/invoice/payout control totals;
2. take a new protected backup and complete a restore drill;
3. establish encrypted off-host backup/PITR or document and explicitly accept
   the residual risk;
4. prepare secret-managed values for privacy, PostHog deletion, compatibility
   sessions, Grafana, exporters, Alertmanager, SPIRE and per-host tokens;
5. stage the dedicated NFS host key for the container identity;
6. prepare an exact real-account/payment preservation and test-data rollover
   report;
7. build immutable images once and deploy those tested digests, not an
   untracked source rebuild;
8. prepare rollback commands and schema compatibility bounds.

### Step 6 — Deploy one known checkpoint

The intended end state is one coherent maintenance checkpoint, not weeks of
dual authority. It still needs explicit gates:

1. stop/finish the one real running workload or preserve it deliberately;
2. deploy code/config/images;
3. run migrations once as the migration identity;
4. start the private observability plane;
5. verify `/healthz`, `/livez`, `/readyz`, `/startupz` all return 200 through
   the public edge;
6. verify alerts, logs, traces, metrics, backup and restore signals;
7. perform the reconciled data rollover;
8. require trusted ingress/per-host identity and activate scheduler/reconciler
   in the same planned cutover window only after their precomputed gates pass;
9. keep a tested rollback available until the observation window closes.

### Step 7 — Post-deploy proof

Prove, do not infer:

- customer registration/login/session/logout;
- deposit, wallet hard stop, invoice, terminal meter, exact provider
  settlement and replay;
- provider compatibility, verification, admission, marketplace visibility and
  first job;
- scheduler placement, lease/fence, agent command/ACK and reconciliation;
- artifact upload/download/delete/hold;
- privacy deletion with a held artifact and all other sinks complete;
- alerts for API/worker outage, queue backlog, meter invariant, stale
  lease/fence, database/backup, outbox/projector and privacy deadline;
- backup restore and replay;
- no real-account/payment loss;
- no legacy-path usage for the agreed evidence window.

Only after this should legacy columns/contracts be dropped and the Track B
checklist marked complete.

## Safety and operational notes

- Keep secrets out of chat, logs, Git and test output.
- Production `.env` contains values with spaces; never source or eval it.
- Hide/override the local `.env` during local smoke tests when it contains
  production-only SSH/NFS settings.
- `/opt/xcelsior` is not a Git repository; use `.deploy_hash` for deployed
  identity.
- Do not run Alembic downgrade in production.
- Do not delete synthetic-looking production rows until the paying customer,
  payment rails and exact wallet control totals have been positively mapped.
- Do not flip scheduler/reconciler/identity flags simply because the code
  exists; each flag must have its data, identity, observability and rollback
  prerequisite proven in the same maintenance plan.
- The old instruction in the previous handoff to leave `bg_worker.py` alone is
  obsolete. It now contains intentional observability and privacy-worker
  changes; preserve and review them.

## Worktree ownership map

Use this to review the shared diff:

- backup/readiness: `scripts/backup-db.sh`,
  `scripts/restore-db-drill.sh`, `infra/systemd/`, `volumes.py`,
  `routes/_deps.py`, probe Nginx/tests;
- settlement: `provider_settlement.py`, migration `080`, billing/Stripe/PayPal
  routes and tests;
- provider hardening: agent/host/marketplace routes, installer, wizard,
  provider UI/tests;
- observability: `control_plane/operational_metrics.py`,
  `infra/observability/`, compose, health metrics, worker heartbeats/tests;
- privacy: migration `081`, `privacy_deletion.py`, `privacy_sinks.py`,
  privacy/auth routes, settings/API frontend/tests;
- host admission draft: migration `082`, `host_admission.py`.

No subsystem has been globally regression-tested together after all parallel
edits. That integration pass is the next agent's first release-readiness gate.
