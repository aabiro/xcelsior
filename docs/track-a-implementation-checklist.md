# Track A — Transactional Core & Concurrency Safeguards: implementation checklist

Derived from `docs/xcelsior-production-control-plane-mcp-blueprint.md`
(§10, §13, §26, Phases 1–5). Each step lists its exit gate; do not start a
step until the previous gate is green. Status reflects the repo at the time
of the last edit to this file.

**Companion documents:**
- [`xcelsior-production-control-plane-mcp-blueprint.md`](./xcelsior-production-control-plane-mcp-blueprint.md)
  — the intended control-plane architecture this checklist implements
  (the §-references throughout point here).
- [`xcelsior-production-data-architecture-companion.md`](./xcelsior-production-data-architecture-companion.md)
  — the production **data-model** companion: per-table ownership,
  database-enforced invariants, retention, and the polyglot-persistence
  boundaries (PostgreSQL authority vs. object storage / Redis / pgvector /
  BigQuery projections). Read it for *where each fact lives and why*; this
  checklist tracks *what has been built*. The Track A tables landed by
  migrations 054–059 (attempts, allocations, fenced leases, commands,
  outbox, observations, reconciliation) are the transactional-authority
  core that companion describes.
- [`site-audit-report-2026-07-18-track-a-rollout.md`](./site-audit-report-2026-07-18-track-a-rollout.md)
  — live production verification of this rollout (shadow sign-off, v2
  fenced worker, rollback drill).

## A1 — Schema evolution (expand phase)

- [x] **A1.1 Migration 054** — `migrations/versions/054_control_plane_core_columns.py`
  - jobs: tenancy (`tenant_id`/`team_id`/`owner_id`), `desired_state`, `phase`,
    reason fields, `generation`/`observed_generation`/`version`,
    `active_attempt_id`, `spec`/`spec_hash`, queue-ordering columns,
    schedule-claim columns, `wallet_hold_id`.
  - hosts: tenancy/region, `administrative_state`/`availability_state`,
    generations/version, `inventory_generation`, observation fields,
    drain fields, `capabilities`/`conditions`.
  - CHECK constraints added `NOT VALID` → `VALIDATE`; batched
    `SKIP LOCKED` backfill with hard verification (abort on unmapped rows).
  - Queue/claim-expiry/tenant/admission indexes.
  - Gate: fresh + production-like DB migrate to head; up→down→up cycle clean. ✔
- [x] **A1.2 Migration 055** — `migrations/versions/055_attempts_allocations_fenced_leases.py`
  - `placement_fencing_token_seq` (global monotonic fence authority).
  - `job_attempts` (+ `uq_job_one_active_attempt` partial unique).
  - `host_gpu_devices` physical inventory (MIG children, stable UUID,
    `uq_host_gpu_uuid` lock-ordering index).
  - `gpu_device_allocations` (+ `uq_gpu_one_exclusive_allocation`,
    release-shape CHECK). Legacy marketplace `gpu_allocations` untouched.
  - `placement_leases` (offered/active/released/expired/fenced,
    `uq_attempt_one_active_lease`, active-shape CHECK, expiry index).
  - `jobs.active_attempt_id` FK (`NOT VALID` → `VALIDATE`, `ON DELETE SET NULL`).
  - Transitional backfill of *active* legacy leases into attempt/lease rows.
  - Gate: schema invariant tests (`tests/test_control_plane_schema.py`) pass. ✔
- [x] **A1.3 Migration 056** — `migrations/versions/056_durable_control_work.py`
  - `agent_commands` evolved in place (expand-only): `command_id`,
    job/attempt/fence/spec-hash refs, priority/not-before, claim
    owner/session/expiry, retry budget (`attempt_count`/`max_attempts`/
    `next_attempt_at`), `idempotency_key` (+ `uq_command_idempotency`
    partial unique), ACK/result/error fields, trace + retention; §9.4
    status CHECK (`pending|claimed|acknowledged|failed|dead_letter|cancelled`)
    and claimed-shape CHECK. v1 drain path untouched.
  - `outbox_events` (unique `(destination_class, idempotency_key)`,
    dispatcher partial index), `api_idempotency_keys` (unique
    `(principal, tenant, route, key)`), `reconciliation_queue`
    (PK-coalesced per resource), `reconciliation_findings`,
    `scheduled_tasks` (durable periodic claims).
  - Gate: up→down→up clean on dev; applied to test; invariant tests in
    `tests/test_control_plane_schema.py` pass; v1 agent_commands
    consumers (agent lifecycle/endpoints, instance flow, bg_worker
    reconcile) pass. ✔
- [x] **A1.4 Migration 057** — `migrations/versions/057_observations_telemetry.py`
  - `host_observations` (immutable per host/session/generation; API
    receipt time is authoritative freshness), `observed_workloads`
    (attempt/fence-keyed container reality), `telemetry_latest`
    (per host/GPU upsert — replaces process-local latest telemetry),
    partitioned `telemetry_samples` (monthly + DEFAULT safety partition;
    `telemetry_partition_maintenance` scheduled task seeded),
    `service_heartbeats` (replica liveness + schema revision).
  - Gate: up→down→up clean; schema tests pass. ✔
- [x] **A1.5 Schema compatibility checks** — `control_plane/schema_compat.py`:
  declared min (057) / optional max revision, env-overridable;
  `assert_schema_compatible`. Tests cover min/max/non-numeric/
  missing-alembic cases. **Readiness wiring landed (2026-07-18):**
  `/readyz` (`routes/health.py`) now runs the gate and 503s when the DB
  is outside the supported range (blueprint ADR-009/§13.8; data-arch
  companion §4.4 rule 2); postgres-only, kill switch
  `XCELSIOR_READYZ_SCHEMA_CHECK`, ready payload reports the resolved
  `{current, minimum, maximum}`. 3 readyz tests. ✔
- [x] **A1.6 Stop runtime DDL drift — first target (from-empty bootstrap)**
  (2026-07-18): pure `alembic upgrade head` from an empty database now
  reaches head. Root cause was `agent_commands` (created only by
  `db._ensure_pg_tables()` runtime DDL while 056 `ALTER`s it). Fix:
  migration 056 starts with `CREATE TABLE IF NOT EXISTS agent_commands`
  (same expand-only pattern as 030/`gpu_pricing`); single ordered path
  `scripts/bootstrap_pg_from_empty.sh` (alembic → optional ensure/seed);
  CI `control-plane` + `test` jobs bootstrap via that path and no longer
  restore `ci-cache/pg_schema.sql`. Gate:
  `tests/test_from_empty_bootstrap.py` (static CREATE-before-ALTER +
  real empty-DB bootstrap ×2). ✔
- [x] **A1.6 tail — no production runtime DDL** (2026-07-19): migration
  `061_residual_runtime_ddl.py` absorbs residual ensure-only objects
  (`job_logs`, `oauth_clients`/`oauth_refresh_tokens`, `team_invites`,
  `users` concurrency/email-change columns, `billing_cycles.token_cost_cad`
  /`model_ref`). `_ensure_pg_tables` is seed-only on Alembic-managed DBs
  (`gpu_pricing` rows only; refuses CREATE/ALTER). `_ensure_oauth_auth_tables`
  is a pure no-op when migrated. Gate: `tests/test_no_runtime_ddl.py`
  (static inventory + instrumented no-DDL startup + residual tables present)
  + from-empty head `061`. ✔

## A2 — Transactional placement (claim → filter → score → reserve → bind)

- [x] **A2.1 Repository layer** — `control_plane/db.py`:
  `control_plane_transaction()` (SET LOCAL statement/lock timeouts,
  commit/rollback envelope, `AmbiguousCommitError` on commit-time
  connection loss — never blindly retried), `run_transaction()` (bounded
  full-jitter retry of `40001`/`40P01`/pre-commit connection errors only),
  `stable_advisory_key()` + `try_advisory_xact_lock()` (§2.5 stable
  sha256-derived key, transaction-scoped, pool/PgBouncer-safe).
  Gate: `tests/test_control_plane_db.py` (13 tests: commit/rollback,
  timeout locality, retry classes, budget, ambiguity, lock exclusion/
  release/reentrancy) pass; pyright clean. ✔
- [x] **A2.2 Stage B queue claim** — `control_plane/scheduler/claim.py`:
  §10.2 claim CTE verbatim (priority DESC, fair-share ASC, queued FIFO,
  `FOR UPDATE SKIP LOCKED`), token CAS release with durable reason +
  bounded backoff, expired-claim sweep. 12 tests incl. two-connection
  SKIP LOCKED exclusivity. ✔
- [x] **A2.3 Stage C/D filters & scoring** —
  `control_plane/scheduler/filters.py` (7 pure versioned hard filters,
  typed `FilterReason`s, aggregate queue-reason payload) and
  `scoring.py` (integer fixed-point components, sha256 tie-break).
  12 tests incl. Hypothesis permutation-invariance properties. ✔
- [x] **A2.4 Stage E reservation transaction** —
  `control_plane/scheduler/reservation.py`: job→host→devices canonical
  lock order, full revalidation, fence from sequence, attempt +
  exclusive device allocations + lease offer + durable start command +
  outbox intents + job projection in one commit; typed
  `ReservationConflict` hierarchy (never retried; transient SQLSTATEs
  bubble to `run_transaction`). 9 tests: atomicity, zero-residue
  conflicts, multi-GPU all-or-nothing, two-claimer race. ✔
- [x] **A2.5 Route all writers through it** — machinery landed dark
  (see Phase 4 section below): projection triggers make every legacy
  writer maintain the 054 columns; the canary partition + authoritative
  tick own queued→assigned for scoped jobs when mode=canary/active.
  All four legacy queue walkers (`process_queue`, `_binpack`, `_filtered`,
  sovereign) skip `_control_plane_owns_job` exactly as claim SQL scopes;
  under `active` legacy walkers assign nothing. Inline API/serverless/
  inference `process_queue()` calls remain as wake paths but are
  partition-safe (owned work is never dual-written). **Operator residual:**
  enable canary/active envs on live prod (P4.4b env flip). Gate:
  `tests/test_scheduler_placement_partition.py`. ✔
- [x] **A2.6 Concurrency proof** — `tests/test_control_plane_concurrency.py`:
  8 spawn-isolated replica processes race 30 jobs over 8 exclusive GPU
  slots; asserts exactly 8 placements, zero double-allocated devices,
  zero multi-attempt jobs, complete attempt/lease/command chain per
  placement, durable `no_capacity` reason on the rest. ✔ (Crash-injection
  variants can extend this harness in Phase 3.)

## Phase 3 — Shadow mode (new pipeline vs legacy comparison)

- [x] **P3.1 Migration 058** — `migrations/versions/058_scheduler_shadow_decisions.py`:
  `scheduler_shadow_decisions` (outcome/comparison shape CHECKs, uncompared
  partial index, per-job/cycle/retention indexes). Expand-only; nothing but
  the shadow runner touches it. Gate: up→down→up clean on dev; applied to
  dev + test. ✔
- [x] **P3.2 Shadow runner** — `control_plane/scheduler/`:
  `config.py` (`XCELSIOR_SCHEDULER_MODE` paused|shadow|canary|active +
  typed env settings), `snapshot.py` (one REPEATABLE READ read of queued
  jobs/fleet/capacity; reads *legacy* truth — runtime writers don't
  maintain the 054 projection columns yet, so ordering is legacy
  `priority DESC, submitted_at ASC` until A2.5), `explain.py` (bounded
  §3.2 explanation for every decision AND non-decision), `shadow.py`
  (ShadowRunner: batch-drained comparator → snapshot → pure
  `simulate_cycle` with in-memory capacity charging → decision persist →
  retention prune; typed comparison classes match_place/match_queue/
  host_mismatch/shadow_placed_legacy_queued/legacy_placed_shadow_queued/
  job_missing/indeterminate; `summarize_comparisons` mismatch-rate
  rollup), `main.py` (standalone replica entrypoint + schema guard).
  Wired: `scheduler.start_shadow_runner()` daemon thread in
  `scheduler_main` (no-op unless mode=shadow + postgres + 058),
  docker-compose env passthrough, `.env.example` block.
  Gate: 8 tests in `tests/test_control_plane_shadow.py` (persisted
  explanations, capacity charging, full comparator matrix, zero
  legacy-table mutation, grace deferral, retention); full control-plane
  suite 128 green; pyright clean. ✔
- [x] **P3.3 Shadow sign-off (Phase 3 exit gate)** — enable
  `XCELSIOR_SCHEDULER_MODE=shadow` in production (deploy applies 058
  automatically), let it run against real traffic, review
  `summarize_comparisons` mismatch classes until reasons are understood
  and signed off. Production review on 2026-07-18: 142 comparisons over
  the 24-hour window, 140 `match_queue`, two
  `legacy_placed_shadow_queued`, zero unexplained mismatches (98.59%
  literal match rate). Both mismatches were successive snapshots of the
  same RTX 2060 job while the only matching GPU was occupied by the v2
  proof attempt; legacy assigned it only after that attempt stopped, so
  the difference is comparison-window timing rather than policy/capacity
  disagreement. Signed off for the current single-GPU canary scope. ✔

## Phase 4 — Transactional scheduler cutover (dark until canary enabled)

- [x] **P4.1 Migration 059 — runtime projection triggers**:
  `BEFORE INSERT OR UPDATE` triggers derive jobs
  `phase`/`desired_state`/`effective_priority`/`queued_at` and hosts
  `administrative_state`/`availability_state` from legacy truth on
  *every* write — covering all ~15 raw `UPDATE jobs SET status` sites
  (billing, reaper, agent routes) that bypass the upsert. Host admission
  now honors the payload `admitted` flag legacy allocation actually
  gates on (054's status-only rule was too loose); `pending` added to
  the CHECK. Drift backfill for rows written since 054. Transitional:
  dropped at contract phase when writers are projection-native (note:
  the trigger clobbers any manual `effective_priority` boost — fairness
  aging must land after its removal). Gate: up→down→up clean; schema
  tests updated (trigger normalizes instead of CHECK raising). ✔
- [x] **P4.2 GPU inventory bootstrap** — `control_plane/inventory.py`:
  projects host payload (`gpu_count`/`gpu_model`/`total_vram_gb`) into
  `host_gpu_devices` rows with stable synthetic `slot:{i}` identities;
  locked in canonical (host_id, gpu_uuid) order; never retires a device
  under an active allocation; bumps `inventory_generation` on change so
  stale scores fail §10.5 revalidation. Real NVML UUIDs replace these
  when the Track B observation pipeline lands. ✔
- [x] **P4.3 Authoritative tick + canary partition + kill switch** —
  `control_plane/scheduler/service.py`: maintenance sweeps (expired
  schedule claims, stale leases, command redelivery) + inventory sync +
  bounded claim→filter→score→reserve loop, walking ranked candidates on
  `ReservationConflict`, releasing claims with durable reasons/backoff.
  Partition is exclusive by construction: claim SQL scope predicate
  (gpu_model ∈ canary set, or payload `{"scheduler": "v2"}` opt-in)
  selects exactly the jobs `SchedulerConfig.owns_job` matches, and all
  four legacy queue walkers (`process_queue`, `_filtered`,
  `_binpack`, jurisdiction) skip that same set; `process_assigned`
  skips attempt-owned jobs (agent start command owns their containers,
  no SSH double-start). Kill switch:
  `XCELSIOR_SCHEDULER_CLAIMS_ENABLED=false` stops claims, sweeps
  continue. Wired into `scheduler_tick`; env/compose passthrough.
  Gate: 11 tests in `tests/test_control_plane_service.py` (trigger
  projections incl. raw-SQL writes, inventory reconcile, partition
  semantics, end-to-end tick placement with attempt/lease/command/
  allocation + explanation, kill switch, durable release, paused no-op);
  full control-plane suite 142 green; legacy scheduler 91 green;
  pyright clean. ✔
- [x] **P4.4a Legacy failover/reaper interlock** — attempt-owned jobs
  (`active_attempt_id IS NOT NULL`; the lease-expiry sweep clears it on
  requeue) are now refused by `requeue_job` (guards ALL legacy failover
  entry points: dead-host failover, legacy `event_store` lease expiry,
  admin requeue) and skipped by the reaper's candidate SELECTs + CAS
  UPDATE. The v2 lease sweep owns that failure class end-to-end
  (attempt fails/lost, allocations released once, durable reason,
  higher fence on retry). User relaunch of a v2 job stays blocked until
  the Phase 5 attempt-termination flow. 2 interlock tests. ✔
- [x] **P4.4b Canary enablement + widen + retire legacy path** (code
  cutover readiness, 2026-07-19; gated on P3.3 shadow sign-off): legacy
  walkers exclusive-skip the transactional ownership partition; under
  `active` they assign nothing; `process_assigned` skips attempt-owned
  jobs and **fails closed** if ownership lookup fails (no SSH double-start
  of fenced work). Host-pin / marketplace direct launch
  (`routes/instances._direct_host_launch_or_defer`) defers owned jobs —
  no inline `assigned` + `run_job`; records `preferred_host_id` and wakes
  the transactional tick. Durable command → lease claim remains start
  authority. Gate: `tests/test_scheduler_placement_partition.py` (partition
  exclusivity, all walkers, host-pin defer, fail-closed SSH skip) +
  service suite. **Operator residual (not claimed by code):** set
  `XCELSIOR_SCHEDULER_MODE=canary` +
  `XCELSIOR_SCHEDULER_CANARY_GPU_MODELS`/`_HOSTS` on live prod, then
  widen → active. Note: billing/serverless raw status writes on *running*
  v2 jobs remain unfenced until fully on `/agent/v2` — keep canary scope
  on fresh workloads until then. ✔

## A3 — Worker lease & fencing engine

- [x] **A3.1 Lease service** — `control_plane/leases.py`:
  `claim_lease`/`renew_lease`/`release_lease` are CAS updates against the
  exact `job+attempt+host+lease+fence` tuple on DB time; rejections are
  typed and diagnosed (`wrong host`, `fence mismatch`, `offer expired`,
  `not claimable`). ✔
- [x] **A3.2 Status fencing** — `require_current_fence`: the §8.1 write
  gate (attempt must be the job's active attempt, on that host, with
  that fence, in an active status) raising `FencingViolation`. Wired
  into worker routes at v2 cutover. ✔
- [x] **A3.3 Lease expiry via fencing** — `expire_stale_leases` sweep:
  offered-past-deadline → attempt `failed` (`lease_claim_timeout`);
  active-past-renewal+grace → attempt `lost`; allocations released
  exactly once, undelivered start command cancelled, job requeued with
  durable reason; retry mints a higher fence and the old tuple fails the
  fence gate (proven in tests). 15 lease tests. ✔
- [x] **A3.4 `worker_agent.py` hard gate (Track B boundary)** — Phase 5
  (see section below): `handle_start_attempt` claims the lease FIRST and
  aborts with a non-retryable NACK on rejection (container never starts);
  every attempt status report carries the full job/attempt/host/fence
  tuple; renewal rejection or a fenced status response = definitive
  authority loss → `docker kill`. Fence on *telemetry* calls remains a
  Phase 5 follow-up (telemetry is diagnostic, not a state write). ✔

## Phase 5 — /agent/v2 fenced worker protocol

- [x] **P5.1 Fenced attempt status service** — `control_plane/attempts.py`:
  `report_attempt_status` applies one worker report inside a txn —
  §8.1 fence gate first, forward-only §9.2 transitions (idempotent
  repeats OK, backward = `out_of_order`), timestamps, and terminal
  settlement (allocations released once, lease released, job projection
  + `active_attempt_id` cleared, outbox event) — all atomic. ✔
- [x] **P5.2 `/agent/v2` routes** — `routes/agent_v2.py` (wired into
  `routes/__init__.py`): `negotiate/{host_id}` (rollout gate:
  `XCELSIOR_AGENT_V2_HOSTS` csv or `*`), `commands/claim` (claim+ACK
  delivery, attempt-bound commands only), `commands/{id}/ack|nack`
  (once-only ACK with result replay; typed NACK → backoff or
  dead-letter), `leases/claim|renew|release` (claim = the §11.2 hard
  gate), `attempts/status` (fenced reports; 409 `fencing_violation` =
  stop your container). Error bodies follow the app's
  `{ok, error:{code}}` contract. ✔
- [x] **P5.3 v1/v2 delivery partition** — v1 destructive drain
  (`GET /agent/commands/{host}`) now excludes attempt-bound commands
  (`attempt_id IS NULL` on GC + claim-delete): a legacy agent can no
  longer destroy a v2 start command it doesn't understand. v2 claim
  takes only attempt-bound commands. Proven by test. ✔
- [x] **P5.4 Worker v2 client + hard gate** — `worker_agent.py`:
  startup protocol negotiation (v1 unless server enrolls the host),
  per-process `WORKER_SESSION_ID`, claim+ACK drain loop dispatching
  `start_attempt` (unknown commands NACK non-retryable). Order enforced:
  lease claim FIRST (abort on rejection — no container), then
  `lease_claimed` report, ACK (execution began; job outlives command
  claim TTL), renewal thread, then the existing `run_job` machinery.
  The v1 `report_job_status` funnel mirrors every transition onto the
  fenced attempt (single hook covers all of run_job's exit points);
  fenced response or renewal rejection → `docker kill` (§11.5), with a
  disconnected-grace window equal to the renewal TTL. ✔
  Gate: 11 HTTP-level tests (`tests/test_agent_v2.py` — negotiation,
  drain partition, ack replay, dead-letter, full lifecycle to terminal
  settlement, wrong-fence everywhere, renewal-after-release, out-of-
  order, failure recording) + 8 worker unit tests
  (`tests/test_worker_agent_v2.py` — gate abort, ack-after-grant,
  malformed NACK, status mirror, fence-loss kill, renewal 409 kill).
  Affected suites 282 green; pyright clean.
- [~] **P5.5 Remaining Phase 5 hardening** — atomic local authority and
  command-result journal now survives restart, records pending terminal
  intent, and replays ACKs without repeating Docker side effects; restart
  adoption requires an exact renewal (transport uncertainty gets one
  bounded lease window, then label-verified stop). v2 containers now use
  attempt-specific names plus job/attempt/fence/spec-hash/managed labels;
  start commands verify the canonical spec hash before lease claim and do
  not re-enter the v1 lease loop. Fenced host observations are live (P6.1).
  Legacy PATCH writes require the exact tuple and the central scheduler
  writer independently rejects unattributed attempt-owned mutations under
  row lock. Billing stop and suspended-wallet paths enqueue a durable
  attempt command. **Terminate/cancel lifecycle controller landed
  (2026-07-18):** `control_plane/lifecycle.py` records
  `lifecycle_intent` + intermediate `stopping`, enqueues fenced
  `stop_attempt` with `preserve=False` (worker stop+rm); ACK projection
  maps intent → `terminated`/`cancelled` (no pre-ACK volume detach).
  Wired through `BillingEngine.terminate_instance` and
  `POST /instances/{id}/cancel`. Gate: `tests/test_fenced_lifecycle_controller.py`
  + updated stop-enqueue tests. Required NFS/encrypted volumes, gVisor,
  and image signatures fail closed at start. **Fresh-attempt resume/
  restart landed (2026-07-18):** `request_fresh_attempt_resume` requeues
  stopped fenced-history jobs (no `start_container`); running restart
  enqueues fenced `stop_attempt` intent=`restart` and ACK projects to
  `queued` for a new attempt. Wired through `start_instance` /
  `restart_instance`. Gate: `tests/test_fresh_attempt_resume_restart.py`.
  **Diagnostic telemetry auth (2026-07-19):** `POST /agent/telemetry`
  already calls `_require_agent_auth(request, host_id=payload.host_id)`
  (production never bypasses; test/ALLOW_UNAUTH only in non-prod). Gate:
  `tests/test_telemetry_auth.py` (real route unauth→401, authed→200,
  spoof host not written; structural gate wiring). ✔
  **Remaining:** deploy/roll out the hardened agent.

## Phase 6 — Observations & reconciler (report-only)

- [x] **P6.1 Observation ingest** — `control_plane/observations.py`:
  immutable full-state snapshots per (host, session, observation
  generation) with at-least-once duplicate collapse; freshens
  `hosts.last_observed_at` (DB receipt time authoritative §12.2);
  PK-coalesced host enqueue into `reconciliation_queue`; 3-day
  retention prune. Route: `POST /agent/v2/observations`. Worker:
  `report_observations_v2` (docker ps over `xcl-*` with attempt/fence
  labels, 60s throttle in the main loop, v2 hosts only). ✔
- [x] **P6.2 Reconciler (report-only)** — `control_plane/reconcile.py`:
  desired (active attempts joined to current job authority) vs observed
  (latest snapshot) per host; deduplicated findings with
  auto-resolution: `attempt_container_missing` (warning, grace window
  for young attempts), `stale_fence_container` (error — the §11.5 kill
  backstop), `unmanaged_workload` (info). Queue processing (`process_due`:
  claim SKIP LOCKED, settle-on-success/backoff-on-error) wired into the
  scheduler service sweeps. Nothing is auto-remediated — actions enable
  per finding type once prod false-positive rates are known (blueprint
  Phase 6 rollout rule). Gate: 9 tests (ingest/dup, route, finding
  matrix incl. dedupe + auto-resolve + grace, queue settle, retention,
  worker docker-ps parsing); 249-test regression green; pyright clean. ✔
- [x] **P6.3a Reconciler action framework + first enforced action** —
  `control_plane/reconcile.py`: `ActionPolicy`/`action_policy_for` resolve
  per-finding-type remediation from
  `XCELSIOR_RECONCILE_ACTION_<TYPE>` (default report-only; only types in
  `_ENFORCEABLE` can be turned on — an operator cannot enable an action
  that doesn't exist). First enforced type `stale_fence_container`:
  `_enqueue_stop_container` writes a durable, idempotency-keyed
  `stop_container` command (attempt_id NULL → delivered by the v1 drain,
  since a revoked fence has no valid authority) inside the reconcile
  transaction, and records `action_taken`/`action_result` on the finding.
  Fires once per occurrence (finding dedupe gates re-action); the
  §11.5 fence-loss kill remains the primary layer, this is the backstop.
  Env passthrough in docker-compose + `.env.example`. Gate: 4 new tests
  (policy resolution incl. fail-safe, report-only default, enforce
  enqueues-once + records action, v1-drain visibility); reconcile suite
  13 green; pyright clean. ✔
- [x] **P6.3b Missing-container action** — second enforceable finding
  type `attempt_container_missing` (default report-only). When enabled,
  the reconciler does not settle the attempt itself: it expedites the
  attempt's active-lease expiry (`_expedite_lease_expiry` stamps
  `expires_at` into the past), so the next `expire_stale_leases` sweep
  performs the one tested terminal settlement (attempt→lost, allocations
  released once, start command cancelled, job requeued, higher fence on
  retry). Single authority for attempt failure (§12); catches the
  "zombie" case the lease-deadline sweep never sees (container gone,
  worker still renewing). Action recorded on the finding; fires once per
  occurrence. 2 tests (report-only leaves lease untouched; enforce
  expedites + lease controller settles). ✔
- [x] **P6.3c Reconciler orphan handling + remaining controllers** —
  orphan-allocation handling landed (2026-07-18): third enforceable
  finding `orphaned_allocation` — a `gpu_device_allocations` row left
  `active` after its attempt reached a terminal state (§8.2 capacity leak
  from a crashed/partial settlement). The per-host reconcile pass detects
  it (DB-internal, observation-independent) and, when enabled, releases
  it (`release_reason='reconciler_orphan'`) so the device is schedulable
  again; auto-resolves once cleared. All three enforceable reconciler
  actions now have docker-compose/`.env.example` passthrough. 3 tests.
  **Stuck-job reaper + VRAM capacity domain cutover (2026-07-19):**
  `control_plane/stuck_jobs.fail_stuck_legacy_job` fails legacy stuck jobs
  only via `scheduler.update_job_status` (CAS `expected_status`, fence
  gate, durable outbox); `reaper.reaper_tick` is a thin invoker that
  still excludes `active_attempt_id IS NOT NULL`. Host free-VRAM repair
  stays sole-authority in `scheduler.reconcile_host_vram` with pure
  policy in `control_plane/capacity.py` (no job lifecycle writes). Gate:
  `tests/test_stuck_job_reaper.py`, `tests/test_host_vram_reconcile.py`. ✔
  **Billing lifecycle dual-writer cutover (2026-07-19):**
  `BillingEngine.stop_instance` routes attempt-owned work through
  `control_plane.lifecycle.request_fenced_stop` (atomic intent +
  preserve stop_attempt); no pre-mark raw SQL dual-write. Legacy stop/
  terminate use guarded `update_job_status` (CAS + outbox). `get_job`
  always stamps `job_id` from PK so upsert cannot silently no-op.
  Gate: `tests/test_billing_lifecycle_domain.py` + billing stop enqueue +
  fenced lifecycle suites. ✔
  **bg_worker stopped-job stop redelivery fence (2026-07-19):**
  `bg_worker.reconcile_paused_stopped_jobs` re-enqueues unfenced
  `stop_container` only for pure-legacy `stopped` jobs. Attempt-owned
  (`active_attempt_id IS NOT NULL`) and fenced-history
  (`EXISTS job_attempts`) jobs are excluded at SQL and again by pure
  `is_fenced_history_job` (same classification as billing start/restart).
  Legacy path keeps throttle, max-attempt, and pending-command dedupe.
  Gate: `tests/test_stopped_job_stop_redelivery.py` (real PG both classes)
  + `tests/test_bg_worker_reconcile.py`. ✔
  **Residual job-scoped host/command fence (2026-07-19):** shared
  `control_plane/job_targets.py` resolves residual container identity
  (`xcl-{job}-{attempt[:8]}` for attempt-owned; legacy `xcl-{job}`;
  fenced-history without live authority refuses unfenced guess). Gated
  sites: SSH reinject, admin reinject, reset, snapshot API +
  registry-down retry, volume hot mount/unmount (incl. wait poll).
  Commands remain on the v1 unfenced drain (worker handlers are
  v1-only) but never target the wrong attempt container. Host-scoped
  `upgrade_agent` / `rollback_agent` / serverless `prepull_image` stay
  host-level. Gate: `tests/test_residual_host_command_sites.py` (real
  PG attempt-owned vs legacy vs fenced-history + static host inventory). ✔
  **Remaining residual (not claimed):** operator agent deploy and live
  canary/active env flips (P5.5). Serverless prepull and unrelated
  process-local producers remain out of scope.
  **Durable scheduled tasks landed (2026-07-19):** `control_plane/scheduled_tasks.py` 
  provides the `scheduled_tasks` table executor (`claim_and_run_tasks`) replacing the 
  process-local background timers in `bg_worker.py` with `SKIP LOCKED` durable 
  execution and database-tracked run states. ✔

## A4 — Durable outbox & command ACK protocol

- [x] **A4.1 Command claim/ACK** — `control_plane/commands.py`:
  `claim_commands` (pending → claimed, SKIP LOCKED, priority order,
  not-before/backoff aware), `ack_command` (once-only terminal ACK;
  duplicate ACK replays the stored result; wrong-host rejected),
  `nack_command` (typed failure → bounded-backoff requeue or
  dead-letter), `redeliver_expired_claims` sweep. 11 tests. The v1
  `DELETE ... RETURNING` drain in `routes/agent.py` is retired at the
  `/agent/v2` cutover (Phase 5), which calls these services. ✔
- [x] **A4.2 Outbox writer** — `control_plane/outbox.py::append_event`:
  idempotent (destination, key) append inside the caller's transaction;
  rollback-atomicity proven in tests. Reservation and lease-expiry paths
  already write through it. ✔
- [x] **A4.3 Outbox dispatcher** — `claim_batch` (SKIP LOCKED, claim
  TTL) + `mark_published`/`mark_failed` (exponential backoff →
  dead-letter) + `OutboxDispatcher.run_once` (per-event settlement,
  handler isolation, unroutable-destination logging). 4 tests incl.
  rival-dispatcher exclusivity and crash-redelivery semantics. ✔
- [x] **A4.4 Migrate side-effect producers (Phase 7)** —
  `control_plane/outbox_runtime.py`: dispatcher runtime (claim →
  deliver → settle loop with backlog drain, retention prune — published
  7d, dead-lettered 14d) started as a scheduler-worker thread
  (`XCELSIOR_OUTBOX_DISPATCHER`, default on). Handlers: `default` maps
  `job.v1.placement_reserved` / `attempt_status_changed` /
  `lease_expired` / **`legacy_status_changed`** to the dashboard's SSE
  vocabulary and publishes on the *existing* `xcelsior_events` NOTIFY
  channel (every API replica already bridges it to its local SSE clients
  via `db.start_pg_listen`; the ssh gateway listens too — zero new
  plumbing, and v2 placements become visible on dashboards for the first
  time); `agent_wake` is a logged no-op until a push channel lands.
  Unknown event types settle silently (forward compatible). All v2
  transactional producers (reservation, lease expiry, fenced attempt
  status) flow end-to-end: state commit → outbox row → NOTIFY → SSE.
  **Legacy `update_job_status` (2026-07-19):** appends
  `job.v1.legacy_status_changed` in the same atomic mutation as the job
  row (Postgres); skips process-local-only `emit_event` when the outbox
  intent is durable — multi-API-replica clients see the same events via
  the dispatcher. Gate: `tests/test_legacy_status_outbox.py` (row
  presence + real LISTEN delivery). ✔
  **Host/job lifecycle residual outbox (2026-07-19):** shared
  `try_append_lifecycle_outbox` (SAVEPOINT-isolated) + SSE projections
  for `host.v1.status_changed` / `host.v1.removed` /
  `job.v1.submitted` / `job.v1.preempted`. Wired on
  `register_host`, `set_host_draining`, `update_host_spot_settings`,
  `remove_host`, `submit_job`, `preempt_job` — same dual-fan-out
  avoidance as legacy status (skip process-local `emit_event` when
  outbox is durable). Gate: `tests/test_host_job_lifecycle_outbox.py`
  (real PG row presence + dispatcher LISTEN for submit/drain). ✔
  **Queue-block + spot-price residual outbox (2026-07-19):**
  `_persist_queue_reason` writes job row + `job.v1.queue_blocked`
  (SSE `job_error`) in one txn before notify; `process_queue_binpack`
  skip path uses that path and skips dual emit when durable.
  `update_spot_prices` appends `pricing.v1.spot_prices_updated`
  (SSE `spot_prices`) after history persist, with bounded payload for
  NOTIFY. Gate: `tests/test_scheduler_residual_outbox.py` (real PG +
  LISTEN for both types). ✔
  Remaining (not this gate, partially closed below): webhook bulk
  intents; residual request-path `broadcast_sse` outside instance
  lifecycle. Operator canary/active and agent deploy unclaimed
  (P5.5).
  **Attempt-scoped usage meters (2026-07-20):** expand-only migration
  `062_usage_meters_attempt_id` adds nullable `usage_meters.attempt_id`
  + partial unique `uq_usage_meters_one_per_attempt`.
  `BillingEngine.meter_job` resolves attempt authority via
  `resolve_meter_attempt_id` (explicit keys → active_attempt_id → latest
  job_attempts), stamps the meter, and is idempotent under re-close of
  the same attempt (no second billable row). Pure-legacy jobs still
  meter with NULL attempt_id. Single INSERT path remains
  `BillingEngine.meter_job`. Gate:
  `tests/test_attempt_scoped_usage_meters.py` (real PG stamp +
  idempotent re-close + legacy + inventory). ✔
  **Concurrency-safe wallet holds (2026-07-20):** migration
  `063_wallet_holds` adds durable `wallet_holds` + job FK. Available
  balance = ledger − active held (non-expired). Launch preflight
  (`routes.instances._wallet_preflight`) creates a hold under wallet
  `FOR UPDATE`; job submit links `jobs.wallet_hold_id`; terminal
  `update_job_status` releases once (idempotent). Concurrent dual-hold
  race admits only one when funds cover a single hold. Gate:
  `tests/test_wallet_holds.py` (real PG race + lifecycle + preflight +
  inventory). ✔
  **Start/restart available-balance + hold expiry (2026-07-20):**
  `wallet_has_available_funds` gates `start_instance` /
  `restart_instance` and API start/restart (ledger positive but fully
  held fails closed). `expire_stale_wallet_holds` CAS held→expired
  (SKIP LOCKED); durable task `wallet_hold_expiry` every 60s in
  `bg_worker`. Gate:
  `tests/test_wallet_hold_expiry_and_start_gate.py` (real PG gate +
  expiry idempotency + inventory). ✔
  **Event-store per-stream hash chain (2026-07-20):**
  `EventStore.append` no longer takes `LOCK TABLE events IN EXCLUSIVE
  MODE`. Chains are scoped to `(entity_type, entity_id)` with
  transaction-scoped `pg_advisory_xact_lock` on
  `stable_advisory_key("events_stream", stream_id)`. Cross-stream
  appends do not contend on a global exclusive table lock; same-stream
  concurrent appends still serialize correctly. Under the stream lock,
  append assigns a causal `timestamp` strictly after the stream head
  (pre-lock timestamps discarded) so head SELECT / `verify_chain`
  order match `prev_hash` link order. `verify_chain` validates each
  stream independently (optional entity filters). Callers
  (`append_user_audit_event`, volume lifecycle `_emit_event`, job
  state machine) remain on the single append path. Gate:
  `tests/test_event_stream_hash_chain.py` (real PG concurrent multi-
  stream + same-stream skew + inverted pre-lock timestamps + broken-
  link + structural no table exclusive lock) + `tests/test_events.py`. ✔
  **Request-path instance lifecycle SSE (2026-07-20):**
  `routes.instances._broadcast_instance_lifecycle_sse` appends durable
  outbox via `enqueue_lifecycle_sse_outbox` (short txn) for
  `job.v1.instance_stopped` / `_started` / `_restarted` /
  `_terminated` / `job.v1.cancelled`; skips process-local
  `broadcast_sse` when durable; falls back only on enqueue failure.
  Wired on `api_stop` / `api_start` / `api_restart` / `api_terminate`
  and both fenced + legacy cancel paths. Dispatcher projections map
  to the historical SSE vocabulary (`instance_*`, `job_cancelled`).
  Gate: `tests/test_request_path_instance_lifecycle_sse.py` (real PG
  row + dual-emit skip + LISTEN for stop/cancel + structural route
  inventory). Remaining process-local request-path sites (hosts
  dual-emit residual, volumes, teams, billing wallet UI, agent
  telemetry, user images, lock/reset, job_log) and webhook bulk
  intents stay unclaimed. ✔


## Cross-cutting gates

- [x] Fix full-suite cross-test contamination (`tests/test_bitcoin.py`
  rebinding `XCELSIOR_DB_BACKEND` to sqlite process-wide — 90 downstream
  failures).
- [x] Serverless advisory-lock fix — `ServerlessRepo.reconcile_lock`
  context manager pins ONE pooled connection for the whole reconcile
  pass and unlocks on that exact session (the old two-checkout
  try/release pair leaked the session lock — a leaked instance was found
  live in the test DB during verification). Per-endpoint
  `pg_try_advisory_xact_lock` (primitives ready in `control_plane/db.py`)
  lands with the Phase 6 per-endpoint reconcile refactor. 4 regression
  tests + serverless suite green. ✔
- [x] CI: `control-plane` job in `.github/workflows/ci.yml` — dedicated
  postgres service; bootstraps via deterministic from-empty path
  (`scripts/ci_bootstrap_pg_schema.sh` → `bootstrap_pg_from_empty.sh` /
  pure `alembic upgrade head`; schema-dump cache removed after A1.6 first
  target), then a **migration-reversibility gate** (`downgrade 053` →
  `upgrade head`) + the 8-replica concurrency stress. Runs alongside the
  full-suite `test` job. ✔
- [x] Local test-DB isolation (2026-07-18) — root-caused an intermittent
  `no_eligible_host` flake: the always-on `xcelsior-test` docker stack
  (scheduler + bg-worker + serverless autoscaler) shared `xcelsior_test`
  with the local pytest suite. Its legacy scheduler assigned leftover
  queued jobs onto pytest fixture hosts and its autoscaler generated
  ~440 endpoints' worth of `serverless-*` scale-up jobs. Fix: pytest now
  runs against a dedicated `xcelsior_pytest` DB (`.env.test`, local),
  created/cloned/stamped by `scripts/setup_pytest_db.sh`; the docker
  stack keeps `xcelsior_test`. Previously-flaky placement suites now
  deterministic across repeated runs. (CI is unaffected — it already
  uses an ephemeral per-job postgres.) ✔


## Phase 8 — Database-Backed Object Storage Catalog & Janitor

- [x] **8.1 Reworked API Routes** — `routes/artifacts.py`:
  Refactored the core `/api/artifacts/upload`, `/api/artifacts/download`, and `/api/artifacts` (list) endpoints to interact directly with the PostgreSQL catalog. Retained S3 listing fallback if DB-backed storage is inactive. Added `/api/artifacts/finalize` (POST) to finalize upload sessions and mark artifacts as available.
- [x] **8.2 State Machine & Catalog Integration** — `artifacts.py`:
  Integrated schema tables and upload state machine (`requested` -> `available`). Enforced metadata persistence (ETag, size, replicas) on finalization. Implemented standalone user upload support with NULL job_id to satisfy relational schema invariants.
- [x] **8.3 Expired Sessions & Orphan GC Background Task** — `bg_worker.py`:
  Added a background `artifact_catalog_janitor` task that runs periodically (every 60s) claiming and cleaning up uncompleted/expired `storage.artifact_upload_sessions` (state `requested` older than 1 hour) and transitioning them to `abandoned`. Also processes durable asynchronous `storage.artifact_deletion_jobs`.
- [x] **8.4 Verification & Testing**:
  Authored comprehensive integration suites `tests/test_artifacts_state_machine.py` and `tests/test_artifacts_janitor.py` simulating full upload-to-download, finalization, expired upload session sweeps, and physical file deletion. All tests pass successfully. ✔


## Phase 9 — Control Plane and MCP UI

- [x] **9.1 Control Plane Backend Admin API Endpoints** — `routes/admin.py`:
  Added 6 highly robust administrative API endpoints to expose the reconciler's open and resolved findings (`GET /api/admin/reconciler/findings`), manual finding dismissal (`POST /api/admin/reconciler/findings/{finding_id}/dismiss`), manual finding enforcement (`POST /api/admin/reconciler/findings/{finding_id}/enforce`), manual host-scoped reconciliation passes (`POST /api/admin/reconciler/reconcile-host/{host_id}`), durable scheduled tasks listing (`GET /api/admin/control-plane/scheduled-tasks`), and transactional active jobs timeline (`GET /api/admin/control-plane/jobs`). Integrated `psycopg.rows.dict_row` to ensure all queries are fully compatible with string-keyed dictionaries.
- [x] **9.2 Control Plane Admin UI Page** — `frontend/src/app/(dashboard)/dashboard/admin/control-plane/page.tsx`:
  Created a beautiful, state-of-the-art administrative page using Client-side React and Lucide icons. Implemented separate high-fidelity sections/tabs for "Scheduler Timelines", "Host Drains & Capacity", "Reconciler Findings", and "Durable Scheduled Tasks". Displays real-time CPU/GPU/VRAM telemetry, transaction-scoped reconciliation findings with interactive action triggers (Drain, Undrain, Dismiss, Enforce, Reconcile Host), and countdowns.
- [x] **9.3 Navigation Link integration** — `frontend/src/app/(dashboard)/dashboard/admin/admin-shell.tsx`:
  Registered "Control Plane" tab into the primary sub-navigation shell of the admin dashboard.
- [x] **9.4 Integration Tests and Coverage** — `tests/test_admin_endpoints_coverage.py`:
  Authored detailed FastAPI test cases (`test_admin_control_plane_endpoints`) covering all new control-plane admin endpoints. Verified findings listing, jobs attempts, scheduled tasks, and host reconciliation passes. All 17 administrative test cases pass successfully. ✔


## Phase 10 — Identity, privilege, and edge hardening

- [x] **10.1 Fail-closed agent identity admission** — `control_plane/identity.py` +
  `routes/agent._require_agent_auth`: production maps `host_id` to a
  registered+admitted host; DB lookup errors return **503** (never
  fail-open). Optional `XCELSIOR_TRUSTED_AGENT_GATEWAY=1` requires
  gateway-set headers and rejects public-injected `X-Worker-*` alone.
  Untrusted identity header strip helper for edge configs. Gate:
  `tests/test_phase10_identity_privilege.py` + telemetry/auth regression. ✔
- [x] **10.2 API privilege drop** — `docker-compose.yml` `api` / `api-blue`:
  removed `cap_add: SYS_ADMIN`; `security_opt: no-new-privileges:true`.
  LUKS remains host-SSH / volume-provisioner. Gate: structural compose
  test (no `- SYS_ADMIN` / no `cap_add` on API blocks). ✔
- [x] **10.3 Ingress separation** — `nginx/mcp-xcelsior.conf` (mcp.xcelsior.ca,
  strips worker identity headers) and `nginx/agent-xcelsior.conf`
  (agent.xcelsior.ca, mTLS client verify, sets trusted gateway headers).
  Gate: structural config tests. ✔
- [x] **10.4 MCP image hygiene** — `mcp/Dockerfile`: locked `npm ci` only
  (no `npm install` fallback), non-root `USER xcelsior`. ✔
- [x] **10.5 SPIRE / provisioner scaffolding (not live mesh claim)** —
  `infra/spire/*` examples + README (fail-closed without SPIRE),
  `infra/envoy/agent-gateway.yaml`, `infra/volume-provisioner/*`.
  Live SPIRE multi-node deploy remains operator residual. ✔

**Phase 10 code-audit fixes (2026-07-21):** gateway identity no longer
trusts client-settable `X-Xcelsior-Agent-Gateway` alone — requires
`XCELSIOR_AGENT_GATEWAY_SECRET` (`X-Xcelsior-Gateway-Auth`); public
`nginx/xcelsior.conf` strips worker identity headers on `/agent/` and
`/host`; API strips untrusted identity headers unless the gateway secret
authenticates. Gate: `tests/test_phase10_identity_privilege.py`.

**Phase 10 residual closeout (2026-07-21, proven in-repo):**
- [x] API/bg-worker/scheduler: no `/exports` mount; `XCELSIOR_VOLUME_PRIVILEGE=host_ssh`
  default forces LUKS over host-SSH; optional `volume-provisioner` profile owns
  `/exports` + SYS_ADMIN. Gate: `tests/test_phase10_residuals.py`.
- [x] Shared fleet bearer: production host mutations reject `api-token` unless
  `XCELSIOR_AGENT_SHARED_BEARER_MIGRATION=1` (+ optional host CSV). Gate: identity
  helpers + `_require_agent_auth` route test.
- [x] MCP Redis-backed rate limit (`mcp/src/rate-limit.ts`): multi-replica INCR;
  fail-closed 503 when backend required/unavailable (never unlimited). Gate:
  `mcp/src/rate-limit.test.ts` (vitest).
- [x] Public nginx strips forgeable gateway headers; agent conf injects secret
  after mTLS (prior + residual tests).

**Still residual (not claimed):** live SPIRE multi-node attestation; Envoy SDS
production; API non-root/read-only image; full public cutover off `/agent/` to
agent.xcelsior.ca only (bearer migration path remains stripped of identity
headers); separate DB roles per service; field-wide bearer rotation (flag defaults off).


