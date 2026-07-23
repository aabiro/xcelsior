# Track B — Product plane, data plane, and operations: implementation checklist

**Status of this document:** authoritative source of truth for all work
specified by the blueprint and data-architecture companion that Track A did
not deliver. Track A closed the *transactional authority core*. Track B
closes everything else, in dependency order, under the same engineering
rules.

**Governing documents (this checklist implements them; it does not amend
them):**

- [`xcelsior-production-control-plane-mcp-blueprint.md`](./xcelsior-production-control-plane-mcp-blueprint.md)
  — the control-plane and MCP architecture. All `§n` references below point
  here unless prefixed `DA§`.
- [`xcelsior-production-data-architecture-companion.md`](./xcelsior-production-data-architecture-companion.md)
  — the data-model, polyglot-persistence, residency, and analytics
  companion. Referenced below as `DA§n`.
- [`track-a-implementation-checklist.md`](./track-a-implementation-checklist.md)
  — the completed predecessor. Track B never re-litigates a Track A gate;
  where Track B changes something Track A built, it says so explicitly and
  names the Track A item.

**Scope boundary — what Track A already owns (do not rebuild):** §2, §8,
§9, §10, §11, §12, §13.1–§13.4, §13.8, §19.1–§19.4, ADR-001…ADR-010,
Phases 1/3/4/5/6, the durable-outbox mechanism of §16.1, wallet holds
(§15.1), attempt-scoped metering (§15.2), and `DA§4.7`. Track B builds
*on* these; it does not replace them.

**Honesty contract:** every `[x]` below means the repository proves it and
a named gate enforces it. `[~]` means partially landed with the exact
residual stated. `[ ]` means not built. No item claims a production deploy
that has not happened; operator actions are listed separately in §B0.4 and
at the end of each phase as **Operator residual**.

---

## B0 — Engineering rules

These are binding for every item in this checklist. Rules 1–14 are carried
forward verbatim in intent from how Track A was executed; rules 15–22 come
from the governing documents' own execution notes (§31, `DA§22`).

### B0.1 Sequencing and evidence

1. **Gates are ordered.** Each item lists an exit gate. Do not start an
   item until the previous item's gate is green. A phase is not complete
   until every gate inside it is green.
2. **A gate is a named, runnable artifact**, not a description — a test
   file, a CI job, or a command whose failure blocks. "Reviewed and looks
   correct" is not a gate.
3. **Prove behaviour by executing it, not by reading the source.** Track A
   set this precedent when the hardened API image was verified by *booting
   it* read-only as uid 10001 — which surfaced a real startup bug that
   reading the Dockerfile would never have found. Denials are proven by
   running the forbidden statement and requiring rejection; positive
   rights by running a permitted one.
4. **Status reflects the repository at the time of the last edit to this
   file.** When an item changes, update its line and its date in the same
   commit as the code.
5. **Separate repo state from operator state.** Anything requiring a
   production env flip, a deploy, a cloud resource, or a fleet action is
   labelled **Operator residual** and is never counted as `[x]`.
6. **Drift found is drift closed in the same pass.** Track A's Phase 11
   found four missing blueprint requirements while implementing a fifth
   and closed all of them. Do the same: if implementing an item reveals a
   governing-document requirement that silently does not exist, add it to
   this checklist and close it before claiming the item.
7. **Never claim a live deploy that has not happened**, and never describe
   scaffolding as a live system. Track A's Phase 10.5 explicitly reads
   "SPIRE / provisioner scaffolding (not live mesh claim)". Match that
   precision.

### B0.2 Change safety

8. **Expand-contract migrations only.** Additive first; backfill in bounded
   `SKIP LOCKED` batches with hard verification that aborts on unmapped
   rows; constraints `NOT VALID` → verify → `VALIDATE`; `CREATE INDEX
   CONCURRENTLY` in autocommit blocks on large tables; `lock_timeout` set.
   Contract only in a later release after measured zero legacy use (§13.8).
9. **Every migration passes up → down → up cleanly on a dev database and a
   production-shaped snapshot** before it is applied anywhere else. One
   Alembic head, always. Verify the real head before assigning a number
   (see §B1).
10. **Alembic is the only production DDL authority** (`ADR-009`, `DA§4.4`).
    Track A made this enforceable in PostgreSQL itself — no runtime role
    holds `CREATE` on a schema (Track A 11.4). Do not weaken that to make
    an item easier; add the migration.
11. **Single authority per state transition.** Every lifecycle mutation
    goes through the one domain service that owns it. No new direct
    lifecycle SQL, ever. If a path needs a transition, it calls the
    service; if the service lacks the case, extend the service.
12. **Real fixes, not suppressions.** The pyright gate is zero-tolerance on
    `reportCallIssue` + `reportArgumentType`. Fix the type error; do not
    add an ignore. Track A restored 34 regressions to 0 this way, and two
    of those "type fixes" removed a real identifier-injection surface.
13. **A test that asserts the opposite of the design is a defect in the
    test.** Track A found one requiring agent identity to fail *open* on a
    DB error and rewrote it to pin §31's fail-closed rule. Pin the design,
    and comment why.
14. **Tests own only their own rows.** No suite may `DELETE FROM` a shared
    or migration-seeded table. Cross-test contamination is a blocking bug,
    not a flake.

### B0.3 Design fidelity

15. **Fail closed for hard requirements** (`ADR-007`, §31). Isolation tier,
    volume attachment, image verification, host identity, capacity, lease
    and fence validity, wallet policy, artifact provider availability,
    rate-limit enforceability, retrieval model identity. A missing SDK,
    model, extension, credential, bucket, or permission produces the
    documented failure or readiness state — never a silent substitution,
    never an empty success, never a local fallback.
16. **Optional degradation must be declared.** If a spec explicitly permits
    degradation, the resource records a `Degraded` condition with reason,
    chosen alternative, and user impact. Undeclared degradation is a bug.
17. **New enforcement lands report-only first.** Track A's reconciler
    actions default to `report_only` and only types on an `_ENFORCEABLE`
    allowlist can be turned on — an operator cannot enable an action that
    does not exist. Every new automated remediation follows this shape:
    ship dark, measure false-positive rate in production, then enable per
    type behind an explicit env var, with malformed values falling back to
    the safe posture.
18. **One authority per fact** (`DA§3`). No business fact is synchronously
    dual-written to two stores. PostgreSQL commits authoritative state and
    its outbox row in one transaction; everything downstream is a
    rebuildable projection with an idempotent consumer and a checkpoint.
19. **MCP calls the API and nothing else** (`ADR-008`, `DA§13.5`). It never
    receives a database URL, Redis URL, bucket credential, warehouse role,
    worker credential, or arbitrary SQL tool.
20. **Feature flags are staged-rollout tools with an owner, an expiry, and
    a removal phase** (§31). They must not maintain two permanent business
    implementations.
21. **Scope discipline** (`DA§22.2`). Implement the requested item
    completely — migration, config, security, observability, tests, and
    deployment wiring — and nothing beyond it. No unrelated refactors.
22. **Every new variable is added to `.env.example`, compose, deploy
    validation, the startup validator, and this document together** (§30,
    `DA§14.3`). Secrets never receive insecure production defaults.

### B0.4 Standing operator residuals inherited from Track A

These are Track A's unclaimed operator actions. Several Track B gates
depend on them; none are Track B's to mark complete.

- Deploy the SPIRE mesh and enrol hosts; flip `XCELSIOR_TRUSTED_AGENT_GATEWAY=1`
  and `XCELSIOR_SPIFFE_STRICT=1`.
- Issue per-host tokens; flip `XCELSIOR_AGENT_HOST_TOKENS=require`.
- Flip `XCELSIOR_AGENT_PUBLIC_INGRESS=deny` with the Nginx retirement snippet.
- Run `scripts/provision_db_roles.py`; set the per-service DSNs.
- Flip `XCELSIOR_SCHEDULER_MODE=canary` → `active` (Track A P4.4b).
- Roll out the hardened worker agent (Track A P5.5).

---

## B1 — Migration ledger reconciliation (blocking prerequisite)

Both governing documents assign migration numbers that the repository has
since spent on other content. `DA§14` and `DA§22.10` both instruct the
implementer to inspect the real head and renumber; this section is that
renumbering, recorded once so no future item guesses.

**Repository head as of 2026-07-22: `069_action_plans_mcp_policy_audit.py`.**

| Document says | Document intended | Repository actually has | Track B resolution |
|---|---|---|---|
| §13.5 `058` | action plans, MCP policy, MCP audit, wallet holds | `058_scheduler_shadow_decisions` | Split: wallet holds landed as `063_wallet_holds` (Track A). Action plans / MCP policy / MCP audit move to **B2** at the live head. |
| §13.6 `059` | partitioned `audit_events_v2` | `059_runtime_projection_triggers` | Audit v2 moves to **B4** at the live head. |
| §13.7 `060` | contract cleanup | `060_shared_state_to_pg` | Contract cleanup **must remain last in the chain** (§13.7, `DA§14`). It moves to **B16**, at whatever the head is then. |
| `DA§6.3` `061` | storage catalog | `061_residual_runtime_ddl`; catalog landed as `064_storage_catalog` | Already satisfied by `064`. Residual columns tracked in **B9.2**. |
| `DA§10.1` `062` | Lightning + Slurm consolidation | `062_usage_meters_attempt_id`; partial consolidation landed as `060_shared_state_to_pg` | `060` created the tables but **not to the companion's contract** — see **B9.3**. |
| `DA§12.1` `063` | outbox projection delivery contracts | `063_wallet_holds` | Moves to **B4** at the live head. |
| `DA§12.7` `064` | deletion / export state | `064_storage_catalog` | Moves to **B12** at the live head. |
| `DA§7.3` `065` | retrieval pgvector | `065_host_agent_tokens` | Moves to **B10** at the live head. |
| `DA§7.6` `066` | semantic cache v2 | *(unused)* | Moves to **B10**, immediately after the retrieval migration. |

- [x] **B1.1 Record the ledger** (2026-07-22) — `migrations/README.md`
  holds the renumbering map above, the nine authoring rules, the
  contract-cleanup-is-last rule, and the §3 anomaly note. Gate:
  `tests/test_migration_ledger.py` (27 tests, pure static parse — no
  database). Every check is driven **both ways**: once against the real
  chain and once against a synthetic chain that violates it, because a
  check never observed to fail is not a gate. Also verified end to end by
  planting a bad migration file in `migrations/versions/` and confirming
  three checks fire with actionable messages. Wired as the first step of
  the CI `control-plane` job so a chain defect is rejected before anything
  touches a database. ✔
  **Anomaly recorded, deliberately not corrected:**
  `060_shared_state_to_pg.py` declares `revision = 'a0985327493e'` (an
  Alembic-generated hash — authored without `--rev-id`), so
  `061`'s `down_revision` points at the hash. Rewriting the id of an
  applied revision is safe only if no database anywhere is stamped exactly
  there; one that is would fail `alembic upgrade head` with "Can't locate
  revision" — an unmigratable deploy traded for cosmetic tidiness. Local
  databases were checked (dev `064`, docker-test `059`, pytest `065` —
  none at the hash) but production is a separate runtime and was not
  verifiable. The test accommodates this one file by id and enforces the
  numeric rule from `066` onward; a second exception fails the suite.
- [x] **B1.2 Production-shaped upgrade rehearsal** (2026-07-22) —
  `tests/test_migration_from_production_snapshot.py` builds a 053-shaped
  database, seeds representative legacy rows (one job per legacy status,
  hosts in every admission shape including one whose payload predates the
  `admitted` flag, an active *and* an expired lease, money-bearing
  `billing_cycles`), upgrades to head, and asserts §26.5 / `DA§16.1`
  properties: no row loss or duplication, zero unprojected rows, the
  status→phase/desired_state distribution per legacy status, tenancy
  projected out of JSONB, queue-ordering columns populated, the active
  legacy lease backfilled into an attempt+lease and the expired one *not*,
  `uq_job_one_active_attempt` holding over migrated data, admitted vs
  pending hosts projecting differently, and money totals unchanged. Plus
  up→down→up over *populated* tables, proving the backfill is idempotent
  across a cycle. Wired into the CI `control-plane` job. ✔
  A sanitized production dump would be a stronger input but is not
  available to CI and must not be pulled into a test environment; seeding
  a 053 schema is reproducible and exercises the same code paths.

**Drift found and closed in the same pass (2026-07-22, B0.1 rule 6):**

- [x] **Migration 054's jobs backfill could not terminate on an unmappable
  legacy status — `alembic upgrade` hung instead of aborting.** The batch
  predicate selected `WHERE phase IS NULL` while the `SET` wrote
  `phase = CASE … ELSE NULL`, so a row with an unknown legacy status was
  re-selected on every pass, `rowcount` never reached 0, and the loop span
  forever holding its locks. The `_verify_backfill()` step that blueprint
  §13.1 requires — and that Track A A1.1 records as "hard verification
  (abort on unmapped rows)" — was therefore **unreachable** in exactly the
  case it exists to catch. A from-empty upgrade cannot reach this path,
  which is why it survived to now. Fix: the batch predicate now also
  requires `(<phase case>) IS NOT NULL`, so unmappable rows are never
  claimed, the loop drains, and `_verify_backfill` raises naming the
  offending status; a pass cap raises rather than looping if the invariant
  is ever broken again. Semantically a no-op for any database that already
  applied 054 (it had zero unmappable rows, or it would have hung), so the
  in-place edit is safe. `_backfill_hosts` was checked and does not share
  the defect. Gate: `test_unknown_legacy_status_aborts_the_migration`,
  which fails fast with a diagnostic instead of a 180-second pytest
  timeout if the hang returns. Track A gates re-run green
  (`test_from_empty_bootstrap` 3, `test_control_plane_schema` 40,
  `test_no_runtime_ddl` 6); pyright clean.

---

## B2 — Unified launch service and versioned API contracts

Blueprint §14, §18, §15.4, Phase 2. **This is the single largest
uncovered blueprint area and the hard prerequisite for B5 (MCP v2).**

**Current state (verified 2026-07-21):** no `control_plane/launch/`, no
`action_plans` table, no `/api/v1/*` route prefix anywhere in `routes/` or
`api.py`, no launch-plan endpoints. `/instance` remains the only launch
path. Wallet holds (§15.1) and idempotency keys (`api_idempotency_keys`,
migration 056) exist from Track A and are the substrate this builds on.

- [x] **B2.1 Migration — action plans, MCP policy, MCP audit** (2026-07-23,
  §13.5 at the live head) — migration `069_action_plans_mcp_policy_audit`
  creates `action_plans` (plan id, action type, principal/client/tenant/
  team, canonical argument JSON + hash, spec hash, quote id, pricing
  version, estimate in **micro-CAD** matching the wallet ledger (068),
  currency, tolerance in bps, required scopes, approval mode, status,
  expiry, approval/consumption timestamps, approved-by/session/method,
  resulting resource, idempotent response, revocation/failure details),
  `mcp_client_policies` (one per `(client_id, tenant_id)`, micro-CAD spend
  ceilings, capability limits, `auto_approve`), and `mcp_tool_audit` (one
  redacted record per tool call, in the audit domain — `db_roles.py`
  assigns it there and the other two to control-plane).
  **Reuse, not a second authority (§13.5):** `wallet_hold_id` is an FK to
  the existing `wallet_holds`; `idempotency_key` is the natural key the
  execute path (B2.5) writes into the existing `api_idempotency_keys`. The
  plan's own `idempotent_response` is *plan-consumption* idempotency (part
  of the §9.5 machine), a different scope from API request replay.
  **§9.5 state machine enforced in the row** by
  `ck_action_plans_state_machine`: approved-and-later carry `approved_at`;
  a consumed plan (`executing`+) carries `consumed_at`; `succeeded` carries
  `job_id` and `idempotent_response`; `failed_*` carries `failure_code`;
  `revoked` carries `revoked_at` — so no service bug can persist an
  impossible plan.
  **OAuth workspace context (drift closed, B0.1 rule 6):**
  `validate_client_credentials_jwt` dropped `workspace_customer_id`/
  `team_id` when building the machine principal, so a client-credentials
  token authorized a launch with **no tenant** — defeating the workspace it
  was issued under. Fixed to carry both plus `tenant_id`, and a new
  `ck_oauth_client_workspace_context` CHECK (added `NOT VALID` →
  `VALIDATE` when clean) makes `workspace_customer_id` mandatory for any
  system-managed or client-credentials client. No second tenant column is
  added — `workspace_customer_id` is the one authority (DA§3).
  Verified up→down→up on the pytest database (069→068→069 clean) and
  from-empty reaches head. Gate: `tests/test_action_plan_schema.py`
  (21 tests, real PostgreSQL, every constraint driven both ways with a
  legal control row). Regressions green: migration-ledger 27,
  from-empty 3, db-service-roles 48, control-plane-schema 40,
  oauth-migration 3, team-tenancy 31, auth-endpoints 39; pyright clean. ✔
- [x] **B2.2 Canonical spec and quoting** (2026-07-23) —
  `control_plane/launch/{canonicalize,validation,quoting,spend_policy}.py`.
  **Canonicalization is deterministic and versioned** (`CANON_SPEC_VERSION`):
  every optional field is defaulted so omitting it hashes identically to
  passing its default; strings are stripped; logically-set collections
  (`exposed_ports`, `volume_ids`) are sorted; auto-launch ports are merged
  exactly as the `JobIn` model does so the two surfaces agree (B2.6).
  **The spec hash is not reinvented:** `spec_hash` reuses the scheduler's
  `canonical_spec_hash`, and the worker's inline copy was extracted to
  `worker_agent.compute_spec_hash` so a drift guard pins the two independent
  implementations together. **Quoting** reuses the existing
  `BillingEngine.estimate_launch_hold_cad` authority (no second pricing
  path), versions it (`pricing_version` encodes the exact rate so a later
  move is detectable — B3.4), and returns micro-CAD; `price_moved_beyond_
  tolerance` protects only against paying *more* (§15.4). **Spend policy**
  evaluates a spec+estimate against an `mcp_client_policies` row for the
  static ceilings and capabilities and the auto-approve grant; the rolling
  hourly/daily counters are deferred to B5.9 and documented as such.
  Gate: `tests/test_launch_canonical_spec.py` (23 tests) — key-order /
  whitespace / optional-default stability (Hypothesis, 200 examples), a
  three-way drift guard asserting `launch == scheduler == worker` over
  generated specs, plus validation/quoting/spend-policy coverage. Pyright
  clean; `test_worker_agent` 76 green after the extraction. ✔
- [x] **B2.3 Action-plan service and preview** (2026-07-23) —
  `control_plane/launch/{action_plans,service}.py` and `POST
  /api/v1/launch-plans` (`routes/action_plans.py`, registered in
  `ALL_ROUTERS`). `preview()` runs §14.1 steps 1–8: canonicalize →
  side-effect-free structural validation → versioned quote (hourly burn +
  worst-case runway, micro-CAD) → **placement feasibility simulation**
  reusing Track A's `take_snapshot` + the pure Stage-C `filter_hosts` in its
  own REPEATABLE READ transaction (no allocation, no lease) → load the
  client's standing policy → persist a `quoted` action plan with the
  canonical args and hash → return plan id, estimate, availability, expiry,
  approval mode, and next action. The transport resolves the `Principal`
  (tenant = effective billing customer, so a plan and the funds execute
  later reserves share one tenant); the service never reads headers.
  An invalid spec returns its problems and persists **no** plan.
  **Canonicalization gained the interactive→`vram_needed=0` defaulting rule**
  the REST path already applies, so the two surfaces stay byte-identical
  (B2.6). Gate: `tests/test_launch_plans.py` (3 tests) — a real HTTP preview
  creates exactly one `action_plans` row and **zero** rows in `job_attempts`,
  `gpu_device_allocations`, `placement_leases`, `wallet_holds`, and `jobs`;
  the stored canonical args + hash match the canonicalizer; an invalid spec
  is 422 with no plan. Pyright clean; canonical-spec 23 and api 157 green. ✔
- [x] **B2.4 Approval** (2026-07-23) — `POST /api/v1/launch-plans/{id}/approve`
  and `/revoke` (§14.2), on `service.approve` / `service.revoke`.
  **Server-bound:** a `standing_policy` plan re-loads its
  `mcp_client_policies` row and self-approves **only** if still inside every
  ceiling — a policy that was removed or tightened after preview forces a
  human (`auto_approval_denied`). A `human` plan is approved only by an
  interactive human (a machine client is rejected `human_approval_required`);
  `confirm:true` is accepted for client symmetry and **ignored** — it never
  approves. The isolation boundary is the tenant/workspace: a cross-tenant id
  returns not-found (no existence leak), while §14.2's human-approves-a-
  machine-plan flow stays allowed within the tenant. Transitions are the
  single authority (`mark_approved`/`mark_revoked`/`mark_expired`, row-locked,
  version-bumping) and the §9.5 CHECK constraints back them. Typed
  `LaunchPlanError`s carry an RFC 9457 `code`/status for B2.8. Gate:
  `tests/test_launch_plan_approval.py` (10 tests) — cross-tenant approve and
  revoke both refused; out-of-policy auto-approval refused; approval and
  revoke idempotent (version does not advance on the no-op); a revoked plan
  cannot be approved; HTTP 404 for an unknown plan. Pyright clean. ✔
- [x] **B2.5 Execute** (2026-07-23) — `POST /api/v1/launch-plans/{id}/execute`
  (§14.3), on `service.execute`. Authenticates the plan's tenant (a cross-tenant
  id is not-found, no existence leak); row-locks the plan `FOR UPDATE` for the
  whole operation; verifies approved / not expired / not consumed / not revoked;
  verifies the canonical-argument hash (`spec_hash_mismatch` refuses a tampered
  spec nobody quoted); re-quotes and enforces tolerance; reserves the wallet
  hold; creates the job through the **one** job authority (`scheduler.submit_job`,
  which itself re-validates the Docker image and enqueues the `job.created`
  lifecycle outbox row idempotently); consumes the plan; returns job id + phase.
  **Exactly-once without a distributed transaction:** the job uses a deterministic
  id (`_upsert_job_row` upserts on replay), the hold is keyed `plan:{id}`
  (idempotent replay in `create_wallet_hold`), and the plan's own status is the
  final gate — so a concurrent execute (loser waits on the row lock, then replays
  `succeeded`) and a crash between hold/submit/consume both converge on one job
  and one hold. Out-of-tolerance price returns `quote_changed` with a fresh
  replacement plan and charges nothing (§15.4). Gate:
  `tests/test_launch_plan_execute.py` (10 tests, real PostgreSQL) — one job + one
  hold on success; repeated execute returns the original job with no second
  effect; **concurrent execute admits exactly one** (thread race, one
  non-idempotent winner); not-approved / revoked / expired all refused (expiry
  durably transitions the row — see the drift note below); spec-hash mismatch
  refused with no hold; insufficient funds is a typed 402 with no job; a
  tolerance breach returns a replacement and leaves the original `approved` and
  uncharged. Pyright clean; B2.4 approval 10 green after the `approve` refactor. ✔
  **Residual (deferred, documented in code):** hold + job + plan-consume are not
  yet one physical transaction — the single-transaction form waits on the
  billing-engine `conn=` variant (B9.5), exactly as the Lightning credit path
  deferred it; the idempotency design above makes the interim crash-safe.
  Rolling hourly/daily spend counters at execute are B5.9; the static policy
  ceiling was bound at approve (B2.4).
- [x] **B2.6 `/instance` no longer schedules inline; specs are equivalent**
  (2026-07-23, §14.4). The general `/instance` path's inline `process_queue()`
  call is removed — the job is enqueued durably and the scheduler claims and
  places it (§10.1), completing the deferral Track A P4.4b began for the
  host-pin path. `/instance` and `/api/v1/launch-plans` share the one
  canonicalizer, so the same input yields the byte-identical canonical spec on
  both surfaces. Gate: `tests/test_launch_instance_adapter.py` (8 tests) — an
  **AST structural test** proving `api_submit_instance` (and every other route
  handler except the four explicitly-named admin queue-runner endpoints) never
  calls `process_queue*`, so a new handler that inline-schedules fails CI; plus
  an **equivalence test** proving a `JobIn` canonicalizes identically however
  the two surfaces reach it (interactive→`vram=0`, ports as a merged set,
  order/whitespace-invariant hash) and that the launch-plan preview persists
  exactly `canonicalize(JobIn)`.
  **Blast radius closed (B0.1 rule 6):** 11 tests across `test_api`, `test_e2e`,
  and `test_integration` had relied on `/instance` assigning inline (BG tasks
  are off under pytest). Each now triggers the scheduler explicitly via the real
  `process_queue` — the production flow — and asserts placement, rather than
  asserting the request handler did the scheduler's job. Full runs green
  (398 passed across the launch/scheduler/e2e suites).
  **Residual (→ B2.7):** `/instance` still owns its submission body (host-pin,
  template resolution, volume validation, wallet preflight, `submit_job`); it
  shares the *canonical spec* with the service but does not yet delegate that
  body to `service.execute`. Collapsing every writer onto the service is B2.7,
  which is where the "one implementation" inventory gate lives.
- [x] **B2.7 Inventory of every job-row writer** (2026-07-23). `submit_job` is
  the single job-creation authority (B0.2 rule 11) — the only path that inserts
  a `jobs` row (`_upsert_job_row` → `DatabaseOps.upsert_job`). Gate:
  `tests/test_job_writer_inventory.py` (2 tests, static AST) enumerates every
  module that calls `submit_job` and asserts the set is **exactly** the launch
  service plus an explicitly-justified allowlist — a new, unclassified writer
  fails CI, and a stale entry that no longer writes also fails (the list cannot
  rot into a rubber stamp). The second test pins the raw `INSERT INTO jobs` /
  `upsert_job` call to the persistence authority (`scheduler.py`/`db.py`) so no
  module grows a second row writer. Current classification: the unified launch
  service is the sanctioned authority; `routes/instances.py` (REST /instance,
  canonical-spec-aligned via B2.6), `serverless/service.py` (distinct
  worker-provisioning lifecycle — B3.1 attempt-binds it), `inference.py` /
  `routes/inference.py` (request/response inference, not an interactive
  instance), `ai_assistant.py` (thin wrapper over the REST launch), and `cli.py`
  (local operator tool) are the justified exceptions. ✔
  **Residual (ongoing consolidation):** the guard *pins* the writer set today;
  collapsing `/instance`'s and serverless' submission bodies to call
  `service.execute` directly (so they share not just the canonical spec but the
  whole preview→hold→consume machine) proceeds surface-by-surface as their
  semantics are reconciled, without ever adding an unclassified writer.
- [~] **B2.8 Versioned API surface and RFC 9457 errors** (§18.1–§18.3,
  §18.5). **Landed (2026-07-23):** the RFC 9457 error framework —
  `routes/problem.py` emits `application/problem+json` with the full field set
  (`type`, `title`, `status`, `detail`, `code`, `retryable`, `retry_after_ms`,
  `trace_id`, `errors`), sets `Retry-After` from `retry_after_ms`, carries a
  W3C `trace_id`, and supports RFC 9457 extension members (the `quote_changed`
  replacement plan). It is **scoped to the typed control-plane error**
  (`LaunchPlanError` / `ProblemException`), so the legacy
  `{"ok": false, "error": {…}}` shape is unchanged — a companion test pins that.
  The whole v1 launch-plan surface (preview / approve / revoke / execute) now
  returns problem+json, and `POST /api/v1/placements/simulate` is added
  (read-only feasibility, reuses the launch service snapshot + Stage-C filter;
  creates no plan/attempt/lease). Gate: `tests/test_v1_problem_json.py` (6) —
  every required field present, retryable codes + `Retry-After`, HTTP 404/409
  problem+json, and legacy shape unchanged; `tests/test_placement_simulate.py`
  (1) — read-only, no rows written. Pyright clean; api 157 + launch suites green.
  **Also landed (2026-07-23): the §3.3 drain-vs-evict split** —
  `routes/control_plane_v1.py` adds `POST /api/v1/hosts/{id}/drain` (stop new
  placements only — **never** preempts; leaves running workloads running),
  `/undrain`, and `/evictions` (a *separate* action requiring the distinct
  `hosts:evict` scope, gated on the host already being drained so drain→evict is
  an explicit two-step operator flow, separately audited as
  `host.workloads_evicted`). Optimistic concurrency via the host `version`
  (`expected_version` mismatch → 409 `version_conflict`); domain errors are
  problem+json. This corrects the §3.3 gap found in the legacy
  `/host/{id}/drain` (which calls `run_drain_preemptions` — "*preempts all
  workloads on a draining host*"): the v1 `drain` does not preempt. Gate:
  `tests/test_control_plane_v1_drain_evict.py` (5) — **drain leaves a running
  workload running**, evict-before-drain refused, drain→evict removes it,
  drain/evict require **distinct** scopes (AST-pinned), unknown host is
  problem+json 404, stale `expected_version` is 409. Pyright clean; host-coverage
  16 green.
  **Also landed (2026-07-23): the customer/operator read endpoints** —
  `GET /api/v1/instances/{id}/control-plane` (phase / desired-state / current
  attempt), `GET /api/v1/instances/{id}/timeline` (per-attempt reserve → command
  → lease → start → end from `job_attempts`), and
  `GET /api/v1/control-plane/reconciliation-findings` (wraps the existing
  `reconciliation_findings` authority). All tenant-scoped: a cross-tenant
  instance id returns **not-found** problem+json, never a 403 permission hint
  (§B5.6 no existence leak); findings need admin or `control_plane:read`. Gate:
  `tests/test_control_plane_v1_drain_evict.py` (+3, now 8) — control-plane +
  timeline shape, cross-tenant not-found (no leak), findings admin-gated with
  invalid-status 422.
  **Also landed (2026-07-23): the operator aggregates + instance
  attempts/retry** — `GET /api/v1/control-plane/queue` (queued instances +
  reasons), `GET /api/v1/control-plane/health` (outbox / findings /
  scheduled-task **live counts**, so a broken-pipeline 0 stays distinguishable
  from a genuine 0 — DA§17), `GET /api/v1/hosts/{id}/capacity`,
  `GET /api/v1/hosts/{id}/observations`, `GET /api/v1/instances/{id}/attempts`,
  `GET /api/v1/instances/{id}/placement-explanation` (the *persisted* scheduler
  explanation — no LLM invents a reason), and
  `POST /api/v1/instances/{id}/retry` (re-enqueues via the one requeue
  authority; never inline-schedules; already-queued/completed → typed 409).
  Operator reads need admin or `control_plane:read`; instance reads/writes are
  tenant-scoped (cross-tenant → not-found, no existence leak). Gate:
  `tests/test_control_plane_v1_drain_evict.py`.
  **Also landed (2026-07-23): reconcile + the served v1 OpenAPI + a contract
  gate.** `POST /api/v1/instances/{id}/reconcile` inserts a durable, coalesced
  request into `reconciliation_queue` and returns — it performs **no** direct
  repair (§3.3 / B5.7); the reconciler claims it out-of-band (`process_due`
  already has a safe no-controller-yet branch for the `job` type). The app's
  `/openapi.json` is a curated public spec that omitted v1, so a dedicated
  `GET /api/v1/openapi.json` (the live FastAPI schema filtered to `/api/v1/*`)
  is served for the generated MCP/dashboard clients (B5.2 / B6.1). Gate:
  `tests/test_v1_openapi_contract.py` (2, a dependency-free schemathesis
  substitute) — every registered v1 route is in the v1 OpenAPI (no non-v1
  leakage), and representative v1 error paths (404/403/422) all emit RFC 9457
  problem+json. **So B2.8 has shipped 15 v1 endpoints + the RFC 9457 framework +
  a served v1 OpenAPI + a contract gate, all gated and pyright-clean.**
  **Residual (the small tail):** (1) giving the reconciler a **job-type
  controller** in `process_due` (the endpoint enqueues correctly; the reconciler
  consuming a `job` reconcile is a Track-A-coupled follow-up); (2) full
  **schemathesis** property-fuzzing (the deterministic contract gate above
  covers the shape; fuzzing would add a frozen-deps change); (3) reconciling the
  **legacy** `/host/{id}/drain` preempt-on-drain behaviour (a behaviour change
  with drain-test blast radius, to be retired behind a flag once the UI moves to
  v1 drain+evict). The v1 `/evictions` uses the existing preempt/requeue path;
  strengthening it to a fully fence-token-invalidating primitive is tracked with
  Track A's fenced lifecycle.

**Drift found and closed in the same pass (2026-07-23, B0.1 rule 6):**

- [x] **Lazy expiry on access was silently rolled back — an expired plan was
  refused but never left `approved`.** Both `service.approve` and
  `service.execute` marked an over-expiry plan with `mark_expired(conn)` and
  then `raise`d inside the *same* `control_plane_transaction`, so the
  `control_plane_transaction` rollback discarded the mark: the plan stayed
  `approved` forever and was re-evaluated on every subsequent call, contradicting
  the §9.5 state machine B2.1 pins in the row. This surfaced only by *executing*
  the path (B0.1 rule 3) — the B2.5 gate's `test_execute_refused_when_expired`
  asserts the row becomes `expired`, which failed on the first run. Fix: both
  paths now persist the terminal transition, let the transaction commit, and
  raise the `plan_expired` refusal *after* the block; pure refusals with no state
  change (not-approved, revoked, spec-hash, auto-approval-denied) still raise
  inside the transaction, where rollback is the correct behaviour. Gate:
  `test_execute_refused_when_expired` (and the B2.4 approval suite, re-run green
  after the `approve`/`_approve_locked` refactor).

**Exit gate (Phase B2):** MCP, dashboard, and REST submit equivalent
canonical specs; repeated execute creates exactly one job and one hold;
simultaneous launches cannot overspend one wallet (extends Track A's
`tests/test_wallet_holds.py` race to the plan path); no API request
handler schedules inline.

---

## B3 — Billing completion

Blueprint §15.3, §15.4, §12.4 billing controller. §15.1 and §15.2 are
Track A.

- [ ] **B3.1 Serverless workers bind to attempts.** `serverless_workers`
  rows link to a fenced `job_attempts` row and its `gpu_device_allocations`
  (§15.3, §5.6). Warm-time accounting is attempt-scoped like compute
  metering already is. Gate: a serverless scale-up produces exactly one
  attempt, one allocation set, and one warm-time meter; a fenced stop
  closes it once.
- [ ] **B3.2 Endpoint and client budgets replace per-request approval**
  (§4.2, §15.3). Server-side enforcement; an inference call never requires
  a human approval click. Gate: budget exhaustion denies with a typed
  problem, not a silent success or an unbounded spend.
- [ ] **B3.3 Billing controller** (§12.4). Ensure exactly one meter per
  accepted running attempt; close orphaned meters after attempt terminal;
  repair missing outbox delivery idempotently; **surface** ledger invariant
  violations rather than concealing them. Runs as a reconciler controller
  under Track A's report-only-then-enforce rule (B0.3 rule 17). Gate:
  injected crash between attempt-running and meter-start converges to
  exactly one meter; an injected duplicate delivery creates no second
  charge.
- [ ] **B3.4 Price-change reapproval end to end** (§15.4). The tolerance
  bound recorded in B2.1 is enforced at B2.5 execute and surfaced in the
  UI (B6) and MCP (B5) as `quote_changed` with a replacement plan.
  Gate: a pricing-snapshot bump beyond tolerance blocks execute and
  produces a new plan; within tolerance it proceeds unchanged.

---

## B4 — Audit v2, event contracts, and persistent streams

Blueprint §13.6, §16.2, §16.3; `DA§12.1`, `DA§12.2`, `DA§2.2`.

**Current state (verified 2026-07-21):** Track A removed the global
`LOCK TABLE events IN EXCLUSIVE MODE` and replaced it with per-stream
advisory-lock hash chains — `DA§12.2`'s first bullet is done. Still open:
`events.archive_old_events()` queries columns `created_at` and
`chain_hash` that the live `events` table does not have (it has
`timestamp`, `prev_hash`, `event_hash`), so the archive path is still
broken exactly as `DA§2.2` described, and `bg_worker` still schedules only
snapshots, never the archive. There is no partitioned audit table, no
signed checkpoint, no event-contract registry, no per-sink delivery state,
and no `Last-Event-ID` cursor on SSE.

- [ ] **B4.1 Migration — partitioned `audit_events_v2`** (§13.6 at the live
  head). New partitioned table, not a rewrite of live `events`: tenant,
  stream type/id, stream sequence, aggregate version, event type,
  actor/client/request/trace ids, redacted immutable payload, per-stream
  `prev_hash`/`event_hash`, `created_at` partition key, unique
  `(stream_id, stream_sequence)` and unique event id. Monthly partitions
  created ahead of time by a `scheduled_tasks` entry (reuse Track A's
  `telemetry_partition_maintenance` pattern), plus a DEFAULT safety
  partition with an alert before it receives rows (`DA§4.5`). Gate:
  up→down→up clean; a test that fills the month boundary and asserts no
  ad-hoc partition creation happens in a request handler.
- [ ] **B4.2 Fix or retire the archive path** (`DA§2.2`, `DA§16.1`). Either
  correct `archive_old_events()` to the live column names and schedule it
  as a durable task, or delete it and replace it with the object-export
  path in B4.4. Do not leave dead code that reads columns which do not
  exist. Gate: a test that runs the retention path against a real
  PostgreSQL and asserts rows move; a static test that fails if any query
  in `events.py` references a column absent from the migration-defined
  schema.
- [ ] **B4.3 Event contract registry** (`DA§12.1`, `DA§13.4`).
  `audit.event_contracts` (event type, version, schema JSON, schema
  sha256, classification, compatibility mode, active, timestamps) and
  `analytics/contracts.py` with the versioned domain-event names from
  §16.2 and `DA§8.3`: `job.v1.created`, `job.v1.placement_reserved`,
  `job.v1.lease_claimed`, `job.v1.running_observed`, `job.v1.terminal`,
  `host.v1.condition_changed`, `command.v1.dead_lettered`,
  `billing.v1.meter_started`, `billing.v1.usage_interval_closed`,
  `billing.v1.wallet_ledger_posted`, `billing.v1.invoice_finalized`,
  `billing.v1.provider_payout_posted`, `serverless.v1.request_completed`,
  `artifact.v1.available`, `artifact.v1.deleted`, `mcp.v1.action_approved`,
  `mcp.v1.tool_completed`. Reconcile against the event names Track A
  already emits (`job.v1.legacy_status_changed`, `host.v1.status_changed`,
  `job.v1.queue_blocked`, `pricing.v1.spot_prices_updated`, the instance
  lifecycle set) — register those, do not rename them mid-flight.
  Gate: a CI contract test that rejects a schema containing a
  `credential_secret`-classified field name or type, and rejects a sink
  mapping with no classification (`DA§13.4`).
- [ ] **B4.4 Per-sink delivery and checkpoints** (`DA§12.1` at the live
  head). Extend Track A's `outbox_events` with `event_version`,
  `tenant_id`, `occurred_at`, `classification`, `payload_sha256`,
  `correlation_id`, `causation_id`, `trace_id`, `fanout_prepared_at`,
  `fanout_attempts`; add `audit.projection_deliveries` and
  `audit.projection_checkpoints`. **Extend, never create a second outbox
  authority.** Match the real migration-056 column names; do not add
  aliases to make the companion's illustrative DDL compile. Two durable
  stages: claim a batch → materialize per-sink delivery rows and set
  `fanout_prepared_at` in one short transaction; then claim delivery rows,
  do external I/O outside the transaction, and record success by stable
  external id. `fanout_prepared_at` means obligations were materialized,
  never that a sink succeeded. Gate: kill the dispatcher at every step and
  show eventual single logical delivery per sink after restart; adding a
  sink later requires an explicit backfill range, proven by test.
- [ ] **B4.5 Signed immutable checkpoints** (§13.6, `DA§12.2`). Periodic
  Merkle root over event ids/hashes, signed with a managed key, stored as
  a manifest in versioned/WORM-capable object storage: sorted event
  ids/hashes, previous manifest hash, schema versions, object generation,
  signing key version, row count, time interval. Verification runs on a
  schedule and during restore drills; signing keys are administratively
  separate from bucket administration. Gate: a tampered or missing object
  fails verification; key rotation preserves verifiability of older
  manifests.
- [ ] **B4.6 Persistent user event streams** (§16.3). SSE reads a persisted
  cursor and tails outbox/audit projections; a client reconnects with
  `Last-Event-ID` without losing transitions. Track A's process-local
  `broadcast_sse` remains only as a latency optimization behind
  persistence, and only where the durable path already fires. Gate: kill
  an API replica mid-stream; the client reconnects and receives every
  transition exactly once in order.
- [ ] **B4.7 Close the residual process-local emitters.** Track A left
  these explicitly unclaimed: hosts dual-emit residual, volumes, teams,
  billing wallet UI, agent telemetry, user images, lock/reset, `job_log`,
  and webhook bulk intents. Route each through
  `try_append_lifecycle_outbox` with the same dual-fan-out avoidance, or
  document why it is legitimately process-local. Gate: a structural
  inventory test listing every `broadcast_sse` / `emit_event` call site
  and its classification, failing on an unclassified new one.

---

## B5 — MCP production v2

Blueprint §3, §17, §26.4, Phase 8. **Depends on B2** — the action-plan
flow cannot exist in MCP before it exists in the API.

**Current state (verified 2026-07-21):** `mcp/src/auth/scopes.ts`
`userHasScope()` returns `true` when the principal has **no scopes** —
the exact unsafe legacy default §5.9 and §8.5 forbid, still live. Zero
`outputSchema` declarations across all eight tool modules. No
`diagnostics.ts`, no `operator.ts`, no `actions.ts`, no generated client,
no `observability/`, no OAuth protected-resource metadata. Redis-backed
rate limiting *is* live and fail-closed (Track A Phase 10 residual
closeout, `mcp/src/rate-limit.ts` + vitest). The `mcp/Dockerfile` is
locked `npm ci`, non-root (Track A 10.4).

- [ ] **B5.1 Default-deny scopes (P0 security).** `userHasScope` denies on
  empty/missing scopes. Add the §17.8 scope vocabulary:
  `instances:operate`, `hosts:operate`, `hosts:evict`, `control_plane:read`,
  `control_plane:operate`, `mcp_actions:approve` alongside the existing
  set. `api` remains a broad legacy scope for existing clients only; new
  clients request explicit scopes and the issuer stops granting `api`.
  Gate: `mcp/src/auth/scopes.test.ts` — empty, undefined, unknown, and
  wrong-scope principals all denied for every tool in the registry, driven
  from the registry so a new tool cannot be added without a scope.
- [ ] **B5.2 Generated typed client** (§17.2). `openapi-typescript` in dev
  generates types from the FastAPI v1 OpenAPI into `mcp/src/client/
  generated/`; `api.ts` supports all required methods, per-route
  deadlines, abort signals, connection reuse, RFC 9457 decoding,
  idempotency and `traceparent` headers, and bounded retry that retries
  safe reads and explicitly idempotent writes only — never a blind replay
  of a mutating request. Gate: a CI diff check failing when the API
  OpenAPI changes without regenerating; respx-style transport tests for
  timeout, 409, 429, and problem+json mapping.
- [ ] **B5.3 Tool contracts** (§17.3). Every tool declares: stable name,
  semantic version in metadata and audit, Zod input schema with bounded
  strings/arrays, MCP `outputSchema`, `structuredContent` plus a concise
  text summary, read-only/destructive/idempotent/open-world annotations,
  required scopes and tenant class, idempotency behaviour, timeout and
  retry policy, redaction policy, and typed API problem mapping. Gate: a
  registry test that fails if any registered tool omits any of these.
- [ ] **B5.4 `create_instance` v2** (§17.4, §17.5, §4.1). Same tool name,
  same original fields, plus optional `plan_id` and `idempotency_key`.
  First call returns `{preview, plan_id, approval_state, canonical_spec,
  estimate, availability, approval_url, expires_at}`. After standing-policy
  or human approval, `confirm:true` + `plan_id` executes. A confirmed call
  *without* a plan returns a structured `approval_required` response with
  an automatically prepared plan — never an unsafe launch and never an
  opaque failure. Publish the change in MCP resources, docs, and UI. No
  indefinite hidden legacy-confirmation branch. Gate: the §26.4 sequence,
  end to end.
- [ ] **B5.5 Serverless MCP behaviour** (§17.6, §4.2).
  `create_serverless_endpoint` uses an action plan (it creates persistent
  spend and capacity policy); `run_serverless_job` stays low friction under
  endpoint/client budgets; `should_i_run_pel_job` stays read-only and
  incorporates current endpoint cost and queue limits; streaming stays on
  the established serverless stream endpoint with trace correlation.
  Gate: no approval prompt on the invocation path; budget denial is typed.
- [ ] **B5.6 Diagnostic tools** (§3.2). `explain_instance_placement`,
  `simulate_instance_placement`, `get_instance_timeline`,
  `get_active_lease`, `get_scheduler_health`, `get_host_capacity`,
  `list_reconciliation_findings`, `get_mcp_action_status`. Each explains
  **persisted** control-plane facts — Track A already stores a bounded
  §3.2 explanation for every decision *and* non-decision
  (`control_plane/scheduler/explain.py`), and the reconciler findings are
  already exposed at `/api/admin/reconciler/findings`. No LLM invents a
  reason. Sensitive host fields, credentials, and private IPs are redacted.
  Gate: tenant-scoped by default; a cross-tenant id returns not-found, not
  a permission hint.
- [ ] **B5.7 Operator mutation tools** (§3.3). `retry_instance`,
  `reconcile_instance`, `drain_host`, `undrain_host`,
  `evict_host_workloads`, `retry_agent_command` — each separately scoped,
  each requiring expected version and idempotency key, each an action plan
  where destructive. `drain_host` and `evict_host_workloads` stay strictly
  separate. `reconcile_instance` enqueues; it never performs direct repair.
  `retry_agent_command` accepts only dead-lettered or retryable commands
  and preserves the original idempotency key and audit history. Gate:
  drain does not stop workloads; evict requires its own scope and plan; a
  stale expected version is refused.
- [ ] **B5.8 Standards-based auth** (§17.7). OAuth protected-resource
  metadata at `/.well-known/oauth-protected-resource`; authorization-server
  metadata discovery; Authorization Code + PKCE for interactive clients;
  client credentials for approved machine agents; RFC 8707 resource
  indicators with an MCP-specific audience; short-lived tokens, refresh
  rotation, revocation, replay detection; asymmetric signing with JWKS
  rotation. Quick Connect remains as the polished onboarding path but its
  generated clients are tenant-bound, short-lived or revocable,
  scope-visible, and governed by spend policy; the principal returned to
  MCP carries workspace/customer/team/client context. Gate: a token
  missing the MCP audience is rejected; a revoked token stops working
  within the documented window; a principal with no tenant is refused.
- [ ] **B5.9 Distributed limits and spend counters** (§17.9). Extend Track
  A's Redis limiter to per-principal + per-client + per-tool rate,
  concurrent long watches, launch attempts, serverless invocation bursts,
  hourly/daily spend policy counters, and abuse lockout — all via atomic
  operations or Lua, never read-modify-write. The durable spend decision
  stays in PostgreSQL; Redis protects traffic, not money. Gate: extend
  `mcp/src/rate-limit.test.ts` to multi-replica concurrency per limit
  class; a Redis outage produces the declared fail-closed behaviour for
  mutating calls, never silent unlimited (§23.2, §31).
- [ ] **B5.10 Audit** (§17.10). Every tool call records timestamp,
  tool+version, transport, client, actor, tenant/team, scopes evaluated,
  redacted canonical argument hash, action plan / idempotency key, API
  route/status/problem type, resource ids created or affected, latency,
  trace id, and approval method. Never a bearer token, secret, registry
  password, environment value, or raw init script. Written through the API
  on the outbox-backed path (B4.4). Gate: a redaction test asserting no
  forbidden field reaches the audit row, driven from the classification
  vocabulary in `DA§13.4`.
- [ ] **B5.11 Resources, prompts, progress, cancellation** (§17.11).
  Versioned read-only resources for pricing methodology, GPU/runtime
  capability definitions, scope documentation, queue-reason catalog, and
  launch policy — not mutable database internals. Resource templates only
  where an access check runs on every read. Playbook prompts invoke the
  same tools rather than embedding stale pricing. Bounded progress for
  launch and watch. **MCP cancellation stops the MCP wait, not the GPU
  job** — destroying compute requires an explicit `cancel_instance`.
  Opaque cursors for large results. `watch_instance` resumable from the
  durable event cursor (B4.6) and returning on a configurable
  phase/timeout. Protocol and server capability versions exposed in health
  and audit. Gate: a cancelled watch leaves the instance running, proven
  by test.
- [ ] **B5.12 MCP observability** (§17.2, §25.3). `pino` structured logs,
  `prom-client` metrics, OTel Node tracing that continues the W3C context
  from the client through to the API. Metrics: calls by tool and outcome,
  p50/p95/p99 latency, auth/scope/rate errors, preview→approval→execute
  conversion, duplicate/idempotent replay rate, launch and serverless
  success, active transports, watch duration. Gate: a trace initiated at
  an MCP tool call is retrievable end to end through API, outbox, and
  scheduler attempt.
- [ ] **B5.13 Hosted E2E and blue/green** (§26.4, §27.3). Real MCP + API +
  PostgreSQL + Redis; the thirteen-step §26.4 sequence through the MCP SDK
  client. Add `mcp-blue` upstreams; start the new image; run a real
  protocol smoke against its direct port; then switch and reload Nginx;
  drain old connections with a deadline; roll back to the prior digest on
  failure. **MCP deploy failure becomes fatal** — `scripts/deploy.sh`
  currently treats it as a warning (§5.9, §31). Gate: `mcp/tests/e2e/*`
  in CI; a deliberately broken MCP image fails the deploy and leaves the
  previous version serving.

**Exit gate (Phase B5):** every current flagship journey passes through
the new API with no feature loss; cross-tenant, scope, approval, and
idempotency tests pass; MCP failure blocks deployment; two replicas pass
load and restart tests.

---

## B6 — Operator and customer UI

Blueprint §20, Phase 9. Track A Phase 9 delivered the admin control-plane
page (`/dashboard/admin/control-plane`) with scheduler timelines, host
drains, reconciler findings, and scheduled tasks, plus its admin API. This
section completes §20.

**Current state (verified 2026-07-21):** `frontend/package.json` has none
of `@tanstack/react-query`, `@tanstack/react-table`,
`@tanstack/react-virtual`, `zod`, `openapi-typescript`, `openapi-fetch`,
`@axe-core/playwright`, or Storybook. Six Playwright specs exist.

- [ ] **B6.1 Typed client foundation** (§24.4). Add the packages above;
  generate `frontend/src/lib/api/generated/*` from the v1 OpenAPI; adopt
  React Query for server state. **Do not add a second component
  framework** — extend the existing tokens, Tailwind 4, CVA, Recharts,
  Framer Motion, Lucide, and Sonner (§4.4, §20.1). Gate: a CI check that
  the generated client is current; a lint rule banning Material UI, Ant,
  and Chakra imports.
- [ ] **B6.2 Admin overview and queue completion** (§20.2). Overview:
  scheduler/reconciler/outbox replica health, queue depth and p50/p95/p99
  queue latency, placements per second, conflicts, retries, failures, GPU
  capacity allocated/free/draining/stale by model and region, reconcile lag
  and open finding severity, command pending/claimed/dead-letter counts,
  MCP launch and serverless success rate. Queue: virtualized table,
  priority and fair-share key, requested resources and policy, durable
  queue reason and failed constraints, age, retry time, scheduling-attempt
  count, and a placement simulation/explanation drawer. Gate: a 10,000-row
  queue renders and scrolls without DOM overload.
- [ ] **B6.3 Placements and hosts** (§20.2, §20.4). Placement: active
  attempt timeline, score waterfall and candidate comparison, allocation/
  lease/command chain, stale and fenced attempts visually separated, trace
  link. Hosts: GPU capacity matrix by physical UUID and MIG child,
  allocatable versus active allocations, freshness and capability
  conditions, drain progress, temperature and utilization trends, command
  delivery and worker version, inventory topology, allocation list with
  tenant-safe identifiers, conditions with remediation, last heartbeat and
  observation session, agent version and rollout status, and a drain
  dialog that explicitly separates "stop new placements" from "evict
  workloads".
- [ ] **B6.4 Reconciliation and MCP activity** (§20.2). Findings feed
  grouped by resource, reason, and severity; desired/observed diff;
  automatic action and result; safe reconcile and retry controls; **no raw
  "fix SQL" button**. MCP: connected client identity, scopes, expiry and
  revoke; action-plan approvals and spend policy; tool success/error/
  latency trends; created-resource links; redacted audit table.
- [ ] **B6.5 Instance detail control-plane section** (§20.3). Plain-language
  current reason ("Queued because no healthy H100 with 80 GB is available
  in Ontario"), phase and conditions, attempt timeline, selected host/GPU
  aliases and score, lease health, desired-versus-observed badge, cost
  quote / wallet hold / live meter, and retry and reconcile actions when
  authorized. Customers see only their own resources and redacted
  infrastructure detail.
- [ ] **B6.6 MCP dashboard** (§20.5). Preserve the current Quick Connect
  flow; add standards-based OAuth connect as primary, a client card with
  scopes/expiry/last-used/revoke/spend policy, an action approval inbox, a
  live `create_instance` preview→approval→launch→watch demo, serverless
  tool examples, and tool audit with actionable fixes. **The token is never
  shown again after creation.**
- [ ] **B6.7 Components** (§20.6). `SchedulerHealthHero`,
  `QueueLatencyChart`, `GpuCapacityMatrix`, `PlacementScoreWaterfall`,
  `ConstraintMatrix`, `AttemptTimeline`, `LeaseStatusBadge`,
  `DesiredObservedDiff`, `ReconciliationFeed`, `CommandDeliveryBadge`,
  `McpActionAuditTable`, `ActionPlanReviewDialog`, `SpendPolicyEditor` —
  each with Storybook states.
- [ ] **B6.8 UX quality gates** (§20.7). WCAG 2.2 AA contrast and focus
  visibility; keyboard-accessible tables, dialogs, tabs, and charts; icon +
  text + colour for status, never colour alone; reduced-motion support;
  dark/light visual snapshots; 375 px through wide operations displays;
  explicitly designed skeleton, empty, stale, partial, permission-denied,
  and error states; screen-reader chart summaries; no raw id without a
  copy affordance and human label; destructive confirmations stating exact
  impact. Gate: `@axe-core/playwright` scans, keyboard-only journeys for
  approval/drain/reconcile, and visual snapshots in CI (§26.6).

**Exit gate (Phase B6):** operators can explain queue, placement, lease,
reconcile, and command state without database access; customers can
understand their own queue reason and MCP approval/spend state; visual and
accessibility gates pass.

---

## B7 — Observability and SLOs

Blueprint §25, Phase 11; `DA§9`.

**Current state (verified 2026-07-21):** `docker-compose.yml` contains
Jaeger only. No OTel Collector, Prometheus, Alertmanager, Grafana, Loki, or
Tempo. `pyproject.toml` has none of `structlog`, `tenacity`, or the OTel
instrumentation packages from §24.1. Track A landed `telemetry_latest`
(shared across replicas) and partitioned `telemetry_samples`, so the
domain-state half of §25.1 exists.

- [ ] **B7.1 Collector and backends** (§25.1, `DA§9`). Deploy the OTel
  Collector, Prometheus, Alertmanager, Grafana, Loki, and Tempo (Jaeger may
  remain as a trace UI during transition). Gate: a compose validation job;
  every service exports to the collector; a synthetic trace is retrievable
  in Tempo.
- [ ] **B7.2 Structured logging** (`DA§9.2`, §24.1). `structlog` JSON logs
  everywhere with timestamp, severity, environment, service, build,
  trace/span, tenant pseudonym where approved, job/attempt/lease/command/
  action ids, error code, and retryability. A **central redaction library**
  removes tokens, secrets, signed URLs, authorization headers, environment
  values, prompt bodies, and private host addresses. Track A's
  `setup_logging` already degrades to console on an unwritable path —
  preserve that. Gate: a test that pipes known secrets through every log
  helper and asserts none survive.
- [ ] **B7.3 Trace propagation** (§25.2, `DA§9.3`). W3C context across
  `MCP tool → API action plan → job/outbox → scheduler attempt → worker
  command → agent start → status/ACK → billing/event`, plus artifact
  upload/finalize. Trace ids stored; sensitive baggage not. Async work
  creates linked spans when the parent is no longer active. Sampling
  preserves errors, destructive MCP operations, slow placements, lease
  conflicts, and reconciliation failures. Gate: an end-to-end trace
  assembled in CI from a real launch.
- [ ] **B7.4 Metrics catalog** (§25.3, `DA§9.1`). Scheduler: queue depth
  and age by class/model/region, claim latency and expiry, filter
  rejections by reason, placement duration and conflict retries,
  **allocation constraint violations (must stay zero)**, preemption plans
  and outcomes, replica heartbeat. Worker: command fetch/claim/ACK latency,
  lease renew success and loss, image pull and start latency, runtime and
  volume preparation failures, observed-versus-desired mismatches, agent
  version and identity expiry. Reconciler: queue age, convergence duration,
  findings by type and severity, actions/retries/dead letters, stale
  observations. MCP: per B5.12. Billing: active meters versus running
  attempts, orphan and missing meter invariants, hold age and expiry,
  ledger lag and failure, wallet hard stops. Plus `DA§9.1`'s PostgreSQL,
  Redis, artifact, outbox, retrieval, and BigQuery indicators.
  **High-cardinality ids (`job_id`, `user_id`, prompt hash, artifact id,
  raw error text) never become metric labels** — they belong in logs and
  traces. Gate: a metric-registry test failing on a label whose
  cardinality class is unbounded.
- [ ] **B7.5 SLOs, error budgets, and alerts** (§25.4, `DA§17`). Encode the
  §25.4 table — four hard invariants at zero (duplicate active exclusive
  allocation; start accepted without valid current attempt/lease/fence;
  stale-fence mutation/route/secret/storage-write/billing acceptance;
  strict workload reassigned before definitive fencing) plus placement
  latency p95 ≤ 2 s, assignment-to-claim p95 ≤ two poll intervals,
  convergence 99% ≤ 60 s, command ACK p95 ≤ 15 s, MCP preview availability
  ≥ 99.95%, approved-launch success ≥ 99.9%, queue entries with a current
  reason 100%, billing meter consistency 100%, stale host removed within
  freshness + 5 s. Add `DA§17`'s data-quality indicators. Multi-window
  burn-rate alerts; page only on actionable user-impacting or invariant
  alerts; route trends to tickets. **A dashboard "0" caused by a broken
  pipeline must be visually distinguishable from a genuine zero**
  (`DA§17`). Gate: an invariant breach fires a page in a staging drill.
- [ ] **B7.6 Resolve `telemetry_snapshots`** (`DA§9.4`). Pick one and
  execute it: bounded downsampled business/SLA history (one- or
  five-minute summaries, partitioned, limited retention), or deprecation
  after B7.1 and B11 land. Do not leave it as an unbounded raw table in the
  authority cluster. Gate: whichever path is chosen is enforced by a
  retention task and a size alert.

---

## B8 — Deployment, edge, database operations, and release gates

Blueprint §21, §22, §23, §27, Phase 11; `DA§4.1`, `DA§4.3`, `DA§4.6`.

- [ ] **B8.1 Replica topology** (§21.1). Two API and two MCP instances;
  scheduler at ≥ 2 active replicas (safe only after Track A's canary→active
  flip); reconciler ≥ 2; outbox dispatcher ≥ 2; maintenance workers as
  row-claiming replicas; volume provisioner isolated and privileged;
  frontend and SSH gateway blue/green. Move ordinary service communication
  to private Docker networks; retain host networking only where a
  Headscale/host-route requirement is proven and documented. Gate: a
  rolling restart of every replica class creates no duplicate allocation,
  no unmetered execution, and no MCP launch outage — run as a repeated
  chaos job, not once.
- [ ] **B8.2 Health semantics completion** (§21.3). Track A landed
  `/livez`, `/readyz`, and `/startupz` for the API. Extend to every
  service: scheduler readiness proves it can claim a synthetic
  non-mutating probe or verify DB primitives plus heartbeat; reconciler
  readiness proves heartbeat and queue access; MCP readiness proves API
  auth metadata/JWKS reachable, Redis reachable if required, and a
  complete tool registry; worker readiness proves identity, API, GPU
  runtime, inventory, and mandatory capability probes. **PID-string health
  checks are removed.** Gate: a test per service that readiness goes false
  when its named dependency is broken and true when repaired.
- [ ] **B8.3 Deployment sequence and rollback** (§21.4, §21.5). Encode the
  thirteen-step sequence in `scripts/deploy.sh`. Rollback: binaries only
  within the declared schema compatibility window (Track A's
  `schema_compat.py` already declares it); a kill switch that stops the new
  write path without corrupting active attempts (Track A's
  `XCELSIOR_SCHEDULER_CLAIMS_ENABLED` is the model); current fenced
  attempts complete or reconcile; **no destructive downgrade during an
  incident** — forward-fix migrations only. Gate: a staged rollback drill.
- [ ] **B8.4 Fatal deployment gates** (§27.2). These abort promotion:
  environment validation, database backup/PITR check where configured,
  Alembic failure or incompatible schema, API readiness, scheduler/
  reconciler/outbox readiness, **MCP build/start/readiness/protocol smoke**,
  frontend readiness, Nginx config test, invariant smoke, and
  signature/SBOM/security policy. Optional visualization may be non-fatal
  only if explicitly classified optional *and* its degraded state alerts.
  Gate: each gate is proven to abort by deliberately breaking it once in
  CI.
- [ ] **B8.5 CI matrix completion** (§27.1). Add to Track A's
  `control-plane`, `test`, `compose`, and `supply-chain` jobs: a
  PostgreSQL + **Redis** service matrix, worker protocol contract tests
  (B14.1), MCP hosted E2E (B5.13), OpenAPI generation diff, frontend
  type/lint/unit/e2e/a11y/visual jobs, `pip-audit` and `npm audit`, and
  CodeQL/SAST. **Note the standing constraint: GitHub Actions has not run
  since ~2026-07-21 due to billing. Until that is restored, every gate
  above must also be runnable locally, and a green push is unverified.**
- [ ] **B8.6 Edge boundaries** (§22). MCP proxy: two keepalive upstreams,
  HTTP/1.1 Streamable HTTP, `proxy_buffering off` and
  `proxy_request_buffering off`, no caching, preserved `Mcp-Session-Id` and
  protocol-version headers, safe request-id and `traceparent`, bounded body
  and header timeouts, long read timeout **only** on the stream route, IP
  edge limits plus Redis principal/tool limits, OAuth metadata served
  without bearer auth, readiness checked before the upstream switch, and
  no sticky sessions. Public: per-route body sizes rather than a global
  500 MB allowance; non-idempotent requests are not retried to another
  upstream unless an idempotency key makes it safe; graceful drain for
  blue/green stream connections; SSE/WebSocket reconnect uses the
  persisted cursor from B4.6. The agent gateway (§22.3) is Track A's.
  Gate: structural config tests plus a live stream-drain drill.
- [ ] **B8.7 PostgreSQL production contract** (§23.1, `DA§4.1`, `DA§4.3`).
  Managed HA or equivalent in the required region; encrypted storage and
  TLS; continuous archiving and PITR; monitored replication lag,
  connections, locks, dead tuples, disk, WAL, and transaction age;
  PgBouncer **transaction** mode — safe only because Track A removed the
  pooled session advisory lock, and gated on a session-state audit;
  per-role `statement_timeout`, `lock_timeout`, and
  `idle_in_transaction_session_timeout` (Track A 11.4 set these per
  service); `application_name` per service and replica (also Track A 11.4);
  query statistics and slow-query review; autovacuum tuned for the
  high-churn queue, command, and outbox tables; automated partition
  lifecycle. Separate pools per workload class (`DA§4.3`): scheduler
  reservation (small, low timeout), API transaction, background/projector,
  retrieval, migration (session-capable, never PgBouncer), operator read.
  Gate: pool saturation and oldest-transaction alerts fire in a drill; a
  PgBouncer session-state audit test.
- [ ] **B8.8 Backup and recovery evidence** (`DA§4.6`). Managed automated
  backups and PITR; encrypted backup storage under a separate backup-admin
  boundary; retention aligned to finance, privacy, and contract; a
  **monthly automated restore into an isolated environment** with migration
  validation and invariant checks; quarterly application-level recovery
  drills covering PostgreSQL, object catalog, objects, and projection
  rebuilds; documented RPO/RTO for zone loss, accidental deletion, bad
  migration, credential compromise, and region loss; evidence of the last
  successful restore, its duration, and its integrity checks. Reclassify
  `scripts/backup-db.sh` as a development/manual logical backup.
  **A backup is proven by a restore, not by an upload.** Gate: the restore
  job is scheduled and its evidence is queryable.
- [ ] **B8.9 Redis production contract** (§23.2, `DA§5.3`). HA managed
  instance; separate logical domains for auth-ephemeral, limits, and cache
  with distinct ACL users and key prefixes (separate instances where
  blast-radius analysis warrants); per-domain eviction policy
  (`noeviction` with bounded TTL for auth and limits, LRU/LFU for cache);
  TLS and private networking. Gate: `DA§16.3` — atomic rate script under
  concurrency, a TTL on every key, no raw secret or PII in a key scan,
  memory-pressure alert before `noeviction` write failure, no retry storm
  into the PostgreSQL pool, and production refusing an in-memory fallback.

---

## B9 — Data plane honesty: Redis, artifacts, shared state

`DA§5`, `DA§6`, `DA§10`, `DA§2.3`, `DA§2.4`, `DA§2.7`.

- [x] **B9.1 Redis rate-limit policy is explicit** (`DA§2.3`, `DA§5.5`).
  `serverless/limits.py` declares `strict-deny` / `upstream-enforced` /
  development-only local deque, production rejects an undefined policy, and
  the Redis path is bounded by a fast socket timeout so an outage fails
  closed rather than stalling. Landed before Track B; recorded here for
  completeness. **Residual:** `DA§5.4`'s key contract (environment,
  version, tenant, hashed identifier; TTL set in the same atomic operation
  as creation) is not yet enforced repository-wide — tracked as B9.1a.
- [x] **B9.1a Key and value contract** (2026-07-22, `DA§5.4`) — new
  `cache_keys.py` builds every key as
  `xc:{env}:v1:{family}[:{public}...][:{sha256(secret)}]`.
  **Security finding closed:** `oauth_service._cache_key` interpolated the
  credential *itself* into the key name —
  `xcelsior:oauth:access_token:<live access token>`, and the same for
  authorization codes, device codes, and the device-poll counter
  (`{device_code}:{ip}`). A Redis key name is not private: `SCAN`,
  `MONITOR`, the slowlog, RDB/AOF files on disk, managed-service support
  tooling, and any keyspace exporter all see it, so read-only access to
  Redis *metadata alone* yielded live bearer tokens. Notably
  `hash_token()` already existed in that module and was already applied to
  refresh tokens — the correct treatment was present and applied
  inconsistently. The serverless limiter had the same shape via
  `dashboard-test:{owner_id}`, where owner ids are frequently email
  addresses (`DA§5.4` forbids those in key names too).
  Also fixed: `RedisAuthCache.incr` issued `INCR` then a conditional
  `EXPIRE` in two round trips, so a crash between them left an **immortal
  key** — for an abuse counter that means a principal stays locked out
  until someone deletes it by hand. Now one `MULTI` with `EXPIRE ... NX`,
  so later increments do not extend the window. Keys now carry the
  environment, so two deployments sharing an instance cannot read each
  other's auth state; `XCELSIOR_AUTH_CACHE_PREFIX` is retired because a
  per-deployment override would defeat the contract.
  Gate: `tests/test_cache_key_contract.py` (27 tests) — including a
  repo-wide static check that no module builds a Redis key by hand, which
  **found a third construction site** (`serverless/limits.py`) that the
  manual survey missed. That one is a process-local deque key rather than
  Redis, and was routed through the same helper anyway: one rule with no
  exception to remember, and already correct if that store is ever shared.
  Public key segments are validated, so `cache_key("ratelimit",
  owner_email)` raises rather than silently reintroducing the leak.
  Regressions green: `test_api` 157, oauth-migration 3, rate-limit-policy
  10, limits-webhooks 9, snapshot-rate-limit 6, startup-validation 26;
  pyright clean. ✔
  **Operator note — this is a key-format change, not a data migration.**
  Keys written under the old format are unreachable and simply expire.
  On deploy: in-flight authorization codes and device codes must be
  re-requested (TTLs 300 s / 900 s), and opaque access tokens are
  re-issued on next use (TTL 900 s). Refresh tokens live in PostgreSQL and
  recover automatically, so clients holding one self-heal. A dual-read
  transition was rejected deliberately: constructing the legacy key
  requires putting the raw secret back into a Redis command, which is the
  exact exposure being removed.
- [~] **B9.2 Artifact catalog completion** (`DA§6`). **Landed:** migration
  `064_storage_catalog` created `storage.artifacts` (full `DA§6.2` state
  CHECK including `upload_authorized`, `uploaded_unverified`, `corrupt`,
  `quarantined`, `abandoned`, `delete_failed`, plus `legal_hold`,
  `retain_until`, `residency_region`, `crc32c`/`sha256`, `object_generation`,
  `version`), `artifact_upload_sessions`, `artifact_replicas`, and
  `artifact_deletion_jobs` with the one-active-deletion partial unique;
  the store fails closed when a configured remote backend is unavailable
  (`StorageUnavailable`, no `file://` fallback, list errors surface as 5xx);
  `boto3` is a declared dependency; a janitor sweeps expired sessions and
  processes deletion jobs. **Correctness and compliance work landed
  2026-07-22 (B9.2a–d below); the remaining residual is the structural
  provider-adapter split, which refactors working code and fixes no live
  bug.** Sub-items:
  - [~] **B9.2a Provider error taxonomy** (2026-07-22, `DA§6.6`/`DA§12.4`).
    The correctness half of the adapter split landed first, because it
    fixes a live orphan bug. `delete_object` returned `False` on any
    failure, and the deletion worker **never checks the return value** — it
    relies on an exception to mark the job `delete_failed` for retry. So a
    failed delete was recorded as a completed one: the catalog said
    `deleted` while the bytes remained, an orphan the inventory scan (B9.2d)
    would later have to find. `delete_object` now **raises**
    `StorageUnavailable` on a genuine failure and treats an already-absent
    object as idempotent success (DELETE's end state already holds). A
    shared `_provider_error_code` helper gives `head_object` and
    `delete_object` one classification of not-found vs. ambiguous
    (`DA§6.6`), instead of each reimplementing the `response['Error']['Code']`
    lookup and drifting. Gate: 4 error-taxonomy tests in
    `tests/test_artifact_finalize_verification.py` driving fake boto
    exceptions. ✔
    **Residual (structural, no live bug):** the `storage/providers/base.py`
    protocol, the `s3.py`/`gcs.py` split, `storage/repository.py`
    compare-and-swap transitions, and `storage/workers.py`. This is a
    refactor of working code into the companion's module layout, deferred
    behind the correctness fixes per B0.3 rule 21.
  - [~] **B9.2b Finalize verification** (2026-07-22, `DA§6.2`/`DA§6.4`/`DA§6.6`).
    **Three defects closed.** Finalize did a provider HEAD but nothing
    else: the session SELECT fetched `artifact_id, tenant_id, principal_id,
    expires_at, completed_at`, so `expected_size_bytes` and
    `expected_sha256` were written at session creation and **never read
    again** — a truncated, substituted, or corrupt object became
    `available` unchallenged, and the caller's claimed checksum was stored
    verbatim without being compared to anything. Now finalize verifies
    size and checksum against the session, marks a mismatch `corrupt` (the
    bytes exist and are wrong — distinct from an upload that never
    arrived), and refuses finalize when a checksum was declared but not
    supplied. A control test proves a *matching* upload still finalizes, so
    the refusals can't pass on a worker that never succeeds.
    (2) `sha256` could be populated from a provider ETag — which for a
    multipart upload is not a content hash — making the column
    unverifiable; only a real supplied/expected hash is stored now.
    (3) `head_object` returned `None` for **every** exception, and finalize
    reads `None` as "object absent" and abandons the artifact — so a
    transient 429/503 during finalize discarded a good upload. It now
    distinguishes a definite 404/NoSuchKey (→ `None`) from an ambiguous
    failure (→ `StorageUnavailable`), per `DA§6.6`.
    **Found while fixing (3): the pre-existing abandon-on-missing never
    persisted either.** The state UPDATE and its `raise` were in the same
    `control_plane_transaction`, so the raise rolled back the UPDATE — the
    error message said "not found" while the row stayed `requested`.
    Fixed with a `_FinalizeRejected` carrier that records the terminal
    state in its own transaction, outside the one being rolled back.
    Gate: `tests/test_artifact_finalize_verification.py` (7 tests); api 157
    green; pyright clean. ✔
    **Residual:** the `upload_authorized → uploading → uploaded_unverified`
    intermediate states, the asynchronous verifier for clients that cannot
    supply a strong checksum, and generation/version precondition checks.
    Signed download **by catalog id only** (never a user-supplied
    bucket/key, never prefix-listing authorization) is a separate route
    audit, still open.
  - [~] **B9.2c Retention, legal hold, and deletion** (`DA§6.5`).
    **Two compliance defects closed (2026-07-22):**
    (1) the catalog deletion worker **never consulted `legal_hold`** — an
    artifact under hold was deleted like any other, which is precisely
    what a hold exists to prevent (`DA§8.8`). It is now a terminal refusal
    with a stated reason rather than a retry, because backing off would
    re-attempt the delete every cycle for the life of the hold and bury
    the signal.
    (2) worse: the legacy object-age sweep ran **unconditionally**,
    outside the catalog branch. It deleted bytes by prefix listing with no
    catalog lookup, no legal-hold check, and no state check — so an
    `available` artifact under hold and inside its retention window lost
    its bytes **with no deletion request at all**, and the catalog row was
    never updated, leaving a row claiming an object that no longer
    existed. `DA§6.5` is explicit that PostgreSQL determines eligibility
    and lifecycle rules are only a safety net; the sweep now runs only
    when the catalog is inactive.
    Also added: `retain_until` defers deletion until the window passes.
    Gate: `tests/test_artifact_retention_authority.py` (4 tests), each
    proven necessary by reverting its fix — including a control test that
    an ordinary artifact *is* still deleted, so the two refusal tests
    cannot pass on a worker that simply never deletes anything.
    **Residual:** per-replica deletion with generation preconditions, the
    deletion tombstone for downstream projections, and `SKIP LOCKED`
    claiming on the deletion queue. Bucket Lock still requires its own
    review before use.
  - [~] **B9.2d Object consistency** (2026-07-22, `DA§12.4`/`DA§6.6`).
    Immutable keys are **already satisfied**: catalog uploads use
    `{artifact_type}/{artifact_id}_{filename}` embedding the random
    `artifact_id`, and migration 064's `UNIQUE (provider, bucket,
    object_key)` makes a collision impossible, so an available object is
    never overwritten in place. Added `ArtifactManager.reconcile_inventory`
    — an off-request-path scan that flags `available`/`expiring` rows whose
    bytes have vanished (`missing_bytes`, error): the catalog insists the
    object exists, a download 404s, and nothing on the request path
    notices. Report-only, records into `reconciliation_findings` (B0.3
    rule 17) — a transient provider blip must not become data loss, so an
    **ambiguous** HEAD (`StorageUnavailable`) never produces a finding,
    only a definite absence does; a test pins that. Deduped, legal-hold
    rows excluded, artifact state never mutated. Wired as the durable
    `artifact_inventory_reconcile` task (600 s). Gate:
    `tests/test_artifact_inventory_reconcile.py` (6 tests). ✔
    **Residual:** orphan-bytes detection (object with no catalog row) needs
    a paginated provider lister, deferred to the B9.2a adapter split;
    generation preconditions on mutating operations.
  - Gate: `DA§16.4` in full — startup with missing SDK/credential/bucket/
    permission/KMS rights; signed-URL expiry and tenant authorization;
    resumable interruption and resume; checksum, size, content-type, and
    generation-race failures; finalize idempotency and duplicate session;
    primary outage with verified replica policy; delete under legal hold,
    precondition failure, and partial replica failure; orphan and
    missing-object detection; lifecycle in a disposable bucket; restore
    catalog plus objects with checksum verification; and **no production
    path returns `file://` or a successful empty listing on provider
    error.**
- [~] **B9.3 Shared state consolidation completion** (`DA§10.1`,
  `DA§2.7`). **Landed:** migration `060_shared_state_to_pg` created
  `ln_deposits` and `slurm_job_mappings` in PostgreSQL, and
  `slurm_adapter.py` reads and writes the table. **Residual — the tables do
  not meet the companion's contract:**
  - [x] **B9.3a Money and time types** (2026-07-22, `DA§4.4` rules 5–6) —
    migration `066_ln_deposits_typed_money_and_time` adds
    `amount_cad_minor BIGINT` (cents), `btc_cad_rate_exact NUMERIC(20,8)`,
    and `created_at_ts`/`expires_at_ts`/`paid_at_ts`/`credited_at_ts`
    `TIMESTAMPTZ`. Expand-only and rolling-deploy safe: a
    `BEFORE INSERT OR UPDATE` projection trigger derives every typed
    column from its legacy twin when not supplied, so an **un-upgraded
    replica still writing only floats cannot leave a money row without its
    exact representation** — the same pattern Track A's migration 059 used.
    Backfill is batched with the termination invariant migration 054 had to
    be repaired for, plus hard verification that aborts on an unprojected
    row *and* on any row losing more than half a cent in conversion —
    a ledger discrepancy must surface, not be rounded away. Zero-epoch
    sentinels (`paid_at = 0`) project to NULL, not 1970. CHECK constraints
    added `NOT VALID` → `VALIDATE`.
    `lightning.py` now writes exact values directly (`cad_to_minor` via
    `Decimal(str(x))` with ROUND_HALF_UP — `int(1.15 * 100)` is **114**,
    which a test pins as the hazard being avoided) and credits the wallet
    from integer cents rather than the stored float.
    Verified by executing against real PostgreSQL: backfill exactness
    (`0.07` → 7 cents), trigger projection on legacy-only INSERT *and*
    UPDATE, explicit typed writes winning over derivation, and down→up over
    a populated table with zero unprojected rows.
    Gate: `tests/test_lightning_deposits.py` (31 tests). Regressions green:
    from-empty 3, production-snapshot 3, no-runtime-DDL 6,
    control-plane-schema 40, db-service-roles 48, bg-worker 2; pyright clean.
    Legacy float columns are retained for the rollout and dropped at
    contract phase, where `created_at_ts` is renamed back to `created_at`
    (**B16.2**). ✔
    **Scope boundary found:** the *wallet* itself is still float-based —
    `wallets.balance_cad`, `wallet_transactions`, and `billing_cycles`
    (`amount_cad`, `rate_per_hour`, `token_cost_cad`) are all
    `double precision`. `DA§4.4` rule 6 applies to them equally, and it is
    a materially larger migration touching the whole billing engine.
    Tracked as **B9.5** below; the Lightning path now hands that engine an
    exact value, which is the most it can do unilaterally.
  - [x] **B9.3b-1 Lightning credit path made functional and exactly-once**
    (2026-07-22). **The module was entirely non-functional against
    PostgreSQL** — migration `060_shared_state_to_pg` moved the table but
    the query layer was never converted or exercised: `dict_row` was
    imported and unused, four functions called `dict(row)` on the pool's
    `tuple_row` connections (`ValueError`), and three carried SQLite `?`
    placeholders (`ProgrammingError`). `get_pending_deposits()` is called
    *outside* any try block, so the first pending row killed the sweep for
    every customer: **no paid Lightning deposit was ever credited.**
    Verified by executing the real path against a real database, not by
    reading it.
    Fixing only the placeholders would have been worse than the bug: with
    `mark_credited` failing after a successful credit, the deposit stays
    `paid` and the 5-second watcher re-credits it forever — and **neither**
    credit callback passed an `idempotency_key`, though
    `BillingEngine.deposit` has supported one all along.
    Landed: per-cursor `dict_row` helpers (never mutating the pooled
    connection's factory), `%s` placeholders throughout, CAS-guarded
    `mark_credited`, per-deposit isolation so one bad row cannot starve the
    fleet, credit-then-mark ordering documented (the reverse loses customer
    money), and a mandatory `credit_idempotency_key(deposit_id)` passed by
    both call sites. Dead `LN_DB_PATH` SQLite config removed. TLS was
    checked and is **already correct** — `ssl.create_default_context()`
    verifies chain and hostname and is passed to `urlopen`, so the
    companion's §2.7 claim is stale; a test now pins it.
    Gate: `tests/test_lightning_deposits.py` (16 tests against real
    PostgreSQL). Each fix was proven necessary by reverting it and
    confirming the suite fails — reverting the `mark_credited` placeholder
    alone fails 7 tests including `test_sweep_does_not_recredit_on_the_next_pass`.
    A structural AST gate fails the build if any credit callback omits the
    idempotency key. Pyright clean; `test_bg_worker` 2,
    `test_billing_lifecycle_domain` 9, `test_wallet_holds` 12 green. ✔
    **Impact: none — the defect was latent** (confirmed with the user
    2026-07-22). Lightning reports unavailable in production and no real
    orders have been placed, so no deposit ever reached `paid` and no
    customer was left uncredited. The bug would have bitten on the first
    real deposit. The `lightning_reconcile` sweep (B9.3b-3) now surfaces
    the condition automatically if it ever recurs.
  - [x] **B9.3b-2 Lightning schema contract** (2026-07-22) — migration
    `067_shared_state_contract` adds `tenant_id` (transitional single-user
    projection from `customer_id`, same rule migration 054 used for jobs),
    an explicit `currency` (an amount without one is not money),
    `wallet_ledger_entry_id` so a credited deposit points at the exact
    ledger row, `UNIQUE NULLS NOT DISTINCT (payment_hash)` so two rows
    cannot claim one provider payment, and a `status` CHECK. Plus an
    **immutability trigger**: `customer_id`, `currency`, `amount_msat`,
    `payment_hash`, and `expires_at_ts` cannot change after insert — those
    are the terms the customer paid against, and an UPDATE editing them
    rewrites history under a settled payment. Lifecycle transitions stay
    allowed (proven by test). `mark_credited` accepts the ledger entry id,
    and `process_ln_deposits` threads it through from the credit
    callback's return value while still working when a callback returns
    nothing. Note `payment_hash` is already NOT NULL from 060, so
    `NULLS NOT DISTINCT` is currently inert — stated anyway so the
    invariant survives a future migration relaxing that column.
    Gate: `tests/test_shared_state_contract.py`, driving each constraint
    both ways. ✔
    **Not done here — single-transaction `paid → credited`.** The
    companion asks for the status flip, the ledger posting, and the outbox
    row in one transaction. `BillingEngine._conn` always checks out its own
    pooled connection, so this needs a `deposit(..., conn=...)` variant —
    a billing-engine change, tracked with **B9.5**. The required
    properties already hold without it: the credit is idempotent on
    `credit_idempotency_key(deposit_id)` and `mark_credited` is a CAS on
    `status = 'paid'`, so a crash anywhere in the sequence replays safely
    (both companion §10.1 gates are covered by tests). The single
    transaction removes the window entirely rather than making it safe.
    **Also outstanding:** the `paid but not credited` / `credited but
    provider unknown` / amount-mismatch reconciliation sweep, and making
    `XCELSIOR_LN_CA_CERT` required in production via B18.1.
  - [x] **B9.3b-3 Lightning reconciliation sweep** (2026-07-22, `DA§10.1`)
    — `control_plane/billing_reconcile.py` covers all four §10.1
    conditions: `paid_not_credited` (error), `credited_without_ledger_entry`
    (warning), `stuck_pending_past_expiry` (info), and `amount_mismatch`
    (error, >½¢ between the exact minor-unit and legacy float columns).
    `paid_not_credited` is the exact condition the pre-2026-07-22 module
    produced on **every** deposit, and it is invisible without this check
    because nothing errors — the row just sits in `paid` while the
    customer waits.
    **Report-only, never repairs** (B0.3 rule 17): crediting from a
    reconciler would be a second, untested money path competing with
    `process_ln_deposits`. `DA§8.7` states the same rule for the warehouse.
    A test asserts the sweep leaves deposit state byte-identical.
    Findings land in the existing `reconciliation_findings` table, so they
    surface in the admin UI Track A Phase 9 already built rather than in a
    second findings authority nobody looks at; same dedupe contract (one
    open finding per resource+type) and auto-resolve when the condition
    clears. Registered as a durable `scheduled_tasks` entry
    (`lightning_reconcile`, 300 s) so it survives restarts and only one
    replica runs it per interval — asserted by an AST gate, since a
    process timer would silently satisfy neither.
    **Bug found and fixed in the same pass:** bounding the scan (blueprint
    §23.1 — a reconciler must not become the oldest transaction in the
    database) initially broke `_resolve_cleared`, which resolved any
    finding absent from the current pass. With a bounded scan, absence
    also means "never looked at", so a backlog past the scan limit would
    silently close real findings — precisely when they matter most.
    Resolution is now scoped to scanned ids; proven by a `scan_limit=1`
    test that fails when the scoping is reverted.
    Gate: `tests/test_ln_deposit_reconcile.py` (16 tests). Regressions
    green: lightning 31, shared-state 25, control-plane-reconcile 18,
    db-service-roles 48, bg-worker 2; pyright clean. ✔
    New env: `XCELSIOR_LN_CREDIT_GRACE_SEC` (300),
    `XCELSIOR_LN_EXPIRY_GRACE_SEC` (3600),
    `XCELSIOR_LN_RECONCILE_SCAN_LIMIT` (5000) — to be added to
    `.env.example` and the startup validator with **B18.1**.
  - [x] **B9.3c Slurm mapping contract** (2026-07-22) — migration `067`
    adds `tenant_id`, `cluster_id`, `xcelsior_attempt_id`, `desired_state`,
    `observed_state`, `submit_idempotency_key`, `version`, and typed
    timestamps, with `UNIQUE (cluster_id, slurm_job_id)`,
    `UNIQUE (xcelsior_attempt_id)`, and
    `UNIQUE (tenant_id, submit_idempotency_key)`. **Three real defects
    closed:**
    (1) `sync_slurm_statuses` ran `DELETE FROM slurm_job_mappings` on
    completion, destroying the only record that an external job ever ran
    for a tenant — companion §10.1 requires "historical mappings remain
    queryable and auditable"; terminal mappings are now stamped with
    `terminal_at` and drop out of the poll set instead.
    (2) `register_slurm_job` was `ON CONFLICT (xcelsior_job_id) DO UPDATE
    SET slurm_job_id` — last-writer-wins, silently re-pointing a job at a
    different external id and orphaning the first; it is now idempotency-
    keyed, version-bumping, and refuses to touch a settled mapping.
    (3) nothing stopped one Slurm job serving two Xcelsior jobs — both
    would act on its status and both would bill for it; now a unique index.
    Added the missing **desired → cluster** direction: `set_slurm_desired_state`
    plus a `cancel_callback` in the sync loop, so a cancellation is
    expressed as state rather than by deleting the row and hoping the
    external job notices. Per-mapping isolation so one unreachable cluster
    cannot starve the rest, and `_load_slurm_map` no longer blanks its
    cache on a transient DB error. `SLURM_MAP_FILE` removed.
    Gate: `tests/test_shared_state_contract.py` (25 tests) +
    `tests/test_slurm_adapter.py` (53). The stale
    `TestSlurmJobMap::test_register_and_load` monkeypatched `SLURM_MAP_FILE`
    — a constant unused since migration 060 — and now drives the real
    table and cleans up its own rows (B0.2 rule 14). ✔
  - Gate (all of B9.3): duplicate Lightning provider notifications create
    exactly one ledger credit; a crash between payment observation and
    response is safe on replay; concurrent Slurm updates preserve one
    mapping per attempt and per external job; **no production code writes an
    `ln_deposits` SQLite file or `slurm_jobs.json`** (static tests). ✔
- [x] **B9.4 Approved SQLite boundary** (2026-07-22, `DA§10.2`) — the
  worker journal is now documented as **NODE-LOCAL RECOVERY STATE — NOT
  CONTROL-PLANE AUTHORITY**, stating what it cannot do (grant a lease,
  extend a fence, make an attempt current — all PostgreSQL rows) and that
  losing it must result in re-adoption or fencing by the control plane,
  never unilateral worker action. Note it is an atomically-replaced JSON
  document, not SQLite: §10.2 *permits* SQLite here, it does not require
  it, and a single-writer snapshot via `os.replace()` gives the needed
  durability without adding a database to the agent bundle.
  Gate: `tests/test_local_state_boundary.py` (6 tests) — an allow-list of
  the three modules that may import `sqlite3` (each with its §10.2
  reason), a staleness check so a retired exception is removed rather than
  left as dead config, an assertion that `lightning.py` and
  `slurm_adapter.py` never regain `sqlite3`/`LN_DB_PATH`/`SLURM_MAP_FILE`,
  and a check that the production startup validator still refuses a
  non-postgres backend — which is the only reason the `db.py` and
  `bitcoin.py` exceptions are safe. `scripts/local/` is exempt as
  workstation tooling, with a companion test asserting it is not
  referenced by any Dockerfile or compose file, so the exemption cannot
  quietly become untrue. Proven to fire by planting a module that imports
  `sqlite3` and confirming failure. ✔
### B9.5 — Ledger-wide money typing (`DA§4.4` rule 6)

Found while landing B9.3a, and **larger than that note implied**: a survey
on 2026-07-22 found **~50 `double precision` money columns across ~25
tables**, not the handful originally listed. Converting all of them in one
pass would be irresponsible, so this is staged by compounding risk.

The defect is demonstrated, not theoretical. Against this database:

```
1000 postings of $0.07  ->  float sum 69.99999999999966   (exact: 70.00)
                            running float balance likewise 69.99999999999966
```

Per-operation error is tiny; the problems are that it **compounds**, and
that `sum()` over the ledger stops equalling the stored running balance —
which is precisely the comparison `DA§8.7`'s finance reconciliation makes.

**Design note (differs from migration 066).** In `066` the legacy float was
authoritative and the typed column was derived. That is wrong here:
deriving `balance_minor` from an already-drifted `balance_cad` yields a
rounded copy of a wrong number. The **minor column must be authoritative**
and the float projected from it, so the arithmetic itself happens in
integers. The projection trigger must therefore be bidirectional — if the
minor column changed, derive the float from it; if only the float changed
(an un-upgraded replica), derive the minor from it — or a rolling deploy
loses the old replica's write.

Sequencing: land **before** B11.4 (finance reconciliation), or the
warehouse reconciles against a source that cannot be exactly summed.

- [x] **B9.5a Wallet ledger core** (2026-07-22) — migration
  `068_wallet_ledger_micro_units` adds `*_micros BIGINT` to `wallets`
  (balance, deposited, spent, refunded, auto-topup amount/threshold),
  `wallet_transactions` (`amount`, `balance_after`), and `wallet_holds`
  (`amount`). `billing.py`'s three balance-arithmetic sites (deposit,
  charge, refund) now increment the integer column, so the accumulation
  itself is exact. Shared `money.py` owns the conversions.
  **The unit is micro-CAD (1e-6), not cents — and that correction came
  from a test.** The first implementation used cents and broke
  `test_billing_tick_integration`: a real per-tick GPU charge of $0.0073
  rounds to $0.01 in cents, a **37% overcharge repeated every tick**.
  Xcelsior meters GPU-seconds and tokens, so the ledger's unit must be at
  least as fine as the smallest amount the business meters. Migration 068
  was amended rather than fixed-forward because it existed only in the
  local pytest database — shipping a known-wrong money migration and
  correcting it later would have been the worse trade. `ln_deposits`
  keeps cents (066): deposits have a $1 minimum, are never sub-cent, and
  whole cents convert to micros exactly, so the two scales meet cleanly.
  **Authority direction is inverted versus 066**, deliberately: deriving
  `balance_micros` from an already-drifted `balance_cad` would just be a
  rounded copy of a wrong number, so the integer column is authoritative
  and the float is projected from it. The trigger is therefore
  **bidirectional** — a change to the micros column derives the float, but
  a write that touches only the float (an un-upgraded replica mid-rolling-
  deploy) derives the micros instead. A one-way trigger would silently
  discard the old replica's write. Both directions are proven by test,
  including that the legacy branch deliberately does *not* reproject the
  float, since clobbering it would discard the very write it exists to
  preserve.
  Gate: `tests/test_wallet_micro_units.py` (17 tests) — exact conversion
  including `int(1.15 * 100) == 114` as the pinned hazard, 1000 × 7¢
  landing on exactly 70.000000 where the float column gives
  69.99999999999966, and a 200-posting random-sequence property test that
  `sum(amount_micros)` equals the stored balance, which is the equality
  `DA§8.7`'s finance reconciliation depends on. Regressions green:
  billing 28, billing-math-properties 15, billing-tick-integration 2 (was
  failing), billing-periodic-harness 3 (was failing), wallet-holds 12,
  hold-expiry 8, lifecycle-domain 9, lightning 31, ln-reconcile 16,
  from-empty 3, production-snapshot 3, db-service-roles 48, api 157;
  pyright clean. ✔
- [ ] **B9.5b+c Settlement chain — ONE coupled migration** (do not split).
  **Coupling confirmed 2026-07-22 by reading `billing.py`:** an invoice is
  not stored money that can be converted in isolation — it is *computed*
  as `SUM(billing_cycles.amount_cad)` grouped into line items
  (`billing.py` ~line 682), then accumulated into the invoice total with
  `ca_total += cost` in Python floats (~line 714), and that total feeds
  `payout_splits` and the `fintrac_reports` regulatory threshold.
  Converting `billing_cycles` to integers only removes drift if the
  invoice aggregation sums the **integer** column and the Python
  accumulation is integer too; otherwise the exact per-row amounts are
  re-summed through a float and the benefit is lost. So these tables move
  together:
  - line items: `billing_cycles` (`amount_cad`, `rate_per_hour`,
    `token_cost_cad`), `usage_meters` (`total_cost_cad`,
    `base_rate_per_hour`), `serverless_jobs.cost_cad`,
    `serverless_token_ledger.cost_cad`;
  - settlement: `invoices` (7 money columns), `payout_ledger`,
    `payout_splits`, `fintrac_reports.trigger_amount_cad`.
  Same expand-contract + bidirectional-trigger pattern as `068`, but the
  work is in the **read path**: `_generate_invoice` must aggregate the
  minor column and carry integers through to the stored invoice and the
  payout split. Per-row storage as a float does not compound the way the
  wallet balance did (each row is written once, not incremented in place),
  which is why B9.5a — the accumulating wallet — was done first and this
  is lower-severity. It is nonetheless **customer-facing and
  FINTRAC-reported**, so it is deliberately *not* rushed: a half-conversion
  that leaves the invoice float-summing an int-authoritative column is
  worse than the status quo. Gate: an invoice generated from a set of
  billing_cycles reconciles to the exact integer sum of those rows, and
  `payout_split` shares sum to the invoice total with zero residual cent.
- [ ] **B9.5d Rate cards** — `gpu_pricing.base_rate_cad`,
  `spot_prices.price`, `storage_billing_rates.rate_cad_per_gb_hr`,
  `reservations`/`reserved_commitments` rates. These are *rates*, not
  amounts: `NUMERIC` rather than minor units, and lower urgency because
  they are inputs recomputed each time rather than accumulated.
- [ ] **B9.5e Contract** — drop the legacy float columns and their
  projection triggers once every reader is on the minor columns
  (folds into **B16.2**).

---

## B10 — Semantic retrieval and semantic cache v2

`DA§7`, companion Phase 4. **Nothing in this section is built.**
`serverless/semantic_cache.py` still serves `difflib.SequenceMatcher`
similarity over the latest 32 prompts — the exact unsafe behaviour
`DA§2.5` documents.

- [ ] **B10.1 Model capability probe first — no schema before evidence**
  (`DA§7.2`, `DA§22.11-12`). The declared inventory is
  `/mnt/storage/models/staging/2026H2/hf/VoyageAI_voyage-3/`,
  `.../VoyageAI_voyage-3-code/`, and
  `.../Cohere_cohere-rerank-v3.5/`. `DA§7.2` records these as **declared
  inventory, not validated runtime capability** — the paths were absent
  from the assessment machine. Before any migration fixes a dimension:
  verify the mount exists in the intended retrieval deployment, the format
  loads in the chosen runtime, the tokenizer and inference code match the
  artifact, and record output dimension, normalization, pooling, max input
  length, query-versus-document input convention, license, GPU/CPU memory,
  warm-up, throughput, batching, concurrency, a deterministic content
  hash, and the reranker's input format, candidate maximum, truncation,
  and output calibration. **There is no silent switch to a remote
  embedding API or a different local model**; a failed probe makes the
  capability unhealthy and leaves queued embedding work retryable with an
  explicit reason. Gate: `retrieval/models.py` probe + a test that a hash
  or dimension mismatch prevents readiness rather than falling back.
- [ ] **B10.2 Migration — pgvector and retrieval schema** (`DA§7.3` at the
  live head). `CREATE EXTENSION vector`; `retrieval` schema;
  `embedding_models`, `sources`, `chunks` (with generated `tsvector` and
  GIN), `chunk_embeddings`, `embedding_jobs`. The vector typmod is set
  **only** from B10.1's verified dimension — do not copy `1024` from the
  companion's example. Create the ANN index as a per-model partial
  expression index after verification; record HNSW build time, index size,
  recall against an exact-search set, write throughput, and query latency.
  `UNIQUE NULLS NOT DISTINCT` requires PostgreSQL 15+; confirm the managed
  major version or substitute explicit partial unique indexes.
- [ ] **B10.3 Retrieval service** (`DA§7.1`, `DA§7.4`, `DA§7.5`, `DA§14.1`).
  `retrieval/contracts.py`, `models.py`, `embeddings.py`, `rerank.py`,
  `chunking.py`, `repository.py`, `worker.py`. Query path: authorize
  tenant/source scope → normalize and exact-key lookup → lexical
  candidates from `tsvector` → vector candidates from pgvector → merge
  20–100 candidates by reciprocal-rank fusion → rerank the small merged set
  → enforce the score/policy threshold → return a source-backed result or
  execute inference. Security filters run in **both** candidate queries,
  before similarity. Indexing is asynchronous and idempotent with
  deterministic chunk ids, `SKIP LOCKED` job claims under a bounded lease,
  and source readiness that flips only when the expected chunk count and
  model revision are complete. **An embedding is derived sensitive data**:
  it inherits the source's tenant, classification, residency, retention,
  and deletion policy; query embeddings are transient and never appear in
  general logs or traces; the local model service never forwards source
  text to an external API. Replace `ai_assistant.py`'s FTS-only
  `search_docs` with the hybrid service and attach citations and source
  versions.
- [ ] **B10.4 Search/vector consistency** (`DA§12.6`). Source row and
  outbox event commit together; embeddings are projections. Reindex keeps
  the old model active while the new version builds under a distinct
  model/source version, passes evaluation, is activated by an atomic
  registry pointer, and ages out old vectors after a rollback window.
  Deletion marks the source unavailable in PostgreSQL immediately so the
  API excludes it before vector deletion completes; a tombstone worker
  removes projections and monitors deletion lag.
- [ ] **B10.5 Migration — semantic cache v2** (`DA§7.6`, immediately after
  B10.2). `retrieval.inference_cache_entries` with the full context
  identity: tenant, endpoint id and revision, exact context hash, base
  model revision, adapter revision, tool schema hash, response schema hash,
  policy revision, sampling hash, retrieval context hash, quality state,
  expiry, invalidation. The exact key is a keyed hash over canonical JSON
  covering tenant policy scope, endpoint id and revision, base model and
  immutable revision, LoRA/adaptor ids and revisions, normalized system/
  developer/user messages, tool definitions and tool-choice policy,
  response schema, safety/policy revision, sampling parameters and seed,
  maximum output settings, retrieval corpus versions, and the cache schema
  version. Sensitive prompt material never appears in a Redis key name.
- [ ] **B10.6 Cache correctness before savings** (`DA§7.6`). Stop serving
  `SequenceMatcher` hits. The first lookup is always the exact key.
  Semantic reuse happens only after an exact miss and only where endpoint
  policy permits, with **default deny** for tool calls or side-effecting
  responses, authentication or authorization decisions, pricing, billing,
  legal, safety, or rapidly changing facts, personalized or sensitive
  prompts without proven isolation and deletion, high-temperature
  generation, and requests whose external retrieval source versions
  differ. A candidate must match every hard context column before distance
  search. Per-endpoint calibrated thresholds, never one global number.
  **Shadow first:** log the candidate and reason without serving it,
  compare correctness and saved cost, then enable per endpoint behind
  explicit policy. `serverless_semantic_cache` stays readable during
  transition; exact-safe entries may migrate, the rest expire.
- Gate (`DA§16.5`): no cross-tenant candidate is returnable even under ANN
  and filter edge cases; a model path, revision, or dimension mismatch
  blocks readiness; results carry valid source/version anchors; a source
  update or deletion removes old results within the defined lag; zero
  side-effect or tool-call reuse; prompt injection in a retrieved document
  cannot alter tool authorization; sensitive content is absent from logs,
  traces, and analytics; exact-key canonicalization survives key-order and
  serialization edge cases; and **pgvector load does not violate the
  scheduler/API database SLOs** (`DA§16.2` — a long retrieval query cannot
  exhaust the scheduler pool or hold control locks).
- [ ] **B10.7 Threshold register — when to move beyond pgvector**
  (`DA§7.7`, `DA§15` Phase 6, `DA§11`). Record the measurements that would
  justify a dedicated vector tier, and refuse the change until one is
  observed: tens of millions of active vectors with continued rapid
  growth; sustained vector query volume causing scheduler or API latency
  or vacuum pressure despite separate pools and replicas; index build or
  rebuild exceeding the maintenance window; filtered ANN recall unable to
  meet the evaluation target; retrieval needing independent regional
  scaling or release cadence; or vector index memory dominating the
  authoritative database economics. If Qdrant is ever added, PostgreSQL
  stays metadata and authorization authority, the outbox projects
  versioned points, query filters carry tenant/source/model revision,
  results are re-authorized against PostgreSQL where necessary, deletion
  tombstones are monitored, and the whole index is rebuildable —
  **never a synchronous dual-write to PostgreSQL and Qdrant in an API
  request**. BigQuery vector search is for offline clustering, evaluation,
  and duplicate analysis only, never the online retrieval path. The same
  register governs the other `DA§15` Phase 6 candidates (AlloyDB,
  Pub/Sub, Kafka, ClickHouse, Bigtable); MongoDB is not on the path at
  all (`DA§11.1`). Gate: a documented decision record per candidate with
  the measured evidence, or an explicit "threshold not reached".

---

## B11 — BigQuery analytical projection

`DA§8`, companion Phase 5. **Do not start before B4 (event contracts and
per-sink delivery) is green** — the outbox is the only ingestion source.

- [ ] **B11.1 Provision through Terraform** (`DA§14.4`, `DA§8.1`). Toronto
  (`northamerica-northeast2`) analytics project, colocated landing bucket,
  datasets, KMS/IAM/network controls, budgets, and monitoring. Choose and
  document one region-loss objective: rebuildable analytics, warm
  analytical recovery, or near-current recovery. **The PostgreSQL ledger
  remains the recovery authority regardless.**
- [ ] **B11.2 Outbox → GCS → BigQuery** (`DA§8.2`). Immutable hourly or
  size-bounded Parquet plus manifests, each carrying contract name and
  version, environment and source service, first/last event time, row
  count, first/last outbox sequence, schema fingerprint, SHA-256/CRC32C,
  exporter build version, tenant-data classification, manifest state, and
  the BigQuery load-job id. Load into `xc_raw`; quarantine schema and
  privacy violations. **No direct BigQuery write inside an API, scheduler,
  billing, or worker transaction, ever.** Storage Write API only when
  freshness requires minutes/seconds, and still with business-level
  deduplication.
- [ ] **B11.3 Warehouse zones and models** (`DA§8.4`, `DA§8.5`, `DA§8.6`).
  `xc_raw`, `xc_staging`, `xc_core`, `xc_marts`, `xc_restricted`,
  `xc_ops_meta` with separate service accounts. Dimensions and facts per
  `DA§8.5`. Partition by business event date, require partition filters,
  cluster on measured filters, set maximum bytes billed on every job.
  Dashboards never query `xc_raw`; direct identity columns never enter
  general marts.
- [ ] **B11.4 Finance reconciliation** (`DA§8.7`). Daily comparison of
  terminal attempts versus closed meters, meter totals versus ledger
  postings, customer charges versus provider payout basis, invoice line
  totals versus ledger references, payment-provider settlement identifiers
  versus PostgreSQL records, missing/duplicate/late/impossible intervals,
  and control-plane price-policy revision versus the projected dimension.
  Findings write back through a restricted API into a PostgreSQL
  `billing_reconciliation_findings` workflow. **No warehouse query updates
  a wallet.**
- [ ] **B11.5 Privacy and governance** (`DA§8.8`, `DA§8.9`). Pseudonymous
  warehouse keys with the identity mapping confined to `xc_restricted`;
  policy tags and column-level security on direct identifiers and
  finance-sensitive columns; row-level access policies for shared
  datasets; VPC Service Controls where the threat model warrants; no public
  dataset exposure or broad service-account keys; workload identity and
  short-lived credentials; logged and reviewed access; per-column retention;
  deletion/tombstone propagation to raw, core, marts, vector indexes,
  object exports, and PostHog; documented time-travel and fail-safe
  behaviour when promising deletion deadlines; auditable legal holds.
  PostHog changes (`DA§8.9`, `DA§2.8`): replace email and name with a
  pseudonymous stable distinct id; stop sending country and province as
  person properties by default; explicit consent and replay masking;
  exclude prompts, logs, artifact names, identity-linked billing amounts,
  host IPs, keys, and MCP arguments; server-side events through a
  redaction contract.
- Gate (`DA§16.6`): PostgreSQL control totals reconcile to warehouse facts
  for a closed test period; duplicate, reordered, and late events produce
  correct facts; **analytics can be disabled for 24 hours with no effect on
  launches, scheduler, worker, or billing**; dataset and bucket location
  and IAM tests prevent cross-boundary access; a full rebuild from
  immutable landing data succeeds; partition filters and cost budgets are
  enforced.

---

## B12 — Cross-store consistency and deletion

`DA§12.3`, `DA§12.7`.

- [ ] **B12.1 Cache consistency** (`DA§12.3`). Cache-aside with a version in
  the value: read Redis, miss reads PostgreSQL, populate with a short TTL,
  mutations commit PostgreSQL and outbox first, invalidation is
  best-effort through the projection. A stale cached value is bounded by
  TTL and **cannot authorize a destructive operation without version
  revalidation**. Security and pricing decisions use an authoritative read
  or a signed, versioned policy snapshot with strict expiry. Correctness
  never depends on an invalidation arriving exactly once.
- [ ] **B12.2 Migration — deletion and export state** (`DA§12.7` at the live
  head). The workflow `requested → validated → authority_deleted/
  anonymized → redis_invalidated → artifacts_deleted or held →
  retrieval_deleted → analytics_deleted/anonymized → posthog_deleted →
  verified → completed`, with each sink recording `not_applicable`,
  `completed`, `legal_hold`, or `failed` plus evidence and deadline, linked
  to the B4.1 audit streams. The API answers request status honestly rather
  than pretending physical deletion is instantaneous across backups and
  time travel. A verifier samples for residual identifiers and raises an
  incident on a missed deadline. Gate: a deletion request with an
  artifact under legal hold reports `legal_hold` for that sink and
  `completed` for the others, and never silently succeeds.

---

## B13 — Security, residency, and infrastructure as code

`DA§13`, `DA§14.4`. Track A closed §19.1–§19.4 (identity, privilege
separation, ingress, supply chain) and `DA§13.5`'s MCP boundary in code.

- [ ] **B13.1 Service identity matrix** (`DA§13.1`). One workload identity
  per deployed service per environment; **no long-lived JSON service-account
  keys** in `.env`, containers, or object storage. Bind each service to the
  minimum external access in the `DA§13.1` table. This composes with Track
  A 11.4's per-service database roles — the DSN split and the cloud
  identity must name the same service.
- [ ] **B13.2 Network boundaries** (`DA§13.2`). Private database and cache
  addresses; no public database IP outside an approved, temporary
  migration; service-to-service authentication and TLS; approved egress for
  object storage and the warehouse; VPC Service Controls around restricted
  analytics; separate production and non-production projects, VPCs, and
  identities; **no production data copied to development without approved
  anonymization**.
- [ ] **B13.3 Encryption and key separation** (`DA§13.3`). Provider-managed
  encryption as the baseline; CMEK where contracts, key separation,
  revocation, audit, or policy justify the operational burden. Separate
  keys per environment and per high-risk domain — database backups,
  artifacts, audit lock, restricted analytics — not one universal key.
  Store KMS resource and version references, never raw keys. Exercise
  rotation and disabled-key recovery.
- [ ] **B13.4 Data classification vocabulary** (`DA§13.4`). Every outbox
  event, artifact type, warehouse column, and log field maps to `public`,
  `internal`, `tenant_confidential`, `personal`, `financial_sensitive`,
  `credential_secret` (**never permitted in events, analytics, or logs**),
  or `regulated_or_legal_hold`. Classification drives allowed sinks,
  retention, masking, region, access review, and deletion. Gate: the CI
  contract test from B4.3 rejects a forbidden field or an unclassified sink
  mapping.
- [ ] **B13.5 Infrastructure as code** (`DA§14.4`). `infra/terraform/` with
  modules for `cloud_sql_postgres`, `memorystore`, `gcs_bucket`,
  `bigquery_dataset`, `kms_keyring`, `workload_identity`, and `monitoring`,
  and environments for `staging-ca` and `production-ca`. Encode region and
  project ids, private service networking, HA/PITR/backup retention/
  deletion protection/maintenance window, justified database flags, cache
  tier and TLS, bucket uniform access and public-access prevention and
  versioning and lifecycle and retention and logging and CMEK, dataset
  location and expiration defaults and IAM and policy tags and budgets, KMS
  separation and rotation, alert policies and routes, service-account
  bindings without broad editor roles, and production deletion protection.
  **Provider consoles are for inspection and emergency operations, not the
  authoritative configuration path.**

---

## B14 — Test strategy completion

Blueprint §26, `DA§16`. Track A covered §26.1 (partially), §26.2, and
§26.5 (partially).

- [ ] **B14.1 Worker contract simulator** (§26.3). A deterministic fake
  worker producing delayed, missing, and duplicate ACKs; old and new
  protocol versions; lease-renewal loss; partial volume, image, and runtime
  preparation; stale inventory generation; duplicate containers; host
  reboot and adoption; partition and reconnect; and **malicious wrong-host,
  wrong-attempt, and wrong-fence reports**. Runs in CI and staging before
  any real GPU canary.
- [ ] **B14.2 Crash-injection placement variants** (§26.2). Track A's
  8-replica concurrency harness is the base; add scheduler death after
  claim, after allocation insert, and before commit; deadlock injection
  validating bounded retry; a stale API expected version; lease expiry
  while the old worker still reports running; an old fence attempting a
  status write or a billing restart; and a command claimant dying before
  ACK. Track A noted these "can extend this harness in Phase 3" — this is
  where they land.
- [ ] **B14.3 Property tests for the new surfaces** (§26.1). Canonical spec
  hashing, action-plan binding and expiry, scope default-deny, price
  tolerance and hold logic, reconcile decision tables. Hypothesis generates
  jobs, host inventories, concurrent transitions, and failure sequences,
  asserting invariants after every generated operation.
- [ ] **B14.4 UI tests** (§26.6). Storybook component states, Vitest
  interaction and data tests, Playwright dark/light/mobile journeys, visual
  snapshots for the MCP, control-plane, instance, and host pages, axe scans,
  keyboard-only approval/drain/reconcile flows, large-queue virtualization,
  and stale/error/partial states.
- [ ] **B14.5 Load and chaos** (§26.7, `DA§16.7`). k6 for API and MCP
  launch-preview and query load; queue and serverless bursts; PostgreSQL
  failover and connection exhaustion; Redis outage; scheduler and
  reconciler rolling restarts; Nginx and MCP blue/green drain; worker
  network partition; stale agent certificate and key rotation; object
  storage and audit sink outage; clock skew (while every lease decision
  stays DB-time based); Toxiproxy for controlled network and database
  faults. Plus `DA§16.7`'s ten game days, each with expected customer
  behaviour, alerts, an owner, recovery steps, and evidence.
  **No GA until invariant tests stay clean under repeated chaos runs, and
  an unplanned fallback is a defect, not resilience.**
- [ ] **B14.6 Test dependencies** (§24.2). Add `testcontainers[postgres,redis]`,
  `pytest-xdist`, `respx`, `schemathesis`, and `pip-audit`. Keep Hypothesis
  and use it heavily. **Note the local constraint: pytest runs against a
  dedicated `xcelsior_pytest` database (Track A), 16 full-suite failures
  are known environment-dependent, and `run-tests.sh` takes one file
  target.** New suites must not regress that isolation.

---

## B15 — Runbooks

Blueprint §32, `DA§18`. **`docs/runbooks/` does not exist.** Every runbook
states trigger, user impact, diagnostic queries, safe actions, **actions
never to take**, rollback and fencing implications, and post-incident
invariant verification.

- [ ] **B15.1 Control plane** (§32): scheduler queue growing with free
  capacity; placement conflict or deadlock spike; allocation invariant
  breach; stale or fenced worker still running; agent command dead-letter;
  host drain stuck; reconcile lag breach.
- [ ] **B15.2 Data plane** (§32, `DA§18.1`–`DA§18.3`): PostgreSQL
  connection saturation, slow or blocked scheduler transaction, deadlock
  spike, storage and WAL growth, zone failover, PITR after operator error,
  bad-migration forward repair, credential rotation, cross-region recovery
  if promised, PgBouncer exhaustion; Redis memory pressure, failover and
  acknowledged ephemeral loss, auth-cache outage policy, rate-limit outage
  policy, hot key and abuse, ACL rotation; object provider unavailable or
  permission denied, signing failure, checksum corruption, missing
  object/catalog mismatch, deletion backlog, accidental lifecycle change,
  KMS key disabled, bucket restore and provider migration.
- [ ] **B15.3 Money and product** (§32): wallet hold or billing meter
  mismatch; MCP auth or JWKS failure; MCP action approval stuck; MCP
  blue/green rollback; worker certificate rotation failure; volume
  provisioner failure.
- [ ] **B15.4 Projections** (§32, `DA§18.4`, `DA§18.5`): audit and outbox
  backlog; telemetry ingestion backlog; Nginx stream drain and certificate
  renewal; backup restore and regional recovery; retrieval model mount or
  load failure, vector index build and rebuild, quality regression
  rollback, stale source and deletion lag, database query overload,
  semantic-cache false hit and emergency disable; oldest outbox lag,
  poison and quarantined contract, GCS manifest mismatch, BigQuery quota
  and load failure, duplicate and late data repair, finance reconciliation
  discrepancy, data deletion failure, warehouse rebuild and regional
  recovery.
- **Standing "never do" list** (`DA§18`): do not edit wallets in BigQuery;
  do not mark outbox events delivered without sink evidence; do not make
  missing artifacts `available`; do not bypass tenant filters to restore
  search; do not enable an unvalidated fallback model.

---

## B16 — Contract migration and legacy removal

Blueprint §13.7, Phase 12. **This phase is last and its migration is
always the head of the chain (§B1).** Nothing here starts until the
relevant legacy-path usage metric has read zero for the agreed retention
period — measured, not assumed.

- [ ] **B16.1 Legacy-use metrics.** Instrument every legacy path so
  "measured zero" is a real number: v1 `/agent/*` calls, direct legacy
  status writes, `/instance` versus `/api/v1/launch-plans`, the legacy
  `leases` table, process-memory preemption, direct scheduler SSH
  execution, broad `api` scope issuance, and `confirm:true` without a plan.
  Gate: a dashboard panel per path plus an alert when a supposedly-dead
  path is used.
- [ ] **B16.2 Contract migration.** Make required normalized columns
  non-null; prevent direct legacy status writes; remove old active-lease
  reads; drop the transitional projection triggers from Track A's migration
  059 — **note Track A's warning: the trigger clobbers any manual
  `effective_priority` boost, so §10.7 fairness aging can only land after
  this removal**; remove obsolete process-memory command and preemption
  state; drop remaining runtime DDL paths; remove production dual-write and
  JSON file state; retain legacy JSONB payload fields only for API
  compatibility with a documented deprecation date; archive or drop old
  event and telemetry structures only after retention and export
  validation.
- [ ] **B16.3 Code removal.** Remove the legacy lease table and path; the
  process-memory command and preemption path (`agent_preempt.py`); direct
  scheduler SSH execution (`process_assigned`/`run_job`); the warning-only
  MCP deploy path; broad legacy `api` scope issuance for new clients, with
  transition clients expired; and old `confirm:true` execution without
  action-plan policy. Decompose the remaining `scheduler.py` and
  `worker_agent.py` compatibility shells into the §11.8 `agent/` module
  layout, preserving signature generation and rollout compatibility through
  `scripts/deploy_worker_agent.sh` at every step.
- [ ] **B16.4 Queue fairness and preemption** (§10.7, §10.8). Only after
  B16.2 removes the trigger: weighted fair share from administrative
  priority class, reservation entitlement, per-team virtual finish time,
  capped age boost, explicit preemptibility, and quota and concurrency
  limits, with the calculated queue key persisted so every replica orders
  work identically and operators see components rather than a mysterious
  score. Preemption becomes a durable plan (§10.8): the scheduler
  identifies victims under a versioned policy; one transaction creates the
  `preemption_plan`, victim stop commands, and the nominated job condition;
  victims receive a grace deadline and checkpoint policy; **the freed GPU
  is not reallocated until observations or a fence prove the victim no
  longer owns it**; a force-stop may follow the deadline under explicit
  policy; and the allocation transfers only in a new reservation
  transaction.
- **Exit gate:** one authoritative path exists for launch, placement,
  execution, transition, billing, command, and audit; legacy-path metrics
  are zero for the agreed retention period; repository architecture docs
  and runbooks match deployed reality.

---

## B17 — Dependency and tooling ledger

§24, `DA§14.2`. Versions are selected at implementation time, pinned in
lockfiles, and updated by automation. **No import fallbacks and no
install-at-runtime behaviour, ever** (§24.6, `DA§14.2`).

| Area | Additions | Item |
|---|---|---|
| Python runtime | `tenacity`, `structlog`, `authlib`, `opentelemetry-instrumentation-{psycopg,httpx,requests,logging}` | B7.2, B5.8 |
| Python dev | `testcontainers[postgres,redis]`, `pytest-xdist`, `respx`, `schemathesis`, `pip-audit` | B14.6 |
| Python data | pgvector adapter; `google-cloud-storage` **or** locked `boto3`/`botocore` (already present); `google-cloud-bigquery` + `pyarrow` if B11 proceeds | B9.2a, B10.2, B11.2 |
| MCP | `openapi-fetch`, `jose`, `ioredis`, `rate-limiter-flexible`, `pino`, `prom-client`, OTel Node packages; `openapi-typescript` in dev | B5.2, B5.9, B5.12 |
| Frontend | `@tanstack/react-{query,table,virtual}`, `zod`, `openapi-fetch` + `openapi-typescript`, `@axe-core/playwright`, Storybook | B6.1 |
| Infra | k6, Toxiproxy, OTel Collector, Prometheus, Alertmanager, Grafana, Loki, Tempo (Syft/Cosign/Trivy already in Track A's `supply-chain` job) | B7.1, B14.5 |

**Lock policy:** Python updates `pyproject.toml` + `uv.lock` and deploys
with `uv sync --frozen`; MCP and frontend update `package.json` +
lockfiles and build with `npm ci` only; Docker builds fail when lockfiles
and manifests disagree; **never `npm ci || npm install`**; never
`pip install` a missing production package at startup; required capability
packages are validated in build and readiness tests. **Select one
production profile per pluggable area** (`DA§14.2`) — do not install every
possible provider SDK. A build fails if a configured extra is absent.

- [ ] **B17.1 Land the ledger** as each item requires it, updating
  `pyproject.toml`/`uv.lock`, `mcp/package.json`/lockfile,
  `frontend/package.json`/lockfile, and the Dockerfiles together.

---

## B18 — Environment and startup-validator contract

§30, `DA§14.3`. Track A built `control_plane/startup_validation.py` with
named checks and documented remediations, enforced in the API lifespan in
production. **Every Track B item that adds a variable extends that
validator in the same commit** (B0.3 rule 22).

Variables Track B introduces, by group:

- **Launch/action plans (B2):** `XCELSIOR_MCP_ACTION_PLAN_TTL_SEC`,
  plan approval mode defaults, price-tolerance defaults.
- **MCP (B5):** `XCELSIOR_MCP_PUBLIC_URL`, `XCELSIOR_MCP_RESOURCE_AUDIENCE`,
  `XCELSIOR_MCP_API_URL`, `XCELSIOR_MCP_REDIS_URL`,
  `XCELSIOR_MCP_RATE_LIMIT_PREFIX`,
  `XCELSIOR_MCP_MAX_WATCHES_PER_PRINCIPAL`, `XCELSIOR_OAUTH_JWKS_URL`,
  `XCELSIOR_OAUTH_ISSUER`.
- **Observability (B7):** `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`,
  `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`,
  `XCELSIOR_METRICS_PORT`, `XCELSIOR_LOG_LEVEL`, `XCELSIOR_LOG_FORMAT=json`.
- **Database (B8.7):** `XCELSIOR_PG_POOL_MIN_SIZE`,
  `XCELSIOR_PG_POOL_MAX_SIZE`, `XCELSIOR_PGBOUNCER_MODE` (the timeout
  variables landed with Track A 11.4).
- **Artifacts (B9.2):** `XCELSIOR_ARTIFACT_BACKEND`, provider-specific
  bucket/endpoint/region variables, `XCELSIOR_ARTIFACT_SIGNED_URL_TTL_SEC`,
  `XCELSIOR_ARTIFACT_REQUIRE_CHECKSUM`, `XCELSIOR_ARTIFACT_LOCAL_DIR`
  (development profile only).
- **Retrieval (B10):** `XCELSIOR_RETRIEVAL_ENABLED`, the three model paths,
  `XCELSIOR_RETRIEVAL_EXPECTED_MODEL_SHA256`,
  `XCELSIOR_RETRIEVAL_EXPECTED_DIMENSION` (populated **only** after a
  verified probe), `XCELSIOR_RETRIEVAL_QUERY_TIMEOUT_MS`,
  `XCELSIOR_RETRIEVAL_MAX_RERANK_CANDIDATES`.
- **Analytics (B11):** `XCELSIOR_ANALYTICS_EXPORT_ENABLED=false` by
  default, `XCELSIOR_ANALYTICS_GCP_PROJECT`,
  `XCELSIOR_ANALYTICS_BQ_LOCATION`, `XCELSIOR_ANALYTICS_BQ_RAW_DATASET`,
  `XCELSIOR_ANALYTICS_GCS_LANDING_BUCKET`,
  `XCELSIOR_ANALYTICS_OUTBOX_BATCH_SIZE`, `XCELSIOR_ANALYTICS_MAX_LAG_SEC`.

- [ ] **B18.1 Extend the production validator** so it additionally rejects
  (§30, `DA§14.3`): a local artifact backend in production; a memory auth
  cache; process-local rate limiting; analytics enabled without project,
  location, bucket, dataset, and workload identity; retrieval enabled
  without every configured model probe and a registered revision; insecure
  Lightning TLS; and secrets in plain environment files where the
  deployment uses a secret manager. Gate: extend
  `tests/test_startup_validation.py` — each condition driven **on and
  off**, matching Track A's 26-test pattern.

---

## B19 — Coverage map

Every section of both governing documents, and where it is owned.

| Blueprint | Owner |
|---|---|
| §1–§2 (concurrency guarantees, advisory-lock fix) | Track A |
| §3 (MCP tool taxonomy and safety boundary) | **B5.6, B5.7** |
| §4 (product contracts) | acceptance criteria across **B2, B3, B5, B6** |
| §5 (current-state evidence) | baseline; re-verified in §B1 and per-section "current state" notes |
| §6, §7 (target architecture, ADRs) | Track A; binding on Track B via §B0 |
| §8, §9 (invariants, state models) | Track A; §9.5 action-plan machine → **B2.1** |
| §10.1–§10.6, §10.9 | Track A |
| §10.7, §10.8 (fairness, preemption) | **B16.4** (blocked on trigger removal) |
| §11 (worker protocol) | Track A; §11.8 modularization → **B16.3** |
| §12 (reconciler) | Track A; billing controller → **B3.3** |
| §13.1–§13.4, §13.8 | Track A |
| §13.5 (action plans, MCP policy/audit) | **B2.1** |
| §13.6 (audit v2) | **B4.1, B4.5** |
| §13.7 (contract cleanup) | **B16.2** |
| §14 (unified launch service) | **B2** |
| §15.1, §15.2 | Track A |
| §15.3, §15.4 | **B3.1–B3.4** |
| §16.1 (outbox mechanism) | Track A |
| §16.2, §16.3 (event contracts, streams) | **B4.3, B4.6** |
| §17 (MCP architecture) | **B5** |
| §18 (versioned API contracts) | **B2.8**; §18.4 worker endpoints = Track A |
| §19.1–§19.6 | Track A |
| §20 (UI) | **B6** (Track A delivered the admin page skeleton) |
| §21 (deployment topology) | **B8.1–B8.4**; §21.3 partially Track A |
| §22 (edge) | **B8.6**; §22.3 agent gateway = Track A |
| §23 (PostgreSQL/Redis ops) | **B8.7, B8.9** |
| §24 (packages) | **B17** |
| §25 (observability, SLOs) | **B7** |
| §26.1 | Track A (partial) + **B14.3** |
| §26.2 | Track A + **B14.2** |
| §26.3, §26.4, §26.6, §26.7 | **B14.1, B5.13, B14.4, B14.5** |
| §26.5 | Track A (from-empty) + **B1.2** (from-production) |
| §27 (CI/CD, release gates) | **B8.4, B8.5**; supply-chain job = Track A |
| §28 Phases 1/3/4/5/6/7(partial)/10 | Track A |
| §28 Phases 0/2/8/9/11/12 | **B1, B2, B5, B6, B7+B8, B16** |
| §29 (file-level change matrix) | distributed across all Track B sections |
| §30 (environment contract) | **B18** |
| §31 (no-fallback rules) | **§B0.3 rule 15**; enforced per item |
| §32 (runbooks) | **B15** |
| §33 (priorities) | reflected in Track B ordering |
| §34 (definition of done) | **§B20** |
| §35 (what not to add) | **§B0.3**, **B11 gate**, `DA§11` |

| Companion | Owner |
|---|---|
| `DA§1`–`DA§3` (verdicts, evidence, authority rules) | binding via **§B0.3 rule 18** |
| `DA§4.1`, `DA§4.3`, `DA§4.5`, `DA§4.6` | **B8.7, B8.8**; partitioning also **B4.1** |
| `DA§4.2` (schemas and roles) | Track A 11.4 |
| `DA§4.4` (schema discipline) | Track A (rules 1–4); rule 5–6 done for `ln_deposits` in **B9.3a**, ledger-wide residual → **B9.5**; rule 7 → **B9.3b-2** |
| `DA§4.7` (reconciler backstops) | Track A |
| `DA§5` (Redis) | **B9.1** landed; **B9.1a**, **B8.9** residual |
| `DA§6` (object storage) | **B9.2** partial; **B9.2a–d** residual |
| `DA§7` (retrieval, semantic cache) | **B10** |
| `DA§8` (BigQuery) | **B11** |
| `DA§9` (telemetry, logs, traces) | **B7** |
| `DA§10.1` (shared SQLite retirement) | **B9.3** partial; **B9.3a–c** residual |
| `DA§10.2`–`DA§10.4` | **B9.4**; volumes = Track A |
| `DA§11` (rejected databases) | binding; **B11** and **B10.7** thresholds |
| `DA§12.1`, `DA§12.2` | **B4.4, B4.5** (global lock already removed by Track A) |
| `DA§12.3`–`DA§12.6` | **B12.1, B9.2d, B11.2, B10.4** |
| `DA§12.7` (deletion workflow) | **B12.2** |
| `DA§13` (security, residency) | **B13**; `DA§13.5` MCP boundary = Track A + **B5** |
| `DA§14` (repository map, deps, env, IaC) | **B1, B17, B18, B13.5** |
| `DA§15` Phases 0–6 | **B1/B9, B9, B4, B10, B11, B10.7/B11 thresholds** |
| `DA§16` (test strategy) | **B14** + per-section gates |
| `DA§17` (data-quality SLOs) | **B7.5** |
| `DA§18` (runbooks) | **B15** |
| `DA§19` (cost discipline) | **B8.7, B11.3, B9.2c** |
| `DA§20`–`DA§22` | binding via **§B0** |

---

## B20 — Definition of done

Track B is complete when every §34 box is true **and** the companion's
data-plane conditions hold. Track A's contribution is marked; the rest is
Track B's.

**Already true (Track A):** two or more schedulers operate concurrently
with zero duplicate active allocation; queue claim, GPU allocation,
attempt, lease offer, command, and outbox are atomic; the worker protocol
never authorizes a start without the exact current attempt/lease/fence and
stale fences cannot mutate control-plane, routing, secrets, storage, or
billing; commands survive fetcher crashes and are ACKed idempotently; the
reconciler converges desired and observed state after every tested
failure; no lifecycle reaper bypasses domain transitions; current
telemetry is shared across API replicas; audit production no longer
globally locks the events table; the API no longer requires `SYS_ADMIN`;
Alembic owns production schema and readiness validates the revision;
production cannot start on SQLite or dual backends.

**Track B must make true:**

- [ ] `create_instance` previews, obtains valid approval or standing
  policy, launches once, and remains watchable through MCP (**B2, B5.4**).
- [ ] Serverless endpoint and job flows retain all current behaviour and
  budget safeguards (**B3.1, B3.2, B5.5**).
- [ ] Cost and wallet checks are concurrency safe *on the plan path*
  (**B2.5**).
- [ ] Strict and non-idempotent workloads are not reassigned before
  definitive host and storage fencing; restartable workloads expose any
  temporary duplicate-execution risk explicitly (**B14.1, B14.2** prove
  it; §8.6 is the contract).
- [ ] Billing starts and stops exactly once per current attempt, proven by
  the invariant dashboard (**B3.3, B7.4**).
- [ ] MCP scopes default deny and tenants are complete in machine
  principals (**B5.1, B5.8**).
- [ ] MCP calls the API only and has structured schemas, audit,
  distributed limits, and traces (**B5.2, B5.3, B5.9, B5.10, B5.12**).
- [ ] MCP deployment failure is fatal and the previous version stays live
  (**B5.13, B8.4**).
- [ ] Hard runtime, storage, identity, and capacity requirements never
  silently fall back (**§B0.3 rule 15**, proven per item).
- [ ] Production cannot start on JSON-file state (**B9.3c, B18.1**).
- [ ] PostgreSQL and Redis backup, restore, failover, and capacity
  procedures are tested (**B8.8, B8.9, B14.5**).
- [ ] Nginx public, MCP, and agent boundaries are explicit and tested
  (**B8.6**; agent = Track A).
- [ ] The UI is complete in dark, light, and mobile, accessible, visually
  reviewed, and typed (**B6**).
- [ ] SLO dashboards, alerts, and runbooks are exercised (**B7.5, B15,
  B14.5**).
- [ ] Legacy paths are removed after measured zero use, not left
  indefinitely (**B16**).
- [ ] No production path returns `file://` or a successful empty listing on
  a provider error (**B9.2**).
- [ ] Every business fact has exactly one authority and every projection is
  rebuildable (**B4.4, B10.4, B11.2, B12**).
- [ ] A deletion request reports honest per-sink status and meets its
  deadline (**B12.2**).

---

## B21 — Execution order

Dependency-ordered. Items inside a stage may proceed in parallel; a stage
does not open until the previous stage's gates are green.

1. **B1** — migration ledger. Blocks every migration below.
2. **B9.1a, B9.3** — data-plane honesty. Money types and shared-state
   contract are P0 correctness; they touch the ledger and must land before
   billing work builds on them.
3. **B2** — unified launch service and v1 API. Blocks B5 and B6.
4. **B4** — audit v2, event contracts, per-sink delivery. Blocks B11 and
   B12; B4.2's broken archive is a live defect.
5. **B3** — billing completion. Depends on B2 and B4.
6. **B5** — MCP v2. **B5.1 (default-deny scopes) is P0 security and should
   ship ahead of the rest of B5, independently.**
7. **B7, B8** — observability and deployment. B7.5's SLOs need B7.4's
   metrics; B8.5's CI needs B14's suites.
8. **B6** — UI. Needs B2.8's API and B7.4's metrics.
9. **B9.2** — artifact catalog completion.
10. **B12** — cross-store consistency and deletion. Needs B4 and B9.2.
11. **B13** — security, residency, IaC. Needs B8's topology.
12. **B10** — retrieval and semantic cache. **B10.1's probe gates
    everything else in the section; do not author B10.2 before it passes.**
13. **B11** — BigQuery. Needs B4 and B13.
14. **B14, B15** — test completion and runbooks, continuous across all
    stages, complete before GA.
15. **B16** — contract migration and legacy removal. Last, and only on
    measured zero legacy use.

**Operator residual for Track B** (never counted as complete here): flip
`XCELSIOR_AGENT_PUBLIC_INGRESS=deny`; provision the managed PostgreSQL,
Redis, object storage, KMS, and BigQuery resources; deploy the
observability stack; restore GitHub Actions billing so CI gates run;
verify the declared retrieval model mounts on the intended deployment
host. Track A's standing residuals in **§B0.4** remain open and several
Track B gates depend on them.
