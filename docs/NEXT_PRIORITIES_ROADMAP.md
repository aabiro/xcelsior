# Xcelsior — Next Priorities Roadmap

*Created: 2026-06-07*  
*Context: Post team-tenancy polish; volumes identified as highest-value app-wide gap.*

This document records the **six highest-priority initiatives** agreed for the next engineering cycle. Each section has checkbox tasks, sub-points, acceptance criteria, dependencies, and estimated effort. Work **#1 (Volumes)** has a dedicated companion checklist: [`VOLUMES_COMPLETION_PLAN.md`](./VOLUMES_COMPLETION_PLAN.md).

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done

---

## Priority ranking (summary)

| # | Initiative | Primary outcome | Est. effort |
|---|------------|-----------------|-------------|
| 1 | Production persistent volumes (NFS) | Real block storage for ML workloads | Done (2026-06-07) |
| 2 | Team tenancy app-wide sweep | B2B-ready shared wallet + RBAC everywhere | 1 week |
| — | PayPal marketplace provider payouts | Second payout rail (deferred — Stripe sufficient for now) | backlog |
| 4 | Web terminal rewrite | Usable in-browser shell on running instances | 1–2 weeks |
| 5 | Mobile performance (F-003 phase 3) | Sub-acceptable TBT on slow 4G marketing pages | 3–5 days |
| 6 | Serverless inference endpoints | Deploy model → REST API (competitive parity) | 3–4 weeks |

---

## 1. Production persistent volumes (NFS-backed)

**Goal:** Volumes are not metadata-only in production. Customers can create, attach, persist data across instance restarts, and be billed correctly — including team workspaces.

**Companion checklist:** [`VOLUMES_COMPLETION_PLAN.md`](./VOLUMES_COMPLETION_PLAN.md)

### 1.1 Infrastructure & configuration

- [x] **NFS storage server provisioned** (prod — VPS pixelenhance-labs `100.64.0.1`)
  - [x] Kernel NFS export `/exports/volumes` on VPS (Mac appliance retired 2026-06-08)
  - [x] Mesh-only via Headscale; workers mount `100.64.0.1:/exports/volumes`
  - [x] Disk capacity plan documented — runbook § Disk capacity (`MAX_VOLUME_GB`, `MAX_TOTAL_STORAGE_GB`, `MAX_VOLUMES_PER_OWNER`) (2026-06-07)
  - [x] LUKS tools in privileged appliance container (`cryptsetup`, `e2fsprogs`)
- [x] **API / scheduler env wired**
  - [x] `XCELSIOR_NFS_SERVER=100.64.0.1` on `api-blue`, `scheduler-worker`, GPU workers
  - [x] `XCELSIOR_NFS_EXPORT_BASE=/exports/volumes` consistent across API, scheduler, volumes engine
  - [x] `XCELSIOR_NFS_SSH_HOST=127.0.0.1` for colocated VPS provision
  - [x] `.env.example` and `docker-compose.yml` updated (`NFS_SSH_CMD_WRAP`)
- [x] **SSH from API to NFS server**
  - [x] API SSH loopback to VPS host for provision/destroy
  - [x] `VolumeEngine._ssh_exec_with_retry` verified in prod (`mode=full`, `reachable=true`)

### 1.2 Backend correctness (team + launch)

- [x] **Instance launch volume validation** uses team billing scope (not personal `user_id`)
  - [x] `routes/instances.py` → `_user_can_access_volume(user, vol)` for `volume_ids` preflight
  - [x] Regression test: team member launches instance with team-owned volume
- [x] **AI assistant `list_volumes` tool** uses `_volume_owner_ids_readable` (team-aware)
- [x] **`list_volumes_for_owner_ids` bug** fixed (`NameError` on loop variable) — 2026-06-07
- [x] **Create volume API** returns `owner_id` in response (full `get_volume` payload)
- [x] **Encrypted volume reopen** after NFS server reboot — `scripts/volumes_reopen_luks.py` + runbook (2026-06-07)
- [x] **Region affinity** warned when attaching cross-region — API `region_warning` + attach UI hint (2026-06-07)

### 1.3 Worker / scheduler attach path

- [x] **Managed volume mount on GPU host** (worker_agent + VPS NFS)
  - [x] Launch with `volume_ids` succeeds in `volumes_e2e_smoke.py`
  - [x] NFS mount at `/mnt/xcelsior-volumes/{volume_id}` on ASUS `aaryn-tuf-rtx2060` (2026-06-07)
  - [x] Bind-mount into container at `/workspace` — persist E2E PASS
  - [x] Data gravity: scheduler `get_volume_host_ids` + binpack 1.3x preference — code in `scheduler.py`
  - [x] Failure mode: required `volume_ids` mount fail → job `failed` (scheduler + worker_agent) (2026-06-07)
- [x] **Detach / terminate cleanup**
  - [x] `detach_all_for_instance` on terminate (`billing.terminate_instance`)
  - [x] Orphan mount cleanup on worker (`cleanup_orphaned_volume_mounts`) — code + periodic thread
- [x] **Attach to running instance** (`POST /api/v2/volumes/{id}/attach`)
  - [x] Live hot-attach via `mount_volume` agent command + `nsenter` bind (2026-06-07)
  - [x] `volumes_e2e_smoke.py --hot-attach` smoke path

### 1.4 Billing & lifecycle

- [x] **Volume storage billing** charges correct `owner_id` (team wallet when in team context)
  - [x] `billing.py` volume tick uses `owner_id` from volumes table
  - [x] Suspended wallet skips volume billing (fail-closed in billing.py)
  - [x] Prod audit: `scripts/volumes_billing_audit.py` (2026-06-07)
- [x] **Stale volume janitor** — `cleanup_stale_volumes` + `reconcile_orphaned_attachments` in bg-worker tick
- [x] **Delete with cryptographic erasure** for encrypted volumes — engine + `--encrypted` E2E PASS; LUKS image removed on delete

### 1.5 Frontend (dashboard)

- [x] **Team context on volumes page** — banner, viewer gates, `xcelsior-team-changed` reload — 2026-06-07
- [x] **Volume status UX** for `error` state (NFS unreachable) with clear Retry CTA — 2026-06-07
- [x] **Launch modal** team-visible volumes + team-changed reload — 2026-06-07
- [x] **i18n** for volumes strings (EN/FR dashboard keys) — 2026-06-07
- [x] **Empty state** explains team vs personal workspace billing — 2026-06-07

### 1.6 Observability & ops

- [x] **NFS health probe** in `readyz` and `/api/nfs/config`
  - [x] Reports `configured`, `reachable`, `export_base`, `mode` (`full` vs `metadata-only`)
  - [x] `XCELSIOR_NFS_REQUIRED=true` fails readiness in production
- [x] **Structured logs** for provision / attach / mount failures (grep-friendly — see runbook)
- [x] **Runbook:** [`VOLUMES_RUNBOOK.md`](./VOLUMES_RUNBOOK.md)
- [x] **Alerts:** export disk >80%/90% — `check_nfs_disk.sh` (VPS) + cron in runbook (2026-06-08)

### 1.7 Testing & E2E

- [x] **Unit / integration tests**
  - [x] Team member create/list/launch/delete volume (`test_team_tenancy_sweep.py` — 153 volume tests green)
  - [x] Launch with team `volume_ids` succeeds
  - [x] Viewer cannot create/mutate; can list
  - [x] Cross-account volume GET returns 404
- [x] **Staging E2E script** [`scripts/volumes_e2e_smoke.py`](../scripts/volumes_e2e_smoke.py)
  - [x] Create → list → get → launch with `volume_ids` → delete — PASS prod 2026-06-08
  - [x] `ops_infra_smoke.py` — PayPal + NFS + volume CRUD PASS
  - [x] Write file → restart instance → file persists — `--persist` PASS on ASUS 2060 (2026-06-07)
  - [x] Encrypted volume round-trip — `--encrypted` PASS prod (2026-06-07)
- [x] **Post-deploy audit** includes `/dashboard/volumes` in Playwright crawl (`generate_reaudit_report.mjs`) (2026-06-07)

### 1.8 Acceptance criteria (volumes “done”)

- [x] Prod `XCELSIOR_NFS_SERVER` set; new volumes provision real storage (`mode=full`)
- [x] Team member can create volume billed to team wallet, launch instance with it, and viewer can see but not mutate — tenancy tests + prod smoke
- [x] Instance launch accepts team-scoped `volume_ids` (e2e smoke)
- [x] Billing tick charges team wallet for team-owned volumes — `volumes_billing_audit.py` (2026-06-07)
- [x] Runbook + health probe documented for on-call
- [x] All volume tests green locally (166 passed); wired into CI via `run-tests.sh`

---

## 2. PayPal marketplace provider payouts *(deferred — not a current priority)*

**Status:** Code landed and tested; **production rollout deferred** — Stripe Connect is sufficient for now. Revisit when provider demand requires a second rail.

**Goal:** GPU providers can onboard PayPal as a disbursement rail alongside Stripe Connect; marketplace jobs record a single `payment_rail` per payout split.

**References:** [`billing-money-path.md`](./billing-money-path.md), [`paypal-marketplace-e2e.md`](./paypal-marketplace-e2e.md), `.env.paypal-sandbox.example`

### 2.1 Backend completion & ship

- [x] **Land uncommitted PayPal work** (review + merge) — 2026-06-07
  - [x] `paypal_connect.py` — seller onboarding, status sync, marketplace capture idempotency
  - [x] Migration `035_paypal_provider_marketplace.py`
  - [x] `routes/billing.py` webhook handlers (`MERCHANT.ONBOARDING.COMPLETED`, etc.)
  - [x] `routes/providers.py` PayPal connect endpoints
  - [x] Idempotency: no double-credit wallet on `PAYMENT.CAPTURE.COMPLETED` for marketplace orders
- [~] **Production env** (ops — code ready)
  - [ ] `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, `PAYPAL_WEBHOOK_ID`
  - [ ] `PAYPAL_MODE=live` (or sandbox for staging)
  - [ ] Webhook URL registered: `https://xcelsior.ca/api/billing/paypal/webhook`
- [x] **Payout API** `POST /api/providers/{id}/payout` with `payment_rail=paypal`
  - [x] Requires provider `paypal_status=active`
  - [x] Single rail per job in `payout_splits` (no double-pay with Stripe)

### 2.2 Frontend (provider dashboard)

- [x] **`PayPalConnectCard`** on provider earnings page — 2026-06-07
  - [x] Connect → redirect → return → status polling
  - [x] Show `onboarding` / `active` / unavailable states
  - [x] i18n EN/FR
- [x] **Payout history** shows `payment_rail` badge (Stripe / PayPal)
- [x] **Error states** for incomplete onboarding (resume + check status CTAs)

### 2.3 Testing & verification

- [x] `tests/test_paypal_connect.py` — all green (incl. capture idempotency)
- [x] `tests/test_paypal_marketplace_webhook.py` — all green
- [x] `tests/test_paypal_platform.py`, `test_paypal_wallet_idempotency.py`
- [x] `tests/test_providers_endpoints_coverage.py` — PayPal provider routes
- [x] `scripts/billing_prod_smoke.py` — PayPal enabled check
- [~] Sandbox E2E per `paypal-marketplace-e2e.md` (manual ops):
  - [x] Customer wallet deposit (unchanged)
  - [ ] Provider onboarding webhook → `active` (staging/prod verify)
  - [ ] Test payout to PayPal seller (staging/prod verify)

### 2.4 Docs & ops

- [x] Provider routes use existing `providers:read` / `providers:write` scopes
- [x] Runbook section in `billing-money-path.md` for PayPal payout failures
- [x] Support macro: “Customer PayPal deposit ≠ provider PayPal payout”

### 2.5 Acceptance criteria

- [~] Provider completes PayPal onboarding in sandbox; status `active` in DB (manual verify)
- [x] Admin/provider can trigger PayPal payout on completed job (API + tests)
- [x] Wallet deposits still credit once per order
- [x] CI PayPal test suite green

---

## 3. Team tenancy app-wide sweep *(next priority after volumes)*

**Goal:** Every customer-scoped resource respects active workspace (personal vs team): wallet, jobs, volumes, artifacts, templates, analytics, AI tools.

**References:** `routes/_deps.py` (`_team_context_for_user`, `_effective_billing_customer_id`), `frontend/src/lib/team-context.ts`

### 3.1 Backend — resource scoping

- [ ] **Instances / jobs** — `_effective_billing_customer_id` on create/list (P0 done)
- [ ] **Volumes** — create/list/mutate scoped (done); launch validation (see #1)
- [ ] **Billing / wallet** — `_require_customer_access`, team admin for deposits (P0 done)
- [ ] **Concurrency pool** — shared per team plan (P2 done)
- [x] **Artifacts** — list/upload/download scoped to team `billing_customer_id` / accessible owner ids (2026-06-08)
- [x] **User image templates** — create/list/delete scoped to team workspace (2026-06-08)
- [x] **Snapshots (instance)** — rate limit and ownership via team billing id (2026-06-08)
- [x] **Inference endpoints** (`routes/inference.py`) — wallet + `owner_id` use `_effective_billing_customer_id` (2026-06-08)
- [ ] **Chat / AI assistant tools** — `list_volumes`, `list_instances`, wallet tools team-aware
- [ ] **SSH keys** — attach to team instances (read scope for viewers)
- [ ] **Analytics API** — usage queries filter by effective billing customer when in team context
- [ ] **Audit logs** — include `team_id`, `team_role` on mutating actions

### 3.2 Backend — auth & workspace switching

- [ ] **Login / register / OAuth** responses include team context fields (login done 2026-06-07)
- [ ] **Single-team personal workspace** — explicit `users.team_id = NULL` not auto-fallback (done)
- [ ] **`PATCH /api/teams/active`** — switch personal ↔ team (done)
- [ ] **Invite accept** → `refreshUser` + settings deep link `#team` (done)

### 3.3 Frontend — consistent UX

- [x] **`TeamContextBanner`** on: billing, instances, volumes, analytics, artifacts, launch modal (2026-06-08)
- [ ] **`TeamSwitcher`** visible when `teams.length >= 1` (done)
- [ ] **Settings workspace picker** for single-team users (done)
- [~] **`xcelsior-team-changed` listener** on: billing, instances, credits, volumes, analytics, artifacts (2026-06-08)
- [ ] **Viewer gates** on all mutation buttons (instances, volumes, launch, billing deposit)
- [ ] **Settings team tab** i18n complete EN/FR (done)

### 3.4 Testing

- [ ] `test_team_tenancy_p0.py` — wallet + job ownership
- [ ] `test_team_tenancy_sweep.py` — roles, concurrency, volumes, login, personal workspace (16 tests)
- [x] `test_team_tenancy_sweep` extended: artifacts, templates, inference (2026-06-08)
- [ ] `frontend/src/__tests__/team-context.test.ts`
- [ ] Security sweep: cross-team IDOR on artifacts, volumes, jobs

### 3.5 Acceptance criteria

- [ ] No customer mutation path uses raw `user_id` where `billing_customer_id` is intended
- [ ] Viewer role read-only on all team resources in API + UI
- [ ] Switching workspace updates wallet, instances, volumes, analytics without full page reload (event-driven)
- [ ] Enterprise demo script: create team → invite member + viewer → shared wallet → launch → viewer blocked

---

## 4. Web terminal rewrite

**Goal:** In-browser terminal on running instances is reliable (no glyph corruption, correct container resolution, reconnect).

**Reference:** [`TERMINAL_PLAN.md`](./TERMINAL_PLAN.md)

### 4.1 Frontend (Phase 1–2)

- [ ] Add xterm addons: `@xterm/addon-webgl`, `@xterm/addon-search`, `@xterm/addon-unicode11`
- [ ] Terminal config: `customGlyphs`, `convertEol: false`, `macOptionIsMeta`
- [ ] WebGL addon with `clearTextureAtlas` on visibility change + resize
- [ ] Unicode11 wide-char support
- [ ] Binary WebSocket for output (`arraybuffer`); JSON for control messages
- [ ] Exponential backoff reconnect (8 attempts, 30s cap)
- [ ] Toolbar: connection status, reconnect, search (Ctrl+Shift+F)

### 4.2 Backend (Phase 3)

- [ ] **Container name resolution** — resolve live container ID, not stale DB name
- [ ] **PTY streaming** — binary frames, backpressure
- [ ] **Rate limits** on terminal WS (per user + per job)
- [ ] **Team access** — viewer read-only or block terminal for viewers (product decision)
- [ ] **Audit** — log terminal session start/end

### 4.3 Infrastructure (Phase 4)

- [ ] **tmux persistence** optional — survive brief disconnects
- [ ] **SSH mesh path** verified under Headscale ACLs

### 4.4 Testing

- [ ] Manual matrix: Ubuntu template, PyTorch template, vim/htop/top, resize, tab background
- [ ] Automated WS smoke test (connect, echo, disconnect)
- [ ] Include terminal route in dashboard Playwright audit

### 4.5 Acceptance criteria

- [ ] New user can open terminal on running instance within 10s, type `ls`, see clean output
- [ ] No React #418 / glyph overlap after tab switch
- [ ] Stale container name does not cause immediate "No such container"

---

## 5. Mobile performance (F-003 phase 3)

**Goal:** Marketing pages usable on slow 4G; desktop TBT already improved (~7s on `/`); mobile TBT still ~28–31s.

**Reference:** [`site-audit-report-2026-06-06-post-recovery.md`](./site-audit-report-2026-06-06-post-recovery.md)

### 5.1 Measurement

- [ ] Re-run perf MCP / Lighthouse post-changes; save to `docs/perf/`
- [ ] Track: `/`, `/pricing`, `/blog`, `/download`, `/gpu-availability`
- [ ] Targets: mobile TBT < 8s (stretch < 5s); desktop TBT < 2s on `/`

### 5.2 Code splitting & deferral

- [ ] Route-level dynamic imports for heavy dashboard-only libs (already partial)
- [ ] Marketing pages: zero dashboard providers on `(marketing)` layout
- [ ] Defer PostHog / GTM further (`lazyOnload` done — verify no regression)
- [ ] `framer-motion` lazy load on marketing (done — verify)
- [ ] Font subsetting / `next/font` preload only critical weights
- [ ] Image priority only for LCP hero

### 5.3 Bundle diet

- [ ] Analyze `@next/bundle-analyzer` report for `/` and `/pricing`
- [ ] Split i18n: marketing loads `en-public` only (dashboard keys not on `/`)
- [ ] Remove or lazy-load unused lucide icons on marketing pages
- [ ] Consider Preact compat or lighter chart lib on public GPU page only

### 5.4 Infrastructure

- [ ] Cloudflare early hints / HTTP3
- [ ] Verify Brotli on HTML+JS
- [ ] Cache headers for static assets (immutable)

### 5.5 Acceptance criteria

- [ ] Mobile slow-4G TBT cut by ≥50% on `/` and `/pricing` vs 2026-06-05 baseline
- [ ] No hydration errors on legal pages
- [ ] `bash scripts/redo_when_prod_up.sh --quick` still 51/51

---

## 6. Serverless inference endpoints

**Goal:** User deploys a model → receives HTTPS REST endpoint (like Replicate/Together); metered billing from wallet.

**Reference:** `routes/inference.py`, `inference.py`, [`competitive_edge.md`](./competitive_edge.md)

### 6.1 Product definition

- [ ] **Scope v1:** GPU-backed inference on dedicated small instance vs true scale-to-zero
- [ ] **Auth:** API key or JWT; rate limits per endpoint
- [ ] **Billing:** per-token or per-request + idle GPU time
- [ ] **Models:** vLLM / TGI containers from templates

### 6.2 Backend

- [ ] **Replace stubs** in `routes/inference.py` — ownership uses `_effective_billing_customer_id`
- [ ] **Endpoint lifecycle:** create → provisioning → active → scaled_down → deleted
- [ ] **Scheduler integration:** inference jobs as specialized instance type or separate pool
- [ ] **OpenAI-compatible** `/v1/chat/completions` routing to user's endpoint (auth scoped — partial done per security sweep)
- [ ] **Worker completion** path for async inference
- [ ] **Wallet preflight** before create/invoke

### 6.3 Frontend

- [ ] **Dashboard `/dashboard/inference`** (or section under instances)
  - [ ] Create endpoint wizard (model, GPU, scaling min/max)
  - [ ] API URL + key display
  - [ ] Usage metrics + cost
- [ ] **Docs** on docs.xcelsior.ca

### 6.4 Testing

- [ ] Integration test: create endpoint (mock GPU), invoke, billed
- [ ] IDOR sweep on inference routes
- [ ] Load test single endpoint latency

### 6.5 Acceptance criteria

- [ ] Customer creates inference endpoint from dashboard, gets URL, successful test request billed to wallet
- [ ] Team workspace: endpoints owned by team billing id; viewer cannot create/delete
- [ ] Documented pricing model

---

## Cross-cutting work (applies to all six)

- [ ] **CI green:** `CI=true XCELSIOR_ENV=test bash run-tests.sh`
- [ ] **Post-deploy:** `bash scripts/redo_when_prod_up.sh --quick` after each prod deploy
- [ ] **Security sweeps:** `tests/test_app_security_sweep.py`, `test_billing_security_sweep.py`, `scripts/audit_route_auth.py`
- [ ] **Fern/OpenAPI** regen if public API surface changes
- [ ] **Frontend SDK** regen for new team/volume/inference types

---

## Suggested execution order

1. **Volumes** (#1) — done; VPS NFS `100.64.0.1` — see [`VOLUMES_COMPLETION_PLAN.md`](./VOLUMES_COMPLETION_PLAN.md)
2. **Team sweep** (#2) — finish artifacts/templates/analytics while team context is fresh
3. **PayPal** — backlog (code ready; prod env/webhook verify when needed)
4. **Terminal** (#4) — retention on interactive instances
5. **Mobile perf** (#5) — conversion on marketing
6. **Inference** (#6) — larger bet; start design while 1–3 land

---

*Update checkboxes in this file as work completes. Volumes detail lives in the companion plan.*