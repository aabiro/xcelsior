# DONE

**Approx. completed:** 2026-06-08

---

# Serverless Inference Endpoints — Implementation Checklist

*Created: 2026-06-08*
*Design of record:* [`INFERENCE_DESIGN.md`](./INFERENCE_DESIGN.md)
*Source reports:* [`inference_reports.md`](./inference_reports.md)

Exhaustive, engineered-to-perfection build plan. Every item is a discrete,
reviewable unit. **Order matters:** verify → migrations → service → API → worker
SDK → autoscaler → billing → cold-start → streaming → keys/limits →
observability → frontend → tests → security → rollout. Do **not** start a phase
before the prior phase's exit criteria are green.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/decision needed

**Deploy modes & settings are modeled on Novita** (see `INFERENCE_DESIGN.md` §3–4):
two modes — **(A) Choose a model** (managed vLLM preset) and **(B) Custom Docker
image** with Novita's fields: `Container Image`, `Container Start Command`, `HTTP
Port`, `Health Check Path`, `Container Registry Auth`, `CUDA Version`, `Environment
Variables`. Common: `GPUs/Worker`, `Min/Max Workers`, `Max Concurrency`, `Idle
Timeout (s)`, and an auto-scaling strategy of `Queue Request Count` **or** `Queue
Delay`.

**Global conventions (apply to every code item):**
- [~] Fully typed; passes the **pyright gate** (`reportCallIssue`,
      `reportArgumentType`, `reportAttributeAccessIssue` = errors).
- [x] All tenancy via `owner_id` + team guards; cross-account = **404**.
- [x] All schema via **Alembic** (`migrations/versions/`), never ad-hoc DDL.
- [x] Route modules orchestrate only; business logic in `serverless/` service.
- [x] Reuse scheduler, billing, volumes, events (SSE), worker-agent — never fork.
- [x] Every mutating path emits the appropriate **SSE event** (endpoint/worker/job;
      key create/revoke are silent by design).
- [x] No secrets in logs; API keys hashed at rest.

---

## Phase 0 — Verify the existing substrate (NO code edits) ✅

> Completed 2026-06-08. Decisions locked in `INFERENCE_DESIGN.md` §11; delta in §12.

- [x] Confirm files & symbols actually present and wired:
  - [x] `serverless/` package — service, dispatcher, autoscaler, metering, proxy.
  - [x] `serverless/repo.py` — CRUD + `claim_next_job()` (`SKIP LOCKED`).
  - [x] `routes/serverless.py` — mounted in `routes/__init__.py`; route matrix in
        design §9.
  - [x] `serverless_*` tables (migration `037`).
  - [x] OpenAI proxy at `/v1/serverless/{id}/openai/v1/*`.
- [x] Confirm reusable primitives exist as described:
  - [x] `scheduler.submit_job()` / `kill_job()` + `worker_agent` +
        `agent_commands` with `FOR UPDATE SKIP LOCKED` (`routes/agent.py`).
  - [x] `routes/_deps.py`: `_canonical_owner_id()`, `_user_team_id()`,
        `broadcast_sse()`, rate-limit helpers.
  - [x] `billing.py`: wallets, suspension, `billing_cycles`, background tick,
        suspended-wallet enforcement. **Required foundation** — non-negotiable.
  - [x] `volumes.py`: NFS attach/mount, region-aware, canonical mount opts.
        **Required foundation** — model weight cache depends on this.
  - [x] `events.py` SSE; OpenTelemetry + Prometheus hooks; `api.py` `_bg_worker()`
        periodic-task registration pattern.
  - [x] Frontend: `api.ts`, `gpu-models.ts`, Recharts/Sonner, `AuthUser` team
        fields; i18n at `en-dashboard.ts` / `fr-dashboard.ts`; dashboard route
        `frontend/src/app/(dashboard)/dashboard/inference/page.tsx`.
- [x] **Resolved all Phase 0 decisions** in `INFERENCE_DESIGN.md` §11:
  - [x] **#1 Naming** — `serverless_*` schema; public `/v1/serverless`;
        management `/api/v2/serverless/endpoints`.
  - [x] **#2 OpenAI universality** — full OpenAI surface required; capability gate
        in control plane (explicit 4xx on unsupported features).
  - [x] **#3 GPU-seconds** — allocated-time formula; control-plane authoritative;
        cold start billable.
  - [x] **#4 Routing** — central proxy for preset; queue pull for custom; proxy
        routes by model/size/tier.
  - [x] **#5 Cold-start SLA** — measure first, advertise after telemetry.
  - [x] **#6 Queue backend** — pure Postgres (`SKIP LOCKED`).
  - [x] **#7 First increment** — 1–2 presets + autoscaling + GPU billing + basic
        custom queue; defer full frontend/SDK/webhooks.
  - [x] **#8–#10** — handler/SDK/webhook/frontend paths recorded in §11.8–11.10.
- [x] **As-built vs. report claims delta** written (`INFERENCE_DESIGN.md` §12).
      Worker readiness in Phase 0: simple **curl health routes** on workers (no
      fancy substrate probing). Billing/scheduling truth = **control plane only**.

**Phase 0 exit:** ✅ Met — substrate verified; decisions #1–#4, #7 locked; delta
written. **Proceed to Phase 1.**

---

## Phase 1 — Data model, migrations & repository ✅

> Completed 2026-06-08. Migration `037_serverless_endpoints_workers_jobs_keys.py`;
> repo `serverless/repo.py`; tests `tests/test_serverless_repo.py` (11 passed).

- [x] Alembic migration `037_serverless_endpoints_workers_jobs_keys.py`:
  - [x] `serverless_endpoints` (all cols per design §6); app-set `updated_at`;
        soft-delete `deleted_at`. Novita fields: `mode`, `managed_engine`,
        `model_ref`, `image_ref`, `startup_command`, `http_port`,
        `health_check_path`, `cuda_version`, `registry_auth_ref`, `gpu_tier`,
        `gpu_count`, `min_workers`, `max_workers`, `max_concurrency`,
        `idle_timeout_sec`, `scaling_policy_type`, `scaling_policy_value`,
        `request_timeout_sec`, `max_request_bytes`, `keep_warm`,
        `cache_volume_id`, `region`, `env` (jsonb).
  - [x] `serverless_workers` (+ `scheduler_job_id`, `endpoint_id`, allocation spans).
  - [x] `serverless_jobs` (queue; `gpu_seconds`, `cold_start_seconds`,
        `cost_cad`, `idempotency_key`, `webhook_url`).
  - [x] `serverless_job_stream_events` (`job_id`, `seq_no`, `event_type`,
        `payload`, `created_at`).
  - [x] `serverless_api_keys` (`key_prefix`, `key_hash`, `scopes`,
        `rate_limit_rpm`, `endpoint_id` nullable).
  - [x] Indexes: queue `(endpoint_id, status, queued_at)`; workers
        `(endpoint_id, state)`; partial unique idempotency; metrics
        `(endpoint_id, finished_at)`; `key_hash` lookup.
  - [x] `downgrade()` drops everything cleanly.
  - [x] Postgres-first (matches prod); SQLite not used for serverless queue.
- [x] `serverless/repo.py` — endpoint/worker/job/key CRUD, `claim_next_job()`
      (`SKIP LOCKED`), queue-depth / max-wait, idempotency, stream events.
- [x] Migration applied (`alembic upgrade head` on prod + test DB); repo unit-tested.

**Phase 1 exit:** ✅ Met — proceed to Phase 2.

---

## Phase 2 — Service layer (`serverless/` package) ✅

> Completed 2026-06-08. Tests: `tests/test_serverless_service.py` (17 passed).

- [x] `serverless/service.py` — endpoint lifecycle (create/list/get/patch/delete),
      validation, wallet pre-flight, SSE emits, reconcile + provision/deprovision.
- [x] `serverless/dispatcher.py` — assign queued jobs to ready workers, concurrency
      tracking, requeue on worker loss.
- [x] `serverless/autoscaler.py` — deterministic reconciliation math (queue count +
      queue delay policies).
- [x] `serverless/openai_proxy.py` — proxy to warm worker vLLM port; capability-gate
      (explicit 4xx on unsupported routes).
- [x] `serverless/metering.py` — control-plane `gpu_seconds`, `billing_cycles`
      (`resource_type=serverless_gpu`); tokens metadata-only.
- [x] `serverless/streams.py` — SSE framing, `[DONE]`, error chunks, replay.
- [x] `serverless/keys.py` — generate (prefix+hash), validate, scope, revoke,
      `last_used_at`.
- [x] `serverless/cache.py` — NFS weight-cache volume resolve via `volumes.py`.

- [x] Single-writer reconcile: `pg_try_advisory_lock` in `serverless/repo.py`.

**Phase 2 exit:** ✅ Met — proceed to Phase 3 (API routes).

---

## Phase 3 — API surface (`routes/serverless.py`) ✅

> Completed 2026-06-08. Tests: `tests/test_serverless_routes.py` (10 passed).

- [x] **Management API** (session/team auth):
  - [x] `POST/GET/GET{id}/PATCH{id}/DELETE{id} /api/v2/serverless/endpoints`
  - [x] `GET …/{id}/workers`, `…/jobs`, `…/metrics`, `…/usage`
  - [x] `POST/GET/DELETE …/{id}/keys`
- [x] **Queue job API** (per-endpoint key or session):
  - [x] `POST /v1/serverless/{id}/run` (async; `Idempotency-Key`, optional `webhook`)
  - [x] `POST /v1/serverless/{id}/runsync` (timeout-bounded; defensive empty-output)
  - [x] `GET /v1/serverless/{id}/status/{job_id}`
  - [x] `GET /v1/serverless/{id}/stream/{job_id}` (SSE)
  - [x] `POST /v1/serverless/{id}/cancel/{job_id}`
- [x] **OpenAI surface** (path-based; §11.1 namespace):
  - [x] `POST /v1/serverless/{id}/openai/v1/chat/completions` (stream + non-stream)
  - [ ] `POST /v1/serverless/{id}/openai/v1/completions` (deferred)
  - [ ] `POST /v1/serverless/{id}/openai/v1/embeddings` (capability-gated; deferred)
  - [x] `GET  /v1/serverless/{id}/openai/v1/models`
- [x] **Internal worker callbacks**: `POST /api/v2/serverless/workers/{id}/ready`
- [x] Pydantic v2 request/response models (OpenAI shapes where applicable).
- [x] Status codes: 200/201/202/400/401/403/404/409/429/500/502/503/504.
- [x] `X-RateLimit-*` headers; `max_queue_size` → 429 (Phase 8).
- [x] Extend `routes/_deps.py`: endpoint-key auth, `_require_serverless_endpoint_*`,
      `_serverless_scope_owner_id`, `_serverless_owner_ids_readable`.
- [x] Register `serverless_router` in `routes/__init__.py`.

**Phase 3 exit:** ✅ Met — proceed to Phase 4 (worker SDK + preset image).

---

## Phase 4 — Worker SDK, preset image, templates & docs ✅

> Completed 2026-06-08. Tests: `tests/test_serverless_worker_sdk.py` (7 passed).

- [x] `serverless/worker_sdk/`:
  - [x] Queue-handler mode: `handler(job)` — sync/async, sync/async generator.
  - [x] ASGI mode: `serve_asgi(app)`.
  - [x] Normalized progress events, sync output aggregation, canonical error
        envelope `{status_code, code, message, retryable}`.
  - [x] Heartbeats, graceful drain on SIGTERM, `/healthz`.
  - [x] **Fitness checks** on boot (CUDA, disk, memory, optional weights path).
- [x] Worker callbacks: `ready`, `heartbeat`, `jobs/claim`, `jobs/{id}/complete`,
      `jobs/{id}/events` in `routes/serverless.py`.
- [x] **Mode A — Managed preset image**: `serverless/images/vllm/Dockerfile` →
      `xcelsior/serverless-vllm:12.4`; `HF_HOME=/xcelsior-cache/...`.
- [x] **Mode B — Custom Docker image**: validated in `serverless/service.py` +
      endpoint schema; echo example in `serverless/examples/echo/`.
- [x] Base custom image `serverless/images/base/Dockerfile` →
      `xcelsior/serverless-base:cuda12.4-py3.12`.
- [x] `docs/serverless/HANDLER_CONTRACT.md`, `QUICKSTART.md`.
- [x] Multi-stage Dockerfiles (slim base + vLLM preset layer).

**Phase 4 exit:** ✅ Met — proceed to Phase 5 (scheduler + worker-agent integration).

---

## Phase 5 — Scheduler & worker-agent integration ✅

> Completed 2026-06-08. Tests: `tests/test_serverless_agent_integration.py` (5),
> `tests/test_platform_env.py`, `tests/test_worker_agent_allowlist.py` (+ serverless suite).

- [x] `serverless/service.py` `provision_worker` submits via `scheduler.submit_job()`
      with `job_type=serverless_worker`, `gpu_model`, `volume_ids`, `exposed_ports`;
      `_set_job_fields` stores `serverless_worker_id`, `serverless_endpoint_id`,
      `environment`, `volume_mounts` (`/xcelsior-cache`), `http_port`,
      `health_check_path`; `scheduler_job_id` on worker row.
- [x] Placement respects GPU tier / VRAM via existing allocate / binpack
      (`gpu_model` from `gpu_tier`, `vram_required_gb`).
- [x] `worker_agent.py`: serverless branch — HTTP health wait, `ready`/`exited`
      callbacks, port mapping + `http_ports` report, platform env injection
      (`XCELSIOR_WORKER_ID`, `XCELSIOR_ENDPOINT_ID`, `XCELSIOR_HTTP_PORT`,
      `XCELSIOR_HEALTH_CHECK_PATH`, `XCELSIOR_API_TOKEN`); skips SSH path.
- [x] `routes/serverless.py`: `POST …/workers/{id}/exited` (requeue in-flight jobs).
- [x] `routes/agent.py`: `prepull_image` agent command; `volume_mounts` merge
      preserves serverless cache paths.
- [x] `serverless/repo.py`: `get_worker_by_scheduler_job_id`.
- [x] `serverless/openai_proxy.py`: `worker_base_url` uses mapped `http_ports`.
- [x] Reuse job logs (`LogForwarder` + `_push_log_lines`) for worker boot/health.

**Phase 5 exit:** ✅ Met — proceed to Phase 6 (autoscaler runtime + reaper).

---

## Phase 6 — Autoscaler runtime, cold-start & failure recovery ✅

> Completed 2026-06-08. Tests: `tests/test_serverless_autoscaler.py` (9),
> `tests/test_bg_worker_reconcile.py` (serverless_reconcile @ 45s).

- [x] `serverless_reconcile` in `api.py` + `bg_worker.py` (45 s, exception-safe);
      `XCELSIOR_SERVERLESS_RECONCILE=false` skips tick; `XCELSIOR_BG_TASKS=false`
      in CI/test disables API-embedded tasks.
- [x] Scale-up on queue/wait breach; `process_queue()` after scale-up batch;
      scale-down with `idle_timeout` + `SCALE_DOWN_COOLDOWN_SEC` + **drain-before-reap**
      (`workers_to_mark_draining` → `DRAINING` → `workers_to_reap`); scale-to-zero.
- [x] Keep-warm (`min_workers≥1`) via `compute_desired_workers` floor.
- [x] **TTL reaper** (`serverless/reaper.py`): stale worker heartbeat →
      `handle_worker_lost` + deprovision; request-timeout jobs failed;
      stuck-`booting` + dead scheduler job → deprovision + reprovision.
- [x] Worker-crash via `worker_exited` + reaper stale heartbeat (requeue in-flight).
- [x] Cold-start: `prepull_image` enqueued on assigned host during reconcile;
      NFS cache via `volume_ids` (Phase 5).
- [x] Anti-thrash: scale-down cooldown + drain grace; synthetic burst tests.

**Phase 6 exit:** ✅ Met — proceed to Phase 7 (GPU-seconds billing).

---

## Phase 7 — Billing (GPU-seconds) integration ✅

> Completed 2026-06-08. Novita-aligned: `running_seconds × ¢/s/worker`.
> Tests: `tests/test_serverless_billing.py` (8 passed).

- [x] `billing.charge_serverless_execution()` + `serverless/metering.py`
      incremental `charge_serverless_execution` → `billing_cycles`
      (`resource_type=serverless_gpu`); `bill_active_serverless_workers` in
      `auto_billing_cycle`.
- [x] Wallet pre-flight on enqueue + OpenAI chat; **suspended wallet → 402**.
- [x] Job completion: `cost_cad=0` on jobs; tokens via `token_cost_metadata` only.
- [x] Cold-start: metadata line item `serverless_gpu_cold_start` (amount in worker
      uptime; flagged for UI/disputes); `cold_start_seconds` on first claimed job.
- [x] No double-charge: incremental slices from `last_billed_period_end`; final
      slice on deprovision only bills unbilled seconds.
- [x] Team endpoints use `_serverless_scope_owner_id` → team wallet `owner_id`.
- [x] `pricing_for_endpoint`: `rate_cents_per_second_per_worker` for dashboard.

**Phase 7 exit:** ✅ Met — proceed to Phase 8 (keys, rate limits, webhooks).

---

## Phase 8 — Keys, rate limits, idempotency, webhooks ✅

> Completed 2026-06-08. Tests: `tests/test_serverless_limits_webhooks.py` (9 passed).

- [x] API key issue/list/revoke; hashed at rest; `key_prefix` shown once.
- [x] Per-key RPM rate limiting + `X-RateLimit-*` headers + 429.
- [x] `max_queue_size` backpressure → 429 (migration `038`, default 100).
- [x] Idempotency-Key dedupe (partial unique + race handling) → retried submit
      returns original.
- [x] Webhooks on terminal state: **HMAC-signed** (`X-Xcelsior-Signature`),
      bounded retries + exponential backoff; `webhook_status` recorded;
      `serverless_webhook_retry` bg task (30 s).
- [ ] (Client SDK, optional) exponential-backoff polling for `/status`; graceful
      404 after retention purge.

**Phase 8 exit:** ✅ Met — proceed to Phase 9 (streaming hardening).

---

## Phase 9 — Streaming hardening ✅

> Completed 2026-06-08. Tests: `tests/test_serverless_streams.py` (9 passed).

- [x] SSE framing correct (`text/event-stream`, `data:` chunks, `: ping`
      heartbeats, terminal `[DONE]`) with anti-buffering headers.
- [x] **Mid-stream error → final error chunk** before close (never silent).
- [x] Client disconnect → `cancel_inflight_job` releases worker concurrency;
      OpenAI proxy stream aborts upstream on disconnect; worker SDK skips
      cancelled jobs.
- [x] Reconnect-safe replay from `serverless_job_stream_events` by `after_seq`.
- [x] `runsync` empty-output defensive handling via `normalize_runsync_output`.

**Phase 9 exit:** ✅ Met — proceed to Phase 10 (observability).

---

## Phase 10 — Observability ✅

> Completed 2026-06-08. Tests: `tests/test_serverless_observability.py` (8 passed).

- [x] SSE events: `serverless_endpoint.created`, `serverless_worker.scaled`,
      `serverless_job.completed` / `.failed` / `.queued` through the existing bus.
- [x] Per-endpoint metrics: requests, success/error rate, `avg_queue_ms`,
      `avg_execution_ms`, `gpu_seconds`, `tokens_per_sec`, active/idle/busy
      workers, cost (`?since_hours=` window).
- [x] Structured per-job logs (`serverless/observability.py`); correlation IDs
      from `X-Correlation-ID` / `X-Request-ID` / job_id.
- [x] OpenTelemetry spans on enqueue/dispatch/claim; Prometheus gauges
      (`serverless_queue_depth`, `serverless_active_workers`,
      `serverless_idle_workers`, `serverless_cold_starts_total`,
      `serverless_job_errors_total`).

**Phase 10 exit:** ✅ Met — proceed to Phase 11 (frontend).

---

## Phase 11 — Frontend (Next.js dashboard) ✅

> Completed 2026-06-08. **Deploy Studio** (not onboarding wizard) at
> `/dashboard/inference/new`; list hub + detail at `/dashboard/inference/[id]`.

- [x] `frontend/src/lib/api.ts`: endpoint CRUD, key CRUD, metrics fetchers,
      sync/async invoke helpers, SSE/EventSource helpers.
- [x] Feature folder `frontend/src/features/serverless/`:
  - [x] `types.ts`
  - [x] `deploy-studio.tsx` — **Novita-modeled Deploy Studio** (step order per
        `INFERENCE_DESIGN.md` §4):
        1. **Deployment method** — segmented `Choose a model` | `Custom Docker image`.
        2. **Source** — model picker (library + HF id) **or** custom fields.
        3. **Hardware** — GPU type (`gpu-models.ts` + marketplace) + `GPUs/Worker`.
        4. **Scaling** — min/max workers, concurrency, idle timeout, scaling policy.
        5. **Environment variables**.
        6. **Review & deploy** — cost estimate + cold-start note. Manual validation;
           team-aware.
  - [x] `endpoint-detail.tsx` — tabs: overview/metrics, workers, jobs, try-it, keys.
  - [x] `metrics-panel.tsx` (Recharts + SSE refresh), `workers-panel.tsx`,
        `logs-panel.tsx`, `try-it-console.tsx` (chat/job + curl + OpenAI snippets),
        `keys-panel.tsx`, `cost-usage-panel.tsx`.
- [x] Dashboard routes: list (`/dashboard/inference`), create (`/new`), detail (`/[id]`).
- [x] **Team-aware**: viewer read-only; reload on `xcelsior-team-changed`; banner.
- [x] **i18n**: `en-dashboard.ts` + `fr-dashboard.ts` (`dash.serverless.*`).
- [x] Loading/error/empty states; responsive; Sonner toasts.

**Phase 11 exit:** ✅ Met — proceed to Phase 12 (tests).

---

## Phase 12 — Tests (write alongside, not after) ✅

> Completed 2026-06-08. Backend: 125 passed (`./run-tests.sh serverless`).
> Frontend: Vitest 10/10 (`npm run test:serverless`); Playwright 2/2
> (`e2e/serverless-smoke.spec.ts` — auth guard + full mocked flow).

- [x] Backend (runnable **without GPUs**, mock scheduler/worker):
  - [x] `tests/test_serverless_routes.py` — management + queue routes, status codes.
  - [x] `tests/test_serverless_tenancy.py` — owner/team scope, viewer 403,
        cross-account 404, key create/revoke.
  - [x] `tests/test_serverless_autoscaler.py` — desired-count math, hysteresis,
        scale-to-zero, drain-before-reap.
  - [x] `tests/test_serverless_billing.py` — GPU-seconds debit, suspended wallet.
  - [x] `tests/test_serverless_openai_compat.py` — chat/completions/embeddings
        (incl. capability-gated failure) + streaming chunks/`[DONE]`.
  - [x] `tests/test_serverless_idempotency.py` — dedupe, webhook HMAC, retries.
  - [x] Queue/dead-letter: `tests/test_serverless_queue_recovery.py` (TTL requeue,
        request-timeout fail, crash recovery).
  - [x] Migration up/down: `tests/test_serverless_repo.py` (037 chain + 038
        `max_queue_size` column).
- [x] Frontend: Vitest (Deploy Studio, list, detail panels, viewer read-only);
      Playwright smoke `e2e/serverless-smoke.spec.ts` (session cookie + API route
      mocks; auth flash-redirect fix in `auth.tsx`).
- [x] `run-tests.sh serverless` target includes all serverless test modules.

**Phase 12 exit:** ✅ Met — full serverless suite green locally without real GPUs.

---

## Phase 13 — Security review ✅

- [x] Cross-account endpoint/job/key access → **404** (`test_serverless_tenancy.py`,
      `TestKeyMismatch404` in `test_serverless_security.py`).
- [x] Viewers cannot create/delete/rotate keys/run mutating tests (API +
      `test_serverless_tenancy.py`; UI Vitest viewer read-only controls).
- [x] API keys hashed; raw shown once (`xcel_` prefix); revocation immediate
      (`serverless/keys.py`, `test_serverless_service.py`).
- [x] Custom containers sandboxed; env/secrets injected, never logged; secrets
      encrypted at rest (`serverless/env_secrets.py`, `encrypt_env_for_storage` on
      create, `decrypt_env_for_worker` on provision).
- [x] Execution + queue timeouts enforced; a job cannot bill forever (TTL)
      (`serverless/reaper.py` `QUEUE_TTL_SEC`, `test_serverless_queue_recovery.py`,
      `TestQueueTTL`).
- [x] Idempotent enqueue; exactly-once-ish billing reconciliation
      (`test_serverless_idempotency.py`, `test_serverless_billing.py`).
- [x] Webhook SSRF guard (no internal IPs); HMAC signature; size caps
      (`serverless/webhooks.py` `_fit_webhook_payload`, `test_serverless_limits_webhooks.py`,
      `TestWebhookBodyCap`).
- [x] Rate-limit + `max_queue_size` abuse protection (`test_serverless_limits_webhooks.py`,
      `max_request_bytes` → 413 on `/run`).
- [x] Path-traversal/symlink guards on cache-volume cleanup (reuse `volumes.py`
      `readlink -f` + NFS prefix guard; shared cache volumes not deleted on endpoint
      delete).

**Phase 13 exit:** ✅ Met — security checklist ticked; `tests/test_serverless_security.py`
(7 passed); full `./run-tests.sh serverless` 132 passed.

---

## Phase 14 — Rollout & iteration ✅

- [x] Feature-flag the product (env + per-owner allowlist) — `serverless/feature.py`,
      `GET /api/v2/serverless/enabled`, route guards in `_deps.py` / `routes/serverless.py`,
      dashboard nav + `ServerlessFeatureGate` layout.
- [x] Config/env documented in `.env.example` (reconcile, TTLs, webhooks, billing
      interval, NVMe cache vars cross-ref).
- [x] **Runbook** `docs/SERVERLESS_RUNBOOK.md`: stuck job, stuck/booting worker,
      scale-down failure, NFS cache miss storms, billing disputes, webhook
      failures, cold-start regressions, emergency disable.
- [x] Post-deploy smoke `scripts/serverless_e2e_smoke.py`: create → run → status →
      stream → cancel → scale-to-zero → delete (exit 1 on any failure).
- [x] Monitor queue depth, cold starts, costs, error rate; tune policy (Prometheus
      metrics + `/metrics` API documented in runbook).
- [x] Phase 15 backlog: predictive scaling (`scaling_predictive.py` + autoscaler),
      TGI/SGLang managed_engine presets (backend + Deploy Studio), GitHub-deploy
      source resolution (`serverless/github_deploy.py`), vanity slugs in API
      (`serverless/vanity.py`), multi-region cache replication env
      (`XCELSIOR_SERVERLESS_CACHE_REPLICATE_REGIONS`), optional Redis RPM
      (`serverless/rate_limit_store.py`). Multi-GPU workers already via `gpu_count`.

**Phase 14 exit:** ✅ Met — flag-gated GA path ready; runbook + smoke script complete;
`tests/test_serverless_feature.py` (4 passed); `./run-tests.sh serverless` 136 passed.

---

## Master checklist (quick scan)

```
Phase 0  Verify substrate      [x] decisions #1–4,#7 + delta note
Phase 1  Data model/migrations [x] 5 tables + indexes + repo
Phase 2  Service layer         [x] service/dispatcher/autoscaler/proxy/metering/streams/keys/cache
Phase 3  API surface           [x] mgmt + queue + OpenAI + deps guards
Phase 4  Worker SDK/preset     [x] 2 modes + vLLM image + templates/docs
Phase 5  Scheduler integration [x] submit via scheduler + agent callbacks
Phase 6  Autoscaler runtime    [x] reconcile + scale-to-zero + reaper + cold-start
Phase 7  Billing (GPU-seconds) [x] charge_serverless_execution + suspend
Phase 8  Keys/limits/idemp/hook[x] keys + 429 + dedupe + webhooks
Phase 9  Streaming hardening   [x] errors + cancel + replay
Phase 10 Observability         [x] metrics + events + OTel/Prom
Phase 11 Frontend              [x] Deploy Studio + detail + try-it + keys + i18n
Phase 12 Tests                 [x] backend + frontend vitest + Playwright smoke
Phase 13 Security              [x] 404/viewer/keys/webhook/TTL
Phase 14 Rollout               [x] flag + runbook + smoke
```

## Definition of done

1. [x] Create endpoint (preset **and** custom) from API + UI.
2. [x] Async `/run` + sync `/runsync` + `/status` + `/stream` + `/cancel` work.
3. [x] OpenAI SDK works by changing base_url + key (preset).
4. [x] Autoscaler: burst → scale-up; idle → scale-to-zero; crash → recover.
5. [x] Billing debits the correct wallet by **GPU-seconds**; suspended wallet
       blocks; team vs. personal correct.
6. [x] Idempotency + rate limits + HMAC webhooks enforced.
7. [x] Streaming: errors always surface; disconnect frees GPU; reconnect replays.
8. [x] Dashboard: live metrics, try-it console, keys, cost; viewer read-only; en/fr.
9. [x] All tests green in CI without real GPUs; pyright gate clean (serverless suite).
10. [x] Runbook + post-deploy smoke complete; feature flag controls GA.

---

## Work log

| Date | Change |
|------|--------|
| 2026-06-08 | Checklist created from the best-of synthesis of the 3 reports. |
| 2026-06-08 | Re-modeled deploy modes + settings + wizard on Novita; design consolidated into `INFERENCE_DESIGN.md` (lowercase duplicate removed). |
| 2026-06-08 | Phase 3 complete — `routes/serverless.py`, `tests/test_serverless_routes.py` (10 passed). |
| 2026-06-08 | Phase 4 complete — `serverless/worker_sdk/`, worker callbacks, Dockerfiles, `tests/test_serverless_worker_sdk.py` (7 passed). |
| 2026-06-08 | Phase 5 complete — `provision_worker` payload enrichment, `worker_agent` serverless branch, `prepull_image` command, `tests/test_serverless_agent_integration.py` (5 passed). |
| 2026-06-08 | Phase 6 complete — `serverless/reaper.py`, drain-before-reap, cooldown, `tests/test_serverless_autoscaler.py` (9 passed). |
| 2026-06-08 | Phase 7 complete — Novita-aligned worker-second billing, incremental cycles, `tests/test_serverless_billing.py` (8 passed). |
| 2026-06-08 | Phase 8 complete — per-key RPM, queue backpressure, idempotency race, HMAC webhooks + retry, `tests/test_serverless_limits_webhooks.py` (9 passed). |
| 2026-06-08 | Phase 9 complete — live SSE poll/replay, disconnect cancel, OpenAI stream abort, `tests/test_serverless_streams.py` (9 passed). |
| 2026-06-08 | Phase 10 complete — metrics API, dot-notation SSE, structured logs, OTel + Prometheus, `tests/test_serverless_observability.py` (8 passed). |
| 2026-06-08 | Phase 12 complete — OpenAI compat, idempotency/webhook retry, queue recovery, migration 038 tests; Vitest 10/10; Playwright smoke 2/2; `./run-tests.sh serverless` 125 passed. |
| 2026-06-08 | Phase 13 complete — `xcel_` key prefix (not `xcsl_`); env encryption/redaction; payload 413; key/endpoint mismatch 404; queue TTL reaper; webhook body fit cap; `tests/test_serverless_security.py` (7); `./run-tests.sh serverless` 132 passed. |
| 2026-06-08 | Phase 14 complete — `XCELSIOR_SERVERLESS_ENABLED` + allowlist; `/api/v2/serverless/enabled`; dashboard feature gate; `docs/SERVERLESS_RUNBOOK.md`; `scripts/serverless_e2e_smoke.py`; `.env.example` serverless block; `tests/test_serverless_feature.py` (4); 136 passed. |
| 2026-06-08 | Loose ends — PayPal live `.env` + `.env.paypal-live.example`; restored legacy `routes/inference.py` (batch jobs); `/api/v2/inference/*` compat aliases → serverless; OpenAPI allowlist updated; `post_merge_smoke.sh` runs serverless suite. |
