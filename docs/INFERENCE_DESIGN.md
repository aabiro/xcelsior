# Serverless Inference Endpoints — Design (canonical)

*Created: 2026-06-08 · merged 2026-06-08 (v1 plan + best-of synthesis + Novita UX modeling)*
*Status: API + worker SDK live — scheduler/agent integration next*
*Roadmap: [NEXT_PRIORITIES_ROADMAP.md](./NEXT_PRIORITIES_ROADMAP.md) §6*
*Source reports: [inference_reports.md](./inference_reports.md) · Build plan: [SERVERLESS_IMPLEMENTATION_CHECKLIST.md](./SERVERLESS_IMPLEMENTATION_CHECKLIST.md)*

> This is the single canonical design doc. It supersedes the earlier lowercase
> `inference_design.md` (removed) to fit the repo's SCREAMING_CASE doc convention.

## TL;DR

Customers create a **serverless inference endpoint** two ways — **(A) choose a
model** (managed preset) or **(B) bring a custom Docker image** — billed to their
active workspace wallet primarily by **GPU-seconds**. Requests hit an
**OpenAI-compatible HTTPS API** plus a **queue job API**, routed to vLLM (preset)
or the customer's container on the existing scheduler. v1 uses **scale-to-zero
with cold-start** and an optional warm `min_workers`. **Settings and wizard UX are
modeled directly on Novita's serverless endpoints.** Team tenancy, wallet
preflight, and viewer gates follow the instance/volume patterns.

---

## 1. Verdict — which architecture we chose

Of the three reports in [`inference_reports.md`](./inference_reports.md):

- **Report C ("Serverless Inference Endpoints for Xcelsior") is the spine** — it
  alone models compute correctly: a serverless worker **is an existing scheduler
  job + a first-class `serverless_workers` row**, not a parallel compute primitive
  or a second orchestration plane.
- **From Report A** (theory): Postgres `FOR UPDATE SKIP LOCKED` queue, TTL/
  dead-letter reaper, vLLM/PagedAttention rationale, NFS weight-cache flashboot,
  worker fitness checks, mid-stream error chunks, `with_cancellation` KV-cache
  release on disconnect.
- **From Report B** (study-first): tech table, idempotency/HMAC-webhook/rate-limit
  specifics, and the open-questions list.

**Stack (2026-06-08):** `serverless/` service package + `routes/serverless.py` +
`serverless_*` Postgres tables. Control-plane authoritative GPU-seconds billing,
Novita-modeled preset/custom endpoints, OpenAI-compatible proxy at
`/v1/serverless/{id}/openai/v1`.

---

## 2. Architecture at a glance

```
                          ┌─────────────────────────────────────────────┐
   client (OpenAI SDK,    │              CONTROL PLANE (FastAPI)          │
   curl, dashboard) ─────▶│  routes/serverless.py → serverless/service     │
                          │  • auth (session/team OR per-endpoint key)    │
                          │  • validate, tenancy guard (owner_id, 404)    │
                          │  • enqueue / proxy / stream / meter           │
                          └───────────────┬───────────────┬──────────────┘
              enqueue durable job          │               │ proxy (preset, warm)
                                           ▼               ▼
                          ┌───────────────────────┐   ┌─────────────────────────┐
                          │   POSTGRES (state)     │   │   WORKER (GPU host)      │
                          │  serverless_endpoints  │   │  scheduler job (submit_  │
                          │  serverless_workers    │   │  job) placed by binpack  │
                          │  serverless_jobs (queue│   │  ├─ A: vLLM OpenAI server │
                          │    FOR UPDATE SKIP LOCK)│   │  └─ B: custom image +     │
                          │  serverless_job_stream │◀──┤     start cmd + HTTP port │
                          │  serverless_api_keys   │   │  health check + heartbeat │
                          └───────────┬────────────┘   └─────────────────────────┘
        reconcile (bg_worker)         │
        autoscaler ───────────────────┘  desired = f(queue request count | queue delay)
        clamp[min,max] · idle_timeout · drain-before-reap · scale-to-zero
```

Three planes, all on existing rails: FastAPI control plane (orchestration) + new
`serverless/` service package (logic); Postgres state + `SKIP LOCKED` queue (no
Redis/Celery/Kafka day one); worker = managed scheduler job + `serverless_workers`
metadata row.

---

## 3. Deployment modes — **modeled on Novita** ⭐

Novita's serverless lets you **either pick a model or supply a Docker image**. We
mirror that exactly.

### Mode A — Choose a model (managed preset)
- User picks a model (curated library entry or a HuggingFace id).
- Platform runs a **managed vLLM OpenAI-server image** with the correct start
  command and the model mounted from the NFS weight cache.
- User only configures hardware + scaling + (optionally) a HF token env var.
- Default preset image: `xcelsior/serverless-vllm:<cuda>`; later presets: TGI,
  SGLang.

### Mode B — Custom Docker image (Novita field-for-field)
The settings below are taken directly from Novita's "Create Serverless Endpoint":

| Setting | Meaning | Example |
|---------|---------|---------|
| **Container Image** | Image repo address (public or private) | `vllm/vllm-openai:latest` |
| **Container Registry Auth** | Credentials for a private image (platform Settings) | — |
| **Container Start Command** | Command/args run at container start | `--model meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096` |
| **HTTP Port** | Port the worker exposes; the load balancer forwards here | `8080` |
| **Health Check Path** | LB health probe; forwards only on HTTP 200 | `/health` |
| **Environment Variables** | Service env (e.g. HF token) | `HUGGING_FACE_HUB_TOKEN=...` |
| **CUDA Version** | Requested CUDA runtime for placement | `12.4` |

Mode B maps onto the worker-SDK ASGI mode: the platform proxies to the container's
`HTTP Port`, health-checks `Health Check Path`, and (for OpenAI-compat) treats the
container as already speaking the OpenAI surface (as `vllm/vllm-openai` does).

### Common settings (both modes)
| Setting | Meaning |
|---------|---------|
| **GPU type / tier** | Hardware class (reuse `gpu-models.ts` catalog) |
| **GPUs / Worker** | GPU count per worker (Novita "GPUs/Worker") |
| **Min Workers** | Warm pool floor (0 = scale-to-zero) |
| **Max Workers** | Scale ceiling |
| **Max Concurrency** | Max concurrent requests **per worker**; overflow → other workers → queue (Novita semantics) |
| **Idle Timeout (seconds)** | Idle duration before a worker goes offline |
| **Auto-scaling strategy** | One of the two Novita strategies (below) |
| **Request timeout / Max request size** | Per-job execution + payload caps |
| **Region** | `ca-east` default; host region affinity |
| **Endpoint name** | Display + slug |

### Auto-scaling strategies — **Novita's two modes**
`scaling_policy_type ∈ { queue_request_count, queue_delay }`:

1. **Queue Max Request Count** (`queue_request_count`): scale **up** when pending
   request count > threshold; scale **down** when below it (+ `Idle Timeout`).
2. **Queue Delay** (`queue_delay`): configure **Queue Delay Time (seconds)**;
   scale **up** when a request's queue-wait exceeds it; scale **down** when below
   it (+ `Idle Timeout`).

`scaling_policy_value` stores the threshold (a count or a delay-seconds), keeping
our reconciliation autoscaler (§6) a thin, deterministic implementation of
Novita's published behavior.

---

## 4. UI/UX — **modeled on Novita's wizard + endpoint console** ⭐

### Create-endpoint wizard (step order mirrors Novita)
1. **Deployment method** — segmented choice: **`Choose a model`** | **`Custom
   Docker image`**.
2. **Source config**
   - Model mode: model picker (library + HF id search), revision.
   - Custom mode: **Container Image**, **Container Start Command**, **HTTP Port**,
     **Health Check Path**, **Container Registry Auth** (if private), **CUDA
     Version**.
3. **Hardware** — GPU type + **GPUs/Worker**.
4. **Scaling** — **Min/Max Workers**, **Max Concurrency**, **Idle Timeout (s)**,
   **Auto-scaling strategy** toggle (`Queue Request Count` | `Queue Delay`) with
   its threshold field.
5. **Environment variables** (+ registry auth surfaced if private image).
6. **Review & deploy** — show estimated cost + cold-start expectation.

### Endpoint detail / console (Novita-style)
- Header: status, **endpoint URL** + **OpenAI base_url**, **API key** with copy.
- **Workers panel**: running / idle / booting counts + per-worker state.
- **Metrics panel**: requests, success/error rate, queue time, exec time,
  `gpu_seconds`, tokens/s, cost (Recharts, SSE-driven).
- **Logs panel**: per-worker/per-job structured logs.
- **Try-it console**: chat or custom payload → streaming preview + auto-generated
  **curl** and **OpenAI-SDK** snippets using the endpoint base_url.
- **Keys** tab; **Cost/usage** view; **Settings** (edit scaling/env).
- Team-aware: viewer read-only; reload on team switch; **en + fr** i18n.

---

## 5. Public API surface

**Namespace:** serverless endpoints live under **`/v1/serverless`** on the public
API host (e.g. `https://api.xcelsior.ai/v1/serverless/...`).

**(a) OpenAI-compatible** (path-based; proxied for preset, capability-gated):
```
POST /v1/serverless/{endpoint_id}/openai/v1/chat/completions
POST /v1/serverless/{endpoint_id}/openai/v1/completions
POST /v1/serverless/{endpoint_id}/openai/v1/embeddings
GET  /v1/serverless/{endpoint_id}/openai/v1/models
```
Customers point any OpenAI SDK at us via **base_url =
`https://api.xcelsior.ai/v1/serverless/{endpoint_id}/openai/v1` + api_key**.
The control plane **must** expose the OpenAI API surface; unsupported capabilities
(e.g. vision on a text-only worker) return an explicit **4xx** — never silent
misrouting (§11.2).

**(b) Queue job API** (guaranteed for every endpoint incl. custom):
```
POST /v1/serverless/{endpoint_id}/run        → {id, status: IN_QUEUE}    (async)
POST /v1/serverless/{endpoint_id}/runsync    → output | error            (sync, timeout-bounded)
GET  /v1/serverless/{endpoint_id}/status/{job_id}
GET  /v1/serverless/{endpoint_id}/stream/{job_id}   (SSE)
POST /v1/serverless/{endpoint_id}/cancel/{job_id}
```
Accepts `Idempotency-Key` + optional `webhook` (webhook/HMAC details §11.9; deferred
past first increment). Status enum: `IN_QUEUE | IN_PROGRESS | COMPLETED | FAILED |
CANCELLED`.

**(c) Management API** (session/team auth):
```
POST|GET|GET/{id}|PATCH/{id}|DELETE/{id}  /api/v2/serverless/endpoints
GET  …/{id}/workers | /jobs | /metrics | /logs
POST|GET|DELETE                            …/{id}/keys
```

**(d) Worker readiness (Phase 0 pattern):** simple **`curl`-able health routes** on
each worker (preset: vLLM `/health`; custom: endpoint `health_check_path`) before
the proxy marks a worker ready. No fancy substrate probing in Phase 0 — HTTP 200 from
the worker health path is the gate.

---

## 6. Data model & autoscaler

**Five Alembic-managed tables** (never ad-hoc DDL): `serverless_endpoints`
(all settings from §3 incl. `mode`, `managed_engine`, `model_ref`, `image_ref`,
`startup_command`, `http_port`, `health_check_path`, `cuda_version`, `gpu_tier`,
`gpu_count`, `min/max_workers`, `max_concurrency`, `idle_timeout_sec`,
`scaling_policy_type`, `scaling_policy_value`, `request_timeout_sec`,
`max_request_bytes`, `keep_warm`, `cache_volume_id`, `region`, `status`,
timestamps, soft-delete); `serverless_workers`; `serverless_jobs` (the queue, full
timing + `gpu_seconds`, `cold_start_seconds`, `cost_cad`, `idempotency_key`);
`serverless_job_stream_events` (replayable SSE); `serverless_api_keys`
(`key_prefix` + `key_hash`, scopes, `rate_limit_rpm`).

**Autoscaler** — deterministic reconciliation loop in `bg_worker` (30–60 s,
bounded, idempotent, single-writer via advisory lock; claims jobs with `SKIP
LOCKED`). Per endpoint: compute free slots `Σ(max_concurrency − current)`, derive
`desired` from the endpoint's Novita strategy (`queue_request_count` or
`queue_delay`), clamp `[min,max]`, scale up immediately on breach, scale down only
after `idle_timeout_sec` + cooldown, **drain before reap**. TTL/dead-letter reaper
requeues stalled jobs; worker crash → requeue + scale event.

---

## 7. Endpoint lifecycle

```
create → provisioning (if min_workers≥1) → active → scaled_down → deleted
         ↓ cold request                    ↑ idle > idle_timeout_sec
         cold_start (image pull + model load + worker) ─┘
```

| State | Customer-visible | Backend |
|-------|------------------|---------|
| `provisioning` | Spinner + worker job id | `submit_job` + queue poll |
| `active` | URL + base_url + API key | ≥0 workers; accepts traffic |
| `scaled_down` | "Cold — first request slower" | 0 workers; next request provisions |
| `error` | Actionable message | Worker failed / wallet empty |
| `deleted` | Gone | Soft-delete + worker terminate |

---

## 8. Billing model

**Primary debit = GPU-seconds allocated** (industry standard: Modal, RunPod, Novita),
replacing token-count as the authoritative charge in the serverless flow. Bill for the
time a GPU is **provisioned to the workload** — actively inferring *or* holding the
model warm — not "active compute only."

| Meter | Unit | Basis |
|-------|------|-------|
| **GPU time** | per second per allocated worker | `(released_at − allocated_at)` in seconds, **control-plane timestamps only**; cost = host `cost_per_hour_cad` × tier × `gpu_count` ÷ 3600 — **authoritative** |
| **Cold start** | included in GPU-seconds | Billable from worker allocation through release; UI shows estimate ("cold start: ~12s, billed") |
| Input tokens | per 1M | `XCELSIOR_INPUT_TOKEN_PRICE` ($0.50 CAD/M) — **observability only** |
| Output tokens | per 1M | `XCELSIOR_OUTPUT_TOKEN_PRICE` ($1.50 CAD/M) — **observability only** |

**Source of truth:** control plane records `allocated_at` when `submit_job` succeeds /
`serverless_workers.state` enters `booting`, and `released_at` on deprovision /
scale-down / job kill. Worker-reported spans are **telemetry only** (utilization,
debugging) — never used for billing (§11.3).

Write usage to `billing_cycles` with `resource_type = serverless_gpu` via
`charge_serverless_execution(...)`. Wallet preflight on enqueue; **suspended wallet
→ reject** (402). Team-owned endpoints debit the **team wallet**. Viewer cannot
create/delete/invoke write-scoped routes.

---

## 9. Current codebase inventory (verified 2026-06-08)

| Layer | File | State |
|-------|------|-------|
| Routes | `routes/serverless.py` | Management `/api/v2/serverless/endpoints`, queue `/v1/serverless/{id}/run|runsync|status|stream|cancel`, OpenAI `/v1/serverless/{id}/openai/v1/*`, worker ready callback |
| Service | `serverless/service.py`, `dispatcher.py`, `autoscaler.py`, `metering.py`, `openai_proxy.py`, `keys.py`, `cache.py`, `streams.py` | Lifecycle, reconcile, GPU billing, proxy |
| Worker SDK | `serverless/worker_sdk/` | Queue `handler(job)`, ASGI `serve_asgi`, fitness, `/healthz` |
| Images | `serverless/images/{base,vllm}/`, `serverless/examples/echo/` | Base + vLLM preset Dockerfiles |
| Repo | `serverless/repo.py` | CRUD + `claim_next_job()` (`SKIP LOCKED`), idempotency, stream events |
| DB | `serverless_*` (5 tables) | Migration `037`; endpoints, workers, jobs, stream events, API keys |
| Tenancy | `routes/_deps.py` | `_serverless_*` guards, `_resolve_serverless_endpoint_auth` (session + `xcel_` keys) |
| Background | `api.py`, `bg_worker.py` | `serverless_reconcile` every 45 s |
| Billing | `billing.py` | `serverless_workers` uptime tick → `resource_type=serverless_gpu` |
| Frontend | `frontend/src/app/(dashboard)/dashboard/inference/page.tsx`, `frontend/src/lib/api.ts` | Create/list/delete + GPU-seconds usage |
| Tests | `tests/test_serverless_{repo,service,routes,worker_sdk}.py`, `test_team_tenancy_sweep.py` | Repo, service, routes, worker SDK, team IDOR |

### Reusable primitives (confirmed present)

| Primitive | Location | Notes |
|-----------|----------|-------|
| Scheduler | `scheduler.submit_job()`, `kill_job()` | Worker placement for `serverless_workers` |
| Agent queue | `routes/agent.py` | `agent_commands` claimed with `FOR UPDATE SKIP LOCKED` |
| Tenancy / SSE | `routes/_deps.py` | `_canonical_owner_id()`, `_user_team_id()`, `broadcast_sse()` |
| Billing | `billing.py` | Wallets, suspension, `billing_cycles`, background tick |
| Volumes | `volumes.py` | NFS attach/mount for weight-cache volumes |
| Observability | `events.py`, `api.py` | SSE bus; `_bg_worker()` periodic-task registration |
| Frontend libs | `frontend/src/lib/api.ts`, `gpu-models.ts` | `apiFetch`, GPU catalog, Recharts/Sonner in dashboard |

### Route matrix (TestClient, 2026-06-08)

| Route | Auth | Notes |
|-------|------|-------|
| `POST/GET/PATCH/DELETE /api/v2/serverless/endpoints` | session | CRUD + wallet preflight |
| `GET …/endpoints/{id}/health`, `…/usage`, `…/metrics`, `…/workers`, `…/jobs` | session | Dashboard + ops |
| `POST/GET/DELETE …/endpoints/{id}/keys` | session | Per-endpoint API keys |
| `POST /v1/serverless/{id}/run` | session or `xcel_` key | Async queue submit |
| `POST /v1/serverless/{id}/runsync` | session or key | Timeout-bounded sync |
| `GET …/status/{job_id}`, `…/stream/{job_id}` | session or key | Poll + SSE |
| `POST …/cancel/{job_id}` | session or key | Cancel in-flight job |
| `POST …/openai/v1/chat/completions` | session or key | Proxy to warm vLLM worker |
| `GET …/openai/v1/models` | session or key | OpenAI model list |

**Next (Phases 5–6):** scheduler/agent callbacks, full autoscaler runtime on real hosts.

---

## 10. Rejected / deferred

Redis/Celery/Kafka (day one) · Kubernetes/external autoscaler · vanity per-endpoint
hostnames (path-based first) · blockchain/tokenized compute · multi-region weight
replication · TGI/SGLang presets (later) · token-based billing as primary debit ·
multi-region failover, fine-tuning pipelines, model-registry marketplace (out of
v1).

---

## 11. Resolved decisions (Phase 0, 2026-06-08)

### 11.1 Naming & schema — **`serverless_*`**

- **`serverless_*` tables** + `serverless/` service package are the product surface.
- **Public API namespace:** `/v1/serverless/{endpoint_id}/…` (§5). Management:
  `/api/v2/serverless/endpoints`.
- All persistence, billing, and routing go through `serverless_*` rows — no parallel
  inference tables or engines.

### 11.2 OpenAI universality & capability gate — **OpenAI spec is table stakes**

- Every serverless endpoint **must** speak the OpenAI API surface (`/v1/chat/completions`,
  `/v1/models`, etc.) via the path-based proxy (§5).
- **Capability gate lives in the control plane:** route by model, size, and user tier;
  reject unsupported features with an explicit **4xx** (e.g. vision request →
  text-only worker). Never silently drop or misroute.

### 11.3 GPU-seconds formula & cold-start billing — **allocated time, control plane**

- **Formula:** `gpu_seconds = released_at − allocated_at` (integer seconds, ceiling),
  per worker allocation. Includes warm hold, active inference, **and cold start**.
- **Billing authority:** control plane timestamps only (`submit_job` / deprovision /
  scale-down). Worker-reported spans → observability metrics, **not** billing.
- **Cold start:** billable — GPU is allocated and unavailable to others for the full
  span. UI transparency: show estimated cold-start seconds and that they are billed.

### 11.4 Routing — **proxy for preset, queue for custom**

| Mode | Path | Mechanism |
|------|------|-----------|
| **A — preset** | OpenAI routes | Central **proxy** to warm worker's vLLM HTTP port; proxy selects worker by model, size, tier |
| **B — custom** | `/run`, `/runsync`, `/stream` | **Queue pull** — worker claims `serverless_jobs` via `SKIP LOCKED` |

Preset overflow (no warm worker) triggers provision + cold start, then proxy. Custom
workloads never bottleneck the shared OpenAI proxy.

### 11.5 Cold-start SLA — **measure first, advertise second**

- Do **not** publish a hard SLA in v1. Collect p50/p99 from first-increment telemetry.
- **Target** (internal): <45 s p99 for 7–8B with NFS flashboot + optional
  `min_workers=1`. Dashboard shows measured cold-start times per endpoint.

### 11.6 Queue backend — **pure Postgres**

- `serverless_jobs` queue with `FOR UPDATE SKIP LOCKED`. No Redis/Celery/Kafka in
  first increment.

### 11.7 First-increment scope — **ruthlessly small**

**Ship:**

- 1–2 popular **preset models** behind the OpenAI-compatible proxy
- Basic **autoscaling** + **scale-to-zero** + cold starts
- **GPU-seconds billing** (control-plane metering → `billing_cycles`)
- **Simple custom worker** support via queue API (`/run`, `/runsync`, `/status`)
- Worker readiness via **curl-able health routes** (§5d)

**Defer** (later phases):

- Full Novita wizard + endpoint console frontend (Phase 12)
- Handler SDK polish, webhooks, idempotency hardening (Phases 5, 10)
- All OpenAI routes beyond `chat/completions` + `models` in first cut
- TGI/SGLang presets, multi-region, vanity hostnames

### 11.8 Handler SDK — **function handler first**

- v1 custom workers: single **function** entrypoint + ASGI HTTP mode for
  OpenAI-native images. Class + `serve()` and advanced streaming envelopes deferred
  to Phase 5.

### 11.9 Idempotency & webhooks — **defer past first increment**

- First cut: optional `Idempotency-Key` storage only (no webhook delivery).
- Phase 10 target: HMAC-SHA256 webhook signatures (Report B), 3 retries with
  exponential backoff, 7-day idempotency-key retention.

### 11.10 Frontend insertion points — **confirmed in-tree**

| Asset | Path |
|-------|------|
| Dashboard page | `frontend/src/app/(dashboard)/dashboard/inference/page.tsx` |
| EN dashboard i18n | `frontend/src/lib/i18n/en-dashboard.ts` |
| FR dashboard i18n | `frontend/src/lib/i18n/fr-dashboard.ts` |
| API client | `frontend/src/lib/api.ts` |
| GPU catalog | `frontend/src/lib/gpu-models.ts` |

Phase 12 renames/extends dashboard routes to `/dashboard/serverless` when the console
ships; first increment is API-only for custom paths.

---

## 12. As-built vs. report claims (delta)

| Report claim | As-built reality | Status |
|--------------|------------------|--------|
| Full serverless queue + `SKIP LOCKED` job table | `serverless_jobs` + `claim_next_job()` | **Done** |
| GPU-second billing | `serverless/metering.py` + `billing.py` worker tick | **Done** (service); job-level charge on complete in Phase 7 |
| OpenAI proxy to warm vLLM | `serverless/openai_proxy.py` wired in `routes/serverless.py` | **Done** |
| `serverless_*` schema | Migration `037` + `serverless/repo.py` | **Done** |
| Management + queue API | `routes/serverless.py` mounted in `routes/__init__.py` | **Done** |
| Novita two-mode wizard | Dashboard create/list/delete only | Phase 12 |
| Scheduler + billing + volumes reusable | `submit_job`, `billing_cycles`, NFS volumes | **Confirmed** |
| Worker agent pull queue | `agent_commands` uses `SKIP LOCKED` | Reuse for `serverless_jobs` (Phase 5) |
| Billing/scheduling truth | Control plane authoritative for money and allocation spans | **Locked** |

---

## 13. Risks

| Risk | Mitigation |
|------|------------|
| Cold start >30 s on large models | Honest UX; `min_workers=1` for prod; preload popular models on NFS cache |
| vLLM OOM on small GPUs | VRAM estimate in `create_endpoint`; block undersized GPU |
| Token metering accuracy | GPU-seconds is authoritative; tokens via vLLM usage logs for display |
| Cross-tenant routing bug | `_require_serverless_endpoint_access`; IDOR sweep |
| Private image creds leak | Container Registry Auth encrypted at rest; never logged |

---

*Next step: [SERVERLESS_IMPLEMENTATION_CHECKLIST.md](./SERVERLESS_IMPLEMENTATION_CHECKLIST.md) — scheduler integration, worker-agent callbacks, autoscaler on real hosts.*

