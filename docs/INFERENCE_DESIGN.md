# Serverless Inference Endpoints — Design (v1)

*Created: 2026-06-08*  
*Status: Ready for implementation*  
*Roadmap: [NEXT_PRIORITIES_ROADMAP.md](./NEXT_PRIORITIES_ROADMAP.md) §6*

## TL;DR

Customers create a **persistent inference endpoint** (model + GPU + scaling bounds) billed to their active workspace wallet. Requests hit an **OpenAI-compatible HTTPS API** routed to vLLM/TGI workers on the scheduler. v1 uses **scale-to-zero with cold-start** (not always-on dedicated instances). Team tenancy, wallet preflight, and viewer gates follow instance patterns.

---

## 1. Product scope (v1)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Deployment model | Scale-to-zero endpoint + warm `min_workers` option | Matches `InferenceEngine` stubs; competitive with Replicate/Together |
| GPU allocation | Scheduler job per worker (`inference-{endpoint_id}`) | Reuses binpack, team billing, NFS optional |
| API surface | OpenAI `/v1/chat/completions` + `/v1/completions` | Industry standard; partial routing exists |
| Auth | OAuth client API key or session JWT | Keys already scoped (`inference:read/write`) |
| Billing | Per-token + per-second GPU idle/compute | `INPUT_TOKEN_PRICE_CAD_PER_M` env in `inference.py` |
| Models | HuggingFace id → vLLM/TGI docker templates | `xcelsior/vllm:latest`, `xcelsior/tgi:latest` |
| Regions | `ca-east` default; honor host region affinity | Consistent with volumes/instances |
| Team | `owner_id` = `_inference_scope_owner_id(user)` | Done in routes; viewer cannot create/delete |

**Out of v1:** multi-region failover, custom autoscaling metrics, fine-tuning pipelines, model registry marketplace.

---

## 2. Current codebase inventory

| Layer | File | State |
|-------|------|-------|
| Engine | `inference.py` | `InferenceEngine` — create/list/delete endpoints, `provision_worker`, token metering stubs |
| Store | `inference_store.py` | Job + result persistence for one-shot `/api/inference` |
| Routes | `routes/inference.py` | v1 sync/async poll, v2 CRUD endpoints, wallet preflight, team gates |
| Frontend | `dashboard/inference/page.tsx` | Create/list/delete UI (wizard partial) |
| Tests | `test_inference_endpoints_coverage.py`, `test_team_tenancy_sweep.py` | Auth + CRUD smoke; team IDOR partial |
| DB | `inference_endpoints` table | Created by engine migrations |

---

## 3. Endpoint lifecycle

```
create → provisioning (if min_workers≥1) → active → scaled_down → deleted
         ↓ cold request                    ↑ idle > scaledown_window
         cold_start (model pull + worker) ─┘
```

| State | Customer-visible | Backend |
|-------|------------------|---------|
| `provisioning` | Spinner + worker job id | `submit_job` + queue poll |
| `active` | URL + API key shown | ≥0 workers; accepts traffic |
| `scaled_down` | "Cold — first request slower" | 0 workers; next request triggers provision |
| `error` | Actionable message | Worker failed / wallet empty |
| `deleted` | Gone | Soft-delete + worker terminate |

---

## 4. Request path

```
Client HTTPS → nginx → FastAPI /v1/chat/completions
  → auth (API key / JWT)
  → resolve endpoint by model or endpoint_id header
  → wallet preflight (balance + not suspended)
  → InferenceEngine.route_request()
      → warm worker with model loaded? → forward
      → else provision_worker (cold) → queue → forward
  → stream SSE or JSON completion
  → meter tokens + GPU seconds → billing tick
```

**Rate limits:** per endpoint `max_concurrent`; per API key global bucket (reuse `_deps` patterns).

---

## 5. Billing model

| Meter | Unit | Default (env) |
|-------|------|---------------|
| Input tokens | per 1M | `XCELSIOR_INPUT_TOKEN_PRICE` ($0.50 CAD/M) |
| Output tokens | per 1M | `XCELSIOR_OUTPUT_TOKEN_PRICE` ($1.50 CAD/M) |
| GPU time | per second while worker running | Host `cost_per_hour_cad` × tier multiplier |
| Cold start | flat fee optional v1.1 | TBD |

Charges post to `owner_id` (team wallet when in team context). Viewer cannot create endpoints or invoke write-scoped routes.

---

## 6. Implementation plan (PR stack)

### PR 1 — Worker completion + routing (backend)
- [ ] Wire `InferenceEngine.route_request` to real worker HTTP (vLLM OpenAI port)
- [ ] Worker callback updates endpoint status `active` / `error`
- [ ] Complete `provision_worker` poll loop (container health via `health_endpoint`)
- [ ] Token counting on response (tiktoken estimate acceptable v1)

### PR 2 — OpenAI proxy (backend)
- [ ] `POST /v1/chat/completions` — map to endpoint; stream via SSE
- [ ] `POST /v1/completions` — legacy text completion
- [ ] API key auth via existing OAuth scope `inference:write`
- [ ] Integration test: mock worker HTTP, assert billed event

### PR 3 — Scale-down reaper (backend)
- [ ] Background tick: idle endpoints past `scaledown_window_sec` → terminate worker job
- [ ] `min_workers >= 1` keeps warm pool
- [ ] Metrics: `xcelsior_inference_cold_starts_total`, `active_endpoints`

### PR 4 — Dashboard polish (frontend)
- [ ] Endpoint wizard: model picker, GPU, min/max workers, cost estimate
- [ ] Show invoke URL + copy curl example
- [ ] Usage panel: requests, tokens, cost (from analytics API)
- [ ] Team banner + viewer gates (mirror instances)

### PR 5 — Docs + ops
- [ ] docs.xcelsior.ca inference guide
- [ ] Runbook: cold start SLA, wallet 402, worker OOM
- [ ] Fern/OpenAPI regen for v2 inference types

---

## 7. Acceptance criteria (§6.5)

- [ ] Customer creates endpoint from dashboard, receives HTTPS URL, test `curl` succeeds
- [ ] Team workspace: endpoint `owner_id` = team billing id; viewer blocked on create/delete/invoke
- [ ] Wallet debited for tokens + GPU seconds; 402 when empty
- [ ] Documented pricing on `/pricing` or docs

---

## 8. Dependencies (complete before starting PR 1)

| Prerequisite | Status |
|--------------|--------|
| Team tenancy sweep | ✅ Done |
| Volumes / NFS | ✅ Done |
| Mobile perf phase 3 | ✅ Code landed; post-deploy perf verify optional |
| Terminal surgical fixes | ✅ Done |
| Wallet + billing engine | ✅ Production |
| Scheduler + worker agent | ✅ Production |

**Gate:** All prerequisites met — **inference implementation may begin at PR 1.**

---

## 9. Risks

| Risk | Mitigation |
|------|------------|
| Cold start >30s on large models | Show honest UX; `min_workers=1` for prod; preload popular models |
| vLLM OOM on small GPUs | VRAM estimate in `create_endpoint`; block undersized GPU |
| Token metering accuracy | tiktoken estimate v1; reconcile with vLLM usage logs v1.1 |
| Cross-tenant routing bug | Reuse `_require_inference_endpoint_access`; IDOR sweep in PR 2 |

---

*Next step: execute PR 1 from this plan. Update checkboxes as PRs land.*