# Serverless Inference — Operations Runbook

*Last updated: 2026-06-08*

Operational guide for on-call and platform admins. Design reference:
[`INFERENCE_DESIGN.md`](./INFERENCE_DESIGN.md). Implementation checklist:
[`SERVERLESS_IMPLEMENTATION_CHECKLIST.md`](./SERVERLESS_IMPLEMENTATION_CHECKLIST.md).

---

## API paths (migration note)

| Legacy | Current |
|--------|---------|
| `/api/v2/inference/endpoints/*` | `/api/v2/serverless/endpoints/*` (compat aliases remain) |
| `/v1/inference/*` | HuggingFace batch jobs (`routes/inference.py`) — unchanged |
| `/api/inference` | Short-lived scheduler inference jobs — unchanged |
| OpenAI chat | `/v1/serverless/{endpoint_id}/openai/v1/chat/completions` |

---

## Rollout & feature flag

Serverless is **off by default** in production.

| Variable | Purpose |
|----------|---------|
| `XCELSIOR_SERVERLESS_ENABLED` | Global kill switch (`true` / `false`) |
| `XCELSIOR_SERVERLESS_ALLOWLIST` | Comma-separated `customer_id` values for phased GA |

**Probe:** `GET /api/v2/serverless/enabled` (session optional) returns
`{ enabled, global_enabled, allowlist_active }`.

**Dashboard:** Inference nav and pages are hidden when `enabled` is false for the
active workspace.

**Post-deploy smoke:**

```bash
python scripts/serverless_e2e_smoke.py --email you@xcelsior.ca --password '...'
# or
python scripts/serverless_e2e_smoke.py --token 'eyJ...' --base-url https://xcelsior.ca
```

---

## Key metrics to watch

Prometheus gauges/counters (when `prometheus_client` is available):

| Metric | Meaning |
|--------|---------|
| `serverless_queue_depth{endpoint_id}` | Backlog pressure |
| `serverless_active_workers{endpoint_id}` | Fleet size |
| `serverless_idle_workers{endpoint_id}` | Ready but unused capacity |
| `serverless_cold_starts_total{endpoint_id}` | Boot latency regressions |
| `serverless_job_errors_total{endpoint_id}` | Terminal failures |
| `serverless_jobs_completed_total{endpoint_id}` | Throughput |

**API:** `GET /api/v2/serverless/endpoints/{id}/metrics?since_hours=24`

**Tune when:**

- Queue depth sustained high → raise `max_workers` or add GPU hosts.
- Cold starts spike → check image pull times, NFS cache hit rate, `BOOTING_TTL`.
- Error rate up → inspect failed jobs (`status=FAILED`), worker logs, reaper stats.

Background reconcile runs every **45 s** (`serverless_reconcile` in `api.py` /
`bg_worker.py`). Disable with `XCELSIOR_SERVERLESS_RECONCILE=false` only during
maintenance.

---

## Stuck job (IN_QUEUE / IN_PROGRESS)

**Symptoms:** Job never leaves `IN_QUEUE` or hangs `IN_PROGRESS`.

**Checks:**

1. Endpoint health: `GET /api/v2/serverless/endpoints/{id}/health`
2. Workers: `GET /api/v2/serverless/endpoints/{id}/workers`
3. Wallet balance — suspended wallets block new work at enqueue time.
4. Queue cap — `max_queue_size` returns 429 `queue_full`.
5. Reaper — stale queued jobs fail after `XCELSIOR_SERVERLESS_QUEUE_TTL_SEC` (default 3600).

**Actions:**

- Cancel client-side: `POST /v1/serverless/{id}/cancel/{job_id}`
- Scale up: patch `min_workers` / `max_workers` if autoscaler is starved.
- If worker crashed: reaper marks heartbeat-stale workers; reconcile provisions replacement.
- Request timeout: jobs past `request_timeout_sec` are failed by reaper.

---

## Stuck / booting worker

**Symptoms:** Worker stays `booting` or `draining` for a long time.

**Checks:**

1. Scheduler job status for linked `scheduler_job_id`
2. Host agent logs on the worker host (`worker_agent.py` serverless branch)
3. `XCELSIOR_SERVERLESS_BOOTING_TTL_SEC` (default 900) — booting workers reaped if exceeded

**Actions:**

- Drain manually via worker terminate API (admin) or endpoint delete (drains fleet).
- Image pull failures: verify registry auth, image allowlist, CUDA version match.
- NFS cache volume attach failures: see cache miss section below.

---

## Scale-down failure

**Symptoms:** Idle workers not terminating; cost not dropping after traffic stops.

**Checks:**

1. `min_workers` > 0 prevents scale-to-zero
2. `XCELSIOR_SERVERLESS_SCALE_DOWN_COOLDOWN_SEC` (default 60) delays scale-down after scale-up
3. `XCELSIOR_SERVERLESS_DRAIN_GRACE_SEC` (default 30) — in-flight jobs must finish
4. Workers with active concurrency block reap until drain completes

**Actions:**

- Patch endpoint: `min_workers: 0`, lower `idle_timeout_sec`
- Wait one reconcile cycle (45 s) + cooldown window
- Force terminate worker row if scheduler job already dead (admin; billing finalizes on release)

---

## NFS cache miss storms

**Symptoms:** Slow cold starts, repeated model downloads, high NFS I/O.

**Checks:**

1. Preset endpoints use `serverless/cache.py` → shared NFS weight-cache volumes
2. Volume exists: dashboard **Volumes** or `GET /api/v2/volumes`
3. Worker NVMe path: `XCELSIOR_NVME_CACHE_DIR`, `XCELSIOR_NVME_CACHE_MAX_GB`
4. Region mismatch — cache volumes are keyed by `(model_ref, revision, region)`

**Actions:**

- Pre-warm: create endpoint with `keep_warm: true` or `min_workers: 1` temporarily
- Verify NFS host reachable from GPU workers (`NFS_SSH_HOST`, export path)
- Cache volume cleanup uses `volumes.py` `readlink -f` + prefix guard — do not bypass

---

## Billing disputes

**Model:** GPU-seconds per worker uptime + cold-start metadata line items
(`serverless/metering.py`). Incremental billing every
`XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC` (default 60).

**Checks:**

1. `GET /api/v2/serverless/endpoints/{id}/usage`
2. Wallet ledger for `serverless_gpu` and `serverless_gpu_cold_start` resource types
3. Job `gpu_seconds`, `cold_start_seconds`, `cost_cad` on status API

**Actions:**

- Suspend abusive owner wallet (blocks new enqueue via wallet preflight)
- Refund via standard billing admin tools; annotate job_id in ledger note
- Dispute on stuck IN_PROGRESS: confirm reaper failed job and billing stopped on terminal state

---

## Webhook failures

**Checks:**

1. Job columns `webhook_status`, `webhook_attempts`, `webhook_last_error`
2. SSRF guard blocks localhost / private IPs (`validate_webhook_url`)
3. HMAC secret: `XCELSIOR_SERVERLESS_WEBHOOK_HMAC_SECRET` (falls back to scheduler token)
4. Body cap: `XCELSIOR_SERVERLESS_WEBHOOK_MAX_BODY_BYTES` (default 65536)

**Actions:**

- Fix receiver URL to public HTTPS endpoint
- Retry pending: background `retry_pending_webhooks` on reconcile path
- After `XCELSIOR_SERVERLESS_WEBHOOK_MAX_ATTEMPTS` (default 3), status → `failed` (manual replay only)

---

## Cold-start regressions

**Symptoms:** `cold_start_seconds` rising, user complaints about first-token latency.

**Checks:**

1. `serverless_cold_starts_total` counter
2. Scale-to-zero policy (`min_workers=0`, `idle_timeout_sec`)
3. Image size / pull time on worker hosts
4. GPU availability — queue waits for scheduler placement

**Actions:**

- Short-term: `min_workers: 1` for SLA-critical endpoints
- Medium-term: prepull images on hosts, verify NFS cache hit rate
- Tune `idle_timeout_sec` vs cost (longer idle keeps warm workers longer)

---

## Emergency disable

1. Set `XCELSIOR_SERVERLESS_ENABLED=false` and reload API / bg_worker
2. Existing endpoints remain in DB but all routes return **404** (hidden)
3. Running scheduler jobs: drain via endpoint delete or admin worker terminate
4. Re-enable per owner with `XCELSIOR_SERVERLESS_ALLOWLIST` before full GA

---

## Related scripts & tests

| Asset | Purpose |
|-------|---------|
| `scripts/serverless_e2e_smoke.py` | Post-deploy API smoke |
| `./run-tests.sh serverless` | Full backend regression (132+ tests) |
| `frontend/e2e/serverless-smoke.spec.ts` | Dashboard UI smoke (mocked API) |
| `tests/test_serverless_feature.py` | Feature-flag unit tests |