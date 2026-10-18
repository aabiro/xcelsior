# Instance lifecycle (stop / start)

User-facing **pause** and **resume** HTTP routes were removed. Use **stop** and **start** instead (RunPod-style semantics).

## User API

| Action | Method | Route | Notes |
|--------|--------|-------|-------|
| Stop | `POST` | `/instances/{job_id}/stop` | Preserves container/volumes; storage billing may continue |
| Start | `POST` | `/instances/{job_id}/start` | Requires wallet balance; resumes compute billing |
| Cancel | `POST` | `/instances/{job_id}/cancel` | Terminates job |

Auth: session cookie or Bearer token; caller must own the job (or platform admin).

## Billing / low balance

When the wallet is empty or suspended, billing calls `stop_instance`, which enqueues the worker command `pause_container` (`docker stop` only — container is kept for `docker start`). This is an **internal** worker primitive, not a public API.

Legacy statuses such as `paused_low_balance` may still appear in DB rows; the dashboard maps them to a single **stopped** surface.

## Workers

Status updates and agent callbacks use `X-Xcelsior-API-Key` / `XCELSIOR_API_TOKEN` (`_require_worker_status_update`), not end-user session auth.

Removed symbols: `pause_instance`, `resume_instance`, `POST .../pause`, `POST .../resume`. See `billing.py` and migration `031_drop_pause_resume_state`.

## Spot (interruptible) instances

Spot jobs are submitted with `pricing_mode: "spot"` on `POST /instance`. They are **preemptible** and billed at the **spot rate locked at assignment** (`spot_rate_cad`), not the host on-demand rate.

| Stage | Status | Notes |
|-------|--------|-------|
| Submit | `queued` | `pricing_mode=spot`, `preemptible=true`, `tier=spot` |
| Assigned | `assigned` → `running` | Host must have `spot_enabled` and free spot GPU slots |
| Preempted | `preempted` → `queued` | Worker or scheduler evicts; volumes detached; auto-requeue |
| Terminal | `completed` / `failed` / `cancelled` | Same as on-demand |

**Preemption path:** `running` → `preempted` (SIGTERM grace) → `queued` with `preempted_at` and `preemption_count` incremented. Capacity-based eviction (on-demand contention) is Phase 7; bid-based eviction is retired.

**API fields on instance detail:** `pricing_mode`, `spot_rate_cad`, `preemptible`, `preempted_at`, `preemption_count`.

**Host payload (scheduler):** `spot_enabled` (default `true`), `spot_gpu_slots` (default = `gpu_count`). Spot jobs may only allocate when the spot pool has headroom.