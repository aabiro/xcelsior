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