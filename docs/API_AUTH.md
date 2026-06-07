# API authentication model

## End users

Dashboard and renter flows use session cookies (`credentials: "include"`) or Bearer tokens from `/api/auth/login`. Scoped routes call `_require_auth` and often `_require_scope` (e.g. `instances:write`, `inference:read`).

Customer-scoped billing paths use `_require_customer_access` so `customer_id` in the URL must match the caller (unless platform admin).

## Workers and platform

Host registration, job status PATCH, inference result callbacks, billing `bill-job`, and agent command drain require `X-Xcelsior-API-Key` / `XCELSIOR_API_TOKEN` (`_require_worker_status_update` or `_require_platform_worker`).

## Ownership checks

Jobs and inference results: `_check_job_access` / `_require_inference_job_access` (owner, `customer_id`, or admin).

Events: `_require_entity_event_access` by entity type.

## Public read endpoints

Some marketplace/SLA/compute-score routes are intentionally unauthenticated reads. Do not put PII or per-user billing data on those paths.

## Hardened routes

- `GET /v1/inference/{job_id}` — requires auth + inference job ownership (was open poll).
- `GET /api/jurisdiction/residency-trace/{job_id}` — requires auth + job ownership.

## Billing + host inventory guards

| Route | Guard |
|-------|-------|
| `POST /api/pricing/reserve` | `_require_customer_access` — caller must own `customer_id` |
| `GET /hosts/ca` | `_require_auth` + `hosts:read` |
| `POST /api/jurisdiction/hosts` | `_require_auth` + `hosts:read` |

CI tracks these in `GUARDED_ROUTE_HANDLERS` (`python scripts/audit_route_auth.py --guarded --strict`).

### Intentional public reads (no session)

- `GET /api/v2/gpu/available`, `GET /api/pricing/reserved-plans`, `GET /api/pricing/estimate`
- `GET /api/sla/targets` (tier definitions only)
- OAuth/login/register under `/api/auth/`