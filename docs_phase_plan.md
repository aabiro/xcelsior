# Xcelsior Reliability Roadmap (Phased)

## Phase 1 — Reliability lock-in
- CI on every PR/push: `pytest`, `ruff`, `black --check`.
- Integration tests for job lifecycle + billing + marketplace mixed cut aggregation.

## Phase 2 — Persistence hardening
- Move mutable state from JSON files to SQLite (`XCELSIOR_DB_PATH`).
- Keep existing scheduler interfaces untouched so API/CLI behavior remains stable.

## Phase 3 — Auth + API safety
- Require token auth in non-dev environments (`XCELSIOR_ENV!=dev/test`).
- Add request validation constraints and consistent error envelopes.
- Add basic in-memory rate limiting (`XCELSIOR_RATE_LIMIT_*`).

## Phase 4 — Observability
- Structured API error responses + scheduler metric snapshot endpoint.
- New endpoints: `/healthz`, `/readyz`, `/metrics`.

## Phase 5 — Deployment baseline
- Container-first deployment with `Dockerfile` + `docker-compose.yml`.
- Shared volume-backed SQLite database for API + worker.

## Phase 6 — Operator UX
- Keep `.env.example` synchronized with runtime variables.
- Add runbooks for failover, requeue, and billing reconciliation.

## TODO — Postgres migration
- Keep SQLite as stable default now; add a Postgres-backed storage adapter in a follow-up phase without breaking public interfaces.
