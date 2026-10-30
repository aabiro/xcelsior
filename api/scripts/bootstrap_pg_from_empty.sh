#!/usr/bin/env bash
# Deterministic from-empty PostgreSQL bootstrap (Alembic head).
#
# Single ordered path to Alembic head without restoring a schema dump:
#   1. alembic upgrade head  (schema authority; creates agent_commands etc.)
#   2. import the versioned Stripe compute catalog (gpu_pricing and
#      stripe_compute_prices), exactly as deploy does after Alembic
#
# Usage:
#   XCELSIOR_POSTGRES_DSN=postgresql://user:pass@host:5432/empty_db \
#     bash scripts/bootstrap_pg_from_empty.sh
#
# Env (same resolution as migrations/env.py / db.resolve_postgres_dsn):
#   XCELSIOR_POSTGRES_DSN | XCELSIOR_PG_DSN | DATABASE_URL
#   PGPASSWORD when using libpq tools for optional checks
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ "$(basename "$ROOT")" == "api" ]] && ROOT="$(dirname "$ROOT")"
export PYTHONPATH="$ROOT/archive:$ROOT/billing:$ROOT/workers:$ROOT/auth:$ROOT/hosts:$ROOT/privacy:$ROOT/api:$ROOT/core:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

echo "from-empty bootstrap: alembic upgrade head"
if command -v uv >/dev/null 2>&1 && [[ -f "$ROOT/uv.lock" ]]; then
  uv run alembic upgrade head
elif [[ -x "$ROOT/.venv/bin/alembic" ]]; then
  "$ROOT/.venv/bin/alembic" upgrade head
else
  alembic upgrade head
fi

# The Stripe catalog is the only writer of gpu_pricing; every customer launch
# quotes it. Offline: this reads billing/stripe_catalog.json, never Stripe.
if [[ "${BOOTSTRAP_SKIP_CATALOG:-0}" != "1" ]]; then
  echo "from-empty bootstrap: import Stripe compute catalog"
  DSN_ENV=XCELSIOR_POSTGRES_DSN
  [[ -z "${XCELSIOR_POSTGRES_DSN:-}" && -n "${XCELSIOR_PG_DSN:-}" ]] && DSN_ENV=XCELSIOR_PG_DSN
  [[ -z "${XCELSIOR_POSTGRES_DSN:-}" && -z "${XCELSIOR_PG_DSN:-}" ]] && DSN_ENV=DATABASE_URL
  CATALOG=(api/scripts/sync_stripe_service_tiers.py --output billing/stripe_catalog.json --import-dsn-env "$DSN_ENV")
  if command -v uv >/dev/null 2>&1 && [[ -f "$ROOT/uv.lock" ]]; then
    uv run python "${CATALOG[@]}"
  else
    python "${CATALOG[@]}"
  fi
fi

echo "from-empty bootstrap: done"
