#!/usr/bin/env bash
# Deterministic from-empty PostgreSQL bootstrap (Track A A1.6).
#
# Single ordered path to Alembic head without restoring a schema dump:
#   1. alembic upgrade head  (schema authority; creates agent_commands etc.)
#   2. optional runtime ensure + seed (gpu_pricing rows and other IF NOT EXISTS)
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
cd "$ROOT"

echo "A1.6 from-empty bootstrap: alembic upgrade head"
if command -v uv >/dev/null 2>&1 && [[ -f "$ROOT/uv.lock" ]]; then
  uv run alembic upgrade head
elif [[ -x "$ROOT/.venv/bin/alembic" ]]; then
  "$ROOT/.venv/bin/alembic" upgrade head
else
  alembic upgrade head
fi

# Seeds and residual IF NOT EXISTS tables that are not migration-seeded
# (gpu_pricing rows are wiped by migration 030; schema-only upgrade has no data).
if [[ "${BOOTSTRAP_SKIP_ENSURE:-0}" != "1" ]]; then
  echo "A1.6 from-empty bootstrap: runtime ensure + reference seed"
  if command -v uv >/dev/null 2>&1 && [[ -f "$ROOT/uv.lock" ]]; then
    uv run python - <<'PY'
from db import _ensure_pg_tables, _get_pg_pool

with _get_pg_pool().connection() as conn:
    _ensure_pg_tables(conn)
    conn.commit()
print("ensure complete")
PY
  else
    python - <<'PY'
from db import _ensure_pg_tables, _get_pg_pool

with _get_pg_pool().connection() as conn:
    _ensure_pg_tables(conn)
    conn.commit()
print("ensure complete")
PY
  fi
fi

echo "A1.6 from-empty bootstrap: done"
