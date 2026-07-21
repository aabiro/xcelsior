#!/usr/bin/env bash
# Bootstrap CI Postgres via the deterministic from-empty path.
#
# Always runs scripts/bootstrap_pg_from_empty.sh (alembic upgrade head +
# ensure/seed). A schema dump cache is no longer required for success;
# optional dump write is retained only as a local debugging artifact when
# CI_WRITE_SCHEMA_CACHE=1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCHEMA_FILE="${1:-ci-cache/pg_schema.sql}"
PGHOST="${PGHOST:-localhost}"
PGUSER="${PGUSER:-xcelsior}"
PGDATABASE="${PGDATABASE:-xcelsior_test}"
export PGPASSWORD="${PGPASSWORD:-test}"

if ! command -v psql >/dev/null 2>&1 || ! command -v pg_dump >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq postgresql-client
  fi
fi

echo "CI Postgres bootstrap: deterministic from-empty path (no schema-dump restore)"
bash "$ROOT/scripts/bootstrap_pg_from_empty.sh"

if [[ "${CI_WRITE_SCHEMA_CACHE:-0}" == "1" ]]; then
  mkdir -p "$(dirname "${SCHEMA_FILE}")"
  pg_dump -h "${PGHOST}" -U "${PGUSER}" -d "${PGDATABASE}" \
    --schema-only --no-owner --no-privileges -f "${SCHEMA_FILE}"
  echo "Wrote optional schema cache to ${SCHEMA_FILE}"
fi
