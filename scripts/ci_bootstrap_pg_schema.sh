#!/usr/bin/env bash
# Bootstrap CI Postgres: restore cached schema dump or migrate + cache.
set -euo pipefail

SCHEMA_FILE="${1:-ci-cache/pg_schema.sql}"
PGHOST="${PGHOST:-localhost}"
PGUSER="${PGUSER:-xcelsior}"
PGDATABASE="${PGDATABASE:-xcelsior_test}"
export PGPASSWORD="${PGPASSWORD:-test}"

if ! command -v psql >/dev/null 2>&1 || ! command -v pg_dump >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq postgresql-client
fi

if [[ -s "${SCHEMA_FILE}" ]]; then
  echo "Restoring cached Postgres schema from ${SCHEMA_FILE}"
  psql -h "${PGHOST}" -U "${PGUSER}" -d "${PGDATABASE}" -v ON_ERROR_STOP=1 -f "${SCHEMA_FILE}"
else
  echo "Schema cache miss — running alembic upgrade head"
  if ! alembic upgrade head; then
    echo "alembic upgrade failed — removing stale schema cache if present" >&2
    rm -f "${SCHEMA_FILE}"
    exit 1
  fi
  mkdir -p "$(dirname "${SCHEMA_FILE}")"
  pg_dump -h "${PGHOST}" -U "${PGUSER}" -d "${PGDATABASE}" \
    --schema-only --no-owner --no-privileges -f "${SCHEMA_FILE}"
  echo "Wrote schema cache to ${SCHEMA_FILE}"
fi