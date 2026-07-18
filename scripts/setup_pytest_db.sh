#!/usr/bin/env bash
# Create an isolated local pytest database, separate from any always-on
# app stack (e.g. the `xcelsior-test` docker compose project) that shares
# `xcelsior_test`.
#
# Why: the local pytest suite and a running app stack must not share a
# database. When they do, the stack's scheduler assigns leftover queued
# jobs onto pytest fixture hosts and its serverless autoscaler generates
# `serverless-*` scale-up jobs — non-deterministically failing placement
# tests with `no_eligible_host`. Point `.env.test` at the DB this script
# creates so pytest owns its data end to end.
#
# Usage:
#   scripts/setup_pytest_db.sh                 # clone schema from xcelsior_test
#   SOURCE_DB=xcelsior scripts/setup_pytest_db.sh
#   TARGET_DB=xcelsior_pytest scripts/setup_pytest_db.sh
#
# Then set in .env.test (gitignored):
#   XCELSIOR_POSTGRES_DB=xcelsior_pytest
#   XCELSIOR_POSTGRES_DSN=postgresql://USER:PASS@HOST:PORT/xcelsior_pytest
#   XCELSIOR_TEST_POSTGRES_DSN=postgresql://USER:PASS@HOST:PORT/xcelsior_pytest
set -euo pipefail

PGHOST="${PGHOST:-localhost}"
PGUSER="${PGUSER:-xcelsior}"
SOURCE_DB="${SOURCE_DB:-xcelsior_test}"
TARGET_DB="${TARGET_DB:-xcelsior_pytest}"
: "${PGPASSWORD:?set PGPASSWORD (the xcelsior role password) before running}"
export PGPASSWORD

echo "Creating ${TARGET_DB} (owner ${PGUSER}) if absent…"
if ! psql -h "$PGHOST" -U "$PGUSER" -d postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname='${TARGET_DB}'" 2>/dev/null | grep -q 1; then
  # CREATE DATABASE needs a role with CREATEDB; the app role usually lacks
  # it, so fall back to the postgres superuser via peer auth.
  if ! psql -h "$PGHOST" -U "$PGUSER" -d postgres -c \
        "CREATE DATABASE ${TARGET_DB} OWNER ${PGUSER}" 2>/dev/null; then
    sudo -u postgres createdb -O "$PGUSER" "$TARGET_DB"
  fi
  echo "  created ${TARGET_DB}"
else
  echo "  ${TARGET_DB} already exists"
fi

echo "Cloning schema (DDL only) from ${SOURCE_DB}…"
pg_dump -h "$PGHOST" -U "$PGUSER" -d "$SOURCE_DB" \
    --schema-only --no-owner --no-privileges \
  | psql -h "$PGHOST" -U "$PGUSER" -d "$TARGET_DB" -q -v ON_ERROR_STOP=0 >/dev/null

# The schema-only dump carries the alembic_version table but not its row;
# stamp the current head so schema-compat / readiness checks pass.
HEAD="$(cd "$(dirname "$0")/.." && \
  XCELSIOR_POSTGRES_DSN="postgresql://${PGUSER}:${PGPASSWORD}@${PGHOST}:5432/${SOURCE_DB}" \
  alembic heads 2>/dev/null | awk '{print $1}' | head -1)"
if [[ -n "$HEAD" ]]; then
  psql -h "$PGHOST" -U "$PGUSER" -d "$TARGET_DB" -q -c \
    "INSERT INTO alembic_version (version_num) VALUES ('${HEAD}')
       ON CONFLICT DO NOTHING;
     UPDATE alembic_version SET version_num='${HEAD}';" >/dev/null
  echo "Stamped alembic_version = ${HEAD}"
fi

echo "Done. ${TARGET_DB} is ready and isolated from ${SOURCE_DB}."
