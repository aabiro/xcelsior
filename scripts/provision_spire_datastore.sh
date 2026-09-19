#!/usr/bin/env bash
# Provision SPIRE's datastore: its own role, its own database, same instance.
#
# Deliberately NOT part of control_plane/db_roles.py. That file models
# least-privilege roles over the application's tables-by-domain layout — "one
# authority per fact" — and SPIRE is not a domain of the application's data. It
# is the root of trust for the mesh, it manages its own schema through its own
# migrations, and it must keep working while the app is broken.
#
# So: shared Postgres instance for operational simplicity, separate DATABASE for
# isolation. Database-level separation means a bug or a compromise on either
# side cannot reach the other — which is exactly the property you want between
# the application and the thing that issues the application's mTLS identities.
#
#   sudo bash scripts/provision_spire_datastore.sh
#
# Idempotent. Prints the DSN line to add to /opt/xcelsior/.env; it does not
# write secrets anywhere itself.
set -euo pipefail

DB=spire
ROLE=spire

command -v psql >/dev/null || { echo "psql not found" >&2; exit 1; }
run() { sudo -u postgres psql -v ON_ERROR_STOP=1 -tAc "$1"; }

if [ "$(run "select 1 from pg_roles where rolname='${ROLE}'")" = "1" ]; then
  echo "role ${ROLE} already exists — leaving its password alone"
  PASSWORD=""
else
  # Generated here and shown once. Not stored by this script: putting it in a
  # file is the operator's decision, and a script that writes secrets to disk
  # tends to leave copies behind.
  PASSWORD="$(openssl rand -base64 32 | tr -d '/+=' | head -c 40)"
  run "create role ${ROLE} with login password '${PASSWORD}'" >/dev/null
  echo "role ${ROLE} created"
fi

if [ "$(run "select 1 from pg_database where datname='${DB}'")" = "1" ]; then
  echo "database ${DB} already exists"
else
  run "create database ${DB} owner ${ROLE}" >/dev/null
  echo "database ${DB} created, owned by ${ROLE}"
fi

# SPIRE creates and migrates its own tables; it needs no rights in any other
# database, and is granted none.
run "revoke all on database ${DB} from public" >/dev/null
echo "public access revoked on ${DB}"

echo
if [ -n "$PASSWORD" ]; then
  echo "Add this to /opt/xcelsior/.env (shown once):"
  echo "SPIRE_DATASTORE_DSN=postgresql://${ROLE}:${PASSWORD}@127.0.0.1:5432/${DB}"
else
  echo "Role already existed, so no DSN is printed. If you do not have the"
  echo "password, rotate it deliberately:"
  echo "  sudo -u postgres psql -c \"alter role ${ROLE} with password '<new>'\""
fi
