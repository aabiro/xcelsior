#!/usr/bin/env bash
# Run docker compose with staging env + overlay.
# Example: ./scripts/run_staging_compose.sh up -d api
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env.staging ]]; then
  echo "Missing .env.staging — create it from .env + .env.staging.secrets first." >&2
  exit 1
fi

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-xcelsior-staging}"

# Loopback only. These services run with `network_mode: host`, so the
# production default of 0.0.0.0 would put a staging API — including the worker
# protocol — on the LAN and the tailnet from a developer machine. Overridable
# for the rare case of testing from another host, but never by default.
export XCELSIOR_API_BIND="${XCELSIOR_API_BIND:-127.0.0.1}"

exec docker compose \
  --env-file .env.staging \
  -f docker-compose.yml \
  -f docker-compose.staging.yml \
  "$@"
