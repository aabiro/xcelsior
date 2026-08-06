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
exec docker compose \
  --env-file .env.staging \
  -f docker-compose.yml \
  -f docker-compose.staging.yml \
  "$@"
