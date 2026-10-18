#!/usr/bin/env bash
# Start worker_agent against local test API (see .env.worker.test).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ ! -f .env.worker.test ]]; then
  echo "Missing .env.worker.test" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source .env.worker.test
set +a

if ! curl -sf "${XCELSIOR_SCHEDULER_URL:-http://localhost:9501}/healthz" >/dev/null; then
  echo "Test API not healthy at ${XCELSIOR_SCHEDULER_URL:-http://localhost:9501} — run: bash scripts/deploy.sh --test" >&2
  exit 1
fi

exec python3 worker_agent.py