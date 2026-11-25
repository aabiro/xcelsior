#!/usr/bin/env bash
# Rehearse the MCP server live: a real API, PostgreSQL and Redis built from
# this checkout, with the MCP server started from source, and every published
# tool called through the MCP SDK the way a connector calls it.
#
#   api/mcp/scripts/rehearse-live.sh                 # every tool
#   api/mcp/scripts/rehearse-live.sh real-stack      # the thirteen-step journey
#
# Nothing here touches production or the docker test stack:
# - the API runs from source on 127.0.0.1:${XCELSIOR_REHEARSAL_API_PORT:-18980}
#   with .env.test only (XCELSIOR_SKIP_DOTENV stops db.py reading .env), so it
#   uses the pytest database, xcelsior_pytest;
# - Redis (for the MCP server) is a throwaway container on 127.0.0.1:${XCELSIOR_REHEARSAL_REDIS_PORT:-6388};
# - money is test money: the wallet is funded through the test-only deposit
#   route, and no payment processor is called.
#
# Exit status is the test run's. Logs: $LOG_DIR (default a temp dir, printed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

WHICH="${1:-every-tool}"
case "$WHICH" in
  every-tool) SPEC="tests/e2e/every-tool.rehearsal.test.ts"; GATE=XCELSIOR_MCP_REHEARSAL ;;
  real-stack) SPEC="tests/e2e/real-stack.test.ts"; GATE=XCELSIOR_MCP_REAL_STACK_E2E ;;
  *) echo "usage: $0 [every-tool|real-stack]" >&2; exit 2 ;;
esac

API_PORT="${XCELSIOR_REHEARSAL_API_PORT:-18980}"
REDIS_PORT="${XCELSIOR_REHEARSAL_REDIS_PORT:-6388}"
REDIS_NAME="xcelsior-rehearsal-redis"
LOG_DIR="${LOG_DIR:-$(mktemp -d -t xcelsior-rehearsal.XXXXXX)}"
API_PID=""

if [ ! -f .env.test ]; then
  echo ".env.test is missing; the rehearsal runs on it and nothing else." >&2
  exit 2
fi
for port in "$API_PORT" "$REDIS_PORT" 18981 18982 18983; do
  if ss -ltn "sport = :$port" | grep -q LISTEN; then
    echo "port $port is already in use; stop what holds it or pick another port." >&2
    exit 2
  fi
done

cleanup() {
  status=$?
  if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
  docker rm -f "$REDIS_NAME" >/dev/null 2>&1 || true
  echo "logs: $LOG_DIR"
  exit "$status"
}
trap cleanup EXIT INT TERM

docker rm -f "$REDIS_NAME" >/dev/null 2>&1 || true
docker run -d --name "$REDIS_NAME" -p "127.0.0.1:${REDIS_PORT}:6379" redis:7-alpine \
  redis-server --save "" --appendonly no >/dev/null
for _ in $(seq 1 50); do
  docker exec "$REDIS_NAME" redis-cli ping 2>/dev/null | grep -q PONG && break
  sleep 0.1
done

set -a
# shellcheck disable=SC1091
source .env.test
set +a
export XCELSIOR_SKIP_DOTENV=1
export XCELSIOR_ENV=test
export XCELSIOR_BG_TASKS=false
export XCELSIOR_ALLOW_UNAUTH_AGENT=0
# As in CI: the API's auth cache in memory. The Redis container is the MCP
# server's alone, and the real-stack journey stops it on purpose to prove
# writes fail closed, which would also sign the test tenants out.
export XCELSIOR_AUTH_CACHE_BACKEND=memory
# Every call here comes from one address, 127.0.0.1, so the API's per-address
# limit (meant for one user each) would throttle the whole run. Real MCP users
# reach the API under their own address: the MCP server forwards it.
export XCELSIOR_RATE_LIMIT_REQUESTS=100000
# Prod and the test stack run the scheduler active. Unset, it is "paused", and
# every launch is refused before it reaches a host.
export XCELSIOR_SCHEDULER_MODE=active
# Instance snapshots need a registry name to record, and a healthy probe of
# it (the fixture's registry-up writes one, as bg_worker would). Nothing is
# pushed: the worker here is the rehearsal's own stand-in.
export XCELSIOR_REGISTRY_URL="${XCELSIOR_REGISTRY_URL:-registry.rehearsal.test/xcelsior}"

# The same package path run-dev.sh and the image use: modules live in these
# directories, not in an installed package.
export PYTHONPATH="$ROOT/archive:$ROOT/billing:$ROOT/workers:$ROOT/auth:$ROOT/hosts:$ROOT/privacy:$ROOT/api:$ROOT/core:$ROOT"
uv run --frozen uvicorn api:app --host 127.0.0.1 --port "$API_PORT" >"$LOG_DIR/api.log" 2>&1 &
API_PID=$!
for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:${API_PORT}/healthz" >/dev/null 2>&1 && break
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "the API exited during startup:" >&2
    tail -40 "$LOG_DIR/api.log" >&2
    exit 1
  fi
  sleep 1
done

cd api/mcp
export "$GATE=1"
export XCELSIOR_MCP_E2E_API_URL="http://127.0.0.1:${API_PORT}"
export XCELSIOR_MCP_E2E_REDIS_URL="redis://127.0.0.1:${REDIS_PORT}/1"
export XCELSIOR_MCP_E2E_MANAGE_REDIS=0
export XCELSIOR_MCP_E2E_REDIS_CONTAINER="$REDIS_NAME"
export XCELSIOR_MCP_E2E_PYTHON="$ROOT/.venv/bin/python"
npx vitest run "$SPEC" 2>&1 | tee "$LOG_DIR/vitest.log"
exit "${PIPESTATUS[0]}"
