#!/usr/bin/env bash
# Stand up staging, mint a real credential, and run the live gates.
#
# Why this exists. Gate §1.3 asks every phase for "at least one assertion that
# runs a real token against the real server", and the reason is stated in the
# plan: *"a mock is what passed while production did not."* That gate had never
# run — its only declared runner was `.github/workflows/live-gates.yml`, which
# cannot execute, and standing the environment up by hand is something nobody
# does twice. This script is the difference between a gate and an intention.
#
#   scripts/run_live_gates.sh
#
# It is idempotent: an already-running staging stack is reused rather than
# rebuilt, and every artefact it creates is a fresh row (a throwaway user, a
# throwaway OAuth client). **It deletes nothing.**
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Staging binds the host network, so it needs a port nothing else holds. 9500 is
# routinely taken by a developer's own API; picking a different one is cheaper
# and far safer than stopping whatever is already there.
PORT="${XCELSIOR_STAGING_PORT:-9600}"
BASE="http://127.0.0.1:${PORT}"
PASSWORD="LiveGateTest123abc!"

log() { printf '\033[36m▸\033[0m %s\n' "$*"; }

if ! curl -sf -m 5 "${BASE}/healthz" >/dev/null 2>&1; then
  log "starting staging on ${PORT}…"
  XCELSIOR_API_PORT="$PORT" ./scripts/run_staging_compose.sh up -d api >/dev/null

  # The auth cache is required for login and is not part of the api service.
  if ! docker ps --format '{{.Names}}' | grep -q '^xcelsior-staging-redis$'; then
    log "starting the staging auth cache…"
    docker run -d --name xcelsior-staging-redis --network host redis:7-alpine >/dev/null
    sleep 3
  fi
  # The URL carries a password; read it from the container rather than pinning a
  # secret in this file.
  PW="$(docker exec xcelsior-staging-api-1 sh -c 'printf "%s" "$XCELSIOR_AUTH_REDIS_URL"' \
        | sed -E 's|redis://:([^@]*)@.*|\1|')"
  [ -n "$PW" ] && docker exec xcelsior-staging-redis redis-cli CONFIG SET requirepass "$PW" >/dev/null

  for _ in $(seq 1 60); do
    curl -sf -m 5 "${BASE}/healthz" >/dev/null 2>&1 && break
    sleep 2
  done
fi

if ! curl -sf -m 5 "${BASE}/healthz" >/dev/null 2>&1; then
  echo "✗ staging never became healthy on ${PORT}; not running the gates." >&2
  echo "  A gate that cannot run must not report green." >&2
  exit 1
fi
log "staging healthy at ${BASE}"

# A NON-admin user. `_refuse_undelegatable_scopes` returns early for an admin by
# design, so an admin token would make every probe "succeed" and the gate would
# report the deployment vulnerable while creating a real operator client.
EMAIL="livegate-$(date +%s)-$RANDOM@xcelsior.ca"
curl -s -m 30 -X POST "${BASE}/api/auth/register" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"${EMAIL}\",\"password\":\"${PASSWORD}\"}" >/dev/null

# Registration requires address verification, which no mailbox will complete
# here. Verifying the row this script just created is not a shortcut around the
# gate — the gate is about scopes, not about email — and it touches nothing that
# existed beforehand.
docker exec xcelsior-staging-api-1 python -c "
from db import UserStore
UserStore.update_user('${EMAIL}', {'email_verified': 1})
" >/dev/null 2>&1

TOKEN="$(curl -s -m 30 -X POST "${BASE}/api/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"${EMAIL}\",\"password\":\"${PASSWORD}\"}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin).get("access_token",""))')"

if [ -z "$TOKEN" ]; then
  echo "✗ could not obtain a session token; not running the gates." >&2
  exit 1
fi
log "minted a non-admin session token for ${EMAIL}"

export XCELSIOR_LIVE_BASE_URL="$BASE"
export XCELSIOR_STAGING_URL="$BASE"
export XCELSIOR_LIVE_USER_TOKEN="$TOKEN"
export XCELSIOR_NONADMIN_TOKEN="$TOKEN"

log "running the live gates"
exec .venv/bin/python -m pytest tests/live/ -q "$@"
