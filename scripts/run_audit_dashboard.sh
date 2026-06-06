#!/usr/bin/env bash
# Dashboard audit: Playwright UI crawl (public) or API probe via SSH origin tunnel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_USER="${XCELSIOR_DEPLOY_USER:-linuxuser}"
REMOTE_HOST="${XCELSIOR_DEPLOY_HOST:-149.28.121.61}"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
TUNNEL_PORT="${AUDIT_TUNNEL_PORT:-19501}"

if [[ ! -f "$PROJECT_DIR/.env.audit" ]]; then
  echo "Missing .env.audit — run: bash scripts/provision_audit_user.sh"
  exit 1
fi

# shellcheck disable=SC1091
set -a
source "$PROJECT_DIR/.env.audit"
set +a

try_public() {
  local code
  code="$(curl -sf -m 20 -o /dev/null -w '%{http_code}' "${AUDIT_BASE:-https://xcelsior.ca}/healthz" 2>/dev/null || echo 000)"
  [[ "$code" == "200" ]]
}

if try_public; then
  echo "[audit] Public site OK — running Playwright dashboard crawl..."
  cd "$PROJECT_DIR/frontend" && node scripts/audit-dashboard.mjs
  exit $?
fi

TAILSCALE_ORIGIN_IP="${AUDIT_ORIGIN_IP:-100.64.0.1}"
TS_API="$(curl -sf -m 10 -o /dev/null -w '%{http_code}' "http://${TAILSCALE_ORIGIN_IP}:9501/healthz" 2>/dev/null || echo 000)"
if [[ "$TS_API" == "200" ]]; then
  echo "[audit] Public down — API probe via Tailscale origin :9501"
  AUDIT_BASE="http://${TAILSCALE_ORIGIN_IP}:9501" node "$PROJECT_DIR/scripts/audit_dashboard_api_check.mjs"
  exit $?
fi

echo "[audit] Public site unreachable — API probe via SSH tunnel :$TUNNEL_PORT → 9501"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -f -N \
  -L "$TUNNEL_PORT:127.0.0.1:9501" "$REMOTE_USER@$REMOTE_HOST"
trap 'pkill -f "ssh.*-L $TUNNEL_PORT:127.0.0.1:9501" 2>/dev/null || true' EXIT

sleep 1
AUDIT_BASE="http://127.0.0.1:$TUNNEL_PORT" node "$PROJECT_DIR/scripts/audit_dashboard_api_check.mjs"