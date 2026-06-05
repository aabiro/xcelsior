#!/usr/bin/env bash
# Provision site-audit@xcelsior.ca on production and sync .env.audit locally.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_USER="${XCELSIOR_DEPLOY_USER:-linuxuser}"
REMOTE_HOST="${XCELSIOR_DEPLOY_HOST:-149.28.121.61}"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
REMOTE_DIR="/opt/xcelsior"

ssh_cmd() {
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$REMOTE_USER@$REMOTE_HOST" "$@"
}

log() { echo "[provision-audit] $*"; }

log "Provisioning audit user in api-blue container..."
OUT="$(ssh_cmd "cd $REMOTE_DIR && docker compose exec -T api-blue python - --provision --json --show-secret" \
  < "$PROJECT_DIR/scripts/ensure_audit_user.py")"
echo "$OUT" | python3 -c "
import json, sys, os
from pathlib import Path
data = json.load(sys.stdin)
prov = data.get('provision', {})
lines = [
    '# MCP dashboard audit — synced from production',
    f\"AUDIT_BASE={prov.get('AUDIT_BASE', 'https://xcelsior.ca')}\",
    f\"AUDIT_EMAIL={prov.get('AUDIT_EMAIL', 'site-audit@xcelsior.ca')}\",
    f\"AUDIT_PASSWORD={prov.get('password', '')}\",
]
if prov.get('AUDIT_USER_ID'):
    lines.append(f\"AUDIT_USER_ID={prov['AUDIT_USER_ID']}\")
if prov.get('AUDIT_CUSTOMER_ID'):
    lines.append(f\"AUDIT_CUSTOMER_ID={prov['AUDIT_CUSTOMER_ID']}\")
path = Path('$PROJECT_DIR') / '.env.audit'
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
os.chmod(path, 0o600)
print(f'Wrote {path}')
"

log "Verifying login on origin (bypasses Cloudflare)..."
EMAIL="$(grep '^AUDIT_EMAIL=' "$PROJECT_DIR/.env.audit" | cut -d= -f2-)"
PASS="$(grep '^AUDIT_PASSWORD=' "$PROJECT_DIR/.env.audit" | cut -d= -f2-)"
ORIGIN_CODE="$(ssh_cmd "curl -s -o /tmp/audit-login.json -w '%{http_code}' -X POST http://127.0.0.1:9501/api/auth/login -H 'Content-Type: application/json' -d '{\"email\":\"$EMAIL\",\"password\":\"$PASS\"}'")"
if [[ "$ORIGIN_CODE" != "200" ]]; then
  echo "Origin login failed: HTTP $ORIGIN_CODE"
  ssh_cmd "cat /tmp/audit-login.json" || true
  exit 1
fi
echo "Origin login OK (HTTP 200)"

log "Done. Run: node frontend/scripts/audit-dashboard.mjs"