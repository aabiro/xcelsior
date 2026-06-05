#!/usr/bin/env bash
# Re-run all production verification after origin/SSH recovery.
# Last blocked: 2026-06-05 — Cloudflare 522, SSH timeout to 149.28.121.61.
#
# Usage:
#   bash scripts/redo_when_prod_up.sh          # full suite
#   bash scripts/redo_when_prod_up.sh --quick  # post-deploy + hydration only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_USER="${XCELSIOR_DEPLOY_USER:-linuxuser}"
REMOTE_HOST="${XCELSIOR_DEPLOY_HOST:-149.28.121.61}"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
BASE="${AUDIT_BASE:-https://xcelsior.ca}"
QUICK=false
[[ "${1:-}" == "--quick" ]] && QUICK=true

log() { echo "[redo-prod] $*"; }
fail() { echo "[redo-prod] FAIL: $*" >&2; exit 1; }

log "Checking public health ($BASE)..."
PUB_CODE="$(curl -sf -m 25 -o /dev/null -w '%{http_code}' "$BASE/healthz" 2>/dev/null || echo 000)"
if [[ "$PUB_CODE" != "200" ]]; then
  fail "Public healthz returned $PUB_CODE (need 200). Retry when origin is back."
fi
log "Public OK ($PUB_CODE)"

if ssh -i "$SSH_KEY" -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE_USER@$REMOTE_HOST" "echo ok" &>/dev/null; then
  ORIGIN_CODE="$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
    "curl -sf -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:9501/healthz" 2>/dev/null || echo 000)"
  log "SSH OK — origin API healthz: $ORIGIN_CODE"
else
  log "SSH unreachable (optional) — continuing with public checks only"
fi

cd "$PROJECT_DIR"

log "1/5 post-deploy (51 checks)..."
node scripts/post_deploy_audit_check.mjs > /tmp/post-deploy-check.json
node -e "const j=require('/tmp/post-deploy-check.json'); if(j.summary.failed) process.exit(1); console.log('  ', j.summary)"

log "2/5 CLI coverage (51 commands)..."
node scripts/audit_cli_coverage.mjs > /tmp/cli-coverage.json
node -e "const j=require('/tmp/cli-coverage.json'); if(j.summary.failed) process.exit(1); console.log('  ', j.summary)"

log "3/5 hydration repro..."
BASE_URL="$BASE" node frontend/scripts/hydration-repro.mjs

if [[ -f "$PROJECT_DIR/.env.audit" ]]; then
  log "4/5 dashboard audit..."
  bash scripts/run_audit_dashboard.sh
else
  log "4/5 dashboard audit — SKIP (no .env.audit; run: bash scripts/provision_audit_user.sh)"
fi

if [[ "$QUICK" == "true" ]]; then
  log "5/5 perf MCP — skipped (--quick)"
else
  if [[ -f /tmp/xcelsior-audit/audit-performance.mjs ]]; then
    log "5/5 F-003 perf MCP..."
    CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS=1 CI=true \
      node /tmp/xcelsior-audit/audit-performance.mjs
  else
    log "5/5 perf MCP — SKIP (/tmp/xcelsior-audit/audit-performance.mjs missing)"
  fi
fi

log "Updating re-audit report..."
REVERIFY_JSON=/tmp/post-deploy-check.json node scripts/generate_reaudit_report.mjs 2>/dev/null || \
  log "generate_reaudit_report.mjs skipped (manual doc may be newer)"

log "Done. Artifacts:"
echo "  /tmp/post-deploy-check.json"
echo "  /tmp/cli-coverage.json"
echo "  /tmp/xcelsior-audit/raw/dashboard-all.json (if dashboard ran)"
echo "  /tmp/xcelsior-audit/raw/perf-all.json (if perf ran)"