#!/usr/bin/env bash
# Apply infra/headscale/acl.hujson on the VPS (fixes Tailscale SSH "lookup local user *").
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ACL_SRC="$PROJECT_DIR/infra/headscale/acl.hujson"
REMOTE_USER="${XCELSIOR_DEPLOY_USER:-linuxuser}"
REMOTE_HOST="${XCELSIOR_DEPLOY_HOST:-100.64.0.1}"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"

log() { echo "[headscale-acl] $*"; }

[[ -f "$ACL_SRC" ]] || { echo "Missing $ACL_SRC" >&2; exit 1; }

ssh_cmd() {
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$REMOTE_USER@$REMOTE_HOST" "$@"
}

log "Copying ACL to $REMOTE_USER@$REMOTE_HOST..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$ACL_SRC" "$REMOTE_USER@$REMOTE_HOST:/tmp/xcelsior-acl.hujson"

log "Installing ACL and reloading headscale..."
ssh_cmd "sudo cp /tmp/xcelsior-acl.hujson /etc/headscale/acl.json && sudo headscale policy set -f /etc/headscale/acl.json && sudo systemctl restart headscale"

log "Verifying SSH (Tailscale)..."
ssh -i "$SSH_KEY" -o ConnectTimeout=15 -o BatchMode=yes "$REMOTE_USER@$REMOTE_HOST" "echo ok && curl -sf -m 8 http://127.0.0.1:9501/healthz"

log "Done. Run: XCELSIOR_DEPLOY_HOST=$REMOTE_HOST bash scripts/deploy.sh"