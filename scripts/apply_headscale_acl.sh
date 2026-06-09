#!/usr/bin/env bash
# Apply infra/headscale/acl.json on the VPS and restart Headscale.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ACL_SRC="${PROJECT_DIR}/infra/headscale/acl.json"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
REMOTE_USER="${XCELSIOR_DEPLOY_USER:-linuxuser}"
REMOTE_HOST="${XCELSIOR_DEPLOY_HOST:-149.28.121.61}"

[[ -f "$ACL_SRC" ]] || { echo "Missing $ACL_SRC"; exit 1; }

scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$ACL_SRC" "${REMOTE_USER}@${REMOTE_HOST}:/tmp/headscale_acl.json"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "${REMOTE_USER}@${REMOTE_HOST}" \
  'sudo cp /tmp/headscale_acl.json /etc/headscale/acl.json && sudo systemctl restart headscale && sleep 2 && sudo systemctl is-active headscale'
echo "✓ Headscale ACL applied"