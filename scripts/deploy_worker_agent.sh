#!/usr/bin/env bash
# Deploy worker_agent.py to a GPU worker over Headscale mesh.
#
# Usage (from dev machine with mesh SSH to worker):
#   bash scripts/deploy_worker_agent.sh
#   bash scripts/deploy_worker_agent.sh --host 100.64.0.6 --user aaryn
#
# Usage (from pixelenhance-labs VPS — repo at /opt/xcelsior, NOT ~/storage/...):
#   cd /opt/xcelsior
#   bash scripts/deploy_worker_agent.sh --host 100.64.0.6 --user aaryn
#
# Note: ASUS worker runs as user aaryn (WorkingDirectory=/home/aaryn/storage/projects/xcelsior).
#       xcelsior@100.64.0.6 often has no SSH key — use aaryn@.
#       VPS cannot reach worker :22 today; deploy from a machine with mesh SSH access.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
WORKER_HOST="${WORKER_HOST:-100.64.0.6}"
WORKER_USER="${WORKER_SSH_USER:-${WORKER_USER:-aaryn}}"
REMOTE_DIR="${WORKER_REMOTE_DIR:-/home/aaryn/storage/projects/xcelsior}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) WORKER_HOST="$2"; shift 2 ;;
    --user) WORKER_USER="$2"; shift 2 ;;
    --key) SSH_KEY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

SRC="$PROJECT_DIR/worker_agent.py"
if [[ ! -f "$SRC" ]]; then
  echo "worker_agent.py not found at $SRC" >&2
  echo "On VPS use: cd /opt/xcelsior && bash scripts/deploy_worker_agent.sh" >&2
  exit 1
fi

TARGET="${WORKER_USER}@${WORKER_HOST}"
echo "Deploying $SRC -> ${TARGET}:${REMOTE_DIR}/worker_agent.py"

scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
  "$SRC" "${TARGET}:${REMOTE_DIR}/worker_agent.py"

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$TARGET" \
  "sudo systemctl restart xcelsior-worker && sleep 2 && systemctl is-active xcelsior-worker"

echo "worker deploy OK (${TARGET})"