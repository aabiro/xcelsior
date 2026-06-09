#!/usr/bin/env bash
# DEV/ADMIN ONLY — not for providers or customer-facing install flows.
# Deploy worker_agent.py to a GPU worker over Headscale mesh.
#
# Usage (from dev machine with mesh SSH to worker):
#   bash scripts/deploy_worker_agent.sh --host <tailscale-ip> --user <ssh-user>
#
# Required: --host (no default — avoids accidental prod deploys)

set -euo pipefail

echo "⚠  DEV/ADMIN ONLY — do not include in provider onboarding artifacts" >&2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
WORKER_HOST=""
WORKER_USER="${WORKER_SSH_USER:-${WORKER_USER:-}}"
REMOTE_DIR="${WORKER_REMOTE_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) WORKER_HOST="$2"; shift 2 ;;
    --user) WORKER_USER="$2"; shift 2 ;;
    --key) SSH_KEY="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$WORKER_HOST" ]]; then
  echo "ERROR: --host is required (no default IP — dev/admin tool only)" >&2
  exit 1
fi
WORKER_USER="${WORKER_USER:-$USER}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/storage/projects/xcelsior}"

SRC="$PROJECT_DIR/worker_agent.py"
if [[ ! -f "$SRC" ]]; then
  echo "worker_agent.py not found at $SRC" >&2
  exit 1
fi

python3 "$SCRIPT_DIR/sign-static-agent.py" >/dev/null

scp -i "$SSH_KEY" "$SRC" "${WORKER_USER}@${WORKER_HOST}:${REMOTE_DIR}/worker_agent.py"
ssh -i "$SSH_KEY" "${WORKER_USER}@${WORKER_HOST}" "sudo systemctl restart xcelsior-worker || true"
echo "Deployed worker_agent.py to ${WORKER_USER}@${WORKER_HOST}:${REMOTE_DIR}"