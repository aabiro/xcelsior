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
SSH_CONTROL_PATH="${XCELSIOR_SSH_CONTROL_PATH:-$HOME/.ssh/cm-worker-%r@%h:%p}"
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

SSH_OPTS=(
  -i "$SSH_KEY"
  -o StrictHostKeyChecking=accept-new
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o ControlPath="$SSH_CONTROL_PATH"
  -o Compression=no
)

DEPLOY_COMPRESS_MODE="${XCELSIOR_DEPLOY_COMPRESS:-zstd}"
case "$DEPLOY_COMPRESS_MODE" in
  auto) DEPLOY_COMPRESS_MODE=zstd ;;
  zstd|gzip|none) ;;
  *) echo "ERROR: Unknown XCELSIOR_DEPLOY_COMPRESS=$DEPLOY_COMPRESS_MODE (use zstd, gzip, or none)" >&2; exit 1 ;;
esac

_ensure_worker_deploy_tools() {
  local -a missing=()
  case "$DEPLOY_COMPRESS_MODE" in
    none) ;;
    gzip) command -v pigz >/dev/null 2>&1 || missing+=(pigz) ;;
    zstd)
      command -v zstd >/dev/null 2>&1 || missing+=(zstd)
      command -v pigz >/dev/null 2>&1 || missing+=(pigz)
      ;;
  esac
  [[ ${#missing[@]} -eq 0 ]] && return 0
  if command -v apt-get >/dev/null 2>&1; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    brew install "${missing[@]}"
    return 0
  fi
  echo "ERROR: Missing deploy packages: ${missing[*]}" >&2
  exit 1
}

RSYNC_COMPRESS_OPTS=()
case "$DEPLOY_COMPRESS_MODE" in
  none) ;;
  gzip) RSYNC_COMPRESS_OPTS=(--compress) ;;
  zstd) RSYNC_COMPRESS_OPTS=(--compress --compress-choice=zstd) ;;
esac

_ensure_worker_deploy_tools

_rsync_shell() {
  printf 'ssh'
  for opt in "${SSH_OPTS[@]}"; do
    printf ' %q' "$opt"
  done
}

cleanup() {
  ssh -O exit -o ControlPath="$SSH_CONTROL_PATH" -i "$SSH_KEY" "${WORKER_USER}@${WORKER_HOST}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

TARGET="${WORKER_USER}@${WORKER_HOST}"
ssh "${SSH_OPTS[@]}" "$TARGET" "true" >/dev/null

python3 "$SCRIPT_DIR/sign-static-agent.py" >/dev/null

rsync -az --partial "${RSYNC_COMPRESS_OPTS[@]}" -e "$(_rsync_shell)" \
  "$SRC" "$TARGET:${REMOTE_DIR}/worker_agent.py"

ssh "${SSH_OPTS[@]}" "$TARGET" "sudo systemctl restart xcelsior-worker || true"
echo "Deployed worker_agent.py to ${WORKER_USER}@${WORKER_HOST}:${REMOTE_DIR}"