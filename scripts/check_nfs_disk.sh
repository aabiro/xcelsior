#!/usr/bin/env bash
# Check VPS NFS export disk usage; warn at 80%/90%.
#
# Usage:
#   bash scripts/check_nfs_disk.sh
#   NFS_SSH_HOST=linuxuser@100.64.0.1 bash scripts/check_nfs_disk.sh
#   WARN_PCT=80 CRIT_PCT=90 bash scripts/check_nfs_disk.sh
#
# Exit: 0 OK, 1 warn (>=WARN_PCT), 2 crit (>=CRIT_PCT)

set -euo pipefail

NFS_HOST="${NFS_SSH_HOST:-linuxuser@100.64.0.1}"
EXPORT_PATH="${NFS_EXPORT_PATH:-/exports/volumes}"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
WARN_PCT="${WARN_PCT:-80}"
CRIT_PCT="${CRIT_PCT:-90}"

log() { echo "[nfs-disk] $*"; }

remote="df -P ${EXPORT_PATH} | tail -1 | awk '{print \$5}' | tr -d '%'"
pct="$(ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$NFS_HOST" "$remote" 2>/dev/null || echo "")"

if [[ -z "$pct" || ! "$pct" =~ ^[0-9]+$ ]]; then
  log "ERROR: could not read disk usage from ${NFS_HOST}:${EXPORT_PATH}"
  exit 2
fi

log "${EXPORT_PATH} on ${NFS_HOST}: ${pct}% used (warn ${WARN_PCT}%, crit ${CRIT_PCT}%)"

if (( pct >= CRIT_PCT )); then
  exit 2
elif (( pct >= WARN_PCT )); then
  exit 1
fi
exit 0