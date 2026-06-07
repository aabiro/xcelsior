#!/usr/bin/env bash
# Check InferenceData disk usage for Mac NFS appliance; warn at 80%/90%.
#
# Usage:
#   bash scripts/check_mac_nfs_disk.sh
#   WARN_PCT=80 CRIT_PCT=90 bash scripts/check_mac_nfs_disk.sh
set -euo pipefail

MAC_HOST="${MAC_NFS_HOST:-aaryn@100.64.0.3}"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
WARN_PCT="${WARN_PCT:-80}"
CRIT_PCT="${CRIT_PCT:-90}"
LUKS_PATH="${INFERENCE_LUKS:-/Volumes/InferenceData/inference.luks}"

log() { echo "[mac-nfs-disk] $*"; }

ssh_mac() {
  ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$MAC_HOST" "$@"
}

read -r USED_PCT AVAIL_KB TOTAL_KB <<<"$(ssh_mac "df -k /Volumes/InferenceData 2>/dev/null | awk 'NR==2 {print \$5, \$4, \$2}' | tr -d '%'")"
USED_PCT="${USED_PCT:-0}"
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
TOTAL_GB=$((TOTAL_KB / 1024 / 1024))

LUKS_ACTUAL="$(ssh_mac "du -k '${LUKS_PATH}' 2>/dev/null | awk '{print \$1}'")"
LUKS_ACTUAL_GB=$((LUKS_ACTUAL / 1024 / 1024))

log "InferenceData: ${USED_PCT}% used (${AVAIL_GB}G free / ${TOTAL_GB}G total)"
log "inference.luks on-disk: ${LUKS_ACTUAL_GB}G"

if [[ "$USED_PCT" -ge "$CRIT_PCT" ]]; then
  log "CRITICAL: disk usage ${USED_PCT}% >= ${CRIT_PCT}%"
  exit 2
fi
if [[ "$USED_PCT" -ge "$WARN_PCT" ]]; then
  log "WARNING: disk usage ${USED_PCT}% >= ${WARN_PCT}%"
  exit 1
fi

log "OK: disk usage below ${WARN_PCT}%"
exit 0