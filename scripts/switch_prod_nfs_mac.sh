#!/usr/bin/env bash
# Point production VPS .env at Mac InferenceData NFS appliance (with backup).
#
# Usage:
#   bash scripts/switch_prod_nfs_mac.sh
#   bash scripts/switch_prod_nfs_mac.sh --dry-run
set -euo pipefail

REMOTE_USER="${XCELSIOR_DEPLOY_USER:-linuxuser}"
REMOTE_HOST="${XCELSIOR_DEPLOY_HOST:-149.28.121.61}"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
REMOTE_ENV="/opt/xcelsior/.env"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

ssh_vps() {
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

BACKUP_SUFFIX="nfs-vps-$(date +%Y%m%d-%H%M%S)"

apply_env() {
  local file="$1"
  grep -v '^XCELSIOR_NFS_' "$file" | grep -v '^#.*NFS' > "${file}.tmp" || true
  cat >>"${file}.tmp" <<'EOF'

# ── NFS Volume Storage (Mac InferenceData appliance) ─────────────────
XCELSIOR_NFS_SERVER=100.64.0.3
XCELSIOR_NFS_SSH_HOST=100.64.0.3
XCELSIOR_NFS_SSH_USER=aaryn
XCELSIOR_NFS_EXPORT_BASE=/exports/volumes
XCELSIOR_NFS_PATH=/exports/volumes
XCELSIOR_NFS_MOUNT=/mnt/xcelsior-volumes
XCELSIOR_NFS_REQUIRED=true
XCELSIOR_NFS_SSH_CMD_WRAP=/Users/aaryn/.local/bin/xcelsior-nfs-exec
XCELSIOR_NFS_MOUNT_OPTS=hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp,nfsvers=4.0,port=12049
EOF
  mv "${file}.tmp" "$file"
}

echo "[switch-nfs] Backing up ${REMOTE_ENV} → ${REMOTE_ENV}.bak.${BACKUP_SUFFIX}"
if [[ "$DRY_RUN" == true ]]; then
  echo "[dry-run] Would set Mac NFS vars on VPS"
  exit 0
fi

ssh_vps "cp '${REMOTE_ENV}' '${REMOTE_ENV}.bak.${BACKUP_SUFFIX}'"
ssh_vps "cat '${REMOTE_ENV}'" > /tmp/xcelsior-prod.env
apply_env /tmp/xcelsior-prod.env
scp -i "$SSH_KEY" /tmp/xcelsior-prod.env "${REMOTE_USER}@${REMOTE_HOST}:/tmp/xcelsior.env.new"
ssh_vps "sudo cp /tmp/xcelsior.env.new '${REMOTE_ENV}' && sudo chown \$(whoami):\$(whoami) '${REMOTE_ENV}'"

echo "[switch-nfs] Restarting API + scheduler to pick up NFS env…"
ssh_vps "cd /opt/xcelsior && docker compose --profile blue up -d --no-deps --force-recreate api-blue scheduler-worker"

echo "[switch-nfs] Waiting for /readyz (avoids transient nginx 404 during container recreate)…"
for i in $(seq 1 30); do
  if curl -sf "https://xcelsior.ca/readyz" | grep -q '"ok":true'; then
    echo "[switch-nfs] readyz OK"
    break
  fi
  [[ "$i" -eq 30 ]] && { echo "[switch-nfs] WARN: readyz not ready after 60s" >&2; exit 1; }
  sleep 2
done

echo "[switch-nfs] Done. Verify:"
echo "  curl -s https://xcelsior.ca/readyz | jq .nfs_volumes"
echo "  python3 scripts/ops_infra_smoke.py"