#!/usr/bin/env bash
# Deploy LUKS + NFS-Ganesha appliance on the Mac InferenceData host.
#
# Usage (from workstation):
#   bash scripts/deploy_mac_nfs_appliance.sh
#
# Or on Mac directly:
#   cd infra/mac-nfs-appliance && docker compose up -d --build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APPLIANCE_DIR="${PROJECT_DIR}/infra/mac-nfs-appliance"
MAC_HOST="${MAC_NFS_HOST:-aaryn@100.64.0.3}"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
REMOTE_DIR="${MAC_NFS_REMOTE_DIR:-/Users/aaryn/xcelsior-mac-nfs}"

log() { echo "[deploy-mac-nfs] $*"; }

ssh_mac() {
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$MAC_HOST" "$@"
}

log "Syncing appliance to ${MAC_HOST}:${REMOTE_DIR}…"
ssh_mac "mkdir -p '${REMOTE_DIR}'"
rsync -az -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new" \
  "${APPLIANCE_DIR}/" "${MAC_HOST}:${REMOTE_DIR}/"

log "Building and starting xcelsior-mac-nfs…"
ssh_mac "export PATH=\"/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:\$PATH\"
open -a Docker 2>/dev/null || true
for i in \$(seq 1 24); do docker info >/dev/null 2>&1 && break; sleep 5; done
docker info >/dev/null 2>&1 || { echo 'Docker not running'; exit 1; }
cd '${REMOTE_DIR}' && docker compose up -d --build"

log "Installing host wrapper ~/.local/bin/xcelsior-nfs-exec…"
ssh_mac "mkdir -p ~/.local/bin
cat > ~/.local/bin/xcelsior-nfs-exec <<'WRAP'
#!/usr/bin/env bash
# Run volume provision commands inside the LUKS+NFS appliance container.
# docker exec -i forwards stdin for LUKS --key-file /dev/stdin.
set -euo pipefail
export PATH=\"/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:\$PATH\"
if [ -t 0 ]; then
  exec docker exec xcelsior-mac-nfs sh -c \"\$*\"
else
  exec docker exec -i xcelsior-mac-nfs sh -c \"\$*\"
fi
WRAP
chmod +x ~/.local/bin/xcelsior-nfs-exec
grep -q '.local/bin' ~/.zshrc 2>/dev/null || echo 'export PATH=\"\${HOME}/.local/bin:\${PATH}\"' >> ~/.zshrc"

log "Smoke test inside container…"
ssh_mac "export PATH=\"/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:\$PATH\"
docker exec xcelsior-mac-nfs sh -c 'mountpoint -q /exports && ls -la /exports/volumes && echo APPLIANCE_OK'"

log "Done. NFS export: 100.64.0.3:/exports/volumes (port 12049, nfsvers=4.0)"
log "Optional: disable macOS nfsd to free :2049 — System Settings → Sharing → File Sharing (NFS) off"
log "Volume SSH wrap: xcelsior-nfs-exec 'mkdir -p /exports/volumes/vol-test'"