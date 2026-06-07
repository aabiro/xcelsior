#!/usr/bin/env bash
# Provision NFS volume storage on the production VPS (colocated with API).
#
# Usage (on VPS as root or via sudo):
#   sudo bash scripts/setup_nfs_vps.sh
#
# Or from workstation:
#   ssh -i ~/.ssh/xcelsior linuxuser@149.28.121.61 'sudo bash -s' < scripts/setup_nfs_vps.sh
#
# After running, set in .env:
#   XCELSIOR_NFS_SERVER=100.64.0.1          # mesh IP workers mount
#   XCELSIOR_NFS_SSH_HOST=127.0.0.1         # API SSH for provision
#   XCELSIOR_NFS_EXPORT_BASE=/exports/volumes
#   XCELSIOR_NFS_PATH=/exports/volumes
#   XCELSIOR_NFS_REQUIRED=true

set -euo pipefail

EXPORT_BASE="${XCELSIOR_NFS_EXPORT_BASE:-/exports/volumes}"
MESH_SUBNET="${XCELSIOR_NFS_CLIENT_SUBNET:-100.64.0.0/10}"
SSH_USER="${XCELSIOR_SSH_USER:-aaryn}"
SSH_PUBKEY_FILE="${XCELSIOR_SSH_PUBKEY_FILE:-/home/linuxuser/.ssh/xcelsior.pub}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root (sudo)" >&2
  exit 1
fi

echo "▸ Installing NFS + LUKS packages…"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nfs-kernel-server cryptsetup e2fsprogs rsync

echo "▸ Creating export directory ${EXPORT_BASE}…"
mkdir -p "${EXPORT_BASE}"
chmod 755 "${EXPORT_BASE}"
chown nobody:nogroup "${EXPORT_BASE}" || true

echo "▸ Configuring /etc/exports…"
EXPORT_LINE="${EXPORT_BASE} ${MESH_SUBNET}(rw,sync,no_subtree_check,no_root_squash)"
if grep -qF "${EXPORT_BASE}" /etc/exports 2>/dev/null; then
  echo "  exports entry already present"
else
  echo "${EXPORT_LINE}" >> /etc/exports
fi
exportfs -ra
systemctl enable --now nfs-server
systemctl restart nfs-server

echo "▸ Ensuring SSH service user ${SSH_USER}…"
if ! id "${SSH_USER}" &>/dev/null; then
  useradd -m -s /bin/bash "${SSH_USER}"
fi
install -d -m 700 -o "${SSH_USER}" -g "${SSH_USER}" "/home/${SSH_USER}/.ssh"
if [[ -f "${SSH_PUBKEY_FILE}" ]]; then
  install -m 600 -o "${SSH_USER}" -g "${SSH_USER}" "${SSH_PUBKEY_FILE}" \
    "/home/${SSH_USER}/.ssh/authorized_keys"
else
  echo "  WARN: ${SSH_PUBKEY_FILE} missing — add xcelsior pubkey to /home/${SSH_USER}/.ssh/authorized_keys"
fi

echo "▸ Sudoers for volume provisioning…"
SUDOERS_FILE="/etc/sudoers.d/xcelsior-volumes"
cat > "${SUDOERS_FILE}" <<EOF
# Xcelsior volume engine — narrow LUKS/mount commands for ${SSH_USER}
${SSH_USER} ALL=(root) NOPASSWD: /usr/sbin/cryptsetup
${SSH_USER} ALL=(root) NOPASSWD: /usr/bin/mount
${SSH_USER} ALL=(root) NOPASSWD: /usr/bin/umount
${SSH_USER} ALL=(root) NOPASSWD: /usr/sbin/mkfs.ext4
EOF
chmod 440 "${SUDOERS_FILE}"

echo "▸ Verifying export…"
exportfs -v | grep -F "${EXPORT_BASE}" || true
test -d "${EXPORT_BASE}" && echo "✓ ${EXPORT_BASE} exists"
systemctl is-active nfs-server && echo "✓ nfs-server active"

echo ""
echo "Done. Set on API host .env:"
echo "  XCELSIOR_NFS_SERVER=<VPS mesh IP, e.g. 100.64.0.1>"
echo "  XCELSIOR_NFS_SSH_HOST=127.0.0.1"
echo "  XCELSIOR_NFS_EXPORT_BASE=${EXPORT_BASE}"
echo "  XCELSIOR_NFS_REQUIRED=true"