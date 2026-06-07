#!/usr/bin/env bash
# Verify / configure Mac InferenceData NFS for Xcelsior persistent volumes.
#
# Run ON THE MAC (as aaryn) after the external SSD is mounted at /Volumes/InferenceData:
#   bash scripts/setup_nfs_mac.sh
#
# Prerequisites (manual — macOS GUI):
#   1. System Settings → General → Sharing → Remote Login: ON
#   2. Allow Tailscale peers (100.64.0.0/10) to reach SSH — check firewall / Tailscale ACLs
#   3. /etc/exports must include InferenceData (this script checks it)
#
# Production .env (when VPS can SSH to Mac):
#   XCELSIOR_NFS_SERVER=100.64.0.3
#   XCELSIOR_NFS_SSH_HOST=100.64.0.3
#   XCELSIOR_NFS_SSH_USER=aaryn
#   XCELSIOR_NFS_EXPORT_BASE=/Volumes/InferenceData
#   XCELSIOR_NFS_REQUIRED=true
#
# Note: macOS has no Homebrew cryptsetup formula — encrypted LUKS volumes require
# the Linux VPS NFS path (scripts/setup_nfs_vps.sh). Mac supports unencrypted volumes.

set -euo pipefail

EXPORT_BASE="${XCELSIOR_NFS_EXPORT_BASE:-/Volumes/InferenceData}"
MESH_NET="${XCELSIOR_NFS_CLIENT_SUBNET:-100.64.0.0}"
MESH_MASK="${XCELSIOR_NFS_CLIENT_MASK:-255.192.0.0}"
SSH_USER="${XCELSIOR_SSH_USER:-aaryn}"

echo "▸ Checking InferenceData mount…"
if [[ ! -d "${EXPORT_BASE}" ]]; then
  echo "ERROR: ${EXPORT_BASE} not mounted — attach external SSD first." >&2
  exit 1
fi
echo "✓ ${EXPORT_BASE} exists"

echo "▸ Checking /etc/exports…"
EXPECTED="${EXPORT_BASE} -network ${MESH_NET} -mask ${MESH_MASK}"
if grep -qF "${EXPORT_BASE}" /etc/exports 2>/dev/null; then
  echo "✓ exports entry present:"
  grep "${EXPORT_BASE}" /etc/exports
else
  echo "WARN: ${EXPORT_BASE} missing from /etc/exports"
  echo "  Add (requires sudo):"
  echo "    ${EXPECTED} -mapall=501:20 -alldirs"
  echo "  Then: sudo nfsd restart"
fi

echo "▸ NFS daemon…"
if pgrep -x nfsd >/dev/null 2>&1; then
  echo "✓ nfsd running"
else
  echo "WARN: nfsd not running — enable NFS sharing in System Settings or: sudo nfsd start"
fi

echo "▸ SSH access (for API volume provision)…"
if systemsetup -getremotelogin 2>/dev/null | grep -qi "on"; then
  echo "✓ Remote Login enabled"
else
  echo "WARN: Remote Login may be off — enable in System Settings → Sharing"
fi

echo "▸ Ensuring SSH authorized_keys for volume provision…"
install -d -m 700 "${HOME}/.ssh"
PUBKEY="${XCELSIOR_SSH_PUBKEY_FILE:-${HOME}/.ssh/xcelsior.pub}"
if [[ -f "${PUBKEY}" ]]; then
  if ! grep -qF "$(cat "${PUBKEY}")" "${HOME}/.ssh/authorized_keys" 2>/dev/null; then
    cat "${PUBKEY}" >> "${HOME}/.ssh/authorized_keys"
    chmod 600 "${HOME}/.ssh/authorized_keys"
    echo "✓ Added ${PUBKEY} to authorized_keys"
  else
    echo "✓ authorized_keys already has xcelsior pubkey"
  fi
else
  echo "  WARN: ${PUBKEY} missing — copy VPS/linuxuser xcelsior.pub here"
fi

echo "▸ Encrypted volumes (LUKS)…"
export PATH="/opt/homebrew/bin:/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin:${PATH}"
if command -v xcelsior-cryptsetup >/dev/null 2>&1; then
  echo "✓ xcelsior-cryptsetup (Docker): $(xcelsior-cryptsetup --version 2>&1 | head -1)"
elif command -v cryptsetup >/dev/null 2>&1; then
  echo "✓ cryptsetup: $(command -v cryptsetup)"
else
  echo "  INFO: run bash scripts/install_cryptsetup_mac.sh (Docker-backed LUKS CLI)"
  echo "  NOTE: LUKS mount/export for NFS still needs Linux — use VPS for encrypted volumes"
fi

echo ""
echo "Done. Verify from VPS:"
echo "  ssh ${SSH_USER}@100.64.0.3 hostname"
echo "  showmount -e 100.64.0.3"