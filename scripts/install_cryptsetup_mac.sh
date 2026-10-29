#!/usr/bin/env bash
# Install a cryptsetup runtime on macOS for Xcelsior LUKS volume provisioning.
#
# Native macOS cannot run cryptsetup (requires Linux device-mapper). This script
# uses Docker Desktop with a pinned Alpine image that includes cryptsetup.
#
# Usage (on Mac, as aaryn):
#   bash scripts/install_cryptsetup_mac.sh
#
# Prerequisites:
#   - Docker Desktop installed and running
#   - /Volumes/InferenceData mounted (external SSD)
#
# After install, volume provision uses:
#   /usr/local/bin/xcelsior-cryptsetup <cryptsetup-args...>

set -euo pipefail

DOCKER_BIN="${DOCKER_BIN:-/usr/local/bin/docker}"
DOCKER_PATH="/Applications/Docker.app/Contents/Resources/bin:${PATH}"
IMAGE="${XCELSIOR_CRYPTSETUP_IMAGE:-alpine:3.20}"
EXPORT_BASE="${XCELSIOR_NFS_EXPORT_BASE:-/Volumes/InferenceData}"
INSTALL_BIN="${INSTALL_BIN:-${HOME}/.local/bin/xcelsior-cryptsetup}"

export PATH="${DOCKER_PATH}"

if [[ ! -d "${EXPORT_BASE}" ]]; then
  echo "ERROR: ${EXPORT_BASE} not mounted" >&2
  exit 1
fi

echo "▸ Starting Docker Desktop if needed…"
open -a Docker 2>/dev/null || true
for _ in $(seq 1 24); do
  if "${DOCKER_BIN}" info >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
"${DOCKER_BIN}" info >/dev/null 2>&1 || {
  echo "ERROR: Docker daemon not running — start Docker Desktop manually" >&2
  exit 1
}

echo "▸ Pulling ${IMAGE}…"
"${DOCKER_BIN}" pull "${IMAGE}"

echo "▸ Verifying LUKS in container…"
"${DOCKER_BIN}" run --rm --privileged -v "${EXPORT_BASE}:/exports" "${IMAGE}" sh -c '
  apk add --no-cache cryptsetup e2fsprogs util-linux >/dev/null
  cryptsetup --version
  IMG=/exports/.xcelsior-luks-selftest.img
  KEY=/tmp/key.bin
  dd if=/dev/urandom of="$KEY" bs=32 count=1 2>/dev/null
  truncate -s 32M "$IMG"
  cryptsetup luksFormat --batch-mode --type luks2 --key-file "$KEY" --key-size 512 "$IMG"
  cryptsetup luksOpen --key-file "$KEY" "$IMG" xcelsior-selftest
  mkfs.ext4 -q /dev/mapper/xcelsior-selftest
  cryptsetup luksClose xcelsior-selftest
  rm -f "$IMG" "$KEY"
  echo "LUKS self-test OK"
'

echo "▸ Installing wrapper ${INSTALL_BIN}…"
mkdir -p "$(dirname "${INSTALL_BIN}")"
cat >"${INSTALL_BIN}" <<EOF
#!/usr/bin/env bash
# Xcelsior cryptsetup wrapper — runs Linux cryptsetup via Docker on macOS.
set -euo pipefail
export PATH="/Applications/Docker.app/Contents/Resources/bin:\${PATH}"
exec ${DOCKER_BIN} run --rm -i --privileged \\
  -v ${EXPORT_BASE}:/exports \\
  ${IMAGE} sh -c 'apk add --no-cache cryptsetup e2fsprogs util-linux >/dev/null 2>&1; exec cryptsetup "\$@"' sh "\$@"
EOF
chmod 755 "${INSTALL_BIN}"

echo "▸ Add to PATH (in ~/.zshrc if missing):"
echo "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
if ! grep -q '.local/bin' "${HOME}/.zshrc" 2>/dev/null; then
  echo 'export PATH="${HOME}/.local/bin:${PATH}"' >>"${HOME}/.zshrc"
  echo "  appended to ~/.zshrc"
fi

echo "▸ Sudoers (optional — run manually with admin password for API SSH provision):"
cat <<EOF
  sudo tee /etc/sudoers.d/xcelsior-volumes <<'SUDO'
aaryn ALL=(root) NOPASSWD: ${INSTALL_BIN}
aaryn ALL=(root) NOPASSWD: /sbin/mount
aaryn ALL=(root) NOPASSWD: /sbin/umount
aaryn ALL=(root) NOPASSWD: /sbin/mkfs.ext4
aaryn ALL=(root) NOPASSWD: /usr/sbin/truncate
SUDO
EOF

echo ""
echo "Done."
echo "  cryptsetup: ${INSTALL_BIN} --version"
echo "  Note: mount/mkfs for LUKS still require Linux workers — Mac NFS is best for unencrypted volumes."
echo "  VPS SSH to Mac: ensure Tailscale ACL allows tag:tagged-devices -> Mac:22"