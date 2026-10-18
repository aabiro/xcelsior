#!/bin/bash
# Long-lived Mac NFS appliance: LUKS open → ext4 mount → NFS-Ganesha export.
set -euo pipefail

LUKS_FILE="${LUKS_FILE:-/luks/inference.luks}"
KEY_FILE="${KEY_FILE:-/key/inference.key}"
MAPPER="${LUKS_MAPPER:-inference}"
MOUNT_POINT="${MOUNT_POINT:-/exports}"
EXPORT_DIR="${EXPORT_DIR:-/exports/volumes}"

log() { echo "[mac-nfs] $*"; }

cleanup() {
  log "shutting down…"
  umount -l "${MOUNT_POINT}" 2>/dev/null || true
  cryptsetup close "${MAPPER}" 2>/dev/null || true
}
trap cleanup SIGTERM SIGINT

if [[ ! -f "${LUKS_FILE}" ]]; then
  log "ERROR: LUKS backing file missing: ${LUKS_FILE}"
  exit 1
fi
if [[ ! -f "${KEY_FILE}" ]]; then
  log "ERROR: keyfile missing: ${KEY_FILE}"
  exit 1
fi

log "opening LUKS ${LUKS_FILE} → /dev/mapper/${MAPPER}"
cryptsetup close "${MAPPER}" 2>/dev/null || true
cryptsetup open --key-file "${KEY_FILE}" "${LUKS_FILE}" "${MAPPER}"

mkdir -p "${MOUNT_POINT}"
umount -l "${MOUNT_POINT}" 2>/dev/null || true
mount -o rw "/dev/mapper/${MAPPER}" "${MOUNT_POINT}"

mkdir -p "${EXPORT_DIR}"
chmod 1777 "${EXPORT_DIR}"
log "mounted ${MOUNT_POINT}; export dir ${EXPORT_DIR}"

# Ganesha needs resolvable hostname, rpcbind, and dbus.
HOSTNAME="${HOSTNAME:-xcelsior-nfs}"
echo "${HOSTNAME}" >/etc/hostname
hostname "${HOSTNAME}" 2>/dev/null || true
grep -qE "[[:space:]]${HOSTNAME}([[:space:]]|$)" /etc/hosts \
  || echo "127.0.0.1 ${HOSTNAME} nfs" >>/etc/hosts

mkdir -p /run/rpcbind /run/dbus
touch /run/rpcbind/rpcbind.xdr /run/rpcbind/portmap.xdr
if ! rpcbind -w; then
  log "ERROR: rpcbind failed to start"
  exit 1
fi
if [ ! -S /run/dbus/system_bus_socket ]; then
  dbus-daemon --system --fork
fi
log "starting NFS-Ganesha (NFSv4 on :2049)…"
ganesha.nfsd -F -L /var/log/ganesha/ganesha.log -f /etc/ganesha/ganesha.conf &
GANESHA_PID=$!
sleep 4
if ! kill -0 "${GANESHA_PID}" 2>/dev/null; then
  log "ERROR: ganesha failed to start"
  tail -30 /var/log/ganesha/ganesha.log 2>/dev/null || true
  exit 1
fi
log "ganesha running (pid ${GANESHA_PID})"
wait "${GANESHA_PID}"