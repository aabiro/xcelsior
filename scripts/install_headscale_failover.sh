#!/usr/bin/env bash
# Install the Headscale standby replica + watchdog on this machine.
# Does not promote. An empty local Headscale is never treated as a replica.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC="$PROJECT_DIR/scripts/headscale_failover.py"
UNIT_DIR="$PROJECT_DIR/infra/systemd"

[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
[[ "$(id -u)" -eq 0 ]] || { echo "run as root" >&2; exit 1; }

install -m 0755 "$SRC" /usr/local/sbin/headscale-failover
install -d -m 0700 /var/backups/headscale
install -m 0644 \
  "$UNIT_DIR/headscale-replicate.service" \
  "$UNIT_DIR/headscale-replicate.timer" \
  "$UNIT_DIR/headscale-failover-watchdog.service" \
  "$UNIT_DIR/headscale-failover-watchdog.timer" \
  /etc/systemd/system/

cat >/etc/default/headscale-failover <<EOF
XCELSIOR_ROOT=$PROJECT_DIR
XCELSIOR_ENV_FILE=$PROJECT_DIR/.env
XCELSIOR_HEADSCALE_HOST=45.76.3.128
XCELSIOR_HEADSCALE_REPLICA_DIR=/var/backups/headscale
XCELSIOR_HEADSCALE_SSH_KEY=/home/aaryn/.ssh/id_ed25519
XCELSIOR_HEADSCALE_USER=root
EOF

systemctl daemon-reload
systemctl enable --now headscale-replicate.timer headscale-failover-watchdog.timer

echo "installed /usr/local/sbin/headscale-failover"
echo "replica dir /var/backups/headscale"
echo
echo "If replica.promotable is false, wait until the VPS is up and run:"
echo "  sudo headscale-failover replicate"
echo "  sudo headscale-failover status --json"
