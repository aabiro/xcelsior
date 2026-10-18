#!/bin/bash
# xcelsior-iptables-fix.sh
#
# Ensures FORWARD/INPUT rules needed for Tailscale + Docker networking are
# present on this GPU host. Idempotent — checks each rule with `-C` before
# `-I`/`-A`, so it's safe to run on every boot and periodically.
#
# Background: on Ubuntu hosts with ufw enabled, FORWARD policy defaults to
# DROP. tailscaled normally installs ACCEPT rules for tailscale0, but they
# can be lost after:
#   - ufw reload
#   - docker engine restart (rewrites FORWARD chain)
#   - kernel/package upgrade
#
# When the rules vanish, inbound Tailscale traffic (e.g. from VPS ssh-gateway
# relay) is silently dropped, producing "host offline" errors in the app.
#
# Run paths:
#   - Boot (via xcelsior-iptables-fix.service, After=tailscaled.service)
#   - Every 60 s (via xcelsior-iptables-fix.timer) as a safety net

set -euo pipefail

IFACE="${XCELSIOR_TS_IFACE:-tailscale0}"
LOG_TAG="xcelsior-iptables-fix"

log() { logger -t "$LOG_TAG" "$@" || echo "[$LOG_TAG] $*" >&2; }

# Wait up to 30 s for the interface to exist on boot so the rules have
# something to attach to.
for _ in $(seq 1 30); do
  if ip link show "$IFACE" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! ip link show "$IFACE" >/dev/null 2>&1; then
  log "interface $IFACE not present — skipping (tailscaled may be down)"
  exit 0
fi

ensure_rule() {
  # args: <chain> <rule...>
  local chain="$1"; shift
  if ! iptables -C "$chain" "$@" 2>/dev/null; then
    iptables -I "$chain" 1 "$@"
    log "inserted: -I $chain 1 $*"
  fi
}

# Allow all traffic to/from the tailscale interface through the FORWARD
# chain. Required for container traffic (docker bridge <-> tailscale0) and
# for the VPS ssh-gateway TCP relay into containers.
ensure_rule FORWARD -i "$IFACE" -j ACCEPT
ensure_rule FORWARD -o "$IFACE" -j ACCEPT

# Allow inbound traffic on tailscale0 to reach local services (sshd, etc.).
ensure_rule INPUT -i "$IFACE" -j ACCEPT

exit 0
