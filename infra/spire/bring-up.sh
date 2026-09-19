#!/usr/bin/env bash
# Bring up the SPIRE mesh in the only order that works.
#
#   sudo bash infra/spire/bring-up.sh
#
# The compose file bind-mounts ./bootstrap.crt, which is the SERVER'S OWN trust
# bundle — so it cannot exist before the server does, and `docker compose up`
# on its own fails: `spire-agent validate` reports
#
#     could not parse trust bundle: open .../bootstrap.crt: no such file
#
# This script does server → bundle → everything else, and refuses to start on a
# half-configured environment rather than leaving a mesh that looks up but
# cannot issue identities.
set -euo pipefail

cd "$(dirname "$0")"
COMPOSE=(docker compose -f docker-compose.spire.yml)
BUNDLE=./bootstrap.crt

fail() { echo "refusing: $*" >&2; exit 1; }

# ── Preconditions, each one a real outage if it is wrong ──────────────────
[ -n "${XCELSIOR_AGENT_GATEWAY_SECRET:-}" ] || fail \
  "XCELSIOR_AGENT_GATEWAY_SECRET is empty. Read it from /opt/xcelsior/.env —
   never generate a new one. The API authenticates the gateway with the value
   it already holds; a fresh secret strips every identity header and refuses
   the whole fleet."

[ -n "${SPIRE_DATASTORE_DSN:-}" ] || fail \
  "SPIRE_DATASTORE_DSN is empty. Run scripts/provision_spire_datastore.sh."

BIND="${SPIRE_BIND_ADDRESS:-100.64.0.1}"
if ! ip -4 addr show 2>/dev/null | grep -q "inet ${BIND}/"; then
  fail "${BIND} is not an address on this host. The server publishes there so
   remote GPU hosts can attest over the tailnet; if the interface is down, bring
   Headscale/tailscale up first (or set SPIRE_BIND_ADDRESS deliberately)."
fi

# ── 1. the server alone ───────────────────────────────────────────────────
echo "== starting spire-server"
"${COMPOSE[@]}" up -d spire-server

echo "== waiting for it to report healthy"
for i in $(seq 1 40); do
  if "${COMPOSE[@]}" exec -T spire-server \
       /opt/spire/bin/spire-server healthcheck \
       -socketPath /tmp/spire-server/private/api.sock >/dev/null 2>&1; then
    echo "   healthy after ${i} checks"
    break
  fi
  [ "$i" -eq 40 ] && fail "spire-server never became healthy; check
   \`docker compose -f infra/spire/docker-compose.spire.yml logs spire-server\`
   — a bad SPIRE_DATASTORE_DSN looks exactly like this."
  sleep 3
done

# ── 2. its trust bundle, which the agent mounts ───────────────────────────
echo "== exporting the trust bundle to ${BUNDLE}"
tmp="$(mktemp)"
"${COMPOSE[@]}" exec -T spire-server \
  /opt/spire/bin/spire-server bundle show \
  -socketPath /tmp/spire-server/private/api.sock > "$tmp"

grep -q "BEGIN CERTIFICATE" "$tmp" || {
  rm -f "$tmp"
  fail "the exported bundle contains no certificate. Writing it anyway would
   give every agent a trust bundle it cannot parse."
}
mv "$tmp" "$BUNDLE"
chmod 0644 "$BUNDLE"
echo "   $(grep -c 'BEGIN CERTIFICATE' "$BUNDLE") certificate(s) written"

# ── 3. the rest of the mesh ───────────────────────────────────────────────
echo "== starting spire-agent and the gateway"
"${COMPOSE[@]}" up -d

echo
echo "== state"
"${COMPOSE[@]}" ps
cat <<'NEXT'

Next:
  1. Register the admitted hosts:
       XCELSIOR_POSTGRES_DSN=... infra/spire/register-host.sh --all
  2. Confirm Envoy is listening on 9443 and healthy.
  3. Only then flip the agent name to Envoy — ONE line in
     nginx/stream-tls-router.conf — per runbooks/tls-ingress-cutover.md.
  4. Only after a real GPU host reconnects through Envoy, set
     XCELSIOR_SPIFFE_STRICT=1 and retire nginx/agent-xcelsior.conf.
NEXT
