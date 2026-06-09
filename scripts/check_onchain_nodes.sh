#!/usr/bin/env bash
# Probe Bitcoin RPC + CLN REST reachability (Tailscale host 100.64.0.6 by default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

BTC_HOST="${XCELSIOR_BTC_RPC_HOST:-127.0.0.1}"
BTC_PORT="${XCELSIOR_BTC_RPC_PORT:-8332}"
LN_URL="${XCELSIOR_LN_CLNREST_URL:-https://127.0.0.1:3010}"

probe_btc_rpc() {
  timeout 5 curl -s --user "${XCELSIOR_BTC_RPC_USER:-}:${XCELSIOR_BTC_RPC_PASS:-}" \
    -d '{"jsonrpc":"1.0","id":"probe","method":"getblockchaininfo","params":[]}' \
    -H "content-type: text/plain;" "http://${BTC_HOST}:${BTC_PORT}/"
}

echo "▸ Bitcoin RPC ${BTC_HOST}:${BTC_PORT}"
btc_body="$(probe_btc_rpc 2>/dev/null || true)"
if [[ -z "$btc_body" ]]; then
  echo "✗ Bitcoin RPC unreachable (node offline or Tailscale route down)"
elif echo "$btc_body" | grep -qE '"chain"|"error":\s*null'; then
  echo "✓ Bitcoin RPC ready"
elif echo "$btc_body" | grep -qE '"code":\s*-28|verifying blocks|loading block index|warming up|starting network|work queue depth exceeded'; then
  echo "◷ Bitcoin RPC reachable but warming up (not ready for deposits yet)"
else
  echo "✗ Bitcoin RPC error: ${btc_body:0:200}"
fi

echo "▸ Lightning CLN REST ${LN_URL}"
if timeout 5 curl -sk -H "Rune: ${XCELSIOR_LN_RUNE:-}" -X POST "${LN_URL}/v1/getinfo" -d "null" >/dev/null 2>&1; then
  echo "✓ CLN REST reachable"
else
  if systemctl is-active lightningd >/dev/null 2>&1 || systemctl show -p ActiveState lightningd 2>/dev/null | grep -q activating; then
    echo "◷ CLN REST not ready (lightningd starting — often waiting for bitcoind)"
  else
    echo "✗ CLN REST unreachable (CLN offline or Tailscale route down)"
  fi
fi