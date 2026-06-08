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

echo "▸ Bitcoin RPC ${BTC_HOST}:${BTC_PORT}"
if timeout 5 curl -sf --user "${XCELSIOR_BTC_RPC_USER:-}:${XCELSIOR_BTC_RPC_PASS:-}" \
  -d '{"jsonrpc":"1.0","id":"probe","method":"getblockchaininfo","params":[]}' \
  -H "content-type: text/plain;" "http://${BTC_HOST}:${BTC_PORT}/" >/dev/null 2>&1; then
  echo "✓ Bitcoin RPC reachable"
else
  echo "✗ Bitcoin RPC unreachable (node offline or Tailscale route down)"
fi

echo "▸ Lightning CLN REST ${LN_URL}"
if timeout 5 curl -sk -H "Rune: ${XCELSIOR_LN_RUNE:-}" "${LN_URL}/v1/getinfo" >/dev/null 2>&1; then
  echo "✓ CLN REST reachable"
else
  echo "✗ CLN REST unreachable (CLN offline or Tailscale route down)"
fi