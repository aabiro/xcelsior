#!/usr/bin/env bash
# Run the GX1 tool-selection eval without GitHub Actions and without a
# production credential.
#
# Why this exists. The eval was wired to `.github/workflows/live-gates.yml`,
# which cannot run — so the only gate that grades the *published tool
# descriptions* had no way to execute, and the baseline in `eval-baseline.json`
# went stale behind ten new tools. A gate nobody can run is not a gate.
#
# What it measures. `scripts/mcp_tool_eval.py` calls `initialize` + `tools/list`
# and then asks a model to choose tools from those schemas. It never executes a
# tool. So a local server built from the working tree publishes byte-identical
# schemas to what a deploy of the same commit would, and grading them is a real
# measurement of the surface.
#
# What it does NOT measure: the deployed surface. Production runs an older
# commit. The captured JSON records `base: http://127.0.0.1:...` so a reader can
# never mistake a local capture for a live one — check that field before
# comparing two baselines.
#
#   ANTHROPIC_API_KEY=sk-... scripts/run_mcp_eval_locally.sh [--samples 3]
#
# Costs real money: roughly $1.30 for the default 3 samples x 30 cases.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=39412
SAMPLES=3
OUT="${REPO}/eval-baseline.json"
ONLY=""
CASE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --samples) SAMPLES="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    # Forwarded so a single category can be re-checked after a description
    # change without paying for the whole set. `mcp_tool_eval.py` has always
    # accepted it; this script did not pass it through, so the only way to
    # verify a one-tool fix was a full capture.
    --only) ONLY="$2"; shift 2 ;;
    --case) CASE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  # Fall back to the repo .env rather than failing, since that is where the key
  # lives on the machines this runs on.
  if [ -f "${REPO}/.env" ]; then
    ANTHROPIC_API_KEY="$(grep -m1 '^ANTHROPIC_API_KEY=' "${REPO}/.env" | cut -d= -f2- | tr -d '"'"'"'')"
    export ANTHROPIC_API_KEY
  fi
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "BLOCKED(env): no ANTHROPIC_API_KEY. The eval cannot run; this is not a pass." >&2
  exit 3
fi

LOG="$(mktemp)"
MCP_DIR="${REPO}/mcp" node "${REPO}/scripts/mcp_local_surface.mjs" > "${LOG}" 2>&1 &
HARNESS=$!
# Always tear the server down, including on failure — a leaked listener silently
# poisons the next run by answering on the same port with the previous commit.
trap 'kill "${HARNESS}" 2>/dev/null || true; wait "${HARNESS}" 2>/dev/null || true' EXIT

# Wait for READY rather than sleeping a fixed interval and connecting anyway.
# That exact pattern let a hosted e2e report green against a server that had
# never started.
for _ in $(seq 1 120); do
  grep -q '^READY ' "${LOG}" && break
  sleep 1
done
if ! grep -q '^READY ' "${LOG}"; then
  echo "FAILED: local MCP surface never came up. Harness log:" >&2
  cat "${LOG}" >&2
  exit 1
fi

echo "Local surface: http://127.0.0.1:${PORT}/mcp"
# NOT `exec`. `exec` replaces this shell, so the EXIT trap above never runs and
# the harness survives the script — which is exactly what happened after the
# first capture: the next run died on EADDRINUSE for port 39411, and the leak
# was invisible because the eval it leaked from had succeeded.
python3 "${REPO}/scripts/mcp_tool_eval.py" \
  --base "http://127.0.0.1:${PORT}/mcp" \
  --token local-eval-token \
  --samples "${SAMPLES}" \
  ${ONLY:+--only "${ONLY}"} \
  ${CASE:+--case "${CASE}"} \
  --out "${OUT}"
status=$?
exit "$status"
