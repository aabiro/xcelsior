#!/usr/bin/env bash
# Publish Fern docs to production (https://xcelsior.docs.buildwithfern.com → docs.xcelsior.ca).
#
# Requires FERN_TOKEN in repo-root .env (see .env.example).
# Usage: ./scripts/publish-fern-docs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing ${ENV_FILE} — set FERN_TOKEN before publishing." >&2
  exit 1
fi

# Read the one key this needs. `.env` is dotenv, not shell: `source`-ing it ran
# any value containing a space as a command (a Redis password did exactly that)
# and would execute anything else the file happened to contain.
if [[ -z "${FERN_TOKEN:-}" ]]; then
  FERN_TOKEN="$(sed -n 's/^FERN_TOKEN=//p' "$ENV_FILE" | tail -n 1)"
  FERN_TOKEN="${FERN_TOKEN%\"}"; FERN_TOKEN="${FERN_TOKEN#\"}"
  FERN_TOKEN="${FERN_TOKEN%\'}"; FERN_TOKEN="${FERN_TOKEN#\'}"
fi
export FERN_TOKEN

if [[ -z "${FERN_TOKEN:-}" ]]; then
  echo "FERN_TOKEN is empty in ${ENV_FILE}." >&2
  exit 1
fi

cd "${PROJECT_DIR}/fern"
exec npx fern generate --docs --no-prompt --force "$@"