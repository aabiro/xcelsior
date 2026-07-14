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

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${FERN_TOKEN:-}" ]]; then
  echo "FERN_TOKEN is empty in ${ENV_FILE}." >&2
  exit 1
fi

cd "${PROJECT_DIR}/fern"
exec npx fern generate --docs --no-prompt --force "$@"