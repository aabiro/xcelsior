#!/usr/bin/env bash
# Regenerate requirements.txt from uv.lock (runtime deps only, for Docker/legacy pip).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
uv export --no-dev --format requirements-txt -o requirements.txt
echo "Wrote ${PROJECT_DIR}/requirements.txt from uv.lock"