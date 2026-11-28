#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
if [[ -n "${XCELSIOR_RUNTIME_SECRET_ID:-}" || -e /etc/xcelsior/runtime-secret-id ||
    "${XCELSIOR_USE_RUNTIME_BREAK_GLASS:-0}" == 1 ]]; then
    exec "${XCELSIOR_MAINTENANCE_PYTHON:-$project_dir/.venv/bin/python}" \
        "$script_dir/with_runtime_secrets.py" "$@"
fi
exec "$@"
