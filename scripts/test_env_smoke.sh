#!/usr/bin/env bash
# Smoke-test local sandbox using .env.test (Stripe/PayPal sandbox, serverless on).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ ! -f .env.test ]]; then
  echo "✗ .env.test not found"
  exit 1
fi

set -a
# shellcheck source=/dev/null
source .env.test
set +a

export XCELSIOR_ENV="${XCELSIOR_ENV:-test}"
export XCELSIOR_TARGET_ENV=test

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d venv ]]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

echo "▸ Test env: XCELSIOR_ENV=$XCELSIOR_ENV PAYPAL_MODE=${PAYPAL_MODE:-?} STRIPE_MODE=${XCELSIOR_STRIPE_MODE:-?}"

python3 - <<'PY'
import os
from serverless.feature import serverless_global_enabled
from stripe_connect import STRIPE_ENABLED, _STRIPE_MODE
print("serverless_enabled", serverless_global_enabled())
print("stripe_enabled", STRIPE_ENABLED, "mode", _STRIPE_MODE)
print("paypal_mode", os.environ.get("PAYPAL_MODE", "?"))
PY

echo "▸ Quick billing + serverless unit tests…"
./run-tests.sh billing -q --tb=line
./run-tests.sh serverless -q --tb=line

echo "✓ Test env smoke finished"