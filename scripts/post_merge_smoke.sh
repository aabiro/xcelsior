#!/usr/bin/env bash
# Post-merge / post-deploy smoke — billing, authz, lifecycle, OAuth machine, OTEL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d venv ]]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

export CI="${CI:-true}"
export XCELSIOR_ENV="${XCELSIOR_ENV:-test}"

PYTEST_TARGETS=(
  tests/test_billing_endpoints_coverage.py
  tests/test_resource_access_control.py
  tests/test_lifecycle.py
  "tests/test_instance_flow.py::TestWalletEmptyStopStart::test_stop_and_start"
  tests/test_oauth_migration.py::TestOAuthMigrationSecurity
  tests/test_phase5_features.py::TestOpenTelemetryIntegration
  tests/test_phase5_features.py::TestOpenTelemetryGracefulDegradation
  tests/test_worker_agent_allowlist.py
)

echo "▸ Post-merge smoke (pytest)…"
python -m pytest "${PYTEST_TARGETS[@]}" -q --tb=line "$@"

if [[ -f .env.test ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env.test
  set +a
fi

if [[ -n "${OTEL_EXPORTER_OTLP_ENDPOINT:-}" ]]; then
  echo "▸ OTEL endpoint configured: ${OTEL_EXPORTER_OTLP_ENDPOINT}"
  if command -v curl >/dev/null 2>&1; then
    host_port="${OTEL_EXPORTER_OTLP_ENDPOINT#*://}"
    host="${host_port%%:*}"
    if curl -sf "http://${host}:16686" >/dev/null 2>&1; then
      echo "✓ Jaeger UI reachable at http://${host}:16686"
    else
      echo "⚠ Jaeger UI not reachable (collector may still accept OTLP on 4317)"
    fi
  fi
else
  echo "⚠ OTEL_EXPORTER_OTLP_ENDPOINT unset — spans in-process only (see docs/OBSERVABILITY.md)"
fi

echo "✓ Post-merge smoke finished"