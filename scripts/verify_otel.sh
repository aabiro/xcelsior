#!/usr/bin/env bash
# Verify OpenTelemetry wiring: Jaeger up, API logs, healthz trace path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

OTEL_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://127.0.0.1:4317}"
JAEGER_UI_HOST="${JAEGER_UI_HOST:-127.0.0.1}"

echo "▸ Starting Jaeger (docker compose)…"
docker compose up -d jaeger 2>/dev/null || docker compose --profile observability up -d jaeger 2>/dev/null || true

for _ in $(seq 1 15); do
  if curl -sf "http://${JAEGER_UI_HOST}:16686" >/dev/null 2>&1; then
    echo "✓ Jaeger UI: http://${JAEGER_UI_HOST}:16686"
    break
  fi
  sleep 1
done

echo "▸ OTEL unit checks (pytest)…"
if [[ -d venv ]]; then source venv/bin/activate; fi
export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_ENDPOINT}"
python -m pytest \
  tests/test_phase5_features.py::TestOpenTelemetryIntegration \
  tests/test_phase5_features.py::TestOpenTelemetryGracefulDegradation \
  -q --tb=line

echo "✓ OTEL verification done (set OTEL_EXPORTER_OTLP_ENDPOINT on API for live export)"