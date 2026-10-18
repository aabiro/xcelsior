# Observability (OpenTelemetry)

Xcelsior instruments the FastAPI API with OpenTelemetry. Spans are created via `otel_span()` in route handlers and exported when configured.

## Local development

`docker compose` includes Jaeger with OTLP enabled:

| Service | Port | Purpose |
|---------|------|---------|
| Jaeger UI | `16686` | Trace search UI |
| OTLP gRPC | `4317` | Span ingest (default for the API) |
| OTLP HTTP | `4318` | Alternate ingest |

Set in `.env` (see `.env.example`):

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
```

On API startup you should see:

```text
OTEL: OpenTelemetry instrumentation active (endpoint=http://127.0.0.1:4317)
```

If `opentelemetry` packages are missing, tracing is disabled with an info log (not a silent success).

If `OTEL_EXPORTER_OTLP_ENDPOINT` is unset or empty, instrumentation still runs but spans are **not** exported (in-process only).

## Production

1. Run an OTLP-compatible backend (Jaeger, Grafana Tempo, Honeycomb, Datadog agent, etc.).
2. Set `OTEL_EXPORTER_OTLP_ENDPOINT` on the API container to the collector gRPC or HTTP endpoint (gRPC is what `api.py` uses today).
3. Ensure the collector is reachable from the API network (not `127.0.0.1` unless the collector is sidecarred).
4. Open the vendor UI and filter by service name `xcelsior-api` (set in `api.py` resource attributes).

`docker-compose.yml` already forwards `OTEL_EXPORTER_OTLP_ENDPOINT` into the shared API environment block; mirror that in Kubernetes manifests or systemd `Environment=` lines.

### Suggested checks after deploy

- API logs contain `OpenTelemetry instrumentation active` with a non-empty endpoint.
- A test request (e.g. `GET /healthz`) produces a trace in the backend within ~1 minute.
- Route spans include attributes such as `job.id` on instance mutations.

### Alerts and dashboards

The repo does not ship Grafana dashboards or Alertmanager rules. After OTLP export works:

- Dashboard: request rate, p95 latency, error ratio by `http.route` or span name.
- Alerts: elevated 5xx rate, OTLP export failures (collector down), missing traces for >N minutes in prod.

Wire alerts in your existing monitoring stack (Prometheus + Alertmanager, Grafana Cloud, etc.) using the same collector or a parallel metrics pipeline.

## Tests

`tests/test_phase5_features.py` asserts OTEL import paths and that export is gated on `OTEL_EXPORTER_OTLP_ENDPOINT`.