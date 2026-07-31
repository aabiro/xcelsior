# Xcelsior private observability stack

The default Compose deployment includes Prometheus, Alertmanager, Grafana,
Loki, Tempo, the OpenTelemetry Collector, node exporter, PostgreSQL exporter,
and Jaeger. Browser-facing ports bind to `127.0.0.1`; application metric
routes are blocked at the public Nginx edge.

Before production startup:

1. Set `XCELSIOR_POSTGRES_EXPORTER_PASSWORD` (or a dedicated exporter DSN
   equivalent) through the deploy secret store.
2. Set `XCELSIOR_GRAFANA_ADMIN_USER` and
   `XCELSIOR_GRAFANA_ADMIN_PASSWORD`, plus a random
   `XCELSIOR_GRAFANA_SECRET_KEY`, through the deploy secret store.
3. Copy `alertmanager/alertmanager.routing.example.yml` outside the repository,
   replace the example receivers with secret-managed endpoints, restrict file
   permissions, and set `XCELSIOR_ALERTMANAGER_CONFIG_PATH` to that absolute
   path.
4. Ensure `/var/lib/node_exporter/textfile_collector` exists and is writable
   only by the backup/restore services. It is mounted read-only into node
   exporter.

Local interfaces:

- Grafana: `http://127.0.0.1:3001`
- Prometheus: `http://127.0.0.1:9091`
- Alertmanager: `http://127.0.0.1:9093`
- Jaeger: `http://127.0.0.1:16686`
- OTLP gRPC/HTTP: `127.0.0.1:4317` / `127.0.0.1:4318`

Prometheus retains 30 days subject to a size cap, Loki retains 14 days, and
Tempo retains seven days. All stateful components use named volumes.
