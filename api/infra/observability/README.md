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

Production now runs the authoritative v2 scheduler in `active` mode. Worker
execution, host commands and runtime status/connection reporting use durable v2
protocols. The authenticated old work, lease, command-drain and status-PATCH
routes return HTTP 410, as do the process-memory preemption poll and admin
enqueue routes. Repair actions remain `report_only`.

The native **Xcelsior Control Plane** dashboard is at
`/d/xcelsior-control-plane/xcelsior-control-plane`. Its 20 panels use raw
PostgreSQL-backed gauges rather than Compose-only recording rules. They show
API reachability, metric availability, scheduling mode, ready hosts, queued work,
active attempts/GPU allocations, command delivery, service heartbeats,
reconciliation, lease/observation health, billing invariants, event delivery,
background failures and metric freshness. Zero jobs is valid; missing metrics
remain `NO DATA`.

For native Grafana, use `grafana/native/control-plane.yml` and
`grafana/dashboards/xcelsior-control-plane.json`. The provider owns only
`/var/lib/grafana/xcelsior-control-plane` in the Xcelsior folder and reuses the
existing datasource UID `prometheus`. Keep directories traversable and files
readable by Grafana. Reload dashboard provisioning with the existing admin
account, then verify all 32 expressions through the datasource proxy and inspect
the rendered panels. Preserve the Tower provider, dashboard and datasource.
Do not apply the Compose datasource provisioning file to the shared native
installation: it contains datasource deletion/replacement directives.

The former retirement dashboard/provider and native alert reference have been
removed. The operator requested a quiescent cutover instead of extending the
retirement-observation project. No retention wait is a deployment prerequisite
for this completed runtime cutover. Historical stopped-job/storage records and
schema remain intact. See the
[executed active cutover](../../docs/review/2026-10-20-active-control-plane-cutover.md).

API containers now start through `/bin/sh api/scripts/start_api.sh`. It creates
a private multiprocess directory before Python imports, then execs Gunicorn.
Deploy this launcher with the image and Compose commands together. Gunicorn
refuses uninitialized shared metrics even with one worker, because that worker
can be replaced. Exited workers' counters remain until the master generation
ends; each new master gets a fresh directory. Container tmpfs owns its lifetime.

The exporter merges additive counter/histogram/summary families across workers
and retains live/default gauge and custom collectors separately. This preserves
the callback-based scheduler policy gauges, which the standard multiprocess
collector cannot evaluate. Export failures return HTTP 503. A single-process
development server exposes `xcelsior_metrics_aggregation_enabled=0`.
`xcelsior_metrics_generation_started_timestamp_seconds` identifies a master
generation; a restart begins a new generation. See the upstream
[multiprocess-mode constraints](https://prometheus.github.io/client_python/multiprocess/).

The earlier `api/scripts/plan_native_prometheus.py` prepares the historical
retirement-observation configuration; do not reapply it to this active rollout.
For future native monitoring changes, preserve existing scrape jobs, Tower
rules and Alertmanager routing, validate the complete installed candidate with
`promtool`, and verify targets and query results after reload. Syntax validation
alone does not prove activation or receiver delivery.
