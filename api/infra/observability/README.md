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

Claude Cowork events (public OTLP log ingest):

- Cowork's exporter runs in the Cowork VM, so it needs a public endpoint:
  `https://otel.xcelsior.ca` (`api/nginx/otel-xcelsior.conf`). The vhost
  accepts only `POST /v1/logs` with `Authorization: Bearer <token>` and proxies
  to the collector's loopback OTLP/HTTP receiver, whose logs pipeline writes to
  Loki. Tests: `tests/test_otel_ingest_vhost.py` (real nginx).
- The token is host-owned: `/etc/nginx/snippets/otel-ingest-token.conf`
  (see `api/nginx/snippets/otel-ingest-token.conf.example`). Without it the
  vhost answers 503. Issue the certificate first
  (`certbot certonly --webroot -w /var/www/certbot -d otel.xcelsior.ca`);
  the installer skips a vhost whose certificate does not exist yet.
- Native Grafana reads Loki through `grafana/native/loki-datasource.yml`
  (Loki on `127.0.0.1:3110`). Query `{service_name="cowork"}`.
- Admin settings > Cowork: endpoint `https://otel.xcelsior.ca`, protocol
  `http/protobuf`, headers `Authorization=Bearer <token>`. Events carry
  `user.email` and `workspace.host_paths` regardless of content capture.

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
executed active cutover record (internal:
`~/xcelsior-internal-docs/api/docs/review/2026-10-20-active-control-plane-cutover.md`).

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

Native alert noise and replica handoffs:

- `configure_native_alerts.py` adds both loopback API targets, merges managed
  control-plane and Tower capacity rules into the installed inventory, and
  preserves receiver credentials and unrelated rules. Run it on prod-hetzner
  with `python3 api/scripts/configure_native_alerts.py` to validate, then add
  `--apply` to back up changed files and reload the two services.
  Concurrent edits abort the apply; failed validation/reload restores backups.
  When Tower integrity rules are installed, it also removes the warning for a
  successful CPU diagnostic completion; diagnostic failures still alert.
- The API alert requires **all** replicas to fail for two minutes. Database
  collection failures alert separately while the API is reachable. Shared
  database gauges aggregate across replicas, so a green/blue swap retains the
  same alert identity. Disabled worker services do not page.
- Existing findings stay firing through up to ten minutes of missing telemetry
  or recovery confirmation. This prevents a short deployment gap from looking
  like repaired billing evidence. A longer telemetry outage is still unknown;
  use the API/metrics availability alert before interpreting resolved findings.
- Native control-plane warnings repeat every 12 hours, critical failures every
  four hours. Parent outages inhibit redundant dependent alerts. Existing
  destinations and resolved notifications are retained.
- Tower disk warnings require both more than 85% usage and less than 100 GiB
  free for 15 minutes. Less than 20 GiB free pages after five minutes regardless
  of percentage. A six-hour trend projecting exhaustion within 24 hours warns
  after 30 minutes, with at least 120 samples and over 80% usage.

Verify the active `up{job="xcelsior-api"}` inventory, loaded rule expressions,
rule evaluation health, both reload-success metrics, and Alertmanager's active
routing after applying. Test real failures and quiet rollouts with:

```sh
promtool test rules tests/fixtures/native_control_plane_alerts.yml tests/fixtures/native_alert_noise.yml
```

The reviewed missing-meter findings for proof jobs `156a1375` and `21eb56eb`
remain durable and report-only. Neither has a captured billing snapshot. Stable
monitoring must not resolve these records or invent historical charges.

On 2026-11-14, Tower's port-8001 inference router was also repaired: its Python
3.12 virtual environment had followed `/usr/bin/python3` to Python 3.14 after
an OS upgrade. Pinning `venv/bin/python3` to `/usr/bin/python3.12` restored its
existing FastAPI/Uvicorn packages. For future interpreter upgrades, rebuild
and validate the environment explicitly; check `/health`, `/metrics`, and
`/v1/models` after restarting the router. `TowerOllamaDown` is the historical
name of this router alert; Ollama itself listens separately on port 11434.
