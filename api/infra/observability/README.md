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

Legacy retirement uses `xcelsior_legacy_execution_total{path,outcome}` and
`xcelsior_legacy_path_retired{path}`. Deploy the API metrics before the updated
alert rule, and verify that every scraped API replica exports every path. The
operations dashboard shows usage and retirement policy separately. Missing
series, failed scrapes, counter resets and gaps need investigation before
claiming a continuous zero-use window.

The 2026-10-20 readiness audit found missing native API scraping and reproduced
counter loss across Gunicorn workers. The approved production follow-up deployed
shared aggregation, the API target and retirement rule, keeping repairs
report-only and scheduler scope canary. Native Grafana also has the dedicated
**Xcelsior Legacy Retirement** dashboard described below. No agreed zero-use
retention window is proven. See
[the original audit](../../docs/review/2026-10-20-legacy-observation-gates.md) and
[the production rollout](../../docs/review/2026-10-20-control-plane-production-rollout.md).

For an existing native Grafana with Prometheus datasource UID `prometheus`,
use `grafana/native/legacy-retirement.yml` with
`grafana/dashboards/xcelsior-legacy-retirement.json`. The provider uses its own
directory, `/var/lib/grafana/xcelsior-retirement`, and the Xcelsior folder. It
does not replace other dashboard providers or datasource settings. The dashboard
uses the raw API series rather than Compose-only recording rules. It has one
usage panel per measured path, per-replica retirement expectations, and checks
for scrape health, aggregation, expected series counts and generation changes.
Missing observations remain `NO DATA`; graphs do not bridge missing values.

The native provider file belongs in
`/etc/grafana/provisioning/dashboards/xcelsior-retirement.yml`, and only the
retirement dashboard belongs in its dedicated directory. Directories must be
traversable and files readable by the Grafana user. Reload dashboard provisioning
with existing authenticated Grafana admin access, then verify the stored
dashboard, every datasource-proxy query and the rendered panels. See the
[executed native dashboard evidence](../../docs/review/2026-10-20-native-legacy-dashboard.md).
Do not apply the Compose datasource provisioning file to the shared native
installation: that file contains datasource deletion/replacement directives.

`requested` counts authenticated calls, including empty command polls and
requests later refused by fencing. For `status_patch`, `used` counts completed
lifecycle mutator calls (including idempotent calls and fenced compatibility
projections). For `command_drain`, it counts rows deleted for delivery, not
worker execution or ACK. On these two paths, `blocked` means a client refusal before a known
effect; `failed` covers server errors or an error after a known effect. A failed
request with uncertain commit state must not be inferred to have had no effect.

Active scheduler mode declares the six execution paths retired: `work_poll`,
`lease_claim`, `lease_renew`, `lease_release`, `command_start`, `command_pause`.
The `status_patch` and `command_drain` handlers still serve v2 workers during
cutover. Only after their separate retirement gates close, list them in
`XCELSIOR_RETIRED_LEGACY_PATHS` to declare the zero-call expectation. The flag
only changes observation policy; it does not disable handlers. Unknown path
names refuse API startup. The alert joins policy on `instance`, `job` and
`path`, so another replica or another path cannot create a false alert.

API containers now start through `/bin/sh api/scripts/start_api.sh`. It creates
a private multiprocess directory before Python imports, then execs Gunicorn.
Deploy this launcher with the image and Compose commands together. Gunicorn
refuses uninitialized shared metrics even with one worker, because that worker
can be replaced. Exited workers' counters remain until the master generation
ends; each new master gets a fresh directory. Container tmpfs owns its lifetime.

The exporter merges additive counter/histogram/summary families across workers
and retains live/default gauge and custom collectors separately. This preserves
the callback-based retirement policy gauges, which the standard multiprocess
collector cannot evaluate. Export failures return HTTP 503. A single-process
development server exposes `xcelsior_metrics_aggregation_enabled=0`; retirement
qualification requires 1 and a stable
`xcelsior_metrics_generation_started_timestamp_seconds` throughout the agreed
window. A new generation invalidates the window. See the upstream
[multiprocess-mode constraints](https://prometheus.github.io/client_python/multiprocess/).

For an existing native Prometheus, prepare changes with
`api/scripts/plan_native_prometheus.py`. It preserves existing scrape jobs,
rules and Alertmanager routing; validates syntax and the isolated retirement
rule group with the installed native `promtool`; and reports only new operations
and hashes. It never reloads a service or modifies its source. An explicit
`--output` creates a mode-0600 candidate without overwriting existing files.
Syntax validation does not install referenced files or prove alert delivery.
