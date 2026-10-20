# Native legacy-retirement dashboard — 2026-10-20

**Superseded:** the operator requested completion of the runtime cutover. This
dashboard and provider were removed and replaced with operational Control Plane
metrics. See the [active cutover record](2026-10-20-active-control-plane-cutover.md).

The shared native Grafana now serves **Xcelsior Legacy Retirement** at
`/d/xcelsior-legacy-retirement/xcelsior-legacy-retirement`, in the Xcelsior
folder. It has one usage panel for each of the eight measured API compatibility
paths, a retirement-expectation table, and collection checks. This closes native
dashboard coverage for these eight paths, while B16.1 remains partial.

The dashboard uses datasource UID `prometheus`, already configured against the
native Prometheus on loopback port 9091. It queries raw API metrics, so it does
not depend on the recording rules, Loki or Tempo from the Compose stack. Every
usage panel separates outcomes and API replicas. The collection panels show
scrape health, aggregation readiness, the expected 32 usage and eight policy
series per replica, generation age, generation changes and scrape continuity.
Missing observations remain `NO DATA`; graphs do not bridge missing values.

## Applied files and preservation

Only the dedicated retirement provider and dashboard were added:

- `/etc/grafana/provisioning/dashboards/xcelsior-retirement.yml`
- `/var/lib/grafana/xcelsior-retirement/xcelsior-legacy-retirement.json`

The provider directory is separate from Tower's existing dashboard directory.
The existing Tower dashboard, provider and datasource files retain their hashes;
the datasource API representation is unchanged. The native Grafana process
remained PID 182240 throughout activation. No Xcelsior application, Tower
serving, VTT or GPU workload restart was needed for this increment.

The initially configured Grafana credentials returned 401. The user supplied
a working existing admin login and explicitly requested environment correction.
Admin authentication succeeded, and the two Grafana admin variables were updated
in both the local and production `.env` files. Unrelated production environment
lines were compared with the pre-rollout copy and matched. Environment files
are ignored by Git; no credential values are recorded in this evidence or
committed. Grafana users and authentication configuration were not reset.

## Executed verification

The candidate first loaded in a bounded, isolated Grafana **13.2.3** instance
using the installed production binary and its own database, provisioning and
data directories. All 16 candidate expressions returned expected nonempty
vectors from live Prometheus, and the complete 32 path/outcome label inventory
was checked. The shared Grafana process was not restarted or reconfigured by
that acceptance instance.

Native activation used the existing admin account and the documented
[dashboard provisioning reload endpoint](https://grafana.com/docs/grafana/latest/developer-resources/api-reference/http-api/api-legacy/admin/#reload-provisioning-configurations).
The stored dashboard is provisioned in the expected folder. All 16 expressions
then passed through Grafana's own authenticated datasource proxy. The browser
inspection scrolled through every query panel; every query returned HTTP 200
and populated data frames, with no page errors or missing-data display.

The rendered policy table needed a frame-to-row transformation. The correction
was checked in isolated Grafana before applying it to the owned dashboard.
The final table shows all eight paths and their current **Callers permitted**
expectation. Browser evidence checks those eight rendered rows as well as the
collection stats and usage graphs. This was a display correction; no policy,
metric or scheduler setting was changed.

JSON/YAML validation and whitespace checks also passed. No GPU acceptance test
or synthetic alert notification was run. The
[machine-readable evidence](2026-10-20-native-legacy-dashboard-evidence.json)
pins the final files, preservation checks, datasource queries and browser results.

## Remaining gates

Production remains the same two-host scheduler canary with repairs in
`report_only`. Work polling and command draining still have callers, and every
retirement policy remains 0. This dashboard does not establish the agreed
continuous zero-use retention window or measure the other B16 legacy paths.
Receiver delivery, the stopped-job/missing-volume recovery disposition,
contract migration and code removal remain open. The earlier
[production rollout record](2026-10-20-control-plane-production-rollout.md)
still describes those boundaries.

To withdraw this dashboard, first set `disableDeletion: false` only in its
dedicated provider and reload dashboard provisioning. Remove only the owned
retirement JSON and reload again so Grafana removes that provisioned dashboard;
verify its UID is absent, then remove its dedicated provider and reload.
Preserve Tower's provider, dashboard directory, datasource and user settings.
No rollback was needed or executed by this successful increment.
