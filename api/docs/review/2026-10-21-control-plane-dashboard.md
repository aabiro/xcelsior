# Xcelsior dashboard: current operations

The native Grafana dashboard `xcelsior-control-plane` now has 30 panels across
seven sections. Its live title remains **Xcelsior Control Plane**, in the
**Xcelsior** folder. The former scheduling-mode tile was removed; no dashboard
query references legacy execution, retirement, canaries or adoption progress.

The dashboard covers:

- API reachability, fresh service identities, ready hosts, attempts and queue;
- queue age, GPU reservations, service heartbeat age and worker command delivery;
- current lease/host observation problems and dependency reachability;
- billing evidence exceptions, event/projection delivery and reconciliation;
- successful database snapshot age, backup/restore proof and completion time;
- CPU, memory, free disk space, GPU/VRAM use and GPU temperature;
- actual firing native Xcelsior alerts, with severity and a warning count.

Queries use the native Prometheus jobs actually present in production.
Database availability comes from successful application database snapshots;
the dashboard does not assume a PostgreSQL exporter is installed. An idle
queue and no active attempts are valid zeroes. Missing observations remain
`NO DATA`. The no-alert count falls back to zero only when Prometheus itself
is being scraped successfully. Historical billing evidence gaps remain visible.

All **50 queries** returned nonempty, finite production data. An authenticated
browser visited all **30 panels**, observed no page or datasource errors, and
captured the dashboard and its capacity/alert section. The existing
observability suite passed **16 tests** in an isolated database harness.
The adjacent evidence JSON records the deployed source hash and receipts.

The file was atomically replaced at
`/var/lib/grafana/xcelsior-control-plane/xcelsior-control-plane.json`; Grafana
loaded it through its existing provider. Its process stayed at PID `182240`.
The previous file is retained outside the provisioning directory at
`/opt/xcelsior-releases/control-plane-dashboard-20261021/before-dashboard.json`.
No application, worker, VTT or serving service was restarted.

Native partition-job freshness and failure alerts are qualified in DB-role
PR #81. They are activated with that job's dedicated-role deployment; this
dashboard does not invent observations for an undeployed job. Repairs remain
`report_only`.

The separate generated endpoint-inventory correction was already committed
and pushed to `main` at `ac1852ad`: **568 of 568 operations classified**.
