# Xcelsior observability runbook

This runbook is for alerts emitted by
`infra/observability/prometheus/alert-rules.yml`. Preserve durable evidence
and fencing authority while diagnosing; an alert is not permission to mutate
ledger or workload state.

## API and worker availability

User impact: API/MCP requests fail, or durable scheduler, reconciler, outbox,
or maintenance work stops progressing.

1. Confirm whether every green/blue target is down in Prometheus `up`, then
   inspect the matching container health and recent Loki logs.
2. For workers, compare `xcelsior_service_expected`,
   `xcelsior_service_heartbeat_fresh_replicas`, and the latest row in
   `service_heartbeats`.
3. Check PostgreSQL reachability before restarting a worker. A shared database
   outage can make every heartbeat stale simultaneously.
4. Restart only the failed replica after capturing logs. Do not start a second
   legacy scheduler or bypass transactional claims to drain the queue.

## Telemetry freshness

User impact: dashboards can show old values or misleading zeros.

1. Check the API scrape target, then
   `xcelsior_control_plane_metrics_available` and
   `xcelsior_control_plane_metrics_last_success_timestamp_seconds`.
2. Inspect API logs for the first failing durable-metrics query and verify that
   migrations are at the expected revision.
3. Check OTel Collector exporter queue failures before restarting Loki or
   Tempo. Persistent queues intentionally replay after recovery.
4. Never treat an absent availability or freshness series as zero work.

## Queue backlog and age

User impact: submitted jobs wait without placement.

1. Compare queue depth and oldest age with scheduler heartbeat freshness.
2. Inspect durable queue reason fields, host observation freshness, admitted
   capacity, and scheduler mode/kill-switch configuration.
3. Repair the failed dependency or capacity condition.
4. Do not hand-assign jobs, clear claims, or requeue strict workloads without
   verifying the current lease and fencing token.

## Billing meter invariants

User impact: work can be unbilled or final cost settlement can remain open.

1. Query the affected terminal attempts and their `usage_meters` by
   `attempt_id`; preserve attempt, meter, wallet, and audit rows.
2. Confirm the `billing_meter_reconcile` scheduled task is fresh and inspect
   its `last_error`.
3. Use only the idempotent billing reconciler for missing meters when its
   enforcement policy is approved. Escalate open terminal meters for manual
   review.
4. Never directly edit wallets, meter totals, or completed timestamps.

## Leases, fences, and observations

User impact: capacity remains reserved, a stale worker may still run, or the
control plane makes decisions from old host state.

1. Join the job, attempt, lease, command journal, latest observation, and open
   reconciliation finding by resource ID.
2. Determine the highest committed fencing token and whether storage or host
   isolation is definitive.
3. Let the lease/reconciliation sweeps perform their idempotent action, or use
   an approved fenced operator action.
4. Never reassign strict work before definitive host and storage fencing.

## Outbox and projectors

User impact: committed state is not reflected in SSE, search, analytics, or
other projection sinks.

1. Separate unpublished `outbox_events`, unprepared fan-out, pending delivery
   receipts, and dead letters by destination/sink.
2. Inspect `attempt_count`, `last_error`, claim expiry, and the sink checkpoint
   before retrying.
3. Repair the sink or dispatcher, then use the bounded idempotent retry path.
4. Never mark an outbox event or receipt delivered without sink evidence, and
   never skip the source outbox to write a downstream projection directly.

## PostgreSQL capacity

User impact: requests and workers slow down or fail as connections or disk
approach exhaustion.

1. Verify `up{job="postgres"}`, connection utilization, deadlock increments,
   database size, and the host filesystem containing PostgreSQL data.
2. Identify connection owners and long transactions before changing pool or
   server limits.
3. Free space by approved retention/backup policy only; preserve audit,
   billing, outbox, and backup evidence.
4. Do not terminate unknown sessions or delete PostgreSQL files directly.

## Backup and restore freshness

User impact: recovery point or recoverability is unproven.

1. Check the node-exporter textfile directory and the four
   `xcelsior_*_last_{success,failure}_timestamp_seconds` series.
2. Inspect the backup/restore timer and service logs. Keep the last known-good
   backup immutable while diagnosing the newest failure.
3. Re-run an isolated restore drill only through the checked-in workflow.
4. Never restore into production, overwrite the source database, or declare a
   backup healthy without a successful restore validation.

## Active v2 control plane

Production uses `XCELSIOR_SCHEDULER_MODE=active` in the API, scheduler and
maintenance worker. All standard placements use the transactional scheduler.
The API advertises host claim/ACK and fenced runtime-metadata features; signed
workers confirm v2 with legacy execution disabled. Old authenticated worker work,
lease, command-drain and status-PATCH mutations return HTTP 410. The old
process-memory preemption poll and admin enqueue routes also return 410; they
cannot announce a request that v2 workers will never consume. Durable fair-share
preemption remains the separately tracked B16.4 feature.

Open **Xcelsior Control Plane** at
`/d/xcelsior-control-plane/xcelsior-control-plane`:

1. Check API up, durable metrics available and V2 scheduling active. These should
   be 1. `NO DATA` is unknown, not a healthy zero.
2. Check ready hosts and worker observations. `xcelsior_ready_hosts` requires
   admission, availability and a fresh v2 observation; registration alone does
   not prove usable capacity.
3. Compare queue depth/age with active attempts and reserved GPU devices. Zero
   jobs is expected on an idle site. Reserved and preparing attempts count as
   active even before a container starts.
4. Inspect pending, claimed and dead-letter worker commands. Pending counts
   exclude expired commands; claimed work remains visible until ACK/NACK or
   claim expiry. NACKs retry with bounded backoff. Dead letters need inspection;
   do not delete rows simply to clear the graph.
5. Check service heartbeat age/fresh replicas, reconciliation backlog/errors,
   expired leases, billing invariants, event delivery and background failures.
   Keep repair policies in the approved `report_only` posture.

Host-management directives now use the same durable command table and protocol
as attempt commands. Results are journaled before ACK; lost ACKs replay the
recorded result. A journal write failure produces a retryable NACK. `health_probe`
is a GPU-free readiness directive that exercises claim, dispatch and durable ACK
without creating a customer job or touching a container.

Runtime connection details, including the public SSH port, are included in the
fenced attempt-status transaction. Stale authority or invalid metadata cannot
change those fields. Repeating a status may update connection metadata without
replaying the lifecycle. The shared runner does not send a compatibility PATCH
for a fenced attempt.

For a worker update on a serving host, use the actual systemd script path and
verify every signed dependency. `XCELSIOR_WORKER_VERIFY_ON_STARTUP=0` explicitly
defers the startup GPU benchmark during that handoff; restore the unit's normal
configuration afterward. Verify the new source hash, v2-only negotiation,
heartbeat/observation and a durable health-probe ACK. Check existing serving and
VTT process identities before and after.

Native Grafana uses the dedicated `grafana/native/control-plane.yml` provider
and existing datasource UID `prometheus`. Preserve other providers/datasources.
Use Grafana's authenticated dashboard-provisioning reload endpoint and verify
all datasource queries and rendered panels. The executed
[cutover and monitoring record](../../docs/review/2026-10-20-active-control-plane-cutover.md)
contains the package identity, checks and rollback boundary.

The former retirement dashboard and native alert are removed. The operator
requested the active cutover while no customer jobs were running, without a
long retirement-observation wait. Historical stopped jobs and missing-volume
metadata remain protected; the active cutover does not pretend to restore or
retire their data. Physical schema contraction is a separate migration and
must preserve that historical disposition.
