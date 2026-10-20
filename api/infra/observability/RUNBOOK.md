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

## Legacy execution retirement

User impact: old execution clients or stopped instances still depend on
compatibility handlers that an active cutover is expected to retire.

1. Use the **Legacy compatibility use** panel to compare requested, used, blocked
   and failed calls. A blocked request is still a caller; an empty command poll
   still prevents a zero-use claim. Status PATCH counts include the fenced
   compatibility projections still sent by v2 workers.
2. Inspect `xcelsior_legacy_execution_enabled` per API replica. Zero means that
   replica's scheduler policy disables legacy starts. The **Legacy retirement
   policy by replica** table shows `xcelsior_legacy_path_retired{path}`. The alert
   joins usage to that policy on the same `instance`, `job` and `path`.
3. Locate the old worker/client, inspect stopped jobs and preserve container and
   storage state before changing resume paths. Deploy the explicit negotiation
   contract before mixed-mode workers; restarting workers refreshes the policy.
4. Require continuous healthy scrapes throughout the agreed retention window.
   Missing data is not zero use. These counters cover work polling, legacy
   lease operations, start/pause command insertion, status PATCH and destructive
   command draining. Other B16 paths, including internal direct status writers,
   still require their own evidence. Start/pause `used` counts authorized queue
   insertion; drain `used` counts rows removed for delivery. Neither is a
   physical execution ACK. Status `used` counts completed lifecycle mutator
   calls, including idempotent calls. Errors after a known effect and server
   errors count as `failed`; uncertain commit results never imply no effect.
5. Keep repairs report-only. Schema/table/code removal requires the B16 migration
   and rollout gates. See the
   [cutover record](../../docs/review/2026-10-20-worker-cutover-boundary.md).

Scheduler active mode alone does not retire `status_patch` or `command_drain`.
Only after their separate removal gates close, declare them through
`XCELSIOR_RETIRED_LEGACY_PATHS` to enable a zero-call alert. This observation
setting does not disable either handler. Unknown path names refuse startup.
Deploy the new API series before its corresponding alert rule, and check every
replica for all eight paths before starting the retention window. See the
[measurement and recovery evidence](../../docs/review/2026-10-20-legacy-observation-gates.md).

For a stopped-job inventory, first obtain a read-only list of job IDs, assigned
host IDs, container names and volume IDs. Independently verify the Docker daemon
name against the assigned host, then run:

```sh
python api/scripts/audit_legacy_containers.py \
  --host-id HOST_ID --expected-daemon-name VERIFIED_DOCKER_NAME \
  --jobs /tmp/stopped-jobs.json --output /tmp/container-inventory.json
```

The tool uses GET requests on the explicit local Docker socket, excludes
environment/command contents, and exits 2 if observation is incomplete. It checks
the full name/label census before accepting a named-container 404 as absence.
Wrong-host, unavailable and malformed evidence stays unknown. Missing containers
still require storage/image/backup recovery checks; do not requeue a stopped job
or delete metadata solely from this report. The
[executed inventory](../../docs/review/2026-10-20-legacy-resume-retirement.md)
records the current twelve-job disposition gates.

Shared-worker qualification now additionally requires
`xcelsior_metrics_aggregation_enabled == 1` on every expected API target.
Compare `xcelsior_metrics_generation_started_timestamp_seconds` across scrapes:
it must be stable and predate the beginning of the agreed retention window.
Require all eight requested-counter paths and all eight retirement-policy paths,
continuous healthy scrapes, and no counter reset or generation change. Worker
replacement within one generation retains counter shards. A master/container
restart starts a new generation and a new qualification window. The standalone
single-process development exporter advertises aggregation 0 and cannot qualify
retirement. Missing/corrupt shared storage or failed collectors return HTTP 503;
the exporter does not silently fall back to a worker-local zero.

On the native production host, prepare rather than replace the existing shared
monitoring configuration:

```sh
python api/scripts/plan_native_prometheus.py \
  --config /etc/prometheus/prometheus.yml \
  --api-target green=127.0.0.1:9500 \
  --rules-path /etc/prometheus/xcelsior-rules/legacy-retirement.yml \
  --rule-source api/infra/observability/prometheus/alert-rules.yml
```

Include blue only while that replica is part of the verified active inventory.
Review the structural additions and source SHA; rerun if the source changes.
The tool validates through stdin and writes no service files. Its syntax-only
check is not runtime activation: the approved rollout must back up the current
configuration, stage the rule file, validate the complete installed candidate,
switch/reload atomically, and verify targets, loaded rules and receiver delivery.
Roll back to the preserved configuration if those checks fail. See
[the executed plan and worker acceptance](../../docs/review/2026-10-20-shared-legacy-metrics.md).
