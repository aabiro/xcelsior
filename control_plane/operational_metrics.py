"""Durable control-plane health signals for Prometheus.

The API's Prometheus endpoint is process-local, while the scheduler,
reconciler, outbox dispatcher, and maintenance worker run in separate
processes.  This module bridges that boundary through the existing
``service_heartbeats`` table and derives operational gauges from durable
PostgreSQL state.

Every snapshot carries an explicit availability gauge and a last-success
timestamp.  An empty queue therefore exports a real zero, while a failed
query exports ``available=0`` and preserves the previous success time.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from control_plane.db import run_transaction

log = logging.getLogger("xcelsior.control_plane.operational_metrics")

_SERVICE_NAMES = ("scheduler", "reconciler", "outbox", "maintenance")
_last_success_timestamp = 0.0
_last_success_lock = threading.Lock()


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prom_label(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def _row_dict(row: object, columns: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {column: row.get(column) for column in columns}
    if row is None:
        return {column: 0 for column in columns}
    return dict(zip(columns, row, strict=False))


_SNAPSHOT_COLUMNS = (
    "queue_depth",
    "queue_oldest_age_seconds",
    "billing_missing_meters",
    "billing_open_terminal_meters",
    "stale_active_leases",
    "stale_offered_leases",
    "stale_host_observations",
    "hosts_missing_observations",
    "oldest_host_observation_age_seconds",
    "reconciliation_queue_depth",
    "reconciliation_oldest_due_age_seconds",
    "reconciliation_queue_errors",
    "stale_fence_findings",
    "critical_findings",
    "outbox_backlog",
    "outbox_oldest_pending_age_seconds",
    "outbox_dead_letters",
    "scheduled_task_failures",
    "maintenance_heartbeat_fresh",
    "maintenance_heartbeat_age_seconds",
)


_SNAPSHOT_SQL = """
WITH
queue_state AS (
    SELECT
        count(*) FILTER (WHERE status = 'queued')::bigint AS queue_depth,
        COALESCE(
            EXTRACT(
                EPOCH FROM (
                    clock_timestamp()
                    - min(COALESCE(queued_at, to_timestamp(submitted_at)))
                      FILTER (WHERE status = 'queued')
                )
            ),
            0
        )::double precision AS queue_oldest_age_seconds
    FROM jobs
),
billing_state AS (
    SELECT
        (
            SELECT count(*)::bigint
            FROM job_attempts a
            LEFT JOIN usage_meters m ON m.attempt_id = a.attempt_id::text
            WHERE a.status IN ('succeeded', 'failed', 'preempted')
              AND a.started_at IS NOT NULL
              AND a.ended_at IS NOT NULL
              AND m.meter_id IS NULL
        ) AS billing_missing_meters,
        (
            SELECT count(*)::bigint
            FROM usage_meters m
            JOIN job_attempts a ON a.attempt_id::text = m.attempt_id
            WHERE m.completed_at IS NULL
              AND a.status IN (
                  'succeeded', 'failed', 'cancelled',
                  'preempted', 'lost', 'fenced'
              )
        ) AS billing_open_terminal_meters
),
lease_state AS (
    SELECT
        count(*) FILTER (
            WHERE status = 'active' AND expires_at <= clock_timestamp()
        )::bigint AS stale_active_leases,
        count(*) FILTER (
            WHERE status = 'offered' AND claim_deadline <= clock_timestamp()
        )::bigint AS stale_offered_leases
    FROM placement_leases
),
observation_state AS (
    SELECT
        count(*) FILTER (
            WHERE administrative_state = 'admitted'
              AND (
                  last_observed_at IS NULL
                  OR last_observed_at
                     < clock_timestamp() - make_interval(secs => %s)
              )
        )::bigint AS stale_host_observations,
        count(*) FILTER (
            WHERE administrative_state = 'admitted'
              AND last_observed_at IS NULL
        )::bigint AS hosts_missing_observations,
        COALESCE(
            EXTRACT(
                EPOCH FROM (
                    clock_timestamp()
                    - min(last_observed_at)
                      FILTER (
                          WHERE administrative_state = 'admitted'
                            AND last_observed_at IS NOT NULL
                      )
                )
            ),
            0
        )::double precision AS oldest_host_observation_age_seconds
    FROM hosts
),
reconciliation_state AS (
    SELECT
        count(*) FILTER (
            WHERE due_at <= clock_timestamp()
        )::bigint AS reconciliation_queue_depth,
        COALESCE(
            EXTRACT(
                EPOCH FROM (
                    clock_timestamp()
                    - min(due_at) FILTER (WHERE due_at <= clock_timestamp())
                )
            ),
            0
        )::double precision AS reconciliation_oldest_due_age_seconds,
        count(*) FILTER (WHERE last_error IS NOT NULL)::bigint
            AS reconciliation_queue_errors
    FROM reconciliation_queue
),
finding_state AS (
    SELECT
        count(*) FILTER (
            WHERE resolved_at IS NULL
              AND finding_type = 'stale_fence_container'
        )::bigint AS stale_fence_findings,
        count(*) FILTER (
            WHERE resolved_at IS NULL
              AND severity IN ('error', 'critical')
        )::bigint AS critical_findings
    FROM reconciliation_findings
),
outbox_state AS (
    SELECT
        count(*) FILTER (
            WHERE published_at IS NULL AND dead_lettered_at IS NULL
        )::bigint AS outbox_backlog,
        COALESCE(
            EXTRACT(
                EPOCH FROM (
                    clock_timestamp()
                    - min(created_at) FILTER (
                        WHERE published_at IS NULL
                          AND dead_lettered_at IS NULL
                    )
                )
            ),
            0
        )::double precision AS outbox_oldest_pending_age_seconds,
        count(*) FILTER (
            WHERE dead_lettered_at IS NOT NULL
        )::bigint AS outbox_dead_letters
    FROM outbox_events
),
scheduled_state AS (
    SELECT
        count(*) FILTER (
            WHERE enabled AND last_status = 'failed'
        )::bigint AS scheduled_task_failures,
        CASE
            WHEN max(last_run_at) FILTER (
                WHERE task_name = 'maintenance_heartbeat'
            ) >= clock_timestamp() - make_interval(secs => %s)
            THEN 1
            ELSE 0
        END::bigint AS maintenance_heartbeat_fresh,
        COALESCE(
            EXTRACT(
                EPOCH FROM (
                    clock_timestamp()
                    - max(last_run_at) FILTER (
                        WHERE task_name = 'maintenance_heartbeat'
                    )
                )
            ),
            0
        )::double precision AS maintenance_heartbeat_age_seconds
    FROM scheduled_tasks
)
SELECT
    queue_state.queue_depth,
    queue_state.queue_oldest_age_seconds,
    billing_state.billing_missing_meters,
    billing_state.billing_open_terminal_meters,
    lease_state.stale_active_leases,
    lease_state.stale_offered_leases,
    observation_state.stale_host_observations,
    observation_state.hosts_missing_observations,
    observation_state.oldest_host_observation_age_seconds,
    reconciliation_state.reconciliation_queue_depth,
    reconciliation_state.reconciliation_oldest_due_age_seconds,
    reconciliation_state.reconciliation_queue_errors,
    finding_state.stale_fence_findings,
    finding_state.critical_findings,
    outbox_state.outbox_backlog,
    outbox_state.outbox_oldest_pending_age_seconds,
    outbox_state.outbox_dead_letters,
    scheduled_state.scheduled_task_failures,
    scheduled_state.maintenance_heartbeat_fresh,
    scheduled_state.maintenance_heartbeat_age_seconds
FROM queue_state
CROSS JOIN billing_state
CROSS JOIN lease_state
CROSS JOIN observation_state
CROSS JOIN reconciliation_state
CROSS JOIN finding_state
CROSS JOIN outbox_state
CROSS JOIN scheduled_state
"""


def _expected_services() -> dict[str, int]:
    scheduler_mode = (os.environ.get("XCELSIOR_SCHEDULER_MODE") or "paused").strip().lower()
    return {
        "scheduler": 1,
        "reconciler": int(scheduler_mode in {"canary", "active"}),
        "outbox": int(_as_bool(os.environ.get("XCELSIOR_OUTBOX_DISPATCHER"), default=True)),
        "maintenance": 1,
    }


def collect_operational_snapshot(
    conn: Any,
    *,
    observation_stale_seconds: int = 300,
    heartbeat_fresh_seconds: int = 60,
) -> dict[str, Any]:
    """Read one internally consistent operational snapshot from PostgreSQL."""
    started = time.monotonic()
    row = conn.execute(
        _SNAPSHOT_SQL,
        (
            max(1, int(observation_stale_seconds)),
            max(1, int(heartbeat_fresh_seconds)),
        ),
    ).fetchone()
    snapshot = _row_dict(row, _SNAPSHOT_COLUMNS)

    heartbeat_rows = conn.execute(
        """
        SELECT
            service,
            count(*) FILTER (
                WHERE last_heartbeat_at
                    >= clock_timestamp() - make_interval(secs => %s)
            )::bigint AS fresh_replicas,
            COALESCE(
                EXTRACT(EPOCH FROM (clock_timestamp() - max(last_heartbeat_at))),
                0
            )::double precision AS latest_age_seconds
        FROM service_heartbeats
        WHERE service = ANY(%s)
        GROUP BY service
        """,
        (max(1, int(heartbeat_fresh_seconds)), list(_SERVICE_NAMES)),
    ).fetchall()

    services: dict[str, dict[str, float]] = {
        service: {"fresh_replicas": 0.0, "latest_age_seconds": 0.0} for service in _SERVICE_NAMES
    }
    for heartbeat_row in heartbeat_rows:
        values = _row_dict(
            heartbeat_row,
            ("service", "fresh_replicas", "latest_age_seconds"),
        )
        service = str(values["service"])
        if service in services:
            services[service] = {
                "fresh_replicas": float(values["fresh_replicas"] or 0),
                "latest_age_seconds": float(values["latest_age_seconds"] or 0),
            }

    # The maintenance worker already owns the durable scheduled-task claim
    # path.  Its no-op heartbeat task proves that executor loop is making
    # progress without granting the billing role a new write into the
    # control-plane heartbeat table.
    services["maintenance"] = {
        "fresh_replicas": float(snapshot["maintenance_heartbeat_fresh"] or 0),
        "latest_age_seconds": float(snapshot["maintenance_heartbeat_age_seconds"] or 0),
    }

    snapshot["services"] = services
    snapshot["expected_services"] = _expected_services()
    snapshot["collection_duration_seconds"] = time.monotonic() - started
    return snapshot


def _metric(lines: list[str], name: str, help_text: str, value: object) -> None:
    lines.extend(
        [
            f"# HELP {name} {help_text}",
            f"# TYPE {name} gauge",
            f"{name} {value}",
        ]
    )


def render_operational_metrics(snapshot: Mapping[str, Any]) -> list[str]:
    """Render a successful snapshot in Prometheus text exposition format."""
    global _last_success_timestamp
    now = time.time()
    with _last_success_lock:
        _last_success_timestamp = now

    lines: list[str] = [""]
    _metric(
        lines,
        "xcelsior_control_plane_metrics_available",
        "1 when durable control-plane metrics were read successfully",
        1,
    )
    _metric(
        lines,
        "xcelsior_control_plane_metrics_last_success_timestamp_seconds",
        "Unix time of the most recent successful durable metrics snapshot",
        f"{now:.3f}",
    )
    _metric(
        lines,
        "xcelsior_control_plane_metrics_collection_duration_seconds",
        "Time spent collecting the durable control-plane metrics snapshot",
        f"{float(snapshot.get('collection_duration_seconds') or 0):.6f}",
    )

    definitions = (
        (
            "xcelsior_queue_oldest_age_seconds",
            "Age of the oldest queued job",
            "queue_oldest_age_seconds",
        ),
        (
            "xcelsior_billing_missing_meters",
            "Billable terminal attempts without a usage meter",
            "billing_missing_meters",
        ),
        (
            "xcelsior_billing_open_terminal_meters",
            "Usage meters still open after their attempt became terminal",
            "billing_open_terminal_meters",
        ),
        (
            "xcelsior_stale_active_leases",
            "Active placement leases past their expiry",
            "stale_active_leases",
        ),
        (
            "xcelsior_stale_offered_leases",
            "Offered placement leases past their claim deadline",
            "stale_offered_leases",
        ),
        (
            "xcelsior_stale_host_observations",
            "Admitted hosts with missing or stale observations",
            "stale_host_observations",
        ),
        (
            "xcelsior_hosts_missing_observations",
            "Admitted hosts that have never produced an observation",
            "hosts_missing_observations",
        ),
        (
            "xcelsior_oldest_host_observation_age_seconds",
            "Age of the oldest admitted-host observation",
            "oldest_host_observation_age_seconds",
        ),
        (
            "xcelsior_reconciliation_queue_depth",
            "Due reconciliation work items",
            "reconciliation_queue_depth",
        ),
        (
            "xcelsior_reconciliation_oldest_due_age_seconds",
            "Age of the oldest due reconciliation item",
            "reconciliation_oldest_due_age_seconds",
        ),
        (
            "xcelsior_reconciliation_queue_errors",
            "Reconciliation queue items carrying a last error",
            "reconciliation_queue_errors",
        ),
        (
            "xcelsior_reconciliation_stale_fence_findings",
            "Open stale-fence-container findings",
            "stale_fence_findings",
        ),
        (
            "xcelsior_reconciliation_critical_findings",
            "Open error or critical reconciliation findings",
            "critical_findings",
        ),
        (
            "xcelsior_outbox_backlog",
            "Unpublished non-dead-lettered outbox events",
            "outbox_backlog",
        ),
        (
            "xcelsior_outbox_oldest_pending_age_seconds",
            "Age of the oldest pending outbox event",
            "outbox_oldest_pending_age_seconds",
        ),
        (
            "xcelsior_outbox_dead_letters",
            "Retained dead-lettered outbox events",
            "outbox_dead_letters",
        ),
        (
            "xcelsior_scheduled_task_failures",
            "Enabled durable scheduled tasks whose latest run failed",
            "scheduled_task_failures",
        ),
    )
    for name, help_text, key in definitions:
        _metric(lines, name, help_text, snapshot.get(key, 0))

    services = snapshot.get("services") or {}
    expected = snapshot.get("expected_services") or {}
    lines.extend(
        [
            "# HELP xcelsior_service_heartbeat_fresh_replicas Service replicas with a heartbeat inside the freshness window",
            "# TYPE xcelsior_service_heartbeat_fresh_replicas gauge",
            "# HELP xcelsior_service_heartbeat_latest_age_seconds Age of the newest heartbeat for a service",
            "# TYPE xcelsior_service_heartbeat_latest_age_seconds gauge",
            "# HELP xcelsior_service_expected Whether this deployment configuration expects the service",
            "# TYPE xcelsior_service_expected gauge",
        ]
    )
    for service in _SERVICE_NAMES:
        service_state = services.get(service) or {}
        label = _prom_label(service)
        lines.append(
            "xcelsior_service_heartbeat_fresh_replicas"
            f"{{service={label}}} {service_state.get('fresh_replicas', 0)}"
        )
        lines.append(
            "xcelsior_service_heartbeat_latest_age_seconds"
            f"{{service={label}}} {service_state.get('latest_age_seconds', 0)}"
        )
        lines.append(f"xcelsior_service_expected{{service={label}}} {expected.get(service, 0)}")
    return lines


def render_operational_metrics_unavailable() -> list[str]:
    """Render explicit failure/freshness signals after a collection error."""
    with _last_success_lock:
        last_success = _last_success_timestamp
    return [
        "",
        "# HELP xcelsior_control_plane_metrics_available 1 when durable control-plane metrics were read successfully",
        "# TYPE xcelsior_control_plane_metrics_available gauge",
        "xcelsior_control_plane_metrics_available 0",
        "# HELP xcelsior_control_plane_metrics_last_success_timestamp_seconds Unix time of the most recent successful durable metrics snapshot",
        "# TYPE xcelsior_control_plane_metrics_last_success_timestamp_seconds gauge",
        f"xcelsior_control_plane_metrics_last_success_timestamp_seconds {last_success:.3f}",
    ]


def heartbeat_once(
    service: str,
    *,
    replica_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Upsert one durable service heartbeat."""
    if service not in _SERVICE_NAMES:
        raise ValueError(f"unsupported heartbeat service: {service}")
    resolved_replica = (
        replica_id
        or os.environ.get("XCELSIOR_REPLICA_ID")
        or f"{socket.gethostname()}-{os.getpid()}"
    )
    payload = json.dumps(dict(details or {}), sort_keys=True)
    service_version = os.environ.get("XCELSIOR_SERVICE_VERSION") or None

    def _write(conn: Any) -> None:
        conn.execute(
            """
            INSERT INTO service_heartbeats (
                service, replica_id, service_version, details
            )
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (service, replica_id) DO UPDATE
            SET last_heartbeat_at = clock_timestamp(),
                service_version = EXCLUDED.service_version,
                details = EXCLUDED.details
            """,
            (service, resolved_replica, service_version, payload),
        )

    run_transaction(_write, what=f"{service}_heartbeat")


@dataclass
class ServiceHeartbeat:
    """Rate-limited heartbeat emitter for a runtime loop."""

    service: str
    replica_id: str | None = None
    interval_seconds: float = 15.0
    details: Mapping[str, Any] | None = None
    _next_due: float = 0.0

    def emit_if_due(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now < self._next_due:
            return False
        self._next_due = now + max(1.0, float(self.interval_seconds))
        heartbeat_once(
            self.service,
            replica_id=self.replica_id,
            details=self.details,
        )
        return True


def start_service_heartbeat(
    service: str,
    *,
    replica_id: str | None = None,
    stop: threading.Event | None = None,
    interval_seconds: float = 15.0,
    details: Mapping[str, Any] | None = None,
) -> threading.Thread:
    """Start a daemon heartbeat for process-level worker availability."""
    stop_event = stop or threading.Event()
    emitter = ServiceHeartbeat(
        service,
        replica_id=replica_id,
        interval_seconds=interval_seconds,
        details=details,
    )

    def _run() -> None:
        while not stop_event.is_set():
            try:
                emitter.emit_if_due(force=True)
            except Exception:
                log.exception("%s heartbeat failed; retrying", service)
            stop_event.wait(max(1.0, float(interval_seconds)))

    thread = threading.Thread(
        target=_run,
        name=f"{service}-heartbeat",
        daemon=True,
    )
    thread.start()
    return thread
