"""Native host drain authority; observation payloads never select the host row."""

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from control_plane.launch.service import PlanConflict
from control_plane.outbox import append_event


def set_draining(conn, *, host_id, draining, expected_version=None):
    with conn.cursor(row_factory=dict_row) as cur:
        host = cur.execute("SELECT * FROM hosts WHERE host_id=%s FOR UPDATE", (host_id,)).fetchone()
        if not host:
            return None
        if host["status"] == "dead":
            raise PlanConflict("host_dead", "cannot change drain state on a dead host")
        if expected_version is not None and host["version"] != expected_version:
            raise PlanConflict(
                "version_conflict", "host changed; re-read before changing drain state"
            )
        admitted = host["admission_state"] == "admitted"
        status = "draining" if draining else "active" if admitted else "pending"
        administrative = (
            "draining" if draining and admitted else "admitted" if admitted else "pending"
        )
        host = cur.execute(
            """UPDATE hosts SET status=%s,administrative_state=%s,
                availability_state=%s,version=version+1,
                payload=(payload-'draining_since') || %s::jsonb || CASE WHEN %s
                    THEN jsonb_build_object('draining_since',EXTRACT(EPOCH FROM clock_timestamp()))
                    ELSE '{}'::jsonb END
            WHERE host_id=%s RETURNING *""",
            (
                status,
                administrative,
                "ready" if status == "active" else "unknown",
                Jsonb(dict(host_id=host_id, status=status)),
                draining,
                host_id,
            ),
        ).fetchone()
    append_event(
        conn,
        aggregate_type="host",
        aggregate_id=host_id,
        event_type="host.v1.status_changed",
        payload=dict(status=status),
        idempotency_key=f'host_status:{host_id}:{host["version"]}',
    )
    return {
        **dict(host["payload"] or {}),
        **{
            key: host[key]
            for key in (
                "host_id",
                "status",
                "version",
                "admission_state",
                "admission_version",
                "admission_decision_id",
                "administrative_state",
                "availability_state",
            )
        },
        "admitted": admitted,
    }


def set_location(conn, *, host_id, country, province, expected_version, confirmed_by):
    """Update declared geography without changing observations or admission.

    Placement takes this same host-row lock and rechecks geography under it.
    Existing allocations prevent a location change; telemetry cannot overwrite
    the subsequently confirmed location.
    """
    from host_metadata import require_location

    location = require_location(country, province)
    with conn.cursor(row_factory=dict_row) as cur:
        host = cur.execute("SELECT * FROM hosts WHERE host_id=%s FOR UPDATE", (host_id,)).fetchone()
        if host is None:
            return None
        if host["version"] != expected_version:
            raise PlanConflict(
                "version_conflict", "Host changed. Reload its details before saving the location."
            )
        changed = any(str(host.get(field) or "") != value for field, value in location.items())
        if (
            changed
            and cur.execute(
                """SELECT 1 FROM (
                SELECT job_id FROM gpu_device_allocations WHERE host_id=%s AND released_at IS NULL
                UNION SELECT job_id FROM jobs WHERE host_id=%s AND phase IN ('scheduled','starting','running')
            ) AS active_workloads LIMIT 1""",
                (host_id, host_id),
            ).fetchone()
        ):
            raise PlanConflict(
                "host_has_workloads",
                "This host has active workloads. Save its new location after they finish.",
            )
        host = cur.execute(
            """UPDATE hosts SET country=%s,province=%s,region=%s,version=version+1,
                payload=COALESCE(payload,'{}'::jsonb) || %s::jsonb || jsonb_build_object(
                    'location_confirmed_at',EXTRACT(EPOCH FROM clock_timestamp()),
                    'location_confirmed_by',%s::text)
                WHERE host_id=%s RETURNING *""",
            (
                location["country"],
                location["province"],
                location["region"],
                Jsonb(location),
                confirmed_by,
                host_id,
            ),
        ).fetchone()
        cur.execute(
            "UPDATE gpu_offers SET region=%s,province=%s WHERE host_id=%s",
            (location["region"], location["province"], host_id),
        )
    append_event(
        conn,
        aggregate_type="host",
        aggregate_id=host_id,
        event_type="host.v1.location_changed",
        payload=location,
        idempotency_key=f'host_location:{host_id}:{host["version"]}',
    )
    return {
        **dict(host["payload"] or {}),
        **{
            field: host[field]
            for field in (
                "host_id",
                "status",
                "version",
                "country",
                "province",
                "region",
                "administrative_state",
                "availability_state",
                "admission_state",
                "admission_version",
                "admission_decision_id",
            )
        },
        "admitted": host["admission_state"] == "admitted",
    }
