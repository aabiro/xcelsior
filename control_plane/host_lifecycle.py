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
