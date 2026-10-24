"""Attempt-fenced runtime metadata; observations cannot choose an authority."""

import hashlib
import json

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from control_plane.job_specs import validate_execution_spec
from control_plane.leases import FencingViolation, require_current_fence
from control_plane.outbox import append_event


def record_runtime_callback(conn, *, job_id, host_id, attempt_id, fencing_token, kind, value):
    if kind not in ("http_ports", "auto_launch_ports"):
        raise ValueError("unknown runtime callback")
    with conn.cursor(row_factory=dict_row) as cur:
        job = cur.execute(
            "SELECT host_id,spec,spec_hash FROM jobs WHERE job_id=%s FOR UPDATE", (job_id,)
        ).fetchone()
        if job is None or job["host_id"] != host_id:
            return False
        require_current_fence(
            conn,
            job_id=job_id,
            host_id=host_id,
            attempt_id=attempt_id,
            fencing_token=fencing_token,
        )
        if not cur.execute(
            """SELECT 1 FROM placement_leases WHERE job_id=%s AND attempt_id=%s
            AND host_id=%s AND fencing_token=%s AND status='active'
            AND expires_at>clock_timestamp()""",
            (job_id, attempt_id, host_id, fencing_token),
        ).fetchone():
            raise FencingViolation("runtime callback requires a live claimed lease")
        spec = validate_execution_spec(job["spec"], job["spec_hash"], require_admission=True)
        ports = value["ports"] if kind == "auto_launch_ports" else value
        allowed = (
            set(spec.get("auto_launch") or [])
            if kind == "auto_launch_ports"
            else {str(port) for port in spec.get("exposed_ports") or []}
        )
        if kind == "auto_launch_ports" and any(
            port != {"jupyter": 8888, "vscode": 8443}.get(name) for name, port in ports.items()
        ):
            raise ValueError("runtime service port differs from its admitted launcher")
        if set(ports) - allowed:
            raise ValueError("runtime callback exceeds the admitted services or ports")
        cur.execute(
            "UPDATE jobs SET payload=jsonb_set(COALESCE(payload,'{}'::jsonb),%s,%s::jsonb,true) "
            "WHERE job_id=%s",
            ([kind], Jsonb(value), job_id),
        )
        # This event carries no service credential. Delivery is shared across replicas.
        safe = {"job_id": job_id, "ports": ports}
        if kind == "auto_launch_ports":
            safe["token_sha"] = value["token_sha"]
        digest = hashlib.sha256(json.dumps(safe, sort_keys=True).encode()).hexdigest()
        append_event(
            conn,
            aggregate_type="job",
            aggregate_id=job_id,
            event_type=f"job.v1.{kind}_ready",
            payload=safe,
            idempotency_key=f"runtime:{attempt_id}:{kind}:{digest}",
        )
    return True
