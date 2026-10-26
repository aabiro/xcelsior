"""Explicit fixture admission for native placement/protocol tests.

Calling this helper is the fixture author's approval of its initial workload.
It does not run automatically, re-admit observations, or qualify history. Tests
of real submission exercise submit_job/DatabaseOps admission directly instead.
"""

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from control_plane.job_specs import canonical_spec_hash, make_execution_spec


def admit_fixture_job(conn, job_id):
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            "SELECT payload,spec,owner_id,tenant_id,team_id FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone()
        if row is None or row["spec"] is not None:
            raise ValueError("fixture approval requires a new unadmitted job")
        approved = dict(row["payload"] or {})
        for field in ("owner_id", "tenant_id", "team_id"):
            if row[field] is not None:
                approved[field] = row[field]
        spec = make_execution_spec(approved)
        cur.execute(
            "UPDATE jobs SET spec=%s,spec_hash=%s WHERE job_id=%s",
            (Jsonb(spec), canonical_spec_hash(spec), job_id),
        )


def insert_schedulable_host(
    conn,
    host_id,
    *,
    model="RTX 4090",
    gpus=1,
    vram_mb=24576,
    payload=None,
    inventory_generation=1,
    observed=True,
):
    """An admitted, freshly observed host with physical (``GPU-``) devices.

    This is the minimum the native scheduler will place on: operator
    admission, ready availability, verified egress isolation, a current
    DB-clock observation and concrete NVML-shaped device identities.
    ``observed=False`` leaves ``last_observed_at`` empty (a host that never
    reported through the v2 observation channel).
    """
    import json as _json
    import time as _time

    conn.execute(
        """INSERT INTO hosts (host_id, status, registered_at, payload,
                              administrative_state, availability_state, admission_state,
                              inventory_generation, capabilities, last_observed_at)
           VALUES (%s, 'active', %s, %s, 'admitted', 'ready', 'admitted', %s,
                   '{"egress_isolation":true}',
                   CASE WHEN %s THEN clock_timestamp() ELSE NULL END)""",
        (
            host_id,
            _time.time(),
            _json.dumps({"admitted": True, "cost_per_hour": 0, **(payload or {})}),
            inventory_generation,
            observed,
        ),
    )
    for index in range(gpus):
        conn.execute(
            """INSERT INTO host_gpu_devices
                   (host_id, gpu_uuid, device_index, model, total_vram_mb,
                    allocatable_vram_mb, health)
               VALUES (%s, %s, %s, %s, %s, %s, 'healthy')""",
            (host_id, f"GPU-{host_id}-{index}", index, model, vram_mb, vram_mb),
        )
    return host_id
