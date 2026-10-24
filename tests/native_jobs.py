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
