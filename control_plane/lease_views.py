"""Tenant-safe diagnostics for the current fenced placement lease."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from psycopg import Connection


def current_lease_view(conn: Connection, job_id: str) -> dict | None:
    """Read only the lease bound to the job's current attempt and host.

    An overdue active lease remains visible for diagnostics until the fenced
    expiry controller settles it. This read never grants or renews authority.
    """
    row = conn.execute(
        """
        SELECT l.lease_id,l.job_id,l.attempt_id,l.host_id,l.status,l.offered_at,
               l.claim_deadline,l.claimed_at,l.last_renewed_at,l.expires_at
          FROM placement_leases l
          JOIN jobs j ON j.job_id=l.job_id
                     AND j.active_attempt_id=l.attempt_id AND j.host_id=l.host_id
          JOIN job_attempts a ON a.attempt_id=l.attempt_id AND a.job_id=j.job_id
                             AND a.host_id=l.host_id AND a.fencing_token=l.fencing_token
         WHERE l.job_id=%s AND l.status IN ('offered','active')
           AND a.status IN ('reserved','command_pending','lease_offered',
                            'lease_claimed','starting','running')
         ORDER BY l.offered_at DESC LIMIT 1
        """,
        (job_id,),
    ).fetchone()
    if row is None:
        return None
    keys = (
        "lease_id",
        "job_id",
        "attempt_id",
        "host_id",
        "status",
        "offered_at",
        "claim_deadline",
        "claimed_at",
        "last_renewed_at",
        "expires_at",
    )
    lease: dict[str, Any] = dict(row) if isinstance(row, dict) else dict(zip(keys, row))
    host_id = str(lease.pop("host_id"))
    lease["host_alias"] = f"host-{hashlib.sha256(host_id.encode()).hexdigest()[:10]}"
    for key, value in list(lease.items()):
        if hasattr(value, "isoformat"):
            lease[key] = value.isoformat()
        elif isinstance(value, UUID):
            lease[key] = str(value)
    return lease
