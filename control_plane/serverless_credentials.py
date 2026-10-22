"""Opaque callback credentials; no fleet or customer credential enters a job."""

import hashlib
import re
import secrets

from psycopg.rows import dict_row

PREFIX = "xsw_"
_CALLBACK = re.compile(
    r"/(?:api/v2|agent/v2)/serverless/workers/[^/]+/"
    r"(?:heartbeat|jobs/claim|jobs/[^/]+(?:/(?:complete|events))?)"
)


def callback_path(path: str, method: str) -> bool:
    if not _CALLBACK.fullmatch(path):
        return False
    if method == "GET":
        return bool(re.fullmatch(r"/(?:api/v2|agent/v2)/serverless/workers/[^/]+/jobs/[^/]+", path))
    return method == "POST" and bool(
        re.fullmatch(
            r"/(?:api/v2|agent/v2)/serverless/workers/[^/]+/(?:heartbeat|jobs/claim|jobs/[^/]+/(?:complete|events))",
            path,
        )
    )


def mint_callback_credential(conn, *, job_id, attempt_id, host_id, fencing_token):
    """Mint only during a successful lease claim; persist the digest, not secret."""
    with conn.cursor(row_factory=dict_row) as cur:
        worker = cur.execute(
            """SELECT w.worker_id,l.expires_at,e.owner_id AS tenant_id FROM serverless_workers w
                 JOIN serverless_endpoints e ON e.endpoint_id=w.endpoint_id
                 JOIN jobs j ON j.job_id=w.scheduler_job_id
                 JOIN job_attempts a ON a.attempt_id=w.attempt_id
                 JOIN placement_leases l ON l.attempt_id=a.attempt_id
                WHERE j.job_id=%s AND a.attempt_id=%s AND a.host_id=%s AND a.fencing_token=%s
                  AND j.active_attempt_id=a.attempt_id AND j.host_id=a.host_id
                  AND w.host_id=a.host_id AND w.released_at IS NULL
                  AND e.owner_id<>''
                  AND COALESCE(j.spec,j.payload)->>'job_type'='serverless_worker'
                  AND COALESCE(j.spec,j.payload)->>'serverless_worker_id'=w.worker_id
                  AND COALESCE(j.spec,j.payload)->>'serverless_endpoint_id'=w.endpoint_id
                  AND l.status='active' AND l.host_id=a.host_id AND l.fencing_token=a.fencing_token""",
            (job_id, attempt_id, host_id, fencing_token),
        ).fetchone()
        if worker is None:
            return None
        token = PREFIX + secrets.token_urlsafe(32)
        cur.execute(
            "INSERT INTO serverless_attempt_credentials(attempt_id,worker_id,tenant_id,token_hash,expires_at) "
            "VALUES(%s,%s,%s,%s,%s)",
            (
                attempt_id,
                worker["worker_id"],
                worker["tenant_id"],
                hashlib.sha256(token.encode()).hexdigest(),
                worker["expires_at"],
            ),
        )
        return token


def authenticate_callback(conn, token: str, worker_id: str) -> bool:
    if not token.startswith(PREFIX) or len(token) > 128:
        return False
    row = conn.execute(
        """SELECT 1 FROM serverless_attempt_credentials c
             JOIN serverless_workers w ON w.worker_id=c.worker_id AND w.attempt_id=c.attempt_id
             JOIN serverless_endpoints e ON e.endpoint_id=w.endpoint_id AND e.owner_id=c.tenant_id
             JOIN job_attempts a ON a.attempt_id=c.attempt_id AND a.host_id=w.host_id
             JOIN jobs j ON j.job_id=a.job_id AND j.active_attempt_id=a.attempt_id
             JOIN placement_leases l ON l.attempt_id=a.attempt_id AND l.job_id=j.job_id
            WHERE c.token_hash=%s AND c.worker_id=%s AND c.revoked_at IS NULL
              AND c.expires_at>clock_timestamp() AND w.released_at IS NULL
              AND w.scheduler_job_id=j.job_id AND j.host_id=a.host_id
              AND a.status IN ('lease_claimed','starting','running')
              AND COALESCE(j.spec,j.payload)->>'job_type'='serverless_worker'
              AND COALESCE(j.spec,j.payload)->>'serverless_worker_id'=w.worker_id
              AND COALESCE(j.spec,j.payload)->>'serverless_endpoint_id'=w.endpoint_id
              AND l.host_id=a.host_id AND l.fencing_token=a.fencing_token
              AND l.status='active' AND l.expires_at>clock_timestamp()""",
        (hashlib.sha256(token.encode()).hexdigest(), worker_id),
    ).fetchone()
    return row is not None


def prune_callback_credentials(conn, *, limit: int = 500) -> int:
    """Remove stale digests after a day; retain live attempts and job history."""
    result = conn.execute(
        """DELETE FROM serverless_attempt_credentials
        WHERE attempt_id IN (
            SELECT c.attempt_id FROM serverless_attempt_credentials c
            JOIN job_attempts a ON a.attempt_id=c.attempt_id
            WHERE c.expires_at<clock_timestamp()-interval '1 day'
              AND a.status IN ('succeeded','failed','cancelled','lost','fenced','preempted')
            ORDER BY c.expires_at LIMIT %s FOR UPDATE OF c SKIP LOCKED
        )""",
        (limit,),
    )
    return result.rowcount
