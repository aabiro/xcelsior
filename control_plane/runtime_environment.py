"""Decrypt execution secrets only within a successful fenced lease claim.

The immutable spec and command retain ciphertext. Plaintext travels in the
authenticated claim response and is never saved in the attempt or outbox.
"""

from control_plane.job_specs import validate_execution_spec
from control_plane.leases import LeaseClaimRejected


def claimed_environment(conn, grant) -> dict:
    row = conn.execute(
        """SELECT j.spec,j.spec_hash FROM jobs j
           JOIN job_attempts a ON a.attempt_id=j.active_attempt_id AND a.job_id=j.job_id
           JOIN placement_leases l ON l.attempt_id=a.attempt_id AND l.job_id=j.job_id
           WHERE j.job_id=%s AND a.attempt_id=%s AND a.host_id=%s
             AND j.host_id=a.host_id AND a.fencing_token=%s
             AND a.spec_hash=j.spec_hash AND a.job_generation=j.generation
             AND l.lease_id=%s AND l.host_id=a.host_id AND l.fencing_token=a.fencing_token
             AND l.status='active' AND l.expires_at>clock_timestamp()""",
        (grant.job_id, grant.attempt_id, grant.host_id, grant.fencing_token, grant.lease_id),
    ).fetchone()
    if row is None:
        raise LeaseClaimRejected("runtime_environment_authority_missing")
    try:
        spec = validate_execution_spec(row[0], row[1], require_admission=True)
        if spec.get("job_type") != "serverless_worker":
            return {}
        stored = spec.get("environment") or {}
        if not isinstance(stored, dict):
            raise ValueError("invalid environment")
        from serverless.env_secrets import decrypt_env_for_worker

        environment = decrypt_env_for_worker(stored)
    except (ValueError, TypeError):
        raise LeaseClaimRejected("runtime_environment_unavailable") from None
    return {
        "runtime_environment": environment,
        "runtime_environment_spec_hash": row[1],
    }
