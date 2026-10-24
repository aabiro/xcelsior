"""Immutable serverless customer prices and exact native execution evidence."""

import math

from control_plane.job_specs import validate_execution_spec

QUOTE_VERSION = "serverless-billing/v1"


def capture_quote(endpoint: dict) -> dict:
    from serverless.metering import get_gpu_rate_per_hour

    count = endpoint.get("gpu_count", 1)
    rate = get_gpu_rate_per_hour(endpoint.get("gpu_tier", ""), endpoint.get("region", ""))
    if (
        type(count) is not int
        or not 1 <= count <= 1024
        or isinstance(rate, bool)
        or not math.isfinite(float(rate))
        or float(rate) <= 0
        or not endpoint.get("owner_id")
        or not endpoint.get("endpoint_id")
    ):
        raise ValueError("serverless admission requires a qualified positive customer price")
    micros = round(float(rate) * 1_000_000)
    if micros <= 0:
        raise ValueError("serverless customer price is below monetary precision")
    return {
        "version": QUOTE_VERSION,
        "owner_id": str(endpoint["owner_id"]),
        "endpoint_id": str(endpoint["endpoint_id"]),
        "gpu_count": count,
        "rate_per_gpu_micros": micros,
        "gpu_tier": str(endpoint.get("gpu_tier") or ""),
        "region": str(endpoint.get("region") or ""),
    }


def require_execution_quote(conn, worker: dict, endpoint: dict) -> dict | None:
    """No legacy worker, observed start, live catalog or caller dict grants billing."""
    row = conn.execute(
        """SELECT j.spec,j.spec_hash,a.spec_hash AS attempt_hash,
                  a.job_generation,j.generation,w.allocated_at,
                  EXTRACT(EPOCH FROM COALESCE(a.ended_at,LEAST(clock_timestamp(),l.expires_at)))::float8 AS execution_end
             FROM serverless_workers w
             JOIN jobs j ON j.job_id=w.scheduler_job_id
             JOIN job_attempts a ON a.attempt_id=w.attempt_id AND a.job_id=j.job_id
             JOIN placement_leases l ON l.attempt_id=a.attempt_id AND l.job_id=j.job_id
                 AND l.host_id=a.host_id AND l.fencing_token=a.fencing_token
            WHERE w.worker_id=%s AND w.endpoint_id=%s AND a.host_id=w.host_id
              AND a.started_at IS NOT NULL AND w.allocated_at>0
              AND w.allocated_at>=EXTRACT(EPOCH FROM a.reserved_at)::float8
              AND w.allocated_at<=EXTRACT(EPOCH FROM a.started_at)::float8
              AND a.started_at<=clock_timestamp()
              AND ((w.released_at IS NULL AND j.active_attempt_id=a.attempt_id
                    AND j.host_id=a.host_id AND a.status IN ('starting','running')
                    AND l.status='active' AND l.expires_at>clock_timestamp())
                   OR (a.ended_at IS NOT NULL AND a.status IN
                        ('succeeded','failed','cancelled','lost','fenced','preempted')
                       AND a.ended_at>=a.started_at AND a.ended_at<=clock_timestamp()
                       AND w.released_at=EXTRACT(EPOCH FROM a.ended_at)::float8))""",
        (worker["worker_id"], endpoint["endpoint_id"]),
    ).fetchone()
    if row is None:
        return None
    spec = validate_execution_spec(row["spec"], row["spec_hash"], require_admission=True)
    quote = spec.get("serverless_billing_quote")
    owner = spec.get("owner_id") or spec.get("owner")
    if (
        row["spec_hash"] != row["attempt_hash"]
        or row["generation"] != row["job_generation"]
        or spec.get("job_type") != "serverless_worker"
        or spec.get("serverless_worker_id") != worker["worker_id"]
        or spec.get("serverless_endpoint_id") != endpoint["endpoint_id"]
        or not isinstance(quote, dict)
        or quote.get("version") != QUOTE_VERSION
        or quote.get("owner_id") != owner
        or owner != endpoint["owner_id"]
        or quote.get("endpoint_id") != endpoint["endpoint_id"]
        or type(quote.get("gpu_count")) is not int
        or quote["gpu_count"] != spec.get("num_gpus")
        or quote["gpu_count"] != worker["gpu_count"]
        or type(quote.get("rate_per_gpu_micros")) is not int
        or quote["rate_per_gpu_micros"] <= 0
    ):
        raise ValueError("serverless billing requires its exact admitted customer quote")
    # Allocation includes startup time, but must agree with native reservation
    # and start evidence. Runtime bounds are separate from the immutable price.
    return {
        **quote,
        "execution_start": float(row["allocated_at"]),
        "execution_end_cap": float(row["execution_end"]),
    }
