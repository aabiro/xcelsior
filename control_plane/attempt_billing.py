"""Normal fenced-attempt metering, independent of report-only repair policy."""

from __future__ import annotations

import math
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def bound_billing_inputs(spec, spec_hash, attempt_hash, quote, host_id):
    """Recover only native admission and the quote captured with its attempt."""
    from control_plane.job_specs import validate_execution_spec
    from billing import resolve_job_pricing_mode

    spec = dict(validate_execution_spec(spec, spec_hash, require_admission=True))
    if not attempt_hash or attempt_hash != spec_hash or not isinstance(quote, dict):
        raise ValueError("billing admission/attempt quote is missing or mismatched")
    owner = spec.get("owner_id") or spec.get("owner")
    mode = resolve_job_pricing_mode(spec)
    count = spec.get("num_gpus", 1)
    if (
        not owner
        or quote.get("owner") != owner
        or quote.get("pricing_mode") != mode
        or type(count) is not int
        or count < 1
        or quote.get("num_gpus") != count
    ):
        raise ValueError("billing quote does not match admitted identity or resources")
    host = quote.get("host")
    if not isinstance(host, dict) or host.get("host_id") != host_id:
        raise ValueError("billing quote does not match the attempt host")
    if spec.get("service_tier_quote") != quote.get("service_tier_quote"):
        raise ValueError("attempt pricing differs from its immutable admission")
    if spec.get("service_tier_quote"):
        from service_tier_pricing import validate_tier_snapshot

        if spec["service_tier_quote"].get("num_gpus") != count:
            raise ValueError("paid quote differs from admitted GPU count")
        validate_tier_snapshot(spec["service_tier_quote"])
    rate = quote.get("spot_rate_cad") if mode == "spot" else host.get("cost_per_hour")
    if rate is None or isinstance(rate, bool) or not math.isfinite(float(rate)) or float(rate) < 0:
        raise ValueError("billing quote has no explicit valid price")
    if mode == "spot" and quote.get("spot_rate_cad") != spec.get("spot_rate_cad"):
        raise ValueError("spot quote differs from admitted locked price")
    ceiling = spec.get("max_price_per_hour")
    if (
        ceiling is not None
        and float(ceiling) > 0
        and float(rate) * (count if mode == "spot" else 1) > float(ceiling)
    ):
        raise ValueError("billing quote exceeds admitted price ceiling")
    return {
        **spec,
        "owner": owner,
        "pricing_mode": mode,
        "num_gpus": count,
        "spot_rate_cad": quote.get("spot_rate_cad"),
    }, dict(host)


def capture_billing_snapshot(payload: dict, owner_id: str | None, host: dict) -> dict | None:
    """Freeze the server-selected owner and explicit rate under reservation locks."""
    from billing import resolve_job_pricing_mode

    owner = owner_id or payload.get("owner")
    if not owner:
        # Historical/internal proof rows have no billable identity. Do not
        # manufacture an owner or charge; their evidence remains reportable.
        return None
    if owner_id and payload.get("owner") and payload["owner"] != owner_id:
        raise ValueError("job owner does not match its authoritative owner")
    mode = resolve_job_pricing_mode(payload)
    raw_rate = payload.get("spot_rate_cad") if mode == "spot" else host.get("cost_per_hour")
    if raw_rate is None or isinstance(raw_rate, bool):
        raise ValueError("billable placement requires an explicit rate")
    rate = float(raw_rate)
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("billable placement requires a finite nonnegative rate")
    count = payload.get("num_gpus", 1)
    if type(count) is not int or count < 1:
        raise ValueError("billable placement requires a positive GPU count")
    ceiling = payload.get("max_price_per_hour")
    tier_quote = payload.get("service_tier_quote")
    if tier_quote:
        from service_tier_pricing import validate_tier_snapshot
        from money import micros_to_cad

        effective = micros_to_cad(validate_tier_snapshot(tier_quote))
        if tier_quote["num_gpus"] != count:
            raise ValueError("GPU count differs from the funded service-tier quote")
        if tier_quote.get("gpu_model") != host.get("gpu_model"):
            from host_metadata import normalize_gpu_model

            if normalize_gpu_model(tier_quote.get("gpu_model")) != normalize_gpu_model(
                host.get("gpu_model", "")
            ):
                raise ValueError("Host GPU differs from the approved Stripe Price")
    else:
        effective = rate * count if mode == "spot" else rate
    if ceiling is not None and float(ceiling) > 0 and effective > float(ceiling):
        raise ValueError("captured rate exceeds the approved ceiling")
    return {
        "owner": str(owner),
        "pricing_mode": mode,
        "num_gpus": count,
        "spot_rate_cad": rate if mode == "spot" else None,
        "service_tier_quote": dict(tier_quote) if tier_quote else None,
        "host": {
            key: value
            for key, value in host.items()
            if value is not None
            and key
            in (
                "host_id",
                "gpu_model",
                "cost_per_hour",
                "compute_score",
                "country",
                "province",
            )
        },
    }


def sync_attempt_billing(
    conn,
    attempt_id: str,
    *,
    released_hold_id: str | None = None,
    bind_release: bool = False,
) -> None:
    """Use bound server timestamps and the captured rate in the status transaction.

    Existing historical attempts without a quote are never retroactively billed.
    Previously opened meters may still settle from their own captured evidence.

    ``bind_release`` settles one fenced attempt after its lease transaction has
    committed. It releases only ``released_hold_id`` and writes job clock fields
    only while that job is still the closed attempt. A newer placement keeps
    its own hold and clock.
    """
    from billing import get_billing_engine, settle_attempt_meter

    with conn.cursor(row_factory=dict_row) as cur:
        attempt = cur.execute(
            """SELECT a.job_id,a.host_id,a.status,a.placement_explanation,
                      j.spec AS payload,j.spec_hash AS job_spec_hash,a.spec_hash AS attempt_hash,
                      EXTRACT(EPOCH FROM a.started_at)::float8 AS started,
                      EXTRACT(EPOCH FROM a.ended_at)::float8 AS ended
                 FROM job_attempts a JOIN jobs j ON j.job_id=a.job_id
                WHERE a.attempt_id=%s""",
            (attempt_id,),
        ).fetchone()
        if attempt is None:
            raise ValueError("metering attempt vanished from the status transaction")
        metadata: dict[str, Any] = {}
        if attempt["started"] is not None:
            metadata["started_at"] = attempt["started"]
        if attempt["ended"] is not None:
            metadata["completed_at"] = attempt["ended"]
        if metadata and bind_release:
            cur.execute(
                "UPDATE jobs SET payload=COALESCE(payload,'{}'::jsonb)||%s::jsonb "
                "WHERE job_id=%s AND payload IS DISTINCT FROM COALESCE(payload,'{}'::jsonb)||%s::jsonb "
                "AND (active_attempt_id IS NULL OR active_attempt_id=%s::uuid) "
                "AND status IN ('completed','failed','terminated','cancelled','stopped','paused')",
                (Jsonb(metadata), attempt["job_id"], Jsonb(metadata), attempt_id),
            )
        elif metadata:
            cur.execute(
                "UPDATE jobs SET payload=COALESCE(payload,'{}'::jsonb)||%s::jsonb "
                "WHERE job_id=%s AND payload IS DISTINCT FROM COALESCE(payload,'{}'::jsonb)||%s::jsonb",
                (Jsonb(metadata), attempt["job_id"], Jsonb(metadata)),
            )
        if attempt["ended"] is not None and bind_release:
            if released_hold_id:
                cur.execute(
                    "UPDATE wallet_holds SET status='released',"
                    "released_at=EXTRACT(EPOCH FROM clock_timestamp()),"
                    "updated_at=EXTRACT(EPOCH FROM clock_timestamp()) "
                    "WHERE hold_id=%s::uuid AND job_id=%s AND status='held'",
                    (released_hold_id, attempt["job_id"]),
                )
                cur.execute(
                    "UPDATE jobs SET wallet_hold_id=NULL WHERE job_id=%s AND wallet_hold_id=%s::uuid",
                    (attempt["job_id"], released_hold_id),
                )
        elif attempt["ended"] is not None:
            # This is normal terminal settlement, using the already locked job
            # transaction. Do not open a second connection to release its hold.
            cur.execute(
                "UPDATE wallet_holds SET status='released',"
                "released_at=EXTRACT(EPOCH FROM clock_timestamp()),"
                "updated_at=EXTRACT(EPOCH FROM clock_timestamp()) "
                "WHERE job_id=%s AND status='held'",
                (attempt["job_id"],),
            )
            cur.execute(
                "UPDATE jobs SET wallet_hold_id=NULL WHERE job_id=%s AND wallet_hold_id IS NOT NULL",
                (attempt["job_id"],),
            )
        if attempt["started"] is None:
            return
        existing = cur.execute(
            "SELECT meter_id FROM usage_meters WHERE attempt_id=%s FOR UPDATE", (attempt_id,)
        ).fetchone()
        quote = (attempt["placement_explanation"] or {}).get("billing_snapshot")
        meter_id = str(existing["meter_id"]) if existing else None
        if meter_id is None and isinstance(quote, dict):
            try:
                inputs, frozen_host = bound_billing_inputs(
                    attempt["payload"],
                    attempt["job_spec_hash"],
                    attempt["attempt_hash"],
                    quote,
                    attempt["host_id"],
                )
            except (ValueError, TypeError, OverflowError):
                return
            job = {
                **inputs,
                "job_id": attempt["job_id"],
                "attempt_id": attempt_id,
                "host_id": attempt["host_id"],
                "owner": quote["owner"],
                "started_at": attempt["started"],
                "pricing_mode": quote["pricing_mode"],
                "num_gpus": quote["num_gpus"],
                "spot_rate_cad": quote["spot_rate_cad"],
                "service_tier_quote": quote.get("service_tier_quote"),
            }
            meter = get_billing_engine().meter_job(job, frozen_host, conn=conn, active=True)
            meter_id = meter.meter_id
        if meter_id and attempt["ended"] is not None:
            if settle_attempt_meter(conn, attempt_id, meter_id) is None:
                raise ValueError("terminal meter does not match its bound attempt evidence")


def pinned_period_rate(
    conn, job_id: str, owner: str, host_id: str, *, terminal_attempt_id=None
) -> tuple[float, str, float, float]:
    """Qualify the exact native attempt and bound billing to its server times."""
    from billing import resolve_compute_rate_cad

    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            """SELECT j.active_attempt_id,j.spec,j.spec_hash,j.generation,
                a.attempt_id,a.spec_hash AS attempt_hash,a.job_generation,a.host_id,
                a.placement_explanation,a.status,
                EXTRACT(EPOCH FROM a.started_at)::float8 AS started_at,
                EXTRACT(EPOCH FROM a.ended_at)::float8 AS ended_at,
                EXTRACT(EPOCH FROM l.expires_at)::float8 AS lease_end,
                l.status AS lease_status,l.fencing_token AS lease_fence,a.fencing_token,
                EXTRACT(EPOCH FROM clock_timestamp())::float8 AS checked_at
              FROM jobs j JOIN job_attempts a ON a.job_id=j.job_id
                  AND a.attempt_id=COALESCE(%s::uuid,j.active_attempt_id)
              LEFT JOIN placement_leases l ON l.attempt_id=a.attempt_id
                  AND l.job_id=a.job_id AND l.host_id=a.host_id
             WHERE j.job_id=%s""",
            (terminal_attempt_id, job_id),
        ).fetchone()
    if row is None:
        raise ValueError("wallet billing requires its exact native attempt")
    quote = (row["placement_explanation"] or {}).get("billing_snapshot")
    inputs, host = bound_billing_inputs(
        row["spec"], row["spec_hash"], row["attempt_hash"], quote, row["host_id"]
    )
    if (
        inputs["owner"] != owner
        or row["host_id"] != host_id
        or inputs.get("job_type") == "serverless_worker"
    ):
        raise ValueError("wallet billing identity differs from native compute execution")
    started = row["started_at"]
    if started is None or started <= 0 or started > row["checked_at"]:
        raise ValueError("wallet billing requires its server-recorded start")
    if terminal_attempt_id is not None:
        if (
            row["ended_at"] is None
            or row["status"]
            not in ("succeeded", "failed", "cancelled", "lost", "fenced", "preempted")
            or not started <= row["ended_at"] <= row["checked_at"]
        ):
            raise ValueError("final billing requires confirmed native terminal evidence")
        end_cap = row["ended_at"]
    else:
        if (
            row["active_attempt_id"] != row["attempt_id"]
            or row["generation"] != row["job_generation"]
            or row["status"] != "running"
            or row["lease_status"] != "active"
            or row["lease_fence"] != row["fencing_token"]
            or row["lease_end"] is None
        ):
            raise ValueError("periodic billing requires its current claimed native lease")
        end_cap = min(row["checked_at"], row["lease_end"])
    rate, mode = resolve_compute_rate_cad(inputs, host)
    return rate, mode, float(started), float(end_cap)
