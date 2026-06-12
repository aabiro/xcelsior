# Xcelsior — Serverless GPU-seconds metering (control-plane authoritative)
# Billing model matches Novita: worker_running_seconds × unit_price_per_worker_per_second.

from __future__ import annotations

import logging
import math
import os
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo

log = logging.getLogger("xcelsior.serverless.metering")

INPUT_TOKEN_PRICE_CAD_PER_M = float(os.environ.get("XCELSIOR_INPUT_TOKEN_PRICE", "0.50"))
OUTPUT_TOKEN_PRICE_CAD_PER_M = float(os.environ.get("XCELSIOR_OUTPUT_TOKEN_PRICE", "1.50"))
MIN_BILLING_INTERVAL_SEC = int(os.environ.get("XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC", "60"))


def compute_gpu_seconds(allocated_at: float | None, released_at: float | None) -> int:
    """Billable GPU-seconds from control-plane allocation span (§8)."""
    if allocated_at is None:
        return 0
    end = released_at if released_at is not None else time.time()
    return max(0, int(math.ceil(end - allocated_at)))


def worker_rate_cad_per_second(rate_per_hour_cad: float, gpu_count: int = 1) -> float:
    """Novita-style unit price: CAD per second for one worker (incl. gpu_count)."""
    if rate_per_hour_cad <= 0:
        return 0.0
    return round((rate_per_hour_cad / 3600.0) * max(1, gpu_count), 8)


def pricing_for_endpoint(ep: dict) -> dict[str, Any]:
    """Quote for UI/API — Novita-aligned ¢/s/worker alongside $/hr."""
    rate_hr = get_gpu_rate_per_hour(
        str(ep.get("gpu_tier") or ""),
        str(ep.get("region") or "ca-east"),
    )
    gpu_count = int(ep.get("gpu_count") or 1)
    rate_sec = worker_rate_cad_per_second(rate_hr, gpu_count)
    return {
        "billing_model": "worker_running_seconds",
        "rate_per_hour_cad": rate_hr,
        "rate_per_second_cad_per_worker": rate_sec,
        "rate_cents_per_second_per_worker": round(rate_sec * 100.0, 4),
        "gpu_count": gpu_count,
        "formula": "cost = running_seconds × rate_cents_per_second_per_worker",
    }


def estimate_cost_cad(gpu_seconds: int, rate_per_hour_cad: float, gpu_count: int = 1) -> float:
    if gpu_seconds <= 0 or rate_per_hour_cad <= 0:
        return 0.0
    return round((gpu_seconds / 3600.0) * rate_per_hour_cad * max(1, gpu_count), 6)


def estimate_worker_cost_for_duration_sec(
    duration_sec: float,
    rate_per_hour_cad: float,
    gpu_count: int = 1,
) -> float:
    """Estimate cost for N seconds of one worker running (Novita calculator)."""
    return estimate_cost_cad(max(0, int(math.ceil(duration_sec))), rate_per_hour_cad, gpu_count)


def token_cost_metadata(input_tokens: int, output_tokens: int) -> dict:
    """Observability-only token costs — never debited as primary meter."""
    in_cost = (input_tokens / 1_000_000.0) * INPUT_TOKEN_PRICE_CAD_PER_M
    out_cost = (output_tokens / 1_000_000.0) * OUTPUT_TOKEN_PRICE_CAD_PER_M
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_cad": round(in_cost, 6),
        "output_cost_cad": round(out_cost, 6),
        "total_token_cost_cad": round(in_cost + out_cost, 6),
    }


def get_gpu_rate_per_hour(gpu_tier: str, region: str) -> float:
    """Lookup host/GPU offer rate — mirrors inference engine pricing."""
    if not gpu_tier:
        return 0.0
    from host_metadata import normalize_region

    region = normalize_region(region)
    from db import _get_pg_pool
    from psycopg.rows import dict_row

    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        row = conn.execute(
            """
            SELECT MIN(ask_cents_per_hour) AS price FROM gpu_offers
            WHERE gpu_model = %s AND region = %s AND available = true
            """,
            (gpu_tier, region),
        ).fetchone()
        if row and row.get("price"):
            return round(float(row["price"]) / 100.0, 4)
        row = conn.execute(
            """
            SELECT MIN((payload->>'cost_per_hour')::float) AS price FROM hosts
            WHERE payload->>'gpu_model' = %s AND status = 'active'
            """,
            (gpu_tier,),
        ).fetchone()
        if row and row.get("price"):
            return round(float(row["price"]), 4)
    log.warning("No pricing data for GPU %s in %s", gpu_tier, region)
    return 0.0


def _billing_job_id(worker: dict) -> str:
    return str(worker.get("scheduler_job_id") or worker.get("worker_id") or "")


def last_billed_period_end(job_id: str) -> float | None:
    if not job_id:
        return None
    from db import _get_pg_pool
    from psycopg.rows import dict_row

    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        row = conn.execute(
            """
            SELECT period_end FROM billing_cycles
            WHERE job_id = %s AND resource_type IN ('serverless_gpu', 'serverless_gpu_cold_start')
            ORDER BY period_end DESC LIMIT 1
            """,
            (job_id,),
        ).fetchone()
    if row and row.get("period_end") is not None:
        return float(row["period_end"])
    return None


def charge_serverless_execution(
    billing_engine,
    repo: ServerlessRepo,
    worker: dict,
    endpoint: dict,
    *,
    period_end: float | None = None,
    final: bool = False,
    resource_type: str = "serverless_gpu",
    description: str | None = None,
) -> dict:
    """
    Debit wallet for an unbilled worker uptime slice (incremental; no double-charge).

    Novita: Worker cost = running_duration_seconds × unit_price_per_second.
    """
    owner_id = str(endpoint.get("owner_id") or "")
    job_id = _billing_job_id(worker)
    now = time.time()
    period_end = float(period_end if period_end is not None else now)
    last_end = last_billed_period_end(job_id)
    period_start = last_end if last_end is not None else float(worker.get("allocated_at") or period_end)
    duration_sec = period_end - period_start

    if duration_sec <= 0:
        return {
            "charged": False,
            "gpu_seconds": 0,
            "amount_cad": 0.0,
            "reason": "zero_duration",
        }
    if not final and duration_sec < MIN_BILLING_INTERVAL_SEC:
        return {
            "charged": False,
            "gpu_seconds": int(duration_sec),
            "amount_cad": 0.0,
            "reason": "below_minimum_interval",
        }

    rate = get_gpu_rate_per_hour(
        str(endpoint.get("gpu_tier") or ""),
        str(endpoint.get("region") or "ca-east"),
    )
    gpu_count = int(endpoint.get("gpu_count") or 1)
    gpu_seconds = max(0, int(math.ceil(duration_sec)))
    amount_cad = estimate_cost_cad(gpu_seconds, rate, gpu_count)
    if amount_cad <= 0 or not owner_id:
        return {
            "charged": False,
            "gpu_seconds": gpu_seconds,
            "amount_cad": 0.0,
            "reason": "zero_amount",
        }

    desc = description or (
        f"Serverless worker: {endpoint.get('name') or endpoint.get('endpoint_id')} "
        f"({gpu_seconds}s @ {rate:.2f}/hr)"
    )
    charge_result = billing_engine.charge(
        owner_id,
        amount_cad,
        job_id=job_id,
        description=desc,
    )

    cycle_id = f"BC-slvr-{int(now)}-{uuid.uuid4().hex[:6]}"
    status = "charged" if charge_result.get("charged") else "failed"

    from db import _get_pg_pool
    from psycopg.rows import dict_row

    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        conn.execute(
            """
            INSERT INTO billing_cycles
                (cycle_id, job_id, customer_id, host_id, resource_type,
                 period_start, period_end, duration_seconds, rate_per_hour,
                 gpu_model, tier, tier_multiplier, amount_cad, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                cycle_id,
                job_id,
                owner_id,
                str(worker.get("host_id") or ""),
                resource_type,
                period_start,
                period_end,
                gpu_seconds,
                rate,
                str(endpoint.get("gpu_tier") or "serverless"),
                "on-demand",
                1.0,
                amount_cad,
                status,
                now,
            ),
        )
        conn.commit()

    if charge_result.get("charged"):
        repo.increment_endpoint_totals(
            str(endpoint["endpoint_id"]),
            gpu_seconds=gpu_seconds,
            cost_cad=amount_cad,
        )

    return {
        "charged": bool(charge_result.get("charged")),
        "gpu_seconds": gpu_seconds,
        "amount_cad": amount_cad,
        "cycle_id": cycle_id,
        "period_start": period_start,
        "period_end": period_end,
        "balance_cad": charge_result.get("balance_cad"),
    }


def record_cold_start_line_item(
    billing_engine,
    repo: ServerlessRepo,
    worker: dict,
    endpoint: dict,
    *,
    cold_start_seconds: int,
    ready_at: float,
) -> dict:
    """
    Metadata-only cold-start billing cycle row (amount included in worker uptime;
    separate line item for disputes/UI, flagged resource_type).
    """
    if cold_start_seconds <= 0:
        return {"recorded": False, "reason": "zero_cold_start"}
    job_id = _billing_job_id(worker)
    alloc = float(worker.get("allocated_at") or ready_at)
    rate = get_gpu_rate_per_hour(
        str(endpoint.get("gpu_tier") or ""),
        str(endpoint.get("region") or "ca-east"),
    )
    gpu_count = int(endpoint.get("gpu_count") or 1)
    amount_cad = estimate_cost_cad(cold_start_seconds, rate, gpu_count)
    owner_id = str(endpoint.get("owner_id") or "")
    now = time.time()
    cycle_id = f"BC-slvr-cs-{int(now)}-{uuid.uuid4().hex[:6]}"

    from db import _get_pg_pool
    from psycopg.rows import dict_row

    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        conn.execute(
            """
            INSERT INTO billing_cycles
                (cycle_id, job_id, customer_id, host_id, resource_type,
                 period_start, period_end, duration_seconds, rate_per_hour,
                 gpu_model, tier, tier_multiplier, amount_cad, status, created_at)
            VALUES (%s, %s, %s, %s, 'serverless_gpu_cold_start', %s, %s, %s, %s, %s, %s, %s, 0, 'metadata', %s)
            """,
            (
                cycle_id,
                job_id,
                owner_id,
                str(worker.get("host_id") or ""),
                alloc,
                ready_at,
                cold_start_seconds,
                rate,
                str(endpoint.get("gpu_tier") or "serverless"),
                "on-demand",
                1.0,
                now,
            ),
        )
        conn.commit()

    return {
        "recorded": True,
        "cycle_id": cycle_id,
        "cold_start_seconds": cold_start_seconds,
        "estimated_cad": amount_cad,
        "note": "included_in_worker_uptime",
    }


def bill_active_serverless_workers(billing_engine, *, now: float | None = None) -> int:
    """Periodic tick: bill incremental slices for all running serverless workers."""
    from db import _get_pg_pool
    from psycopg.rows import dict_row
    from serverless.repo import ServerlessRepo

    now = now or time.time()
    repo = ServerlessRepo()
    billed = 0
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        rows = conn.execute(
            """
            SELECT w.*, e.endpoint_id, e.owner_id, e.name, e.gpu_tier, e.region, e.gpu_count
            FROM serverless_workers w
            JOIN serverless_endpoints e ON e.endpoint_id = w.endpoint_id
            WHERE w.state IN ('booting', 'ready', 'idle', 'draining')
              AND w.allocated_at IS NOT NULL
              AND e.deleted_at = 0
            """,
        ).fetchall()

    for row in rows:
        worker = dict(row)
        endpoint = {
            "endpoint_id": worker["endpoint_id"],
            "owner_id": worker["owner_id"],
            "name": worker.get("name"),
            "gpu_tier": worker.get("gpu_tier"),
            "region": worker.get("region"),
            "gpu_count": worker.get("gpu_count"),
        }
        try:
            result = charge_serverless_execution(
                billing_engine,
                repo,
                worker,
                endpoint,
                period_end=now,
                final=False,
            )
            if result.get("charged"):
                billed += 1
        except Exception as e:
            log.error(
                "Serverless billing error for %s: %s",
                worker.get("worker_id"),
                e,
            )
    return billed


def charge_serverless_worker(
    repo: ServerlessRepo,
    worker: dict,
    endpoint: dict,
    *,
    released_at: float | None = None,
) -> dict:
    """Final slice on worker deprovision — only unbilled seconds since last cycle."""
    from billing import get_billing_engine

    return charge_serverless_execution(
        get_billing_engine(),
        repo,
        worker,
        endpoint,
        period_end=released_at,
        final=True,
        description=(
            f"Serverless worker final: {endpoint.get('name') or endpoint.get('endpoint_id')}"
        ),
    )
