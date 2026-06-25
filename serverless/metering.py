# Xcelsior — Serverless GPU-seconds metering (control-plane authoritative)
# Billing model matches Novita: worker_running_seconds × unit_price_per_worker_per_second.

from __future__ import annotations

import logging
import math
import os
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo

log = logging.getLogger("xcelsior.serverless.metering")

# Fallback flat token prices (used only when model size can't be inferred).
INPUT_TOKEN_PRICE_CAD_PER_M = float(os.environ.get("XCELSIOR_INPUT_TOKEN_PRICE", "0.50"))
OUTPUT_TOKEN_PRICE_CAD_PER_M = float(os.environ.get("XCELSIOR_OUTPUT_TOKEN_PRICE", "1.50"))
MIN_BILLING_INTERVAL_SEC = int(os.environ.get("XCELSIOR_SERVERLESS_MIN_BILLING_INTERVAL_SEC", "60"))

# Blended serverless meter: when enabled, a worker's billing slice is charged the
# higher of (GPU-seconds cost, token cost) so low-utilization workers are protected
# and high-throughput workers are priced by tokens. Default OFF — flip on only after
# the dual-write validation period (see token-accrual wiring).
BLENDED_BILLING_ENABLED = os.environ.get("XCELSIOR_SERVERLESS_BLENDED_BILLING", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Size-tiered token pricing (CAD per million tokens), by model parameter count.
# (max_params_b_inclusive, input_price, output_price)
_TOKEN_PRICE_BANDS: list[tuple[float, float, float]] = [
    (9.0, 0.15, 0.45),          # ≤ 9B
    (34.0, 0.35, 1.05),         # 10–34B
    (80.0, 0.70, 2.10),         # 35–80B
    (float("inf"), 1.10, 3.30),  # 80B+ / MoE
]
# Model families that should price at the top band even without an explicit size.
_LARGE_MODEL_MARKERS = ("deepseek", "mixtral", "8x7b", "8x22b", "wizardlm-2-8x")
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


def infer_model_params_b(model_ref: str | None) -> float | None:
    """Best-effort parameter count (billions) parsed from a model id/name.

    Recognizes e.g. ``Llama-3.1-70B``, ``mistral-7b``, ``gemma-2-9b-it``; MoE /
    very large families (DeepSeek, Mixtral) map to the top band. Returns None when
    the size can't be determined, so callers fall back to flat pricing.
    """
    if not model_ref:
        return None
    s = model_ref.lower()
    if any(marker in s for marker in _LARGE_MODEL_MARKERS):
        return float("inf")
    m = _PARAM_RE.search(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def token_prices_for_model(model_ref: str | None) -> tuple[float, float]:
    """(input, output) CAD per million tokens for the model's size band.

    Unknown size → the flat env fallback prices (back-compat).
    """
    params_b = infer_model_params_b(model_ref)
    if params_b is None:
        return INPUT_TOKEN_PRICE_CAD_PER_M, OUTPUT_TOKEN_PRICE_CAD_PER_M
    for max_b, in_price, out_price in _TOKEN_PRICE_BANDS:
        if params_b <= max_b:
            return in_price, out_price
    return INPUT_TOKEN_PRICE_CAD_PER_M, OUTPUT_TOKEN_PRICE_CAD_PER_M


def blended_period_amount(gpu_amount_cad: float, token_cost_cad: float) -> float:
    """Blended meter for a billing slice: the higher of GPU-seconds or token cost."""
    return round(max(float(gpu_amount_cad or 0.0), float(token_cost_cad or 0.0)), 6)


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


def token_cost_metadata(
    input_tokens: int, output_tokens: int, model_ref: str | None = None
) -> dict:
    """Size-tiered token costs for a request. Observability until blended billing
    is enabled, at which point this feeds the blended meter. ``model_ref`` selects
    the size band; omitted → flat fallback prices."""
    in_price, out_price = token_prices_for_model(model_ref)
    in_cost = (input_tokens / 1_000_000.0) * in_price
    out_cost = (output_tokens / 1_000_000.0) * out_price
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price_cad_per_m": in_price,
        "output_price_cad_per_m": out_price,
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
    token_cost_cad: float = 0.0,
) -> dict:
    """
    Debit wallet for an unbilled worker uptime slice (incremental; no double-charge).

    Novita: Worker cost = running_duration_seconds × unit_price_per_second.

    When ``BLENDED_BILLING_ENABLED``, the slice is charged the higher of the
    GPU-seconds cost and the token cost accrued for the period (``token_cost_cad``);
    both are always recorded for dual-write validation before the flag is flipped.
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
    gpu_amount_cad = estimate_cost_cad(gpu_seconds, rate, gpu_count)
    # Blended meter: charge the higher of GPU-seconds vs. token cost (flag-gated).
    blended_cad = blended_period_amount(gpu_amount_cad, token_cost_cad)
    amount_cad = blended_cad if BLENDED_BILLING_ENABLED else gpu_amount_cad
    if BLENDED_BILLING_ENABLED and token_cost_cad > 0:
        log.info(
            "blended-meter job=%s gpu=%.6f token=%.6f charged=%.6f",
            job_id, gpu_amount_cad, token_cost_cad, amount_cad,
        )
    if amount_cad <= 0 or not owner_id:
        return {
            "charged": False,
            "gpu_seconds": gpu_seconds,
            "amount_cad": 0.0,
            "gpu_amount_cad": gpu_amount_cad,
            "token_cost_cad": round(float(token_cost_cad or 0.0), 6),
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
        "gpu_amount_cad": gpu_amount_cad,
        "token_cost_cad": round(float(token_cost_cad or 0.0), 6),
        "blended_amount_cad": blended_cad,
        "blended_billing": BLENDED_BILLING_ENABLED,
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
