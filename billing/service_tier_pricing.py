"""Stripe-backed compute prices shared by launch quotes and attempt billing.

Stripe owns monetary amounts. Runtime code reads an imported, versioned catalog;
it never contacts Stripe while holding a placement transaction open. Accepted
quotes carry immutable Stripe Price identities into the attempt snapshot.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from money import micros_to_cad

POLICY_VERSION = "service-tiers-v1"
QUEUE_BID_POLICY_VERSION = "tier-bid-aging-v1"
SERVICE_TIERS = {
    "standard": {"label": "Standard", "priority": 1, "scope": "allocated_gpus"},
    "premium": {"label": "Premium", "priority": 2, "scope": "allocated_gpus"},
    "dedicated": {"label": "Dedicated", "priority": 3, "scope": "physical_host"},
}
# Preemptible spot capacity queues and ranks below every purchased tier, so a
# Standard on-demand job can preempt it.
SPOT_PRIORITY = 0
# Used only when provisioning the approved Stripe catalog, never for runtime
# monetary calculations. Runtime additions are the difference between Prices.
APPROVED_PRICE_FACTORS = {
    "premium": Decimal("1.60"),
    "dedicated": Decimal("2.80"),
    "gpu-exclusive": Decimal("0.40"),
}


class TierPricingUnavailable(ValueError):
    pass


class TierFundingRequired(ValueError):
    pass


class TierSchedulingUnavailable(ValueError):
    pass


def normalize_tier(tier: str | None) -> str:
    value = str(tier or "standard").strip().lower()
    if value not in SERVICE_TIERS:
        raise ValueError("Choose Standard, Premium, or Dedicated")
    return value


def stripe_amount_micros(price: dict[str, Any]) -> int:
    """Stripe decimal cents -> exact ledger micro-CAD; refuse unsupported prices."""
    if price.get("currency") != "cad" or price.get("billing_scheme") != "per_unit":
        raise ValueError("Compute prices must use CAD per-unit pricing")
    amount = price.get("unit_amount_decimal")
    if amount is None:
        amount = price.get("unit_amount")
    if amount is None or isinstance(amount, bool):
        raise ValueError("Stripe price has no explicit unit amount")
    micros = Decimal(str(amount)) * 10_000
    if not micros.is_finite() or micros <= 0 or micros != micros.to_integral_value():
        raise ValueError("Stripe price must be positive and representable in micro-CAD")
    return int(micros)


def tier_catalog(
    gpu_model: str,
    mode: str = "on_demand",
    *,
    vram_gb: int | None = None,
    price_id: str | None = None,
    min_vram_gb: float | None = None,
) -> dict[str, Any]:
    """Return all selectable tiers for the same canonical GPU variant.

    Without an exact ``vram_gb`` or ``price_id``, the smallest variant holding
    ``min_vram_gb`` is chosen: reservation binds devices of the priced
    variant's VRAM, so a smaller variant could never place the request.
    """
    from db import pg_connection
    from psycopg.rows import dict_row
    from host_metadata import normalize_gpu_model

    gpu_model = normalize_gpu_model(gpu_model)

    with pg_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            base = cur.execute(
                """SELECT p.gpu_model, p.vram_gb, p.form_factor, p.high_frequency,
                          s.stripe_price_id, s.stripe_product_id, s.lookup_key,
                          s.unit_amount_micros, s.catalog_version
                     FROM gpu_pricing p JOIN stripe_compute_prices s
                       ON s.stripe_price_id=p.stripe_price_id
                    WHERE p.gpu_model=%s AND p.pricing_mode=%s AND p.tier='standard'
                      AND p.active AND s.active
                      AND (%s::integer IS NULL OR p.vram_gb=%s)
                      AND (%s::numeric IS NULL OR p.vram_gb>=%s)
                      AND (%s::text IS NULL OR EXISTS (
                          SELECT 1 FROM gpu_pricing chosen WHERE chosen.stripe_price_id=%s
                           AND chosen.gpu_model=p.gpu_model AND chosen.vram_gb=p.vram_gb
                           AND chosen.form_factor=p.form_factor AND chosen.high_frequency=p.high_frequency
                           AND chosen.pricing_mode=p.pricing_mode AND chosen.active))
                    ORDER BY p.vram_gb, p.high_frequency, p.form_factor LIMIT 1""",
                (gpu_model, mode, vram_gb, vram_gb, min_vram_gb, min_vram_gb, price_id, price_id),
            ).fetchone()
            if base is None:
                raise TierPricingUnavailable("Stripe compute pricing is unavailable for this GPU")
            prefix = str(base["lookup_key"]).rsplit(":", 2)[0]
            mode_key = str(base["lookup_key"]).rsplit(":", 1)[1]
            rows = cur.execute(
                """SELECT stripe_price_id, stripe_product_id, lookup_key,
                          unit_amount_micros, catalog_version
                     FROM stripe_compute_prices
                    WHERE active AND lookup_key=ANY(%s)""",
                (
                    [f"{prefix}:{tier}:{mode_key}" for tier in (*SERVICE_TIERS, "gpu-exclusive")]
                    + [f"{prefix}:queue-bid:all"],
                ),
            ).fetchall()
    by_tier = {str(row["lookup_key"]).rsplit(":", 2)[1]: dict(row) for row in rows}
    base_micros = int(base["unit_amount_micros"])
    tiers = []
    for tier, policy in SERVICE_TIERS.items():
        row = by_tier.get(tier)
        if row is None:
            continue
        amount = int(row["unit_amount_micros"])
        if amount < base_micros:
            raise TierPricingUnavailable("Tier price is below its Standard compute price")
        tiers.append(
            {
                "tier": tier,
                **policy,
                **row,
                "base_rate_micros": base_micros,
                "base_stripe_price_id": base["stripe_price_id"],
                "tier_addition_micros": amount - base_micros,
                "rate_per_gpu_cad": micros_to_cad(amount),
                "tier_addition_per_gpu_cad": micros_to_cad(amount - base_micros),
                "policy_version": POLICY_VERSION,
            }
        )
    addon = by_tier.get("gpu-exclusive")
    return {
        "ok": True,
        "currency": "CAD",
        "gpu_model": gpu_model,
        "mode": mode,
        "vram_gb": base["vram_gb"],
        "form_factor": base["form_factor"],
        "high_frequency": base["high_frequency"],
        "base_rate_per_gpu_cad": micros_to_cad(base_micros),
        "base_rate_micros": base_micros,
        "tiers": tiers,
        "gpu_exclusive_addon": (
            {**addon, "rate_per_gpu_cad": micros_to_cad(addon["unit_amount_micros"])}
            if addon
            else None
        ),
        "queue_bid_unit": by_tier.get("queue-bid"),
        "queue_bid_policy": {
            "version": QUEUE_BID_POLICY_VERSION,
            "aging_interval_seconds": 900,
            "aging_fraction_bps": 1000,
            "max_tier_promotion": 1,
        },
    }


def quote_tier(
    gpu_model: str,
    mode: str = "on_demand",
    *,
    tier: str = "standard",
    num_gpus: int = 1,
    price_id: str | None = None,
    gpu_exclusive: bool = False,
    vram_gb: int | None = None,
    gpu_exclusive_price_id: str | None = None,
    queue_bid_units: int = 0,
    queue_bid_price_id: str | None = None,
    min_vram_gb: float | None = None,
) -> dict[str, Any]:
    tier = normalize_tier(tier)
    if type(num_gpus) is not int or num_gpus < 1:
        raise ValueError("A quote requires a positive GPU count")
    if type(queue_bid_units) is not int or not 0 <= queue_bid_units <= 1000000:
        raise ValueError("A bid requires a whole number of Stripe bid units between 0 and 1000000")
    try:
        catalog = tier_catalog(
            gpu_model, mode, vram_gb=vram_gb, price_id=price_id, min_vram_gb=min_vram_gb
        )
    except TierPricingUnavailable as exc:
        if price_id:
            raise TierPricingUnavailable(
                "The selected Stripe Price changed or its GPU variant is unavailable; refresh the launch quote"
            ) from exc
        raise
    selected = next((item for item in catalog["tiers"] if item["tier"] == tier), None)
    if selected is None:
        raise TierPricingUnavailable("This service tier has no active Stripe Price")
    if price_id and selected["stripe_price_id"] != price_id:
        raise TierPricingUnavailable("The selected Stripe Price changed; refresh the launch quote")
    addon = catalog["gpu_exclusive_addon"] if gpu_exclusive and tier != "dedicated" else None
    if gpu_exclusive and tier != "dedicated" and addon is None:
        raise TierPricingUnavailable("GPU exclusivity has no active Stripe Price")
    if gpu_exclusive_price_id and (not addon or addon["stripe_price_id"] != gpu_exclusive_price_id):
        raise TierPricingUnavailable(
            "The GPU exclusivity Stripe Price changed; refresh the launch quote"
        )
    addon_micros = int(addon["unit_amount_micros"]) if addon else 0
    bid_unit = catalog["queue_bid_unit"] if queue_bid_units else None
    if queue_bid_units and not bid_unit:
        raise TierPricingUnavailable("Queue bidding has no active Stripe unit Price")
    if queue_bid_price_id and (not bid_unit or bid_unit["stripe_price_id"] != queue_bid_price_id):
        raise TierPricingUnavailable("The queue bid Stripe Price changed; refresh the launch quote")
    bid_micros = int(bid_unit["unit_amount_micros"]) * queue_bid_units if bid_unit else 0
    rate_micros = (int(selected["unit_amount_micros"]) + addon_micros + bid_micros) * num_gpus
    next_tier = next(
        (item for item in catalog["tiers"] if item["priority"] == selected["priority"] + 1), None
    )
    if queue_bid_units and selected["priority"] < 3 and next_tier is None:
        raise TierPricingUnavailable("Queue aging requires the next tier's Stripe Price")
    return {
        **selected,
        "currency": "CAD",
        "gpu_model": catalog["gpu_model"],
        "mode": mode,
        "num_gpus": num_gpus,
        "vram_gb": catalog["vram_gb"],
        "form_factor": catalog["form_factor"],
        "high_frequency": catalog["high_frequency"],
        "gpu_exclusive": bool(gpu_exclusive or tier == "dedicated"),
        "host_exclusive": tier == "dedicated",
        "gpu_exclusive_price_id": addon["stripe_price_id"] if addon else None,
        "gpu_exclusive_rate_micros": addon_micros,
        "queue_bid_policy_version": QUEUE_BID_POLICY_VERSION,
        "queue_bid_units": queue_bid_units,
        "queue_bid_unit_micros": int(bid_unit["unit_amount_micros"]) if bid_unit else 0,
        "queue_bid_micros": bid_micros,
        "queue_bid_price_id": bid_unit["stripe_price_id"] if bid_unit else None,
        "next_tier_price_id": next_tier["stripe_price_id"] if next_tier else None,
        "next_tier_gap_micros": (
            int(next_tier["unit_amount_micros"]) - int(selected["unit_amount_micros"])
            if next_tier
            else None
        ),
        "total_rate_micros": rate_micros,
        "total_per_hour_cad": micros_to_cad(rate_micros),
    }


def validate_tier_snapshot(snapshot: dict[str, Any]) -> int:
    """Validate a server-written quote and return its exact hourly ledger rate."""
    if snapshot.get("policy_version") != POLICY_VERSION or snapshot.get("currency") != "CAD":
        raise ValueError("Invalid service-tier pricing policy or currency")
    tier = normalize_tier(snapshot.get("tier"))
    if not str(snapshot.get("stripe_price_id", "")).startswith("price_"):
        raise ValueError("A service tier requires its Stripe Price identity")
    count = snapshot.get("num_gpus")
    unit = snapshot.get("unit_amount_micros")
    addon = snapshot.get("gpu_exclusive_rate_micros", 0)
    bid = snapshot.get("queue_bid_micros", 0)
    bid_units = snapshot.get("queue_bid_units", 0)
    bid_unit_micros = snapshot.get("queue_bid_unit_micros", 0)
    total = snapshot.get("total_rate_micros")
    if any(
        type(value) is not int
        for value in (count, unit, addon, total, bid, bid_units, bid_unit_micros)
    ):
        raise ValueError("Service-tier money and GPU counts must be integers")
    if (
        count < 1
        or unit <= 0
        or addon < 0
        or bid < 0
        or bid_units < 0
        or bid_unit_micros < 0
        or bid != bid_units * bid_unit_micros
        or total != (unit + addon + bid) * count
    ):
        raise ValueError("Service-tier price breakdown does not match its total")
    if bool(snapshot.get("host_exclusive")) != (tier == "dedicated"):
        raise ValueError("Dedicated requires physical-host exclusivity")
    if snapshot.get("priority") != SERVICE_TIERS[tier]["priority"]:
        raise ValueError("Queue priority must come from the purchased service tier")
    if addon and (
        not snapshot.get("gpu_exclusive")
        or not str(snapshot.get("gpu_exclusive_price_id", "")).startswith("price_")
    ):
        raise ValueError("GPU exclusivity requires its paid Stripe Price")
    if tier == "dedicated" and (addon or not snapshot.get("gpu_exclusive")):
        raise ValueError("Dedicated includes GPU exclusivity without an extra fee")
    if bid and (
        snapshot.get("queue_bid_policy_version") != QUEUE_BID_POLICY_VERSION
        or not str(snapshot.get("queue_bid_price_id", "")).startswith("price_")
    ):
        raise ValueError("Queue bids require a paid Stripe unit Price and queue policy")
    gap = snapshot.get("next_tier_gap_micros")
    if (
        bid
        and SERVICE_TIERS[tier]["priority"] < 3
        and (type(gap) is not int or gap <= 0 or not snapshot.get("next_tier_price_id"))
    ):
        raise ValueError("Queue aging requires a positive, pinned next-tier price gap")
    return total


def usage_charge_micros(hourly_rate_micros: int, duration_seconds: float) -> int:
    seconds = Decimal(str(duration_seconds))
    if (
        not seconds.is_finite()
        or seconds < 0
        or type(hourly_rate_micros) is not int
        or hourly_rate_micros < 0
    ):
        raise ValueError("Usage requires a finite nonnegative duration and an integer ledger rate")
    return int(
        (Decimal(hourly_rate_micros) * seconds / 3600).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    )


def validate_persisted_quote(conn, snapshot: dict[str, Any]) -> int:
    """Check locked server pricing against immutable source rows, even if retired."""
    from psycopg.rows import dict_row

    total = validate_tier_snapshot(snapshot)
    ids = [
        snapshot["stripe_price_id"],
        snapshot.get("gpu_exclusive_price_id"),
        snapshot.get("queue_bid_price_id"),
        snapshot.get("next_tier_price_id"),
    ]
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT stripe_price_id,unit_amount_micros,lookup_key,stripe_product_id FROM stripe_compute_prices WHERE stripe_price_id=ANY(%s)",
            ([pid for pid in ids if pid],),
        ).fetchall()
    prices = {row["stripe_price_id"]: row for row in rows}
    main = prices.get(ids[0])
    if (
        not main
        or main["unit_amount_micros"] != snapshot["unit_amount_micros"]
        or main["lookup_key"].rsplit(":", 2)[1] != snapshot["tier"]
    ):
        raise ValueError("The service-tier snapshot differs from its immutable Stripe Price")
    if snapshot.get("gpu_exclusive_rate_micros"):
        addon = prices.get(ids[1])
        if (
            not addon
            or addon["unit_amount_micros"] != snapshot["gpu_exclusive_rate_micros"]
            or addon["lookup_key"].rsplit(":", 2)[1] != "gpu-exclusive"
            or addon["stripe_product_id"] != main["stripe_product_id"]
        ):
            raise ValueError("The exclusivity snapshot differs from its immutable Stripe Price")
    if snapshot.get("queue_bid_micros"):
        bid = prices.get(ids[2])
        if (
            not bid
            or bid["unit_amount_micros"] != snapshot["queue_bid_unit_micros"]
            or bid["lookup_key"].rsplit(":", 2)[1] != "queue-bid"
            or bid["stripe_product_id"] != main["stripe_product_id"]
        ):
            raise ValueError("The queue bid differs from its immutable Stripe Price")
    if snapshot.get("next_tier_price_id"):
        next_price = prices.get(ids[3])
        if (
            not next_price
            or next_price["stripe_product_id"] != main["stripe_product_id"]
            or next_price["unit_amount_micros"] - main["unit_amount_micros"]
            != snapshot.get("next_tier_gap_micros")
            or SERVICE_TIERS.get(next_price["lookup_key"].rsplit(":", 2)[1], {}).get("priority")
            != SERVICE_TIERS[snapshot["tier"]]["priority"] + 1
        ):
            raise ValueError("The aging tier threshold differs from its Stripe Prices")
    return total


def fund_tier_submission(
    owner: str, job_id: str, quote: dict[str, Any], *, hold_id: str | None = None
) -> str:
    """Every customer submission funds the exact quote before becoming claimable."""
    from billing import get_billing_engine
    from db import pg_connection
    from psycopg.rows import dict_row
    import time

    required = validate_tier_snapshot(quote)
    with pg_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            prior = cur.execute(
                """SELECT hold_id,amount_micros,job_id FROM wallet_holds
                    WHERE customer_id=%s AND status='held' AND expires_at>%s
                      AND ((%s::uuid IS NOT NULL AND hold_id=%s::uuid)
                        OR (%s::uuid IS NULL AND job_id=%s)) LIMIT 1""",
                (owner, time.time(), hold_id, hold_id, hold_id, job_id),
            ).fetchone()
    billing = get_billing_engine()
    if prior:
        if prior["job_id"] not in (None, "", job_id) or int(prior["amount_micros"]) < required:
            raise TierFundingRequired("Wallet hold does not cover this service-tier quote")
        hold_id = str(prior["hold_id"])
    elif hold_id:
        raise TierFundingRequired("The service-tier funding hold is expired or unavailable")
    else:
        result = billing.create_wallet_hold(
            owner, micros_to_cad(required), job_id=job_id, idempotency_key=f"service-tier:{job_id}"
        )
        if not result.get("held"):
            raise TierFundingRequired(
                "Insufficient available wallet funds for the selected service tier"
            )
        hold_id = str(result["hold_id"])
    billing.link_wallet_hold_to_job(hold_id, job_id)
    # Re-check after idempotent create/link: a concurrent release or an expired
    # replay must not turn into an unfunded queue admission.
    with pg_connection() as conn:
        valid = conn.execute(
            """SELECT 1 FROM wallet_holds WHERE hold_id=%s::uuid AND customer_id=%s
                 AND job_id=%s AND status='held' AND expires_at>%s AND amount_micros>=%s""",
            (hold_id, owner, job_id, time.time(), required),
        ).fetchone()
    if not valid:
        raise TierFundingRequired("The service-tier funding hold is expired or unavailable")
    return hold_id


def refresh_pending_tier_funding(*, scope_gpu_models=None, limit=25) -> None:
    """Renew expired queue holds without changing the accepted price or FIFO time."""
    from db import pg_connection
    from psycopg.rows import dict_row

    with pg_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        jobs = cur.execute(
            """SELECT j.job_id,COALESCE(j.owner_id,j.spec->>'owner') AS owner,
                      j.spec->'service_tier_quote' AS quote
                 FROM jobs j WHERE j.phase='pending' AND j.desired_state='running'
                  AND jsonb_typeof(j.spec->'service_tier_quote')='object'
                  AND (next_schedule_at IS NULL OR next_schedule_at<=clock_timestamp())
                  AND (%s OR lower(COALESCE(j.spec->>'gpu_model',''))=ANY(%s)
                       OR lower(COALESCE(j.spec->>'scheduler',''))='v2')
                  AND NOT EXISTS (SELECT 1 FROM wallet_holds h WHERE h.job_id=j.job_id
                       AND h.customer_id=COALESCE(j.owner_id,j.spec->>'owner')
                       AND h.status='held' AND h.expires_at>EXTRACT(EPOCH FROM clock_timestamp())
                       AND h.amount_micros>=(j.spec->'service_tier_quote'->>'total_rate_micros')::bigint)
                 ORDER BY j.queued_at,j.job_id LIMIT %s""",
            (scope_gpu_models is None, [m.lower() for m in (scope_gpu_models or [])], limit),
        ).fetchall()
    for job in jobs:
        try:
            hold_id = fund_tier_submission(job["owner"], job["job_id"], job["quote"])
            with pg_connection() as conn:
                conn.execute(
                    "UPDATE jobs SET wallet_hold_id=%s::uuid WHERE job_id=%s AND phase='pending'",
                    (hold_id, job["job_id"]),
                )
        except (TierFundingRequired, ValueError):
            with pg_connection() as conn:
                conn.execute(
                    """UPDATE jobs SET reason_code='funding_required',
                              next_schedule_at=clock_timestamp()+interval '30 seconds'
                         WHERE job_id=%s AND phase='pending'""",
                    (job["job_id"],),
                )
