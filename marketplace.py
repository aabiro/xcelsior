# Xcelsior GPU Marketplace Engine
# Proper GPU offer/allocation system replacing JSON-file marketplace.
#
# Per REPORT_FEATURE_FINAL.md + REPORT_XCELSIOR_TECHNICAL_FINAL.md:
# - Provider-set ask prices with platform floor pricing
# - Spot pricing via supply/demand curve (e^(k * surge))
# - Reserved instance commitments (1/3/6/12 month terms)
# - Anti-double-sell via gpu_allocations table
# - Spot price history recording for transparency

import logging
import math
import os
import time
import uuid
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("xcelsior.marketplace")

# ── Configuration ─────────────────────────────────────────────────────

SPOT_SENSITIVITY = float(os.environ.get("XCELSIOR_SPOT_K", "0.5"))
SPOT_THRESHOLD = float(os.environ.get("XCELSIOR_SPOT_THRESHOLD", "0.8"))
SPOT_UPDATE_INTERVAL = int(os.environ.get("XCELSIOR_SPOT_INTERVAL_SEC", "600"))  # 10 min
PLATFORM_CUT = float(os.environ.get("XCELSIOR_PLATFORM_CUT", "0.15"))

# Reserved instance discount schedule (Phase 2.3)
RESERVED_DISCOUNTS = {
    1: 0.20,  # 1-month: 20% off
    3: 0.30,  # 3-month: 30% off
    6: 0.40,  # 6-month: 40% off
}


class MarketplaceEngine:
    """GPU marketplace with offers, allocations, spot pricing, and reservations."""

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ── GPU Offers (Provider Listings) ────────────────────────────────

    def upsert_offer(
        self,
        provider_id: str,
        host_id: str,
        gpu_model: str,
        vram_gb: int,
        ask_cents_per_hour: int,
        gpu_count_total: int = 1,
        region: str = "",
        province: str = "",
        spot_multiplier: float = 0.6,
    ) -> dict:
        """Create or update a GPU offer from a provider."""
        now = time.time()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT offer_id FROM gpu_offers WHERE host_id = %s AND gpu_model = %s",
                (host_id, gpu_model),
            ).fetchone()

            if existing:
                offer_id = existing["offer_id"]
                conn.execute(
                    """UPDATE gpu_offers SET
                        ask_cents_per_hour = %s, gpu_count_total = %s,
                        gpu_count_available = %s, vram_gb = %s,
                        spot_multiplier = %s, region = %s, province = %s,
                        available = true, updated_at = %s
                       WHERE offer_id = %s""",
                    (
                        ask_cents_per_hour,
                        gpu_count_total,
                        gpu_count_total,
                        vram_gb,
                        spot_multiplier,
                        region,
                        province,
                        now,
                        offer_id,
                    ),
                )
            else:
                offer_id = f"offer-{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """INSERT INTO gpu_offers
                       (offer_id, provider_id, host_id, gpu_model, gpu_count_total,
                        gpu_count_available, vram_gb, ask_cents_per_hour, spot_multiplier,
                        currency, region, province, available, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'CAD', %s, %s, true, %s, %s)""",
                    (
                        offer_id,
                        provider_id,
                        host_id,
                        gpu_model,
                        gpu_count_total,
                        gpu_count_total,
                        vram_gb,
                        ask_cents_per_hour,
                        spot_multiplier,
                        region,
                        province,
                        now,
                        now,
                    ),
                )

        log.info(
            "GPU offer %s: %s x%d @ %d¢/hr from %s",
            offer_id,
            gpu_model,
            gpu_count_total,
            ask_cents_per_hour,
            provider_id,
        )
        return {
            "offer_id": offer_id,
            "gpu_model": gpu_model,
            "ask_cents_per_hour": ask_cents_per_hour,
        }

    def search_offers(
        self,
        gpu_model: str = "",
        min_vram_gb: int = 0,
        max_price_cents: int = 0,
        region: str = "",
        canada_only: bool = False,
        sort_by: str = "price",  # price, vram, reliability
        limit: int = 50,
    ) -> list[dict]:
        """Search available GPU offers with filters."""
        conditions = ["available = true"]
        params = []

        if gpu_model:
            conditions.append("gpu_model ILIKE %s")
            params.append(f"%{gpu_model}%")
        if min_vram_gb > 0:
            conditions.append("vram_gb >= %s")
            params.append(min_vram_gb)
        if max_price_cents > 0:
            conditions.append("ask_cents_per_hour <= %s")
            params.append(max_price_cents)
        if region:
            conditions.append("region = %s")
            params.append(region)
        if canada_only:
            conditions.append("province != ''")

        where = " AND ".join(conditions)

        order_map = {
            "price": "ask_cents_per_hour ASC",
            "vram": "vram_gb DESC",
            "reliability": "reliability_score DESC",
        }
        order = order_map.get(sort_by, "ask_cents_per_hour ASC")

        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM gpu_offers WHERE {where} ORDER BY {order} LIMIT %s",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def get_offer(self, offer_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM gpu_offers WHERE offer_id = %s",
                (offer_id,),
            ).fetchone()
            return dict(row) if row else None

    def disable_offer(self, offer_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE gpu_offers SET available = false, updated_at = %s WHERE offer_id = %s",
                (time.time(), offer_id),
            )

    def sync_offer_from_host(self, host: dict, provider_id: str = ""):
        """Auto-create/update a gpu_offer from a registered host dict."""
        cost = host.get("cost_per_hour", 0.20)
        self.upsert_offer(
            provider_id=provider_id or host.get("owner", ""),
            host_id=host["host_id"],
            gpu_model=host.get("gpu_model", "unknown"),
            vram_gb=int(host.get("total_vram_gb", 0)),
            ask_cents_per_hour=int(cost * 100),
            region=host.get("region", ""),
            province=host.get("province", ""),
        )

    # ── GPU Allocations (Anti-Double-Sell) ────────────────────────────

    def allocate_gpu(
        self,
        offer_id: str,
        job_id: str,
        gpu_count: int = 1,
        allocation_type: str = "on_demand",
    ) -> Optional[dict]:
        """Allocate GPU(s) from an offer for a job. Returns None if unavailable.

        Uses SELECT ... FOR UPDATE to prevent race conditions.
        """
        now = time.time()
        with self._conn() as conn:
            # Lock the offer row
            offer = conn.execute(
                "SELECT * FROM gpu_offers WHERE offer_id = %s FOR UPDATE",
                (offer_id,),
            ).fetchone()

            if not offer or not offer["available"]:
                return None
            if offer["gpu_count_available"] < gpu_count:
                return None

            # Check no existing allocation for this job
            existing = conn.execute(
                "SELECT allocation_id FROM gpu_allocations WHERE job_id = %s AND released_at = 0",
                (job_id,),
            ).fetchone()
            if existing:
                return {"allocation_id": existing["allocation_id"], "already_allocated": True}

            allocation_id = f"alloc-{uuid.uuid4().hex[:12]}"
            price_cents = offer["ask_cents_per_hour"]

            # Apply spot multiplier for spot allocations
            if allocation_type == "spot":
                price_cents = int(price_cents * offer.get("spot_multiplier", 0.6))

            conn.execute(
                """INSERT INTO gpu_allocations
                   (allocation_id, offer_id, job_id, gpu_count,
                    price_cents_per_hour, allocation_type, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (allocation_id, offer_id, job_id, gpu_count, price_cents, allocation_type, now),
            )

            new_available = offer["gpu_count_available"] - gpu_count
            conn.execute(
                """UPDATE gpu_offers
                   SET gpu_count_available = %s, available = %s, updated_at = %s
                   WHERE offer_id = %s""",
                (new_available, new_available > 0, now, offer_id),
            )

        log.info(
            "GPU allocated: %s job=%s offer=%s %dx @ %d¢/hr (%s)",
            allocation_id,
            job_id,
            offer_id,
            gpu_count,
            price_cents,
            allocation_type,
        )
        return {
            "allocation_id": allocation_id,
            "offer_id": offer_id,
            "job_id": job_id,
            "gpu_count": gpu_count,
            "price_cents_per_hour": price_cents,
            "allocation_type": allocation_type,
        }

    def release_allocation(self, job_id: str):
        """Release GPU allocation when a job completes or is cancelled."""
        now = time.time()
        with self._conn() as conn:
            alloc = conn.execute(
                "SELECT * FROM gpu_allocations WHERE job_id = %s AND released_at = 0 FOR UPDATE",
                (job_id,),
            ).fetchone()
            if not alloc:
                return

            conn.execute(
                "UPDATE gpu_allocations SET released_at = %s WHERE allocation_id = %s",
                (now, alloc["allocation_id"]),
            )

            # Restore availability
            conn.execute(
                """UPDATE gpu_offers
                   SET gpu_count_available = gpu_count_available + %s,
                       available = true, updated_at = %s
                   WHERE offer_id = %s""",
                (alloc["gpu_count"], now, alloc["offer_id"]),
            )
        log.info("GPU released: job=%s allocation=%s", job_id, alloc["allocation_id"])

    # ── Spot Pricing Engine ───────────────────────────────────────────

    def compute_spot_price(
        self,
        base_price_cents: int,
        demand: int,
        supply: int,
        k: float = SPOT_SENSITIVITY,
        threshold: float = SPOT_THRESHOLD,
    ) -> int:
        """Compute spot price in cents using supply/demand curve.

        Formula: spot = base * (1 + demand_factor)
        Demand factor capped: min(0.5, queue_depth / (available_gpus * 2))
        Per Phase 2.2: max 50% surge above base price.
        """
        if supply <= 0:
            return int(base_price_cents * 1.5)  # Cap at 50% surge

        # Demand factor: capped at 0.5 (50% max surge)
        demand_factor = min(0.5, demand / (supply * 2))
        multiplier = 1.0 + demand_factor
        return max(1, int(round(base_price_cents * multiplier)))

    def update_spot_prices(self) -> dict:
        """Recalculate spot prices for all GPU types and record history.

        Returns {gpu_model: price_cents}.
        """
        with self._conn() as conn:
            # Count supply per GPU model
            supply_rows = conn.execute(
                """SELECT gpu_model, SUM(gpu_count_available) as supply,
                          MIN(ask_cents_per_hour) as min_ask
                   FROM gpu_offers WHERE available = true
                   GROUP BY gpu_model""",
            ).fetchall()

            # Count demand per GPU model (running + queued)
            demand_rows = conn.execute(
                """SELECT COALESCE(j.payload->>'gpu_model', 'unknown') as gpu_model, COUNT(*) as demand
                   FROM jobs j
                   WHERE j.status IN ('running', 'queued')
                   GROUP BY j.payload->>'gpu_model'""",
            ).fetchall()

        supply_map = {
            r["gpu_model"]: (int(r["supply"] or 0), int(r["min_ask"] or 20)) for r in supply_rows
        }
        demand_map = {r["gpu_model"]: int(r["demand"] or 0) for r in demand_rows}

        spot_prices = {}
        now = time.time()

        all_gpus = set(supply_map.keys()) | set(demand_map.keys())
        for gpu in all_gpus:
            supply, base_cents = supply_map.get(gpu, (0, 20))
            demand = demand_map.get(gpu, 0)
            price = self.compute_spot_price(base_cents, demand, supply)
            spot_prices[gpu] = price

            # Record to history
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO spot_price_history
                       (gpu_model, clearing_price_cents, supply_count, demand_count, recorded_at)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (gpu, price, supply, demand, now),
                )

        log.info("SPOT PRICES: %s", {k: f"{v}¢" for k, v in spot_prices.items()})
        return spot_prices

    def get_spot_price_history(
        self,
        gpu_model: str,
        hours: int = 24,
        limit: int = 200,
    ) -> list[dict]:
        """Get spot price history for a GPU model."""
        cutoff = time.time() - (hours * 3600)
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT gpu_model, clearing_price_cents, recorded_at
                   FROM spot_price_history
                   WHERE gpu_model = %s AND recorded_at >= %s
                   ORDER BY recorded_at DESC LIMIT %s""",
                (gpu_model, cutoff, limit),
            ).fetchall()
            return [
                {
                    "gpu_model": r["gpu_model"],
                    "spot_cents": r["clearing_price_cents"],
                    "recorded_at": r["recorded_at"],
                }
                for r in rows
            ]

    def get_current_spot_prices(self) -> dict:
        """Get latest spot price per GPU model (dict format for internal use)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT ON (gpu_model) gpu_model, clearing_price_cents
                   FROM spot_price_history
                   ORDER BY gpu_model, recorded_at DESC""",
            ).fetchall()
            return {r["gpu_model"]: r["clearing_price_cents"] for r in rows}

    def get_current_spot_prices_list(self) -> list[dict]:
        """Get latest spot price per GPU model (array format for API)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT ON (gpu_model) gpu_model, clearing_price_cents, recorded_at
                   FROM spot_price_history
                   ORDER BY gpu_model, recorded_at DESC""",
            ).fetchall()
            return [
                {
                    "gpu_model": r["gpu_model"],
                    "spot_cents": r["clearing_price_cents"],
                    "recorded_at": r["recorded_at"],
                }
                for r in rows
            ]

    # ── Reserved Instances ────────────────────────────────────────────

    def create_reservation(
        self,
        customer_id: str,
        gpu_model: str,
        gpu_count: int,
        period_months: int,
    ) -> dict:
        """Create a reserved instance commitment.

        Locks in a discount for a term commitment:
        1-month=10%, 3-month=20%, 6-month=30%, 12-month=40%
        """
        discount_pct = RESERVED_DISCOUNTS.get(period_months)
        if discount_pct is None:
            raise ValueError(
                f"Invalid period: {period_months}. Options: {list(RESERVED_DISCOUNTS.keys())}"
            )

        # Find the current on-demand rate for this GPU
        with self._conn() as conn:
            offer = conn.execute(
                """SELECT MIN(ask_cents_per_hour) as min_price
                   FROM gpu_offers
                   WHERE gpu_model ILIKE %s AND available = true""",
                (f"%{gpu_model}%",),
            ).fetchone()

        base_cents = int(offer["min_price"]) if offer and offer["min_price"] else 20
        discounted_cents = int(base_cents * (1 - discount_pct))
        monthly_rate_cad = round(
            (discounted_cents / 100) * 24 * 30.44 * gpu_count, 2
        )  # ~hours/month

        now = time.time()
        starts_at = now
        ends_at = now + (period_months * 30.44 * 86400)
        reservation_id = f"res-{uuid.uuid4().hex[:12]}"

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reservations
                   (reservation_id, customer_id, gpu_model, gpu_count,
                    period_months, discount_pct, monthly_rate_cad,
                    starts_at, ends_at, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', %s)""",
                (
                    reservation_id,
                    customer_id,
                    gpu_model,
                    gpu_count,
                    period_months,
                    discount_pct,
                    monthly_rate_cad,
                    starts_at,
                    ends_at,
                    now,
                ),
            )

        log.info(
            "RESERVATION %s: %s x%d for %d months @ $%.2f/month (%d%% off)",
            reservation_id,
            gpu_model,
            gpu_count,
            period_months,
            monthly_rate_cad,
            int(discount_pct * 100),
        )
        return {
            "reservation_id": reservation_id,
            "gpu_model": gpu_model,
            "gpu_count": gpu_count,
            "period_months": period_months,
            "discount_pct": discount_pct,
            "monthly_rate_cad": monthly_rate_cad,
            "starts_at": starts_at,
            "ends_at": ends_at,
        }

    def get_customer_reservations(self, customer_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reservations WHERE customer_id = %s ORDER BY created_at DESC",
                (customer_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def cancel_reservation(self, reservation_id: str, customer_id: str) -> dict:
        """Cancel a reservation early and compute termination fee.

        Per Phase 2.3: Early termination fee = remaining_months * monthly_rate * 0.5
        """
        with self._conn() as conn:
            res = conn.execute(
                "SELECT * FROM reservations WHERE reservation_id = %s AND customer_id = %s",
                (reservation_id, customer_id),
            ).fetchone()

        if not res:
            return {"error": "Reservation not found"}
        res = dict(res)

        if res["status"] != "active":
            return {"error": f"Cannot cancel reservation with status '{res['status']}'"}

        now = time.time()
        ends_at = float(res["ends_at"])
        remaining_sec = max(0, ends_at - now)
        remaining_months = remaining_sec / (30 * 24 * 3600)
        monthly_rate = float(res["monthly_rate_cad"])

        # Early termination fee: remaining_months * monthly_rate * 50%
        termination_fee = round(remaining_months * monthly_rate * 0.5, 2)

        with self._conn() as conn:
            conn.execute(
                """UPDATE reservations
                   SET status = 'cancelled', updated_at = %s
                   WHERE reservation_id = %s""",
                (now, reservation_id),
            )

        log.info(
            "RESERVATION %s cancelled: fee=$%.2f (%.1f months remaining @ $%.2f/mo)",
            reservation_id,
            termination_fee,
            remaining_months,
            monthly_rate,
        )

        return {
            "reservation_id": reservation_id,
            "status": "cancelled",
            "remaining_months": round(remaining_months, 1),
            "monthly_rate_cad": monthly_rate,
            "early_termination_fee_cad": termination_fee,
        }

    def expire_reservations(self) -> int:
        """Expire reservations past their end date. Called periodically."""
        now = time.time()
        with self._conn() as conn:
            result = conn.execute(
                "UPDATE reservations SET status = 'expired' WHERE status = 'active' AND ends_at < %s",
                (now,),
            )
            return result.rowcount

    def get_effective_rate(self, customer_id: str, gpu_model: str) -> tuple[int, str]:
        """Get the best rate for a customer, considering reservations.

        Returns (price_cents_per_hour, pricing_type).
        """
        # Check for active reservation
        with self._conn() as conn:
            res = conn.execute(
                """SELECT discount_pct FROM reservations
                   WHERE customer_id = %s AND gpu_model ILIKE %s AND status = 'active'
                   ORDER BY discount_pct DESC LIMIT 1""",
                (customer_id, f"%{gpu_model}%"),
            ).fetchone()

        # Get base price
        with self._conn() as conn:
            offer = conn.execute(
                """SELECT MIN(ask_cents_per_hour) as min_price
                   FROM gpu_offers
                   WHERE gpu_model ILIKE %s AND available = true""",
                (f"%{gpu_model}%",),
            ).fetchone()

        base_cents = int(offer["min_price"]) if offer and offer["min_price"] else 20

        if res:
            discounted = int(base_cents * (1 - res["discount_pct"]))
            return (discounted, "reserved")

        # Check spot price
        spot_prices = self.get_current_spot_prices()
        spot = spot_prices.get(gpu_model)
        if spot and spot < base_cents:
            return (spot, "spot")

        return (base_cents, "on_demand")

    # ── Marketplace Stats ─────────────────────────────────────────────

    def get_marketplace_stats(self) -> dict:
        with self._conn() as conn:
            offers = conn.execute(
                """SELECT
                    COUNT(*) as total_offers,
                    COUNT(*) FILTER (WHERE available = true) as available_offers,
                    SUM(gpu_count_total) as total_gpus,
                    SUM(gpu_count_available) as available_gpus,
                    COUNT(DISTINCT gpu_model) as gpu_models,
                    COUNT(DISTINCT provider_id) as providers,
                    MIN(ask_cents_per_hour) as cheapest_cents,
                    AVG(ask_cents_per_hour) as avg_cents
                   FROM gpu_offers""",
            ).fetchone()

            active_allocs = conn.execute(
                """SELECT COUNT(*) as active,
                          SUM(gpu_count) as gpus_allocated
                   FROM gpu_allocations WHERE released_at = 0""",
            ).fetchone()

            reservations = conn.execute(
                "SELECT COUNT(*) as active FROM reservations WHERE status = 'active'",
            ).fetchone()

        return {
            "total_offers": offers["total_offers"] or 0,
            "available_offers": offers["available_offers"] or 0,
            "total_gpus": int(offers["total_gpus"] or 0),
            "available_gpus": int(offers["available_gpus"] or 0),
            "gpu_models": offers["gpu_models"] or 0,
            "providers": offers["providers"] or 0,
            "cheapest_cad_per_hour": round((offers["cheapest_cents"] or 0) / 100, 2),
            "avg_cad_per_hour": round((offers["avg_cents"] or 0) / 100, 2),
            "active_allocations": active_allocs["active"] or 0,
            "gpus_in_use": int(active_allocs["gpus_allocated"] or 0),
            "active_reservations": reservations["active"] or 0,
            "currency": "CAD",
        }


# ── Singleton ─────────────────────────────────────────────────────────

_marketplace_engine: Optional[MarketplaceEngine] = None


def get_marketplace_engine() -> MarketplaceEngine:
    global _marketplace_engine
    if _marketplace_engine is None:
        _marketplace_engine = MarketplaceEngine()
    return _marketplace_engine
