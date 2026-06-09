"""Unified spot pricing — single source of truth for live spot rates.

Platform catalog (gpu_pricing), supply/demand dynamics, and provider floors
are combined here. History is persisted to spot_price_history (PostgreSQL).
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("xcelsior.spot_pricing")

SPOT_UPDATE_INTERVAL = int(os.environ.get("XCELSIOR_SPOT_UPDATE_INTERVAL", "600"))

# In-memory cache for environments without PG or between DB reads.
_latest_quotes: dict[str, "SpotQuote"] = {}
_history_cache: list[dict[str, Any]] = []


@dataclass
class SpotQuote:
    gpu_model: str
    rate_cad: float
    on_demand_cad: float
    savings_pct: float
    supply: int
    demand: int
    spot_cents: int
    provider_floor_cents: int
    as_of: float

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "gpu_model": self.gpu_model,
            "spot_cents": self.spot_cents,
            "rate_cad": round(self.rate_cad, 4),
            "on_demand_cad": round(self.on_demand_cad, 4),
            "savings_pct": self.savings_pct,
            "supply": self.supply,
            "demand": self.demand,
            "provider_floor_cents": self.provider_floor_cents,
            "recorded_at": self.as_of,
        }


def demand_factor(demand: int, supply: int) -> float:
    """Capped surge multiplier component (max +50%)."""
    if supply <= 0:
        return 0.5
    return min(0.5, demand / (supply * 2))


def apply_surge(base_cad: float, demand: int, supply: int) -> float:
    """Apply supply/demand surge to a CAD/hr base rate."""
    if base_cad <= 0:
        return 0.0
    multiplier = 1.0 + demand_factor(demand, supply)
    return round(base_cad * multiplier, 4)


def _pg_available() -> bool:
    try:
        from db import _get_pg_pool

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.execute("SELECT 1 FROM spot_price_history LIMIT 0")
        return True
    except Exception:
        return False


def get_platform_rate(
    gpu_model: str,
    pricing_mode: str = "spot",
    tier: str = "standard",
) -> float | None:
    """Look up canonical platform rate from gpu_pricing."""
    if not gpu_model:
        return None
    try:
        from db import pg_connection

        with pg_connection() as conn:
            row = conn.execute(
                """SELECT base_rate_cad FROM gpu_pricing
                   WHERE gpu_model = %s AND tier = %s AND pricing_mode = %s
                     AND active = TRUE
                   ORDER BY vram_gb ASC
                   LIMIT 1""",
                (gpu_model, tier, pricing_mode),
            ).fetchone()
        if row:
            return float(row[0])
    except Exception as exc:
        log.debug("gpu_pricing lookup failed for %s/%s: %s", gpu_model, pricing_mode, exc)
    return None


def get_platform_spot_base(
    gpu_model: str,
    tier: str = "standard",
    **_kwargs: Any,
) -> float | None:
    return get_platform_rate(gpu_model, pricing_mode="spot", tier=tier)


def get_platform_on_demand_base(gpu_model: str, tier: str = "standard") -> float | None:
    return get_platform_rate(gpu_model, pricing_mode="on_demand", tier=tier)


def _host_supply_by_gpu() -> dict[str, int]:
    from scheduler import load_hosts

    supply: dict[str, int] = defaultdict(int)
    for host in load_hosts(active_only=True):
        gpu = (host.get("gpu_model") or "unknown").strip()
        supply[gpu] += max(1, int(host.get("num_gpus") or 1))
    return dict(supply)


def _host_min_cost_by_gpu() -> dict[str, float]:
    from scheduler import load_hosts

    mins: dict[str, float] = {}
    for host in load_hosts(active_only=True):
        gpu = (host.get("gpu_model") or "unknown").strip()
        cost = float(host.get("cost_per_hour") or 0.20)
        if gpu not in mins or cost < mins[gpu]:
            mins[gpu] = cost
    return mins


def _marketplace_supply_by_gpu() -> dict[str, int]:
    if not _pg_available():
        return {}
    try:
        from db import pg_connection

        with pg_connection() as conn:
            rows = conn.execute(
                """SELECT gpu_model, SUM(gpu_count_available) AS supply
                   FROM gpu_offers
                   WHERE available = TRUE AND spot_enabled = TRUE
                   GROUP BY gpu_model""",
            ).fetchall()
        return {r[0]: int(r[1] or 0) for r in rows if r[0]}
    except Exception as exc:
        log.debug("marketplace supply query failed: %s", exc)
        return {}


def _provider_floor_cents_by_gpu() -> dict[str, int]:
    if not _pg_available():
        return {}
    try:
        from db import pg_connection

        with pg_connection() as conn:
            rows = conn.execute(
                """SELECT gpu_model, MAX(spot_min_cents) AS floor_cents
                   FROM gpu_offers
                   WHERE available = TRUE AND spot_enabled = TRUE
                   GROUP BY gpu_model""",
            ).fetchall()
        return {r[0]: int(r[1] or 0) for r in rows if r[0]}
    except Exception as exc:
        log.debug("provider floor query failed: %s", exc)
        return {}


def _job_demand_by_gpu() -> dict[str, int]:
    from scheduler import load_hosts, load_jobs

    hosts = load_hosts(active_only=False)
    host_map = {h["host_id"]: h for h in hosts}
    host_supply = _host_supply_by_gpu()
    offer_supply = _marketplace_supply_by_gpu()
    supply_gpus = set(host_supply) | set(offer_supply)

    demand: dict[str, int] = defaultdict(int)
    for job in load_jobs():
        if job.get("status") not in ("running", "queued"):
            continue
        gm = (job.get("gpu_model") or "").strip()
        if gm:
            demand[gm] += 1
            continue
        host_id = job.get("host_id")
        if host_id and host_id in host_map:
            gpu = (host_map[host_id].get("gpu_model") or "unknown").strip()
            demand[gpu] += 1
        elif supply_gpus:
            for gpu in supply_gpus:
                demand[gpu] += 1
    return dict(demand)


def get_supply_demand(gpu_model: str) -> tuple[int, int]:
    """Merged supply and demand for a GPU model."""
    host_supply = _host_supply_by_gpu()
    offer_supply = _marketplace_supply_by_gpu()
    supply = host_supply.get(gpu_model, 0) + offer_supply.get(gpu_model, 0)
    demand = _job_demand_by_gpu().get(gpu_model, 0)
    return supply, demand


def _catalog_gpu_models() -> set[str]:
    models: set[str] = set()
    try:
        from db import pg_connection

        with pg_connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT gpu_model FROM gpu_pricing
                   WHERE active = TRUE AND pricing_mode = 'spot'""",
            ).fetchall()
        models.update(r[0] for r in rows if r[0])
    except Exception:
        pass
    models.update(_host_supply_by_gpu())
    models.update(_marketplace_supply_by_gpu())
    models.update(_job_demand_by_gpu())
    return {m for m in models if m and m != "unknown"}


def compute_live_spot_quote(
    gpu_model: str,
    tier: str = "standard",
    *,
    supply: int | None = None,
    demand: int | None = None,
) -> SpotQuote:
    """Compute a live spot quote for one GPU model."""
    if supply is None or demand is None:
        supply, demand = get_supply_demand(gpu_model)

    on_demand_cad = get_platform_on_demand_base(gpu_model, tier=tier)
    host_mins = _host_min_cost_by_gpu()
    if on_demand_cad is None:
        on_demand_cad = host_mins.get(gpu_model, 0.20)

    spot_base = get_platform_spot_base(gpu_model, tier=tier)
    if spot_base is None:
        spot_base = round(on_demand_cad * 0.4, 4)

    rate_cad = apply_surge(spot_base, demand, supply)
    rate_cad = min(rate_cad, on_demand_cad)

    floors = _provider_floor_cents_by_gpu()
    floor_cents = floors.get(gpu_model, 0)
    floor_cad = floor_cents / 100.0
    if floor_cad > 0:
        rate_cad = max(rate_cad, floor_cad)
        rate_cad = min(rate_cad, on_demand_cad)

    spot_cents = max(1, int(round(rate_cad * 100)))
    rate_cad = spot_cents / 100.0

    savings_pct = 0.0
    if on_demand_cad > 0:
        savings_pct = round(max(0.0, (1.0 - rate_cad / on_demand_cad) * 100.0), 1)

    now = time.time()
    return SpotQuote(
        gpu_model=gpu_model,
        rate_cad=rate_cad,
        on_demand_cad=on_demand_cad,
        savings_pct=savings_pct,
        supply=supply,
        demand=demand,
        spot_cents=spot_cents,
        provider_floor_cents=floor_cents,
        as_of=now,
    )


def record_spot_history(quote: SpotQuote) -> None:
    """Persist one quote row to spot_price_history."""
    global _history_cache
    entry = {
        "gpu_model": quote.gpu_model,
        "price": quote.rate_cad,
        "spot_cents": quote.spot_cents,
        "supply": quote.supply,
        "demand": quote.demand,
        "computed_at": quote.as_of,
    }
    _history_cache.append(entry)
    if len(_history_cache) > 1000:
        _history_cache = _history_cache[-1000:]

    if not _pg_available():
        return
    try:
        from db import pg_connection

        with pg_connection() as conn:
            conn.execute(
                """INSERT INTO spot_price_history
                   (gpu_model, clearing_price_cents, supply_count, demand_count, recorded_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    quote.gpu_model,
                    quote.spot_cents,
                    quote.supply,
                    quote.demand,
                    quote.as_of,
                ),
            )
    except Exception as exc:
        log.warning("spot_price_history insert failed: %s", exc)


def update_all_spot_prices() -> dict[str, float]:
    """Recalculate all spot prices and record history. Returns {gpu_model: rate_cad}."""
    global _latest_quotes
    out: dict[str, float] = {}
    for gpu_model in sorted(_catalog_gpu_models()):
        quote = compute_live_spot_quote(gpu_model)
        record_spot_history(quote)
        _latest_quotes[gpu_model] = quote
        out[gpu_model] = quote.rate_cad

    if out:
        log.info("SPOT PRICES: %s", {k: f"${v:.4f}/hr" for k, v in out.items()})
    return out


def get_current_spot_prices() -> dict[str, float]:
    """Latest spot rate per GPU model in CAD/hr."""
    if _pg_available():
        try:
            from db import pg_connection

            with pg_connection() as conn:
                rows = conn.execute(
                    """SELECT DISTINCT ON (gpu_model) gpu_model, clearing_price_cents
                       FROM spot_price_history
                       ORDER BY gpu_model, recorded_at DESC""",
                ).fetchall()
            if rows:
                return {r[0]: int(r[1]) / 100.0 for r in rows}
        except Exception as exc:
            log.debug("spot_price_history read failed: %s", exc)

    if _latest_quotes:
        return {k: q.rate_cad for k, q in _latest_quotes.items()}

    # Cold start: compute live without persisting.
    return {gpu: compute_live_spot_quote(gpu).rate_cad for gpu in _catalog_gpu_models()}


def get_current_spot_prices_list() -> list[dict[str, Any]]:
    """API-friendly list of current spot quotes."""
    prices = get_current_spot_prices()
    if _latest_quotes:
        return [_latest_quotes[gpu].to_api_dict() for gpu in sorted(_latest_quotes)]
    return [
        compute_live_spot_quote(gpu).to_api_dict()
        for gpu in sorted(prices)
    ]


def get_spot_price_history(
    gpu_model: str,
    hours: int = 24,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Historical spot prices for charts."""
    cutoff = time.time() - (hours * 3600)
    if _pg_available():
        try:
            from db import pg_connection

            with pg_connection() as conn:
                rows = conn.execute(
                    """SELECT gpu_model, clearing_price_cents, recorded_at
                       FROM spot_price_history
                       WHERE gpu_model = %s AND recorded_at >= %s
                       ORDER BY recorded_at DESC LIMIT %s""",
                    (gpu_model, cutoff, limit),
                ).fetchall()
            return [
                {
                    "gpu_model": r[0],
                    "spot_cents": int(r[1]),
                    "recorded_at": r[2],
                }
                for r in rows
            ]
        except Exception as exc:
            log.debug("spot history read failed: %s", exc)

    return [
        {
            "gpu_model": e["gpu_model"],
            "spot_cents": e.get("spot_cents", int(round(e["price"] * 100))),
            "recorded_at": e["computed_at"],
        }
        for e in reversed(_history_cache)
        if e.get("gpu_model") == gpu_model and e.get("computed_at", 0) >= cutoff
    ][:limit]


def get_in_memory_history() -> list[dict[str, Any]]:
    """Test/compat helper — recent in-memory history entries."""
    return list(_history_cache)


def lock_spot_rate_for_job(job: dict[str, Any], host_gpu_model: str | None = None) -> float:
    """Return locked spot CAD/hr for assignment; persists on job dict."""
    gpu_model = (
        (host_gpu_model or "").strip()
        or (job.get("gpu_model") or "").strip()
        or (job.get("host_gpu_model") or "").strip()
    )
    if not gpu_model:
        gpu_model = "RTX 4090"
    quote = compute_live_spot_quote(gpu_model)
    job["spot_rate_cad"] = quote.rate_cad
    job["pricing_mode"] = "spot"
    return quote.rate_cad


def compute_spot_price_cents(base_price_cents: int, demand: int, supply: int) -> int:
    """Marketplace-compat cents helper (delegates surge math)."""
    base_cad = base_price_cents / 100.0
    rate_cad = apply_surge(base_cad, demand, supply)
    return max(1, int(round(rate_cad * 100)))