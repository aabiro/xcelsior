"""Routes: marketplace."""

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from routes._deps import (
    _effective_billing_customer_id,
    _get_current_user,
    _require_admin,
    _require_auth,
    _require_platform_worker,
    _require_scope,
)
from host_evidence import attach_evidence, reputation_score
from marketplace import get_marketplace_engine
from scheduler import (
    get_marketplace,
    list_hosts,
    list_rig,
    marketplace_bill,
    marketplace_stats,
    unlist_rig,
)

router = APIRouter()


@router.get("/marketplace/search", tags=["Marketplace"])
def api_marketplace_search(
    gpu_model: str | None = None,
    min_vram: float | None = None,
    max_price: float | None = None,
    min_reputation: int | None = None,
    sort_by: str = "price",
    limit: int = Query(50, ge=1, le=1000),
):
    """Search marketplace listings with filters and sorting."""
    listings = _live_priced_listings(get_marketplace(active_only=True))

    if gpu_model:
        listings = [l for l in listings if gpu_model.lower() in (l.get("gpu_model", "")).lower()]
    if min_vram is not None:
        listings = [l for l in listings if (l.get("vram_gb", 0) or 0) >= min_vram]
    if max_price is not None:
        listings = [l for l in listings if _listing_price(l) <= max_price]
    if min_reputation is not None:
        # Listings never carried `reputation_score`, so any floor above zero
        # returned nothing. The score is the host's evidence (host_evidence).
        listings = [l for l in listings if (reputation_score(l) or 0) >= min_reputation]

    # Sort
    sort_keys = {
        "price": _listing_price,
        "vram": lambda x: -(x.get("vram_gb", 0) or 0),
        "reputation": lambda x: -(reputation_score(x) or 0),
        "score": lambda x: -(x.get("compute_score", 0) or 0),
    }
    if sort_by in sort_keys:
        listings.sort(key=sort_keys[sort_by])

    return {
        "ok": True,
        "total": len(listings),
        "listings": listings[:limit],
        "filters_applied": {
            "gpu_model": gpu_model,
            "min_vram": min_vram,
            "max_price": max_price,
            "min_reputation": min_reputation,
            "sort_by": sort_by,
        },
    }


# ── Model: RigListing ──


class RigListing(BaseModel):
    host_id: str
    gpu_model: str
    vram_gb: float
    price_per_hour: float
    description: str = ""
    owner: str = "anonymous"
    region: str = ""
    province: str = ""


def _require_marketplace_eligible_host(host_id: str) -> dict:
    host = next(
        (entry for entry in list_hosts(active_only=False) if entry.get("host_id") == host_id),
        None,
    )
    if host is None:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    if host.get("admitted") is not True or host.get("status") != "active":
        raise HTTPException(
            status_code=409,
            detail="Host must complete authoritative worker verification before listing",
        )
    return host


@router.post("/marketplace/list", tags=["Marketplace"])
def api_list_rig(rig: RigListing, request: Request):
    """List an active, authoritatively admitted rig on the marketplace."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope

    _require_scope(user, "marketplace:write")
    _require_marketplace_eligible_host(rig.host_id)
    listing = list_rig(
        rig.host_id,
        rig.gpu_model,
        rig.vram_gb,
        rig.price_per_hour,
        rig.description,
        rig.owner,
        rig.region,
        rig.province,
    )
    return {"ok": True, "listing": listing}


@router.delete("/marketplace/{host_id}", tags=["Marketplace"])
def api_unlist_rig(host_id: str, request: Request):
    """Remove a rig from the marketplace."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope

    _require_scope(user, "marketplace:write")
    if not unlist_rig(host_id):
        raise HTTPException(status_code=404, detail=f"Listing {host_id} not found")
    return {"ok": True, "unlisted": host_id}


@router.get("/marketplace", tags=["Marketplace"])
def api_get_marketplace(request: Request, active_only: bool = True):
    """Browse marketplace listings."""
    user = _get_current_user(request) if request else None
    if user:
        from routes._deps import _require_scope

        _require_scope(user, "marketplace:read")
    listings = get_marketplace(active_only=active_only)
    return {"listings": _live_priced_listings(listings) if active_only else listings}


def _live_priced_listings(listings: list[dict]) -> list[dict]:
    """Listings a renter can act on, each with the price a launch is billed.

    A listing shows only while its host is live (status ``active``) and
    admitted, checked when it is read, so a listing left behind by a host that
    died or was never admitted cannot surface. ``price_per_hour`` is the host's
    own ask; a launch is billed the Standard tier price, which is what
    ``price_per_hour_cad`` carries (the same source as marketplace search and
    available GPUs). Each listing carries its host's ``evidence`` badge, and
    while verified-only placement is on a host that is not currently verified
    is not listed (``only_placeable``).
    """
    from money import micros_to_cad
    from service_tier_pricing import listed_rate_lookup

    hosts = {h.get("host_id"): h for h in list_hosts(active_only=False)}
    rate = listed_rate_lookup()
    live = []
    for listing in listings:
        host = hosts.get(listing.get("host_id"))
        if not host or host.get("status") != "active" or host.get("admitted") is not True:
            continue
        micros = rate(str(listing.get("gpu_model") or ""), listing.get("vram_gb"))
        live.append(
            {
                **listing,
                "rate_micros_per_hour": micros,
                "price_per_hour_cad": micros_to_cad(micros) if micros is not None else None,
                "price_source": "standard_tier",
            }
        )
    from host_evidence import only_placeable

    return only_placeable(attach_evidence(live))


def _listing_price(listing: dict) -> float:
    price = listing.get("price_per_hour_cad")
    return float(price) if price is not None else 999.0


@router.post("/marketplace/bill/{job_id}", tags=["Marketplace"])
def api_marketplace_bill(job_id: str, request: Request):
    """Bill a marketplace job — split between host and platform."""
    user = _require_auth(request)
    _require_scope(user, "marketplace:write")
    _require_platform_worker(user)
    result = marketplace_bill(job_id)
    if not result:
        raise HTTPException(status_code=400, detail=f"Could not bill marketplace job {job_id}")
    return {"ok": True, "bill": result}


@router.get("/marketplace/stats", tags=["Marketplace"])
def api_marketplace_stats(request: Request):
    """Marketplace aggregate stats (platform admin only)."""
    _require_admin(request)
    return {"stats": marketplace_stats()}


# ── Model: GPUOfferCreate ──


class GPUOfferCreate(BaseModel):
    host_id: str = Field(min_length=1, max_length=64)
    gpu_model: str = Field(min_length=1, max_length=64)
    gpu_count_total: int = Field(default=1, ge=1, le=64)
    vram_gb: float = Field(default=0, ge=0, le=1024)
    ask_cents_per_hour: int = Field(default=20, ge=1, le=1_000_000)
    region: str = Field(default="", max_length=32)
    spot_enabled: bool = True
    spot_min_cents: int = Field(default=10, ge=0, le=1_000_000)


@router.post("/api/v2/marketplace/offers", tags=["Marketplace v2"])
def api_marketplace_create_offer(body: GPUOfferCreate, request: Request):
    """Create or update a GPU offer on the marketplace."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope

    _require_scope(user, "marketplace:write")
    _require_marketplace_eligible_host(body.host_id)
    me = get_marketplace_engine()
    offer = me.upsert_offer(
        provider_id=user.get("user_id", user.get("email", "")),
        host_id=body.host_id,
        gpu_model=body.gpu_model,
        gpu_count_total=body.gpu_count_total,
        vram_gb=body.vram_gb,
        ask_cents_per_hour=body.ask_cents_per_hour,
        region=body.region,
        spot_enabled=body.spot_enabled,
        spot_min_cents=body.spot_min_cents,
    )
    return {"ok": True, "offer": offer}


# ── Model: MarketplaceSearchParams ──


class MarketplaceSearchParams(BaseModel):
    gpu_model: str = Field(default="", max_length=64)
    min_vram_gb: float = Field(default=0, ge=0, le=1024)
    max_price_cents: int = Field(default=0, ge=0)
    region: str = Field(default="", max_length=32)
    spot_available: bool = False
    # The MCP tool sent this from the start; the model had no such field, so
    # pydantic dropped it and every host matched.
    min_reputation: float | None = Field(default=None, ge=0, le=10_000)
    verified_only: bool = False
    sort_by: str = Field(default="price", pattern="^(price|reputation|region|vram)$")
    limit: int = Field(default=50, ge=1, le=500)


@router.post("/api/v2/marketplace/search", tags=["Marketplace v2"])
def api_marketplace_search_v2(body: MarketplaceSearchParams):
    """Search available GPU offers with filters.

    Suffixed because the v1 handler above is also called
    `api_marketplace_search`, and the second `def` shadowed the first at module
    scope. Both routes worked — the decorator holds each function object — but
    `UNTESTED_ENDPOINTS.md` scores coverage by whether a handler's *name*
    appears under `tests/`, so a single test naming `api_marketplace_search`
    marked both routes tested when only one was.
    """
    me = get_marketplace_engine()
    # Every live offer, then filter and sort here: the price, reputation and
    # verification a renter filters on are not gpu_offers columns, and a SQL
    # LIMIT before them would cut the list in the wrong order.
    offers = me.search_offers(
        gpu_model=body.gpu_model,
        min_vram_gb=body.min_vram_gb,
        region=body.region,
        spot_available=body.spot_available,
        limit=500,
    )
    _with_listed_price(offers)
    attach_evidence(offers)
    from host_evidence import only_placeable

    # With verified-only placement on, an offer on an unverified host cannot
    # be launched, so it is not offered.
    offers = only_placeable(offers)
    if body.max_price_cents > 0:
        # Against the price a renter pays, not the host's ask.
        ceiling = body.max_price_cents * 10_000
        offers = [
            o
            for o in offers
            if o["rate_micros_per_hour"] is not None and o["rate_micros_per_hour"] <= ceiling
        ]
    if body.min_reputation is not None:
        offers = [o for o in offers if (reputation_score(o) or 0) >= body.min_reputation]
    if body.verified_only:
        offers = [o for o in offers if o["evidence"]["verification"] == "verified"]
    sort_keys = {
        "price": lambda o: (o["rate_micros_per_hour"] is None, o["rate_micros_per_hour"] or 0),
        "reputation": lambda o: -(reputation_score(o) or 0),
        "region": lambda o: str(o.get("region") or ""),
        "vram": lambda o: -(o.get("vram_gb") or 0),
    }
    offers.sort(key=sort_keys[body.sort_by])
    offers = offers[: body.limit]
    return {"ok": True, "offers": offers, "count": len(offers)}


@router.get("/api/v2/hosts/{host_id}/evidence", tags=["Marketplace v2"])
def api_host_evidence(host_id: str, request: Request):
    """One host's evidence: verification, benchmark history, reputation, job success.

    The same read model the listings badge from (``host_evidence``), in full,
    with the dates and verdicts of up to ten earlier benchmark runs. Signed-in
    users; nothing here is private to the host (no de-verification reason,
    hold or GPU identity).
    """
    user = _require_auth(request)
    _require_scope(user, "marketplace:read")
    from db import pg_connection

    with pg_connection() as conn:
        known = conn.execute("SELECT 1 FROM hosts WHERE host_id = %s", (host_id,)).fetchone()
    if not known:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    from host_evidence import evidence_for_hosts

    evidence = evidence_for_hosts([host_id], history=10).get(host_id)
    if evidence is None:
        raise HTTPException(status_code=503, detail="Host evidence is unavailable")
    return {"ok": True, **evidence}


def _with_listed_price(offers: list[dict]) -> None:
    """Every offer carries the price a renter pays, from the one catalog source.

    ``ask_cents_per_hour`` is the host's own ask. It is not what a launch is
    billed: that comes from the Standard tier price (``listed_rate_micros``),
    so it is the number shown as ``price_per_hour_cad``.
    """
    from money import micros_to_cad
    from service_tier_pricing import listed_rate_lookup

    rate = listed_rate_lookup()
    for offer in offers:
        micros = rate(str(offer.get("gpu_model") or ""), offer.get("vram_gb"))
        offer["rate_micros_per_hour"] = micros
        offer["price_per_hour_cad"] = micros_to_cad(micros) if micros is not None else None
        offer["price_source"] = "standard_tier"


@router.get("/api/v2/marketplace/spot-prices", tags=["Marketplace v2"])
def api_marketplace_spot_prices():
    """Get current spot prices for all GPU models."""
    me = get_marketplace_engine()
    prices = me.get_current_spot_prices_list()
    return {"ok": True, "spot_prices": prices}


@router.get("/api/v2/marketplace/spot-prices/{gpu_model}/history", tags=["Marketplace v2"])
def api_marketplace_spot_history(gpu_model: str, hours: int = 24):
    """Get spot price history for a GPU model."""
    me = get_marketplace_engine()
    history = me.get_spot_price_history(gpu_model, hours=hours)
    return {"ok": True, "gpu_model": gpu_model, "history": history}


@router.get("/api/v2/marketplace/stats", tags=["Marketplace v2"])
def api_marketplace_stats_v2():
    """Get marketplace aggregate statistics."""
    me = get_marketplace_engine()
    stats = me.get_marketplace_stats()
    return {"ok": True, **stats}


# ── Model: ReservationCreate ──


class ReservationCreate(BaseModel):
    gpu_model: str
    gpu_count: int = 1
    period_months: int = 1


@router.post("/api/v2/marketplace/reservations", tags=["Marketplace v2"])
def api_marketplace_create_reservation(body: ReservationCreate, request: Request):
    """Create a reserved instance commitment."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope

    _require_scope(user, "marketplace:write")
    me = get_marketplace_engine()
    try:
        res = me.create_reservation(
            customer_id=user.get("user_id", user.get("email", "")),
            gpu_model=body.gpu_model,
            gpu_count=body.gpu_count,
            period_months=body.period_months,
        )
        return {"ok": True, "reservation": res}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/api/v2/marketplace/reservations/{reservation_id}", tags=["Marketplace v2"])
def api_marketplace_cancel_reservation(reservation_id: str, request: Request):
    """Cancel a reserved instance commitment early.

    Computes early termination fee: remaining_months * monthly_rate * 50%.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope

    _require_scope(user, "marketplace:write")
    me = get_marketplace_engine()
    result = me.cancel_reservation(
        reservation_id=reservation_id,
        customer_id=user.get("user_id", user.get("email", "")),
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"ok": True, **result}


# ── Model: AllocateGPURequest ──


class AllocateGPURequest(BaseModel):
    offer_id: str
    job_id: str
    gpu_count: int = 1
    spot: bool = False


@router.post("/api/v2/marketplace/allocate", tags=["Marketplace v2"])
def api_marketplace_allocate(body: AllocateGPURequest, request: Request):
    """Allocate GPUs from an offer for a job. Atomic — prevents double-sell."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope

    _require_scope(user, "marketplace:write")
    me = get_marketplace_engine()
    alloc = me.allocate_gpu(
        body.offer_id,
        body.job_id,
        body.gpu_count,
        allocation_type="spot" if body.spot else "on_demand",
    )
    if not alloc:
        raise HTTPException(409, "Offer not available or insufficient GPUs")
    return {"ok": True, "allocation": alloc}


@router.post("/api/v2/marketplace/release/{allocation_id}", tags=["Marketplace v2"])
def api_marketplace_release(allocation_id: str, request: Request):
    """Release a GPU allocation (job completed/failed)."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope

    _require_scope(user, "marketplace:write")
    me = get_marketplace_engine()
    # `release_allocation` takes a **job id** and queries `WHERE job_id = %s`,
    # so passing an allocation id matched nothing and this returned `ok: true`
    # having done nothing at all. Correcting the lookup alone would have made it
    # work *and* made it releasable by anyone holding `marketplace:write`, since
    # the table had no owner — which is why migration 101 came first.
    released = me.release_allocation_by_id(
        allocation_id, owner_id=_effective_billing_customer_id(user)
    )
    if not released:
        # Not-found rather than forbidden: a foreign allocation and a
        # non-existent one must be indistinguishable, or this becomes a way to
        # enumerate other tenants' allocation ids. Already-released lands here
        # too, which is correct — there was nothing to release.
        raise HTTPException(404, "No such active allocation")
    return {"ok": True, "allocation_id": allocation_id, "released": True}
