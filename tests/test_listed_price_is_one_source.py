"""Every listing shows the price a launch is billed (Phase 2.5, 2026-11-22).

Prod showed the RTX 2060 at $0.08/hr in available GPUs (the Standard tier price
a launch is billed) and at 50 cents in marketplace search (the host's own ask,
which nothing bills). The dashboard marketplace showed the ask as well. All three
now read ``service_tier_pricing.listed_rate_micros``.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def billed_at_8_cents(monkeypatch):
    import service_tier_pricing

    calls = []

    def listed(gpu_model, vram_gb=None, mode="on_demand"):
        calls.append((gpu_model, vram_gb, mode))
        return 80_000 if gpu_model == "RTX 2060" else None

    monkeypatch.setattr(service_tier_pricing, "listed_rate_micros", listed)
    return calls


def test_marketplace_search_shows_the_billed_price_not_the_ask(billed_at_8_cents):
    from routes.marketplace import _with_listed_price

    offers = [{"gpu_model": "RTX 2060", "vram_gb": 6, "ask_cents_per_hour": 50}]
    _with_listed_price(offers)
    assert offers[0]["price_per_hour_cad"] == 0.08
    assert offers[0]["rate_micros_per_hour"] == 80_000
    assert offers[0]["ask_cents_per_hour"] == 50  # the host's ask, kept and labelled


def test_dashboard_listings_show_live_hosts_at_the_billed_price(billed_at_8_cents, monkeypatch):
    import routes.marketplace as route

    monkeypatch.setattr(
        route,
        "list_hosts",
        lambda active_only=False: [
            {"host_id": "live", "status": "active", "admitted": True},
            {"host_id": "pending", "status": "active", "admitted": False},
            {"host_id": "dead", "status": "dead", "admitted": True},
        ],
    )
    listings = [
        {"host_id": h, "gpu_model": "RTX 2060", "vram_gb": 6, "price_per_hour": 0.5}
        for h in ("live", "pending", "dead", "gone")
    ]
    shown = route._live_priced_listings(listings)
    assert [listing["host_id"] for listing in shown] == ["live"]
    assert shown[0]["price_per_hour_cad"] == 0.08
    # One catalog read for the whole page, not one per listing.
    assert len(billed_at_8_cents) == 1


def test_an_unpriced_model_reads_as_unpriced(billed_at_8_cents):
    from routes.marketplace import _with_listed_price

    offers = [{"gpu_model": "H100", "vram_gb": 80, "ask_cents_per_hour": 20}]
    _with_listed_price(offers)
    assert offers[0]["price_per_hour_cad"] is None


def test_every_listing_route_reads_the_one_source():
    for relative in ("api/routes/marketplace.py", "api/routes/gpu.py"):
        text = (ROOT / relative).read_text()
        assert "listed_rate_lookup" in text, relative
    # gpu.py used to call tier_catalog itself.
    assert "tier_catalog(" not in (ROOT / "api/routes/gpu.py").read_text()


@pytest.mark.needs_db
def test_the_listed_price_is_the_price_a_launch_is_quoted():
    from service_tier_pricing import listed_rate_micros, quote_tier
    from tests._db_helpers import price_test_gpu

    model = price_test_gpu(f"RTX 2060 listed-{uuid.uuid4().hex[:6]}", vram_gb=6)
    quote = quote_tier(model, "on_demand", tier="standard", vram_gb=6)
    assert listed_rate_micros(model, 6) == int(quote["unit_amount_micros"])
    assert listed_rate_micros(f"no-such-gpu-{uuid.uuid4().hex[:6]}") is None
