"""Offers surface only while their host is live and admitted (Phase 2.5, 2026-11-22).

Prod's marketplace showed H100 and B200 offers at 20 cents/hr from hosts that
were never admitted, next to the two real hosts, and spot supply counted those
offers as GPUs plus every real host twice (its host row and its offer). The
filter is applied when offers are read, so a stale row cannot come back.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from marketplace import LIVE_OFFER_HOST_SQL, MarketplaceEngine  # noqa: E402

pytestmark = pytest.mark.needs_db


def _pool():
    from db import _get_pg_pool

    return _get_pg_pool()


@pytest.fixture
def fleet():
    """One live admitted host, one never admitted, one dead; an offer on each."""
    tag = uuid.uuid4().hex[:8]
    model = f"RTX 3060 live-{tag}"
    now = time.time()
    hosts = {
        "live": (f"live-{tag}", "active", "admitted"),
        "pending": (f"pending-{tag}", "active", "pending"),
        "dead": (f"dead-{tag}", "dead", "admitted"),
    }
    offers = {}
    with _pool().connection() as conn:
        for name, (host_id, status, admission) in hosts.items():
            conn.execute(
                """INSERT INTO hosts (host_id, status, registered_at, payload, admission_state,
                       administrative_state)
                   VALUES (%s, %s, %s, %s::jsonb, %s, %s)""",
                (
                    host_id,
                    status,
                    now,
                    f'{{"host_id": "{host_id}", "gpu_model": "{model}", "num_gpus": 1}}',
                    admission,
                    admission,
                ),
            )
            offer_id = f"offer-{name}-{tag}"
            conn.execute(
                """INSERT INTO gpu_offers (offer_id, provider_id, host_id, gpu_model, vram_gb,
                       gpu_count_total, gpu_count_available, ask_cents_per_hour, available,
                       spot_enabled, spot_min_cents, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, 12, 1, 1, %s, TRUE, TRUE, %s, %s, %s)""",
                (
                    offer_id,
                    f"prov-{tag}",
                    host_id,
                    model,
                    35 if name == "live" else 20,
                    0 if name == "live" else 150,
                    now,
                    now,
                ),
            )
            offers[name] = offer_id
        conn.commit()
    yield {"model": model, "hosts": hosts, "offers": offers}
    with _pool().connection() as conn:
        conn.execute("DELETE FROM gpu_offers WHERE offer_id = ANY(%s)", (list(offers.values()),))
        conn.execute("DELETE FROM hosts WHERE host_id = ANY(%s)", ([h[0] for h in hosts.values()],))
        conn.commit()


def test_search_lists_only_the_live_admitted_host(fleet):
    found = MarketplaceEngine().search_offers(gpu_model=fleet["model"])
    assert [o["offer_id"] for o in found] == [fleet["offers"]["live"]]


def test_a_stale_offer_cannot_be_fetched_or_allocated(fleet):
    me = MarketplaceEngine()
    for name in ("pending", "dead"):
        assert me.get_offer(fleet["offers"][name]) is None
        assert me.allocate_gpu(fleet["offers"][name], f"job-{name}", gpu_count=1) is None
    assert me.get_offer(fleet["offers"]["live"])["offer_id"] == fleet["offers"]["live"]


def test_stale_offers_set_no_price(fleet):
    """The stale offers ask 20 cents against the live host's 35: they must not be the price."""
    me = MarketplaceEngine()
    cents, _ = me.get_effective_rate(f"cust-{uuid.uuid4().hex[:6]}", fleet["model"])
    assert cents == 35


def test_stale_offers_set_no_spot_floor(fleet):
    import spot_pricing

    floors = spot_pricing._provider_floor_cents_by_gpu()
    assert floors.get(spot_pricing._canonical_gpu_model(fleet["model"]), 0) == 0


def test_spot_supply_counts_each_live_host_once(fleet):
    """The live host has a host row and an offer: one GPU, not two. The others: none."""
    import spot_pricing

    supply, _ = spot_pricing.get_supply_demand(fleet["model"])
    assert supply == 1


def test_the_predicate_is_one_constant_every_reader_uses():
    """Search, fetch, allocation, prices, stats, spot floors and serverless all read it."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    marketplace = (root / "billing/marketplace.py").read_text()
    assert (
        marketplace.count("{LIVE_OFFER_HOST_SQL}") + marketplace.count("LIVE_OFFER_HOST_SQL]") >= 6
    )
    for relative in ("billing/spot_pricing.py", "workers/inference.py"):
        assert "LIVE_OFFER_HOST_SQL" in (root / relative).read_text(), relative
    assert "status = 'active'" in LIVE_OFFER_HOST_SQL
    assert "admission_state = 'admitted'" in LIVE_OFFER_HOST_SQL
    assert "_marketplace_supply_by_gpu" not in (root / "billing/spot_pricing.py").read_text()
