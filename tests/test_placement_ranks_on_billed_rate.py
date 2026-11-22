"""Placement and host pages use the set rate, never a host's asking price.

A launch is billed the catalog rate for its GPU (``quote_tier``) and the host is
paid that charge less the platform cut, so a host's registered ``cost_per_hour``
changes neither price nor pay. Placement still ranked and capped on it: a host
that asked less won more jobs at the same pay, and a renter's price cap was
checked against a number nobody is charged. The hosts page showed it as the
host's rate. Each now reads the listed rate.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = Path(__file__).resolve().parents[1]

RATES = {"RTX 3060": 120_000, "RTX 2060": 80_000}


@pytest.fixture
def set_rates(monkeypatch):
    import service_tier_pricing

    def listed(gpu_model, vram_gb=None, mode="on_demand"):
        return RATES.get(gpu_model)

    monkeypatch.setattr(service_tier_pricing, "listed_rate_micros", listed)


def _host(host_id, **over):
    base = {
        "host_id": host_id,
        "status": "active",
        "administrative_state": "admitted",
        "gpu_model": "RTX 3060",
        "total_vram_gb": 12,
        "gpu_count": 1,
        "free_gpu_count": 1,
        "free_vram_gb": 12.0,
        "region": "ca-on",
        "inventory_generation": 1,
        "capabilities": {"egress_isolation": True},
    }
    base.update(over)
    return base


def test_hosts_are_priced_at_the_listed_rate(set_rates):
    from service_tier_pricing import price_hosts_at_listed_rate

    asks_little = _host("a", cost_per_hour=0.01)
    unpriced = _host("b", gpu_model="Mystery GPU", cost_per_hour=0.33)
    priced = price_hosts_at_listed_rate([asks_little, unpriced])
    assert priced[0]["cost_per_hour"] == 0.12
    assert priced[1]["cost_per_hour"] == 0.33  # no catalog row: keeps its own figure
    assert asks_little["cost_per_hour"] == 0.01  # copies, not mutated in place


def test_a_low_ask_no_longer_wins_the_job(set_rates):
    from control_plane.scheduler.scoring import rank_candidates
    from service_tier_pricing import price_hosts_at_listed_rate

    cheap_ask = _host("cheap-ask", cost_per_hour=0.01, reliability_score=0.2)
    reliable = _host("reliable", cost_per_hour=0.50, reliability_score=0.99)
    job = {"job_id": "j1", "gpu_model": "RTX 3060", "num_gpus": 1, "vram_needed_gb": 8.0}
    ranked = rank_candidates(job, price_hosts_at_listed_rate([cheap_ask, reliable]))
    assert ranked[0].host_id == "reliable"
    assert ranked[0].components["cost"] == ranked[1].components["cost"]


def test_a_price_cap_is_checked_against_the_billed_rate(set_rates):
    from control_plane.scheduler.filters import evaluate_host
    from service_tier_pricing import price_hosts_at_listed_rate

    # The ask (0.50) is over the cap; the rate a launch is billed (0.12) is not.
    [host] = price_hosts_at_listed_rate([_host("h", cost_per_hour=0.50)])
    job = {"job_id": "j1", "gpu_model": "RTX 3060", "num_gpus": 1, "vram_needed_gb": 8.0}
    codes = {r.code for r in evaluate_host({**job, "max_price_per_hour": 0.15}, host)}
    assert "price_exceeds_approved" not in codes
    codes = {r.code for r in evaluate_host({**job, "max_price_per_hour": 0.10}, host)}
    assert "price_exceeds_approved" in codes


def test_the_host_page_shows_what_renters_pay_and_what_the_host_keeps(set_rates, monkeypatch):
    from host_metadata import enrich_host_for_api

    monkeypatch.setenv("XCELSIOR_PLATFORM_CUT", "0.20")
    out = enrich_host_for_api(_host("h", cost_per_hour=0.50))
    assert out["renter_rate_per_hour"] == 0.12
    assert out["cost_per_hour"] == out["price_per_hour"] == 0.12
    assert out["host_earns_per_hour"] == 0.096
    assert out["platform_cut_pct"] == 20.0

    unpriced = enrich_host_for_api(_host("u", gpu_model="Mystery GPU", cost_per_hour=0.5))
    assert unpriced["renter_rate_per_hour"] is None
    assert unpriced["host_earns_per_hour"] is None


def test_every_placement_path_prices_at_the_listed_rate():
    snapshot = (ROOT / "control_plane/scheduler/snapshot.py").read_text()
    assert "price_hosts_at_listed_rate(hosts)" in snapshot
    scheduler = (ROOT / "workers/scheduler.py").read_text()
    # allocate, allocate_binpack and allocate_best_host each price their candidates.
    assert scheduler.count("candidates = _priced_at_listed_rate(candidates)") == 3


def test_the_wizard_asks_no_price():
    flow = (ROOT / "api/wizard/src/wizard-flow.ts").read_text()
    assert '"custom-rate"' not in flow
    assert "COMPETITIVE_DISCOUNT" not in flow
