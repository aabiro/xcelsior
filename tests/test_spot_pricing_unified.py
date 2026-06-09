"""Unified spot_pricing service tests."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

import spot_pricing as sp
import scheduler


class TestSpotPricingMath:
    def test_zero_supply_surge_cap(self):
        rate = sp.apply_surge(1.0, demand=5, supply=0)
        assert rate == 1.5

    def test_low_demand_near_base(self):
        rate = sp.apply_surge(1.0, demand=1, supply=10)
        assert 1.0 <= rate <= 1.1

    def test_spot_never_exceeds_on_demand_in_quote(self):
        quote = sp.compute_live_spot_quote(
            "RTX 4090",
            supply=0,
            demand=100,
        )
        assert quote.rate_cad <= quote.on_demand_cad

    def test_provider_floor_respected(self, monkeypatch):
        monkeypatch.setattr(sp, "_provider_floor_cents_by_gpu", lambda: {"RTX 4090": 50})
        quote = sp.compute_live_spot_quote("RTX 4090", supply=100, demand=0)
        assert quote.spot_cents >= 50

    def test_marketplace_cents_helper_matches_surge(self):
        assert sp.compute_spot_price_cents(100, demand=10, supply=10) == 150


class TestSpotPricingIntegration:
    def setup_method(self):
        scheduler.register_host("sp1", "127.0.0.1", "RTX 4090", 24, 24, cost_per_hour=0.55)
        sp._latest_quotes.clear()
        sp._history_cache.clear()

    def test_update_all_populates_cache_and_history(self):
        prices = sp.update_all_spot_prices()
        assert "RTX 4090" in prices
        assert prices["RTX 4090"] > 0
        history = sp.get_in_memory_history()
        assert any(h["gpu_model"] == "RTX 4090" for h in history)

    def test_scheduler_delegates_to_unified_service(self):
        scheduler.update_spot_prices()
        current = scheduler.get_current_spot_prices()
        assert "RTX 4090" in current

    def test_lock_spot_rate_for_job(self):
        job = scheduler.submit_job("spot-lock", 8, pricing_mode="spot", gpu_model="RTX 4090")
        rate = sp.lock_spot_rate_for_job(job, host_gpu_model="RTX 4090")
        assert rate > 0
        assert job.get("spot_rate_cad") == rate
        assert job.get("pricing_mode") == "spot"


class TestSpotPricingProperties:
    def test_spot_rate_lte_on_demand_property(self):
        from hypothesis import given
        from hypothesis import strategies as st

        @given(
            demand=st.integers(min_value=0, max_value=200),
            supply=st.integers(min_value=0, max_value=200),
        )
        def _prop(demand, supply):
            quote = sp.compute_live_spot_quote("RTX 4090", supply=supply, demand=demand)
            assert quote.rate_cad <= quote.on_demand_cad + 1e-9

        _prop()