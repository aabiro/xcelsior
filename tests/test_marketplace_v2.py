"""Tests for GPU marketplace: spot pricing, reservations, constants.

DB-dependent tests require PostgreSQL and are marked with @pytest.mark.pg.
"""

import os
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from marketplace import MarketplaceEngine, RESERVED_DISCOUNTS, SPOT_SENSITIVITY, SPOT_THRESHOLD

# Check if PostgreSQL is available with required schema
try:
    from db import _get_pg_pool
    _pool = _get_pg_pool()
    with _pool.connection() as _conn:
        _conn.execute("SELECT 1 FROM gpu_offers LIMIT 0")
    _PG_AVAILABLE = True
except Exception:
    _PG_AVAILABLE = False

pg = pytest.mark.skipif(not _PG_AVAILABLE, reason="PostgreSQL not available")


def _engine() -> MarketplaceEngine:
    """Create an engine instance (methods that don't use DB work without PG)."""
    return MarketplaceEngine()


class TestSpotPricing:
    """Verify spot price computation (pure math, no DB needed)."""

    def test_spot_price_zero_demand(self):
        """At zero demand, spot price equals base price."""
        me = _engine()
        price = me.compute_spot_price(base_price_cents=100, demand=0, supply=10)
        assert price == 100

    def test_spot_price_increases_with_demand(self):
        """Higher demand/supply ratio → higher spot price."""
        me = _engine()
        low = me.compute_spot_price(base_price_cents=100, demand=5, supply=10)
        high = me.compute_spot_price(base_price_cents=100, demand=15, supply=10)
        assert high > low

    def test_spot_price_exponential_at_high_demand(self):
        """Extreme demand should produce significantly higher prices."""
        me = _engine()
        normal = me.compute_spot_price(base_price_cents=100, demand=8, supply=10)
        extreme = me.compute_spot_price(base_price_cents=100, demand=20, supply=10)
        assert extreme > normal * 1.5

    def test_spot_price_zero_supply_caps(self):
        """Zero supply should cap at 3x base price."""
        me = _engine()
        price = me.compute_spot_price(base_price_cents=100, demand=10, supply=0)
        assert price == 300

    def test_spot_price_always_positive(self):
        """Spot price should always be at least 1 cent."""
        me = _engine()
        price = me.compute_spot_price(base_price_cents=1, demand=0, supply=100)
        assert price >= 1


class TestSpotPricingConstants:
    """Verify spot pricing constants are sensible."""

    def test_sensitivity_positive(self):
        assert SPOT_SENSITIVITY > 0

    def test_threshold_between_0_and_1(self):
        assert 0 < SPOT_THRESHOLD < 1


class TestReservedDiscounts:
    """Verify reservation discount tiers."""

    def test_one_month_discount(self):
        assert RESERVED_DISCOUNTS[1] == pytest.approx(0.10, abs=0.01)

    def test_three_month_discount(self):
        assert RESERVED_DISCOUNTS[3] == pytest.approx(0.20, abs=0.01)

    def test_six_month_discount(self):
        assert RESERVED_DISCOUNTS[6] == pytest.approx(0.30, abs=0.01)

    def test_twelve_month_discount(self):
        assert RESERVED_DISCOUNTS[12] == pytest.approx(0.40, abs=0.01)

    def test_discount_increases_with_term(self):
        discounts = [RESERVED_DISCOUNTS[m] for m in sorted(RESERVED_DISCOUNTS.keys())]
        for i in range(1, len(discounts)):
            assert discounts[i] > discounts[i - 1]

    def test_all_four_tiers_present(self):
        assert set(RESERVED_DISCOUNTS.keys()) == {1, 3, 6, 12}


class TestMarketplaceStats:
    """Verify marketplace stats endpoint returns expected structure."""

    @pg
    def test_stats_structure(self):
        from marketplace import get_marketplace_engine
        me = get_marketplace_engine()
        stats = me.get_marketplace_stats()
        assert "total_offers" in stats
        assert "total_gpus" in stats
        assert isinstance(stats["total_offers"], int)
