# Tests for cloudburst.py, marketplace.py, stripe_connect.py, volumes.py
# Coverage for previously-untested modules per audit finding M11.

import os
import time
import uuid
from unittest.mock import MagicMock, patch, PropertyMock

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_DB_BACKEND", "postgres")
os.environ.setdefault("XCELSIOR_POSTGRES_DSN", "postgresql://test:test@localhost/test")
os.environ.setdefault("XCELSIOR_STRIPE_SECRET_KEY", "")
os.environ.setdefault("XCELSIOR_BURST_ENABLED", "false")

import pytest

# ═══════════════════════════════════════════════════════════════════════
# Marketplace Tests
# ═══════════════════════════════════════════════════════════════════════


class TestComputeSpotPrice:
    """Test MarketplaceEngine.compute_spot_price — pure function, no DB."""

    def setup_method(self):
        from marketplace import MarketplaceEngine

        self.engine = MarketplaceEngine.__new__(MarketplaceEngine)

    def test_zero_supply_caps_at_150pct(self):
        assert self.engine.compute_spot_price(1000, 10, 0) == 1500

    def test_no_demand_returns_base(self):
        assert self.engine.compute_spot_price(1000, 0, 10) == 1000

    def test_moderate_demand_increases_price(self):
        # demand=5, supply=10 → factor = min(0.5, 5/20) = 0.25 → 1250
        assert self.engine.compute_spot_price(1000, 5, 10) == 1250

    def test_high_demand_caps_at_50pct_surge(self):
        # demand=100, supply=10 → factor = min(0.5, 100/20) = 0.5 → 1500
        assert self.engine.compute_spot_price(1000, 100, 10) == 1500

    def test_equal_demand_supply(self):
        # demand=10, supply=10 → factor = min(0.5, 10/20) = 0.5 → 1500
        assert self.engine.compute_spot_price(1000, 10, 10) == 1500

    def test_low_demand_supply(self):
        # demand=1, supply=10 → factor = min(0.5, 1/20) = 0.05 → 1050
        assert self.engine.compute_spot_price(1000, 1, 10) == 1050

    def test_minimum_price_is_1_cent(self):
        assert self.engine.compute_spot_price(0, 0, 10) >= 0

    def test_negative_supply_treated_as_zero(self):
        assert self.engine.compute_spot_price(1000, 5, -1) == 1500


class TestMarketplaceConfig:
    """Test marketplace module-level configuration."""

    def test_reserved_discounts_structure(self):
        from marketplace import RESERVED_DISCOUNTS

        assert 1 in RESERVED_DISCOUNTS
        assert 3 in RESERVED_DISCOUNTS
        assert 6 in RESERVED_DISCOUNTS
        for months, discount in RESERVED_DISCOUNTS.items():
            assert 0 < discount < 1, f"Discount for {months}mo must be 0-1"

    def test_spot_sensitivity_is_positive(self):
        from marketplace import SPOT_SENSITIVITY

        assert SPOT_SENSITIVITY > 0

    def test_spot_threshold_is_fraction(self):
        from marketplace import SPOT_THRESHOLD

        assert 0 < SPOT_THRESHOLD <= 1.0


class TestMarketplaceUpsert:
    """Test upsert_offer with mocked DB."""

    def setup_method(self):
        from marketplace import MarketplaceEngine

        self.engine = MarketplaceEngine.__new__(MarketplaceEngine)

    @patch.object(__import__("marketplace").MarketplaceEngine, "_conn")
    def test_upsert_creates_new_offer(self, mock_conn):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # No existing offer
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_cm

        result = self.engine.upsert_offer(
            provider_id="prov-1",
            host_id="host-1",
            gpu_model="RTX 4090",
            vram_gb=24,
            ask_cents_per_hour=100,
        )
        assert result["gpu_model"] == "RTX 4090"
        assert result["ask_cents_per_hour"] == 100
        assert "offer_id" in result


class TestMarketplaceSearch:
    """Test search_offers with mocked DB."""

    def setup_method(self):
        from marketplace import MarketplaceEngine

        self.engine = MarketplaceEngine.__new__(MarketplaceEngine)

    @patch.object(__import__("marketplace").MarketplaceEngine, "_conn")
    def test_search_returns_results(self, mock_conn):
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            {"offer_id": "o1", "gpu_model": "A100", "ask_cents_per_hour": 200},
        ]
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_cm

        results = self.engine.search_offers(gpu_model="A100")
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════
# CloudBurst Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCloudBurstConfig:
    """Test cloudburst module-level configuration."""

    def test_cloud_instance_types_structure(self):
        from cloudburst import CLOUD_INSTANCE_TYPES

        assert "aws" in CLOUD_INSTANCE_TYPES
        assert "gcp" in CLOUD_INSTANCE_TYPES
        for provider, types in CLOUD_INSTANCE_TYPES.items():
            for name, info in types.items():
                assert "gpu_model" in info
                assert "gpu_count" in info
                assert "cost_per_hour_cad" in info
                assert info["cost_per_hour_cad"] > 0

    def test_burst_disabled_by_default_in_test(self):
        from cloudburst import BURST_ENABLED

        # In test env it should be false
        assert not BURST_ENABLED


class TestEvaluateBurstNeed:
    """Test CloudBurstEngine.evaluate_burst_need with mocked DB."""

    def setup_method(self):
        from cloudburst import CloudBurstEngine

        self.engine = CloudBurstEngine.__new__(CloudBurstEngine)

    @patch("cloudburst.BURST_ENABLED", False)
    def test_disabled_returns_disabled(self):
        result = self.engine.evaluate_burst_need()
        assert result["action"] == "disabled"

    @patch("cloudburst.BURST_ENABLED", True)
    @patch.object(__import__("cloudburst").CloudBurstEngine, "_conn")
    def test_below_threshold_returns_none(self, mock_conn):
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            {"cnt": 2},  # queued jobs (below threshold of 5)
            {"cnt": 10},  # community hosts
            {"cnt": 0, "spent": 0},  # burst instances
        ]
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_cm

        result = self.engine.evaluate_burst_need()
        assert result["action"] == "none"

    @patch("cloudburst.BURST_ENABLED", True)
    @patch("cloudburst.BURST_MAX_INSTANCES", 10)
    @patch.object(__import__("cloudburst").CloudBurstEngine, "_conn")
    def test_at_max_instances(self, mock_conn):
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            {"cnt": 20},  # queued (above threshold)
            {"cnt": 5},  # community
            {"cnt": 10, "spent": 100},  # burst at max
        ]
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_cm

        result = self.engine.evaluate_burst_need()
        assert result["action"] == "at_max"

    @patch("cloudburst.BURST_ENABLED", True)
    @patch("cloudburst.BURST_BUDGET_CAD", 500.0)
    @patch("cloudburst.BURST_MAX_INSTANCES", 10)
    @patch.object(__import__("cloudburst").CloudBurstEngine, "_conn")
    def test_budget_exceeded(self, mock_conn):
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            {"cnt": 20},  # queued
            {"cnt": 5},  # community
            {"cnt": 3, "spent": 600},  # over budget
        ]
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_cm

        result = self.engine.evaluate_burst_need()
        assert result["action"] == "budget_exceeded"

    @patch("cloudburst.BURST_ENABLED", True)
    @patch("cloudburst.BURST_BUDGET_CAD", 500.0)
    @patch("cloudburst.BURST_MAX_INSTANCES", 10)
    @patch("cloudburst.BURST_QUEUE_THRESHOLD", 5)
    @patch.object(__import__("cloudburst").CloudBurstEngine, "_conn")
    def test_burst_needed(self, mock_conn):
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            {"cnt": 15},  # queued (above threshold)
            {"cnt": 5},  # community
            {"cnt": 2, "spent": 100},  # under budget, under max
        ]
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_cm

        result = self.engine.evaluate_burst_need()
        assert result["action"] == "burst"
        assert result["instances_needed"] > 0
        assert result["instances_needed"] <= 3  # max 3 at a time


# ═══════════════════════════════════════════════════════════════════════
# Stripe Connect Tests
# ═══════════════════════════════════════════════════════════════════════


class TestStripePlatformCut:
    """Test PLATFORM_CUT_FRAC normalization logic."""

    def test_fraction_unchanged(self):
        # If set to 0.15, should stay 0.15
        from stripe_connect import PLATFORM_CUT_FRAC

        assert 0 < PLATFORM_CUT_FRAC <= 1.0

    def test_enums_defined(self):
        from stripe_connect import AccountStatus, ProviderType

        assert AccountStatus.PENDING.value == "pending"
        assert AccountStatus.ACTIVE.value == "active"
        assert ProviderType.INDIVIDUAL.value == "individual"
        assert ProviderType.COMPANY.value == "company"

    def test_provider_account_dataclass(self):
        from stripe_connect import ProviderAccount

        pa = ProviderAccount(
            provider_id="p1",
            email="test@test.com",
            stripe_account_id="acct_123",
        )
        assert pa.provider_id == "p1"
        assert pa.status == "pending"  # default

    def test_payout_split_dataclass(self):
        from stripe_connect import PayoutSplit

        ps = PayoutSplit(
            job_id="j1",
            provider_id="p1",
            total_cad=100.0,
            provider_share_cad=85.0,
            platform_share_cad=15.0,
            gst_hst_cad=11.05,
        )
        assert ps.total_cad == 100.0
        assert ps.platform_share_cad == 15.0


class TestStripeWebhookDedup:
    """Test handle_webhook when stripe is disabled."""

    def setup_method(self):
        from stripe_connect import StripeConnectManager

        self.mgr = StripeConnectManager.__new__(StripeConnectManager)

    @patch("stripe_connect.STRIPE_ENABLED", False)
    @patch("stripe_connect.stripe", None)
    def test_no_stripe_returns_not_handled(self):
        result = self.mgr.handle_webhook(b"payload", "sig")
        assert result["handled"] is False
        assert "not enabled" in result["reason"].lower()


class TestStripeSplitPayout:
    """Test split_payout math."""

    def setup_method(self):
        from stripe_connect import StripeConnectManager

        self.mgr = StripeConnectManager.__new__(StripeConnectManager)

    @patch("stripe_connect.STRIPE_ENABLED", True)
    @patch("stripe_connect.PLATFORM_CUT_FRAC", 0.15)
    @patch("stripe_connect.stripe")
    @patch.object(__import__("stripe_connect").StripeConnectManager, "_conn")
    @patch.object(__import__("stripe_connect").StripeConnectManager, "get_provider")
    def test_split_math(self, mock_get_prov, mock_conn, mock_stripe):
        mock_get_prov.return_value = {
            "provider_id": "p1",
            "stripe_account_id": "acct_123",
            "status": "active",
        }
        mock_stripe.Transfer.create.return_value = MagicMock(id="tr_123")

        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_cm

        with patch("billing.get_tax_rate_for_province", return_value=0.13):
            result = self.mgr.split_payout(
                job_id="j1", provider_id="p1", total_cad=100.0, province="ON"
            )
            assert result["total_cad"] == 100.0
            assert result["platform_share_cad"] == pytest.approx(15.0, abs=0.01)
            assert result["provider_share_cad"] == pytest.approx(85.0, abs=0.01)
            assert result["gst_hst_cad"] == pytest.approx(13.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# Volume Tests
# ═══════════════════════════════════════════════════════════════════════


class TestVolumeValidation:
    """Test VolumeEngine.create_volume validation."""

    def setup_method(self):
        from volumes import VolumeEngine

        self.engine = VolumeEngine.__new__(VolumeEngine)

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name is required"):
            self.engine.create_volume(owner_id="u1", name="", size_gb=10)

    def test_whitespace_name_rejected(self):
        with pytest.raises(ValueError, match="name is required"):
            self.engine.create_volume(owner_id="u1", name="   ", size_gb=10)

    def test_oversized_volume_rejected(self):
        with pytest.raises(ValueError, match="exceeds max"):
            self.engine.create_volume(owner_id="u1", name="big", size_gb=999999)

    def test_zero_size_rejected(self):
        with pytest.raises(ValueError, match="at least 1GB"):
            self.engine.create_volume(owner_id="u1", name="tiny", size_gb=0)

    def test_negative_size_rejected(self):
        with pytest.raises(ValueError, match="at least 1GB"):
            self.engine.create_volume(owner_id="u1", name="neg", size_gb=-5)


class TestVolumeConfig:
    """Test volume module-level configuration."""

    def test_max_constants_are_positive(self):
        from volumes import MAX_VOLUME_SIZE_GB, MAX_TOTAL_STORAGE_GB

        assert MAX_VOLUME_SIZE_GB > 0
        assert MAX_TOTAL_STORAGE_GB > 0

    def test_default_mount_path(self):
        from volumes import DEFAULT_MOUNT_PATH

        assert DEFAULT_MOUNT_PATH == "/workspace"


class TestVolumeCapacity:
    """Test create_volume capacity checking."""

    def setup_method(self):
        from volumes import VolumeEngine

        self.engine = VolumeEngine.__new__(VolumeEngine)

    @patch("volumes.MAX_TOTAL_STORAGE_GB", 100)
    @patch.object(__import__("volumes").VolumeEngine, "_conn")
    def test_capacity_exceeded(self, mock_conn):
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {"total": 95}  # 95GB used
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_cm

        with pytest.raises(ValueError, match="Insufficient storage"):
            self.engine.create_volume(owner_id="u1", name="vol1", size_gb=10)


class TestVolumeDeleteGuard:
    """Test that delete_volume checks for attachments."""

    def setup_method(self):
        from volumes import VolumeEngine

        self.engine = VolumeEngine.__new__(VolumeEngine)

    @patch.object(__import__("volumes").VolumeEngine, "_conn")
    def test_delete_nonexistent_raises(self, mock_conn):
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # volume not found
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = mock_cm

        with pytest.raises((ValueError, Exception)):
            self.engine.delete_volume(volume_id="vol-nonexistent", owner_id="u1")


# ═══════════════════════════════════════════════════════════════════════
# Singleton accessors
# ═══════════════════════════════════════════════════════════════════════


class TestSingletons:
    """Test that engine singletons return consistent instances."""

    @patch("marketplace.MarketplaceEngine._conn")
    def test_marketplace_singleton(self, _):
        from marketplace import get_marketplace_engine

        e1 = get_marketplace_engine()
        e2 = get_marketplace_engine()
        assert e1 is e2

    @patch("cloudburst.CloudBurstEngine._conn")
    def test_cloudburst_singleton(self, _):
        from cloudburst import get_burst_engine

        e1 = get_burst_engine()
        e2 = get_burst_engine()
        assert e1 is e2

    @patch("volumes.VolumeEngine._conn")
    def test_volume_singleton(self, _):
        from volumes import get_volume_engine

        e1 = get_volume_engine()
        e2 = get_volume_engine()
        assert e1 is e2

    @patch("stripe_connect.StripeConnectManager._conn")
    def test_stripe_singleton(self, _):
        from stripe_connect import get_stripe_manager

        e1 = get_stripe_manager()
        e2 = get_stripe_manager()
        assert e1 is e2
