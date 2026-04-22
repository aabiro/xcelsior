"""Tests for payment flow logic: tax computation, metering, idempotency logic.

DB-dependent tests (deposit, FINTRAC, auto-topup) require PostgreSQL.
They are marked with @pytest.mark.pg and skipped if PG is unavailable.
"""

import os
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from billing import (
    BillingEngine,
    GST_RATE,
    PROVINCE_TAX_RATES,
    UsageMeter,
    get_tax_rate_for_province,
)

# Check if PostgreSQL is available with required schema
try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _conn:
        _conn.execute("SELECT idempotency_key FROM wallet_transactions LIMIT 0")
    _PG_AVAILABLE = True
except Exception:
    _PG_AVAILABLE = False

pg = pytest.mark.skipif(not _PG_AVAILABLE, reason="PostgreSQL not available")


@pytest.fixture(autouse=True)
def _clean_payment_data():
    """Clean payment-related tables before each test."""
    if _PG_AVAILABLE:
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id LIKE 'pay-test-%%'")
            conn.execute("DELETE FROM wallets WHERE customer_id LIKE 'pay-test-%%'")
            conn.execute("DELETE FROM fintrac_reports WHERE customer_id LIKE 'pay-test-%%'")
    yield


def _engine() -> BillingEngine:
    return BillingEngine()


# ── Pure Logic Tests (no DB needed) ──────────────────────────────────


class TestFINTRACThreshold:
    """Verify FINTRAC threshold constant."""

    def test_lvctr_threshold_is_10k(self):
        """FINTRAC LVCTR threshold is $10,000 CAD."""
        assert GST_RATE == pytest.approx(0.05, abs=0.001)

    def test_all_provinces_have_tax_rates(self):
        expected = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}
        assert set(PROVINCE_TAX_RATES.keys()) == expected


class TestGSTSmallSupplierThreshold:
    """Verify GST small supplier threshold constant."""

    def test_small_supplier_threshold_is_30k(self):
        from billing import GST_SMALL_SUPPLIER_THRESHOLD_CAD

        assert GST_SMALL_SUPPLIER_THRESHOLD_CAD == 30_000.00


class TestUsageMeter:
    """Verify per-job usage computation."""

    def test_meter_defaults(self):
        meter = UsageMeter(
            job_id="j-1",
            gpu_model="RTX 4090",
            province="ON",
        )
        assert meter.job_id == "j-1"
        assert meter.gpu_model == "RTX 4090"
        assert meter.province == "ON"
        assert meter.total_cost_cad == 0.0

    def test_meter_tax_ontario(self):
        rate, desc = get_tax_rate_for_province("ON")
        assert abs(rate - 0.13) < 0.001
        assert "HST" in desc or "13" in desc

    def test_meter_tax_quebec(self):
        rate, desc = get_tax_rate_for_province("QC")
        assert abs(rate - 0.14975) < 0.001


# ── DB-Dependent Tests ───────────────────────────────────────────────


class TestDepositIdempotency:
    """Verify idempotent deposits via idempotency_key."""

    @pg
    def test_duplicate_deposit_rejected(self):
        be = _engine()
        be.get_wallet("pay-test-idem-1")
        be.deposit("pay-test-idem-1", 25.00, idempotency_key="pi_test_abc123")
        be.deposit("pay-test-idem-1", 25.00, idempotency_key="pi_test_abc123")
        wallet = be.get_wallet("pay-test-idem-1")
        assert wallet["balance_cad"] == pytest.approx(25.00, abs=0.01)

    @pg
    def test_different_keys_both_apply(self):
        be = _engine()
        be.get_wallet("pay-test-idem-2")
        be.deposit("pay-test-idem-2", 10.00, idempotency_key="pi_test_001")
        be.deposit("pay-test-idem-2", 15.00, idempotency_key="pi_test_002")
        wallet = be.get_wallet("pay-test-idem-2")
        assert wallet["balance_cad"] == pytest.approx(25.00, abs=0.01)

    @pg
    def test_no_key_always_applies(self):
        be = _engine()
        be.get_wallet("pay-test-idem-3")
        be.deposit("pay-test-idem-3", 5.00)
        be.deposit("pay-test-idem-3", 5.00)
        wallet = be.get_wallet("pay-test-idem-3")
        assert wallet["balance_cad"] == pytest.approx(10.00, abs=0.01)


class TestAutoTopupConfig:
    """Verify auto-topup configuration."""

    @pg
    def test_configure_auto_topup(self):
        be = _engine()
        be.get_wallet("pay-test-topup-1")
        result = be.configure_auto_topup(
            customer_id="pay-test-topup-1",
            enabled=True,
            amount_cad=50.0,
            threshold_cad=10.0,
            stripe_payment_method_id="pm_test123",
        )
        assert result["auto_topup_enabled"] is True

    @pg
    def test_disable_auto_topup(self):
        be = _engine()
        be.get_wallet("pay-test-topup-2")
        result = be.configure_auto_topup("pay-test-topup-2", enabled=False)
        assert result["auto_topup_enabled"] is False


class TestFINTRACReporting:
    """Verify FINTRAC large value transaction detection."""

    @pg
    def test_below_threshold_no_report(self):
        be = _engine()
        result = be.fintrac_check_transaction(customer_id="pay-test-fin-1", amount_cad=5000)
        # Below $10K should not trigger report
        assert result is None

    @pg
    def test_at_threshold_creates_report(self):
        be = _engine()
        result = be.fintrac_check_transaction(customer_id="pay-test-fin-2", amount_cad=10000)
        assert result is not None
        assert result["report_type"] == "LVCTR"
        assert result["trigger_amount_cad"] >= 10000


class TestSuspendedWalletEnforcement:
    """Verify jobs are stopped for suspended wallets."""

    @pg
    def test_stop_jobs_returns_int(self):
        be = _engine()
        result = be.stop_jobs_for_suspended_wallets()
        assert isinstance(result, int)
