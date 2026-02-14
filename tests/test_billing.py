"""Tests for Xcelsior billing engine — tax rates, wallets, invoices, CAF exports."""

import os
import tempfile
import time

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_billing_test_")
_tmpdir = _tmp_ctx.name
os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ["XCELSIOR_BILLING_DB"] = os.path.join(_tmpdir, "billing_test.db")

from billing import (
    BillingEngine,
    GST_RATE,
    PROVINCE_TAX_RATES,
    UsageMeter,
    get_tax_rate_for_province,
)


def _engine() -> BillingEngine:
    """Isolated billing engine per test group."""
    return BillingEngine(db_path=os.path.join(_tmpdir, f"billing_{os.urandom(4).hex()}.db"))


# ── Province Tax Rates ───────────────────────────────────────────────


class TestProvinceTaxRates:
    """Verify all 13 provinces/territories have correct GST/HST/QST rates."""

    EXPECTED_PROVINCES = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}

    def test_all_13_present(self):
        assert set(PROVINCE_TAX_RATES.keys()) == self.EXPECTED_PROVINCES

    def test_ontario_hst_13pct(self):
        rate, _desc = PROVINCE_TAX_RATES["ON"]
        assert abs(rate - 0.13) < 0.001

    def test_quebec_combined_14975(self):
        rate, _desc = PROVINCE_TAX_RATES["QC"]
        assert abs(rate - 0.14975) < 0.001

    def test_alberta_gst_only_5pct(self):
        rate, _desc = PROVINCE_TAX_RATES["AB"]
        assert abs(rate - 0.05) < 0.001

    def test_bc_12pct(self):
        rate, _desc = PROVINCE_TAX_RATES["BC"]
        assert abs(rate - 0.12) < 0.001

    def test_atlantic_provinces_15pct(self):
        for prov in ("NB", "NL", "NS", "PE"):
            rate, _desc = PROVINCE_TAX_RATES[prov]
            assert abs(rate - 0.15) < 0.001, f"{prov} should be 15%"

    def test_territories_5pct(self):
        for terr in ("NT", "NU", "YT"):
            rate, _desc = PROVINCE_TAX_RATES[terr]
            assert abs(rate - 0.05) < 0.001, f"{terr} should be 5%"

    def test_all_have_description(self):
        for code, (rate, desc) in PROVINCE_TAX_RATES.items():
            assert desc, f"{code} missing description"


# ── Tax Rate Lookup ──────────────────────────────────────────────────


class TestGetTaxRate:
    """Tax rate lookup for different provinces."""

    def test_ontario_rate(self):
        rate, desc = get_tax_rate_for_province("ON")
        assert abs(rate - 0.13) < 0.001
        assert "13" in desc

    def test_alberta_rate(self):
        rate, desc = get_tax_rate_for_province("AB")
        assert abs(rate - 0.05) < 0.001

    def test_quebec_rate(self):
        rate, desc = get_tax_rate_for_province("QC")
        assert abs(rate - 0.14975) < 0.001

    def test_unknown_province_defaults_gst(self):
        rate, desc = get_tax_rate_for_province("XX")
        assert abs(rate - GST_RATE) < 0.001

    def test_case_insensitive(self):
        rate1, _ = get_tax_rate_for_province("on")
        rate2, _ = get_tax_rate_for_province("ON")
        assert rate1 == rate2


# ── Wallet Operations ────────────────────────────────────────────────


class TestWallets:
    """Customer wallet deposit/charge/balance via BillingEngine."""

    def test_get_wallet_creates_new(self):
        eng = _engine()
        w = eng.get_wallet("cust-new")
        assert w["customer_id"] == "cust-new"
        assert w["balance_cad"] == 0.0

    def test_deposit_increases_balance(self):
        eng = _engine()
        eng.get_wallet("cust-dep")
        eng.deposit("cust-dep", 50.0)
        w = eng.get_wallet("cust-dep")
        assert w["balance_cad"] >= 50.0

    def test_charge_reduces_balance(self):
        eng = _engine()
        eng.get_wallet("cust-charge")
        eng.deposit("cust-charge", 100.0)
        result = eng.charge("cust-charge", 30.0, job_id="j1")
        assert result["charged"] is True
        assert result["balance_cad"] == pytest.approx(70.0, abs=0.01)

    def test_insufficient_balance_triggers_grace(self):
        eng = _engine()
        eng.get_wallet("cust-grace")
        result = eng.charge("cust-grace", 10.0)
        assert result["charged"] is False
        assert "grace" in result.get("action", "")


# ── Metering ─────────────────────────────────────────────────────────


class TestMetering:
    """Per-job usage metering."""

    def test_meter_job_returns_usage(self):
        eng = _engine()
        now = time.time()
        job = {"job_id": "j-meter-1", "started_at": now - 3600, "completed_at": now, "owner": "u1"}
        host = {"host_id": "h1", "gpu_model": "RTX 4090", "cost_per_hour": 0.50, "country": "CA"}
        meter = eng.meter_job(job, host)
        assert isinstance(meter, UsageMeter)
        assert meter.job_id == "j-meter-1"
        assert meter.total_cost_cad > 0

    def test_canadian_compute_flagged(self):
        eng = _engine()
        now = time.time()
        job = {"job_id": "j-ca", "started_at": now - 60, "completed_at": now, "owner": "u1"}
        host = {"host_id": "h-ca", "gpu_model": "A100", "cost_per_hour": 1.0, "country": "CA"}
        meter = eng.meter_job(job, host)
        assert meter.is_canadian_compute is True


# ── Invoice Generation ───────────────────────────────────────────────


class TestInvoicing:
    """Invoice generation with tax breakdown."""

    def test_generate_invoice_empty(self):
        eng = _engine()
        inv = eng.generate_invoice("cust-inv-1", "Test User", 0, time.time())
        assert inv.customer_id == "cust-inv-1"
        assert inv.subtotal_cad == 0.0

    def test_invoice_has_tax_fields(self):
        eng = _engine()
        inv = eng.generate_invoice("cust-inv-2", "Test", 0, time.time(), customer_province="ON")
        assert abs(inv.tax_rate - 0.13) < 0.001


# ── Payout ────────────────────────────────────────────────────────────


class TestPayouts:
    """Stripe Connect–ready payout recording."""

    def test_record_payout_structure(self):
        eng = _engine()
        result = eng.record_payout("prov-1", "j-pay", 100.0)
        assert result["provider_id"] == "prov-1"
        assert result["platform_fee_cad"] == pytest.approx(15.0, abs=0.01)
        assert result["provider_payout_cad"] == pytest.approx(85.0, abs=0.01)


# ── Refunds ──────────────────────────────────────────────────────────


class TestRefunds:
    """Automated refund processing based on failure classification."""

    def test_no_usage_record_returns_false(self):
        eng = _engine()
        result = eng.process_refund("nonexistent-job", exit_code=1)
        assert result["refund"] is False

    def test_hardware_error_full_refund(self):
        eng = _engine()
        now = time.time()
        job = {"job_id": "j-ref-hw", "started_at": now - 60, "completed_at": now, "owner": "u1"}
        host = {"host_id": "h1", "gpu_model": "A100", "cost_per_hour": 1.0}
        eng.meter_job(job, host)
        result = eng.process_refund("j-ref-hw", exit_code=-1, failure_reason="hardware failure")
        assert result["refund"] is True
        assert result["refund_percentage"] == 1.0

    def test_user_oom_no_refund(self):
        eng = _engine()
        now = time.time()
        job = {"job_id": "j-ref-oom", "started_at": now - 60, "completed_at": now, "owner": "u1"}
        host = {"host_id": "h1", "gpu_model": "A100", "cost_per_hour": 1.0}
        eng.meter_job(job, host)
        result = eng.process_refund("j-ref-oom", exit_code=137)
        assert result["refund"] is False
        assert result["classification"] == "user_oom"


# ── CAF Export ────────────────────────────────────────────────────────


class TestCAFExport:
    """Canadian AI Compute Access Fund export format."""

    def test_export_caf_report_structure(self):
        eng = _engine()
        report = eng.export_caf_report("cust-caf", 0, time.time())
        assert report["report_type"].startswith("AI Compute Access Fund")
        assert "summary" in report
        assert "line_items" in report

    def test_export_caf_csv(self):
        eng = _engine()
        csv_str = eng.export_caf_csv("cust-csv", 0, time.time())
        assert "Job ID" in csv_str  # Header present
