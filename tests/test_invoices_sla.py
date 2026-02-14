"""Tests for Phase 8.9 — Invoice listing and SLA hosts-summary endpoints."""

import logging
import os
import tempfile
import time

import pytest

# Use a TemporaryDirectory (auto-cleaned) for all data files
_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_inv_sla_test_")
_tmpdir = _tmp_ctx.name
os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ.setdefault("XCELSIOR_BILLING_DB", os.path.join(_tmpdir, "billing.db"))

import scheduler

# Patch file paths to use temp directory
scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")

# Reconfigure the logger so the FileHandler writes to the temp dir
for _h in scheduler.log.handlers[:]:
    if isinstance(_h, logging.FileHandler):
        scheduler.log.removeHandler(_h)
        _h.close()
_fh = logging.FileHandler(scheduler.LOG_FILE)
_fh.setLevel(logging.INFO)
_fh.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
scheduler.log.addHandler(_fh)

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

GOOD_VERSIONS = {
    "runc": "1.2.4",
    "nvidia_ctk": "1.17.8",
    "nvidia_driver": "560.35.03",
    "docker": "27.1.1",
}


def _register_host(host_id="h1", ip="10.0.0.1", gpu="RTX 4090", vram=24,
                    country="CA", province="ON", versions=None, **extra):
    data = {
        "host_id": host_id, "ip": ip, "gpu_model": gpu,
        "total_vram_gb": vram, "free_vram_gb": vram,
        "country": country, "province": province,
    }
    if versions is not None:
        data["versions"] = versions
    data.update(extra)
    return client.put("/host", json=data)


@pytest.fixture(autouse=True)
def clean_data():
    for f in (
        scheduler.HOSTS_FILE,
        scheduler.JOBS_FILE,
        scheduler.BILLING_FILE,
        scheduler.MARKETPLACE_FILE,
        scheduler.AUTOSCALE_POOL_FILE,
        scheduler.SPOT_PRICES_FILE,
        scheduler.COMPUTE_SCORES_FILE,
        os.environ["XCELSIOR_DB_PATH"],
    ):
        if os.path.exists(f):
            os.remove(f)
    import api as api_mod
    api_mod._host_telemetry.clear()
    api_mod._RATE_BUCKETS.clear()
    yield


# ═══════════════════════════════════════════════════════════════════════
# Invoice List Endpoint
# ═══════════════════════════════════════════════════════════════════════


class TestInvoiceList:
    """Tests for GET /api/billing/invoices/{customer_id}."""

    def test_invoice_list_returns_ok(self):
        """Basic call returns ok with invoices array."""
        r = client.get("/api/billing/invoices/cust-test-001")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "invoices" in d
        assert isinstance(d["invoices"], list)
        assert "count" in d

    def test_invoice_list_empty_no_usage(self):
        """Customer with no usage should get empty invoices."""
        r = client.get("/api/billing/invoices/nonexistent-customer-xyz")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["count"] == 0
        assert d["invoices"] == []

    def test_invoice_list_limit_param(self):
        """Limit parameter should be honoured."""
        r = client.get("/api/billing/invoices/cust-test-001?limit=3")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        # Even if no usage, should still return ok
        assert isinstance(d["invoices"], list)

    def test_invoice_list_default_limit_12(self):
        """Default limit should be 12 (max 12 invoices returned)."""
        r = client.get("/api/billing/invoices/cust-test-001")
        assert r.status_code == 200
        d = r.json()
        # Count should not exceed 12
        assert d["count"] <= 12

    def test_invoice_fields_when_present(self):
        """Invoices should have required fields when they exist."""
        r = client.get("/api/billing/invoices/cust-test-001?limit=6")
        d = r.json()
        for inv in d["invoices"]:
            assert "invoice_id" in inv
            assert "period_start" in inv
            assert "period_end" in inv
            assert "total_cad" in inv
            assert "subtotal_cad" in inv
            assert "tax_cad" in inv
            assert "tax_rate" in inv
            assert "line_items" in inv
            assert "caf_eligible_cad" in inv
            assert "status" in inv


# ═══════════════════════════════════════════════════════════════════════
# SLA Hosts Summary Endpoint
# ═══════════════════════════════════════════════════════════════════════


class TestSLAHostsSummary:
    """Tests for GET /api/sla/hosts-summary."""

    def test_summary_empty_no_hosts(self):
        """When no hosts are registered, returns empty list."""
        r = client.get("/api/sla/hosts-summary")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["hosts"] == []
        assert d["count"] == 0

    def test_summary_single_host(self):
        """Register one host and check summary includes it."""
        _register_host("sla-h1", gpu="A100", vram=80, versions=GOOD_VERSIONS)
        r = client.get("/api/sla/hosts-summary")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["count"] >= 1
        found = [h for h in d["hosts"] if h["host_id"] == "sla-h1"]
        assert len(found) == 1
        h = found[0]
        assert h["gpu_model"] == "A100"
        assert "uptime_30d_pct" in h
        assert "violation_count" in h
        assert "sla_tier" in h
        assert h["country"] == "CA"
        assert h["province"] == "ON"

    def test_summary_multiple_hosts(self):
        """Register multiple hosts and verify all appear."""
        _register_host("sla-m1", ip="10.0.0.1", gpu="RTX 4090", vram=24, versions=GOOD_VERSIONS)
        _register_host("sla-m2", ip="10.0.0.2", gpu="A100", vram=80, versions=GOOD_VERSIONS, country="CA", province="QC")
        _register_host("sla-m3", ip="10.0.0.3", gpu="H100", vram=80, versions=GOOD_VERSIONS, country="CA", province="BC")
        r = client.get("/api/sla/hosts-summary")
        assert r.status_code == 200
        d = r.json()
        assert d["count"] >= 3
        host_ids = {h["host_id"] for h in d["hosts"]}
        assert "sla-m1" in host_ids
        assert "sla-m2" in host_ids
        assert "sla-m3" in host_ids

    def test_summary_fields_structure(self):
        """Each host in summary should have all required fields."""
        _register_host("sla-f1", versions=GOOD_VERSIONS)
        r = client.get("/api/sla/hosts-summary")
        d = r.json()
        for h in d["hosts"]:
            assert "host_id" in h
            assert "gpu_model" in h
            assert "status" in h
            assert "sla_tier" in h
            assert "uptime_30d_pct" in h
            assert isinstance(h["uptime_30d_pct"], (int, float))
            assert "violation_count" in h
            assert isinstance(h["violation_count"], int)
            assert "last_violation" in h  # can be None
            assert "country" in h
            assert "province" in h

    def test_summary_uptime_is_percentage(self):
        """Uptime should be between 0 and 100 (or 0 and 1 as proportion)."""
        _register_host("sla-u1", versions=GOOD_VERSIONS)
        r = client.get("/api/sla/hosts-summary")
        d = r.json()
        for h in d["hosts"]:
            assert 0 <= h["uptime_30d_pct"] <= 100

    def test_summary_host_with_pending_status(self):
        """Hosts without versions should appear with pending status."""
        _register_host("sla-p1")  # No versions → pending
        r = client.get("/api/sla/hosts-summary")
        d = r.json()
        found = [h for h in d["hosts"] if h["host_id"] == "sla-p1"]
        assert len(found) == 1
        assert found[0]["status"] == "pending"

    def test_summary_different_provinces(self):
        """Province info should be carried through to summary."""
        _register_host("sla-qc1", ip="10.0.1.1", versions=GOOD_VERSIONS, province="QC")
        _register_host("sla-bc1", ip="10.0.1.2", versions=GOOD_VERSIONS, province="BC")
        r = client.get("/api/sla/hosts-summary")
        d = r.json()
        provinces = {h["host_id"]: h["province"] for h in d["hosts"]}
        assert provinces.get("sla-qc1") == "QC"
        assert provinces.get("sla-bc1") == "BC"
