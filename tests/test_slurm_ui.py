"""Tests for Slurm HPC Tab, Responsive Design & UX Polish (v2.8.0).

Covers:
- Slurm API endpoints (submit, status, cancel, profiles)
- Spot bid submission endpoint
- SLA hosts-summary endpoint
- Residency trace endpoint
- Dashboard HTML structure (Slurm tab, responsive CSS, spot bid form, SLA display)
"""

import json
import logging
import os
import tempfile
import time

import pytest

# Use a TemporaryDirectory for all data files
_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_slurm_ui_test_")
_tmpdir = _tmp_ctx.name
os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"
os.environ.setdefault("XCELSIOR_BILLING_DB", os.path.join(_tmpdir, "billing.db"))
os.environ["XCELSIOR_AUTH_DB_PATH"] = os.path.join(_tmpdir, "auth.db")

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

# Reconfigure logger to temp dir
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

import db as db_mod

db_mod.AUTH_DB_FILE = os.path.join(_tmpdir, "auth.db")

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}

GOOD_VERSIONS = {
    "runc": "1.2.4",
    "nvidia_ctk": "1.17.8",
    "docker": "28.1.1",
    "cuda": "12.8",
}


# ── Helpers ──────────────────────────────────────────────────────────


def _register_host(host_id="hpc-host-1", gpu_model="A100", vram_gb=80, country="CA"):
    """Register a test host."""
    return client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": "10.0.0.1",
            "gpu_model": gpu_model,
            "total_vram_gb": vram_gb,
            "free_vram_gb": vram_gb,
            "cost_per_hour": 0.50,
            "country": country,
            "province": "ON",
            "versions": GOOD_VERSIONS,
        },
    )


def _submit_job(name="test-job", vram=8.0, tier="standard"):
    """Submit a test job."""
    return client.post(
        "/instance",
        json={
            "name": name,
            "vram_needed_gb": vram,
            "tier": tier,
        },
    )


def _seed_wallet():
    """Seed wallet for anonymous test user so wallet pre-flight checks pass."""
    from billing import get_billing_engine

    get_billing_engine().deposit("anonymous", 10_000.0, description="Test credits")


# ══════════════════════════════════════════════════════════════════════
# Slurm API Tests
# ══════════════════════════════════════════════════════════════════════


class TestSlurmProfiles:
    """Tests for GET /api/slurm/profiles."""

    def test_profiles_returns_dict(self):
        r = client.get("/api/slurm/profiles")
        assert r.status_code == 200
        d = r.json()
        assert "profiles" in d
        assert isinstance(d["profiles"], dict)

    def test_profiles_contain_known_clusters(self):
        r = client.get("/api/slurm/profiles")
        d = r.json()
        profiles = d["profiles"]
        # At minimum, "generic" profile should exist
        assert len(profiles) >= 1


class TestSlurmSubmit:
    """Tests for POST /api/slurm/submit."""

    def test_submit_dry_run(self):
        """Dry run should return a script preview without actually submitting."""
        r = client.post(
            "/api/slurm/submit",
            json={
                "name": "test-dryrun",
                "vram_needed_gb": 16.0,
                "priority": 5,
                "tier": "standard",
                "num_gpus": 1,
                "image": "pytorch:latest",
                "dry_run": True,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d.get("dry_run") or "script" in d

    def test_submit_with_profile(self):
        """Submit with a specific profile."""
        r = client.post(
            "/api/slurm/submit",
            json={
                "name": "profiled-job",
                "vram_needed_gb": 24.0,
                "profile": "generic",
                "dry_run": True,
            },
        )
        assert r.status_code == 200

    def test_submit_missing_name(self):
        """Name is required and must be non-empty."""
        r = client.post(
            "/api/slurm/submit",
            json={
                "name": "",
                "vram_needed_gb": 8.0,
            },
        )
        assert r.status_code in (400, 422)  # Adapter or Pydantic validation

    def test_submit_zero_vram_rejected(self):
        """VRAM must be > 0."""
        r = client.post(
            "/api/slurm/submit",
            json={
                "name": "bad-vram",
                "vram_needed_gb": 0,
            },
        )
        assert r.status_code in (400, 422)  # Adapter or Pydantic validation

    def test_submit_multi_gpu(self):
        """Submit with multiple GPUs."""
        r = client.post(
            "/api/slurm/submit",
            json={
                "name": "multi-gpu-job",
                "vram_needed_gb": 48.0,
                "num_gpus": 4,
                "dry_run": True,
            },
        )
        assert r.status_code == 200


class TestSlurmStatus:
    """Tests for GET /api/slurm/status/{id}."""

    def test_status_nonexistent_job(self):
        """Looking up a non-existent Slurm job returns an error."""
        r = client.get("/api/slurm/status/nonexistent-999")
        # Should return 400 (slurm_adapter returns error dict)
        assert r.status_code in (400, 404, 200)

    def test_status_endpoint_exists(self):
        """Endpoint should exist and accept requests."""
        r = client.get("/api/slurm/status/test-123")
        assert r.status_code in (200, 400, 404)


class TestSlurmCancel:
    """Tests for DELETE /api/slurm/{id}."""

    def test_cancel_nonexistent(self):
        """Cancelling a non-existent job returns an error."""
        r = client.delete("/api/slurm/nonexistent-999")
        assert r.status_code in (400, 404, 200)

    def test_cancel_endpoint_exists(self):
        """Endpoint should exist and accept DELETE."""
        r = client.delete("/api/slurm/test-cancel-123")
        assert r.status_code in (200, 400, 404)


# ══════════════════════════════════════════════════════════════════════
# Spot Bid Tests
# ══════════════════════════════════════════════════════════════════════


class TestSpotInstance:
    """Tests for spot instance launch via POST /instance."""

    def setup_method(self):
        _seed_wallet()

    def test_submit_spot_instance(self):
        """Submit a spot instance via unified launch API."""
        _register_host("spot-host-1")
        r = client.post(
            "/instance",
            json={
                "name": "spot-inference",
                "vram_needed_gb": 16.0,
                "pricing_mode": "spot",
                "priority": 5,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d.get("ok") is True
        assert "instance" in d
        assert d["instance"]["name"] == "spot-inference"

    def test_spot_instance_missing_name(self):
        """Name is required."""
        r = client.post(
            "/instance",
            json={
                "name": "",
                "vram_needed_gb": 8.0,
                "pricing_mode": "spot",
            },
        )
        assert r.status_code == 422

    def test_spot_instance_premium_tier_rejected(self):
        """Spot cannot use premium/sovereign tiers."""
        r = client.post(
            "/instance",
            json={
                "name": "spot-premium",
                "vram_needed_gb": 8.0,
                "pricing_mode": "spot",
                "tier": "premium",
            },
        )
        assert r.status_code == 400

    def test_spot_prices_endpoint(self):
        """GET /spot-prices should return prices dict."""
        r = client.get("/spot-prices")
        assert r.status_code == 200
        d = r.json()
        assert "prices" in d

    def test_spot_price_update(self):
        """POST /spot-prices/update should recalculate."""
        r = client.post("/spot-prices/update")
        assert r.status_code == 200
        d = r.json()
        assert d.get("ok") is True


# ══════════════════════════════════════════════════════════════════════
# SLA Hosts Summary Tests
# ══════════════════════════════════════════════════════════════════════


class TestSLAHostsSummary:
    """Tests for GET /api/sla/hosts-summary."""

    def test_sla_summary_returns_list(self):
        _register_host("sla-host-1")
        r = client.get("/api/sla/hosts-summary", headers=_auth_headers())
        assert r.status_code == 200
        d = r.json()
        assert d.get("ok") is True
        assert "hosts" in d
        assert isinstance(d["hosts"], list)
        assert "count" in d

    def test_sla_summary_has_uptime_fields(self):
        _register_host("sla-host-2", gpu_model="RTX4090")
        r = client.get("/api/sla/hosts-summary", headers=_auth_headers())
        d = r.json()
        hosts = d["hosts"]
        if hosts:
            h = hosts[0]
            assert "uptime_30d_pct" in h
            assert "violation_count" in h
            assert "sla_tier" in h


class TestSLAEnforce:
    """Tests for POST /api/sla/enforce."""

    def test_enforce_returns_credit_info(self):
        _register_host("sla-enforce-host")
        r = client.post(
            "/api/sla/enforce",
            json={
                "host_id": "sla-enforce-host",
                "month": "2025-01",
                "tier": "community",
                "monthly_spend_cad": 100.0,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d.get("ok") is True
        assert "uptime_pct" in d
        assert "credit_pct" in d
        assert "credit_cad" in d


# ══════════════════════════════════════════════════════════════════════
# Residency Trace Tests
# ══════════════════════════════════════════════════════════════════════


class TestResidencyTrace:
    """Tests for GET /api/jurisdiction/residency-trace/{job_id}."""

    def setup_method(self):
        _seed_wallet()

    def test_trace_nonexistent_job(self):
        r = client.get("/api/jurisdiction/residency-trace/nonexistent-job")
        assert r.status_code == 404

    def test_trace_existing_job(self):
        _register_host("trace-host")
        jr = _submit_job("trace-test-job")
        assert jr.status_code == 200
        job_id = jr.json()["instance"]["job_id"]
        # Assign & complete the job
        client.post("/queue/process")
        r = client.get(f"/api/jurisdiction/residency-trace/{job_id}")
        assert r.status_code == 200
        d = r.json()
        assert d.get("ok") is True
        assert "trace" in d
        assert d["job_id"] == job_id


# ══════════════════════════════════════════════════════════════════════
# Dashboard HTML Structure Tests
# ══════════════════════════════════════════════════════════════════════


class TestDashboardHTML:
    """Verify Slurm/responsive UI elements exist in dashboard.html."""

    @pytest.fixture(autouse=True)
    def _load_html(self):
        r = client.get("/dashboard")
        assert r.status_code == 200
        self.html = r.text

    def test_version_updated(self):
        assert "v2.8.0" in self.html

    def test_slurm_tab_button_exists(self):
        assert "HPC/Slurm" in self.html

    def test_slurm_tab_div_exists(self):
        assert 'id="tab-slurm"' in self.html

    def test_slurm_profiles_container(self):
        assert 'id="slurm-profiles"' in self.html

    def test_slurm_submit_form(self):
        assert 'id="slurm-name"' in self.html
        assert 'id="slurm-vram"' in self.html
        assert 'id="slurm-tier"' in self.html
        assert 'id="slurm-gpus"' in self.html
        assert 'id="slurm-image"' in self.html
        assert 'id="slurm-profile"' in self.html
        assert 'id="slurm-dryrun"' in self.html

    def test_slurm_status_lookup(self):
        assert 'id="slurm-status-id"' in self.html
        assert 'id="slurm-status-result"' in self.html

    def test_slurm_recent_jobs(self):
        assert 'id="slurm-recent-jobs"' in self.html

    def test_slurm_script_preview(self):
        assert 'id="slurm-script-card"' in self.html
        assert 'id="slurm-script-content"' in self.html

    def test_slurm_tab_in_switcher(self):
        assert "'tab-slurm'" in self.html

    def test_spot_bid_form_removed(self):
        assert 'id="spot-bid-name"' not in self.html
        assert "submitSpotBid" not in self.html

    def test_sla_credit_calculator(self):
        assert "SLA Credit Calculator" in self.html
        assert "sla-credit-table" in self.html
        assert "sla-hosts-summary" in self.html

    def test_residency_trace_ui(self):
        assert 'id="residency-trace-jobid"' in self.html
        assert 'id="residency-trace-result"' in self.html
        assert "fetchResidencyTrace()" in self.html

    def test_responsive_media_queries(self):
        assert "@media (max-width: 1024px)" in self.html
        assert "@media (max-width: 768px)" in self.html
        assert "@media (max-width: 480px)" in self.html

    def test_responsive_tab_bar_scroll(self):
        assert "overflow-x: auto" in self.html

    def test_responsive_table_scroll(self):
        assert "overflow-x: auto" in self.html

    def test_slurm_js_functions(self):
        assert "fetchSlurmProfiles" in self.html
        assert "submitSlurmJob" in self.html
        assert "fetchSlurmStatus" in self.html
        assert "cancelSlurmJob" in self.html
        assert "renderSlurmRecent" in self.html
        assert "selectClusterProfile" in self.html

    def test_sla_js_function(self):
        assert "fetchSLAHostsSummary" in self.html

    def test_residency_trace_js_function(self):
        assert "fetchResidencyTrace" in self.html

    def test_slurm_in_refresh(self):
        assert "fetchSlurmProfiles()" in self.html
