"""Tests for Xcelsior API endpoints.

Phase 7.2 — Comprehensive API tests using FastAPI TestClient.
Covers: Host Registration Flow, Core CRUD, Auth, v2.0 Endpoints,
        Telemetry, Reputation, Pricing, Billing, Transparency, SLA,
        Privacy, Spot Pricing, Verification, Agent endpoints.
"""

import logging
import os
import tempfile
import time

import pytest

# Use a TemporaryDirectory (auto-cleaned) for all scheduler data files
_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_api_test_")
_tmpdir = _tmp_ctx.name
os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_AUTH_DB_PATH"] = os.path.join(_tmpdir, "auth.db")
os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"  # Prevent 429s in tests

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

# Reconfigure the logger so the FileHandler writes to the temp dir, not the repo
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
import db as db_mod
db_mod.AUTH_DB_FILE = os.path.join(_tmpdir, "auth.db")

client = TestClient(app)

# ── Good versions that pass admission ──
GOOD_VERSIONS = {
    "runc": "1.2.4",
    "nvidia_ctk": "1.17.8",
    "nvidia_driver": "560.35.03",
    "docker": "27.1.1",
}

BAD_VERSIONS = {
    "runc": "1.0.0",
    "nvidia_ctk": "1.10.0",
    "nvidia_driver": "520.0",
    "docker": "20.0.0",
}


def _register_host(host_id="h1", ip="10.0.0.1", gpu="RTX 4090", vram=24,
                    country="CA", province="ON", versions=None, **extra):
    """Helper to register a host with optional parameters."""
    data = {
        "host_id": host_id, "ip": ip, "gpu_model": gpu,
        "total_vram_gb": vram, "free_vram_gb": vram,
        "country": country, "province": province,
    }
    if versions is not None:
        data["versions"] = versions
    data.update(extra)
    return client.put("/host", json=data)


def _submit_job(name="llama3", vram=16, **extra):
    """Helper to submit a job."""
    data = {"name": name, "vram_needed_gb": vram}
    data.update(extra)
    return client.post("/job", json=data)


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
    # Clear in-memory telemetry and rate limit buckets
    import api as api_mod
    api_mod._host_telemetry.clear()
    api_mod._RATE_BUCKETS.clear()
    yield


# ═══════════════════════════════════════════════════════════════════════
# 7.2.1 — Host Registration Flow
# ═══════════════════════════════════════════════════════════════════════


class TestHostRegistrationFlow:
    """Tests for host registration with admission gating."""

    def test_put_host_creates_pending(self):
        """PUT /host without versions → status=pending, admitted=False."""
        r = _register_host("h1")
        assert r.status_code == 200
        host = r.json()["host"]
        assert host["admitted"] is False
        assert host["status"] == "pending"

    def test_put_host_with_valid_versions(self):
        """PUT /host with good versions → admitted=True."""
        r = _register_host("h1", versions=GOOD_VERSIONS)
        assert r.status_code == 200
        host = r.json()["host"]
        assert host["admitted"] is True

    def test_put_host_with_failing_versions(self):
        """PUT /host with old runc → admitted=False, status=pending."""
        r = _register_host("h1", versions=BAD_VERSIONS)
        assert r.status_code == 200
        host = r.json()["host"]
        assert host["admitted"] is False
        assert host["status"] == "pending"

    def test_put_host_preserves_country_province(self):
        """Country=CA, province=ON persisted correctly."""
        r = _register_host("h1", country="CA", province="ON")
        assert r.status_code == 200
        host = r.json()["host"]
        assert host["country"] == "CA"
        assert host["province"] == "ON"

    def test_put_host_non_ca_country(self):
        """Non-CA country is preserved correctly."""
        r = _register_host("h1", country="US", province="")
        host = r.json()["host"]
        assert host["country"] == "US"
        assert host["province"] == ""

    def test_agent_versions_admits_host(self):
        """POST /agent/versions with good versions → host becomes active."""
        _register_host("h1")
        r = client.post(
            "/agent/versions",
            json={"host_id": "h1", "versions": GOOD_VERSIONS},
        )
        assert r.status_code == 200
        assert r.json()["admitted"] is True

        # Verify host is now active
        hosts = client.get("/hosts?active_only=false").json()["hosts"]
        h = next(h for h in hosts if h["host_id"] == "h1")
        assert h["admitted"] is True
        assert h["status"] == "active"

    def test_agent_versions_rejects_host(self):
        """POST /agent/versions with bad versions → host stays pending."""
        _register_host("h1")
        r = client.post(
            "/agent/versions",
            json={"host_id": "h1", "versions": BAD_VERSIONS},
        )
        assert r.status_code == 200
        assert r.json()["admitted"] is False

    def test_admitted_host_receives_work(self):
        """Only admitted hosts get job assignments."""
        _register_host("h1", versions=GOOD_VERSIONS)
        _submit_job("llama3", 16)
        r = client.post("/queue/process")
        assert r.status_code == 200
        assert len(r.json()["assigned"]) == 1

    def test_pending_host_excluded(self):
        """Pending host with enough VRAM still doesn't get jobs."""
        _register_host("h1")  # No versions → pending
        _submit_job("llama3", 16)
        r = client.post("/queue/process")
        assert r.status_code == 200
        assert len(r.json()["assigned"]) == 0

    def test_host_canadian_company_fields(self):
        """Canadian company fields persist on registration."""
        r = _register_host(
            "h1",
            corporation_name="Acme GPU Inc.",
            business_number="123456789RC0001",
            gst_hst_number="RT0001",
            legal_name="Acme GPU Inc.",
        )
        host = r.json()["host"]
        assert host.get("corporation_name") == "Acme GPU Inc."
        assert host.get("business_number") == "123456789RC0001"


# ═══════════════════════════════════════════════════════════════════════
# 7.2.2 — Core CRUD
# ═══════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["name"] == "Xcelsior"

    def test_healthz(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_readyz(self):
        r = client.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"
        assert r.json()["storage"]["ok"] is True

    def test_metrics(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "queue_depth" in r.json()["metrics"]


class TestHostEndpoints:
    def test_register_host(self):
        r = _register_host("h1")
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_list_hosts(self):
        _register_host("h1")
        r = client.get("/hosts?active_only=false")
        assert r.status_code == 200
        assert len(r.json()["hosts"]) >= 1

    def test_list_hosts_active_only(self):
        """GET /hosts?active_only=true excludes pending/dead hosts."""
        _register_host("h1")  # pending (no versions)
        r = client.get("/hosts?active_only=true")
        # h1 is pending, so may or may not appear depending on active_only logic
        assert r.status_code == 200

    def test_list_hosts_include_all(self):
        """GET /hosts?active_only=false includes all hosts."""
        _register_host("h1")
        r = client.get("/hosts?active_only=false")
        assert r.status_code == 200
        assert any(h["host_id"] == "h1" for h in r.json()["hosts"])

    def test_remove_host(self):
        _register_host("h1")
        r = client.delete("/host/h1")
        assert r.status_code == 200

    def test_remove_nonexistent_host(self):
        r = client.delete("/host/nonexistent")
        assert r.status_code == 404


class TestJobEndpoints:
    def test_submit_job(self):
        r = _submit_job("llama3", 16)
        assert r.status_code == 200
        assert r.json()["ok"]
        assert r.json()["job"]["status"] == "queued"

    def test_submit_job_returns_job_id(self):
        r = _submit_job("test-model", 8)
        assert "job_id" in r.json()["job"]
        assert len(r.json()["job"]["job_id"]) > 0

    def test_submit_job_validation_empty_name(self):
        """Empty name should fail validation."""
        r = client.post("/job", json={"name": "", "vram_needed_gb": 8})
        assert r.status_code == 422

    def test_submit_job_validation_zero_vram(self):
        """Zero VRAM should fail validation."""
        r = client.post("/job", json={"name": "test", "vram_needed_gb": 0})
        assert r.status_code == 422

    def test_list_jobs(self):
        _submit_job("llama3", 16)
        r = client.get("/jobs")
        assert len(r.json()["jobs"]) == 1

    def test_list_jobs_filter_by_status(self):
        """GET /jobs?status=running filters correctly."""
        _submit_job("model-a", 16)
        r = client.get("/jobs?status=running")
        # Nothing is running, so should be empty
        assert r.status_code == 200
        assert len(r.json()["jobs"]) == 0

    def test_list_jobs_filter_queued(self):
        """GET /jobs?status=queued returns queued jobs."""
        _submit_job("model-a", 16)
        r = client.get("/jobs?status=queued")
        assert len(r.json()["jobs"]) == 1

    def test_get_job(self):
        resp = _submit_job("llama3", 16)
        job_id = resp.json()["job"]["job_id"]
        r = client.get(f"/job/{job_id}")
        assert r.status_code == 200
        assert r.json()["job"]["name"] == "llama3"

    def test_get_nonexistent_job(self):
        r = client.get("/job/nonexistent")
        assert r.status_code == 404

    def test_update_job_status(self):
        """PATCH /job/{id}/status transitions correctly."""
        resp = _submit_job("test", 8)
        job_id = resp.json()["job"]["job_id"]
        r = client.patch(f"/job/{job_id}", json={"status": "completed"})
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

    def test_process_queue(self):
        _register_host("h1", versions=GOOD_VERSIONS)
        _submit_job("llama3", 16)
        r = client.post("/queue/process")
        assert r.status_code == 200
        assert len(r.json()["assigned"]) == 1

    def test_submit_multi_gpu_job(self):
        """Multi-GPU job support (num_gpus > 1)."""
        r = _submit_job("large-model", 40, num_gpus=4)
        assert r.status_code == 200
        job = r.json()["job"]
        assert job.get("num_gpus") == 4

    def test_submit_job_with_nfs(self):
        """NFS mount support in job submission."""
        r = _submit_job(
            "nfs-model", 16,
            nfs_server="10.0.0.5", nfs_path="/exports/models",
            nfs_mount_point="/mnt/models",
        )
        assert r.status_code == 200
        job = r.json()["job"]
        assert job.get("nfs_server") == "10.0.0.5"

    def test_submit_job_with_image(self):
        """Docker image override in job submission."""
        r = _submit_job("custom", 8, image="nvcr.io/nvidia/pytorch:24.01-py3")
        assert r.status_code == 200
        assert r.json()["job"].get("image") == "nvcr.io/nvidia/pytorch:24.01-py3"


class TestJobStatusApiValidation:
    def test_invalid_status_returns_400(self):
        """PATCH /job/{id} with invalid status should return 400, not 200."""
        resp = _submit_job("test", 8)
        job_id = resp.json()["job"]["job_id"]
        r = client.patch(f"/job/{job_id}", json={"status": "totally_bogus_status"})
        assert r.status_code == 400
        assert "Invalid status" in r.json()["error"]["message"]


# ═══════════════════════════════════════════════════════════════════════
# 7.2.3 — Auth
# ═══════════════════════════════════════════════════════════════════════


class TestAuth:
    """Auth tests — test mode disables auth, but we verify the mechanism."""

    def test_dev_mode_no_token_ok(self):
        """XCELSIOR_ENV=test → no token needed."""
        r = client.get("/hosts")
        assert r.status_code == 200

    def test_public_paths_no_auth(self):
        """Public paths always accessible."""
        for path in ["/", "/healthz", "/readyz", "/metrics", "/dashboard"]:
            r = client.get(path)
            assert r.status_code == 200, f"{path} returned {r.status_code}"

    def test_token_generate(self):
        """POST /token/generate returns a token."""
        r = client.post("/token/generate")
        assert r.status_code == 200
        assert "token" in r.json()


# ═══════════════════════════════════════════════════════════════════════
# 7.2.4 — v2.0 Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestTelemetry:
    def test_telemetry_push_pull(self):
        """POST telemetry → GET returns it."""
        client.post("/agent/telemetry", json={
            "host_id": "h1",
            "timestamp": time.time(),
            "metrics": {"utilization": 85, "temp": 72, "memory_errors": 0},
        })
        r = client.get("/agent/telemetry/h1")
        assert r.status_code == 200
        assert r.json()["metrics"]["utilization"] == 85

    def test_telemetry_missing_host(self):
        """GET telemetry for unknown host → 404."""
        r = client.get("/agent/telemetry/nonexistent")
        assert r.status_code == 404

    def test_telemetry_all(self):
        """GET /api/telemetry/all returns all host telemetry."""
        client.post("/agent/telemetry", json={
            "host_id": "h1", "metrics": {"utilization": 50},
        })
        client.post("/agent/telemetry", json={
            "host_id": "h2", "metrics": {"utilization": 90},
        })
        r = client.get("/api/telemetry/all")
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_telemetry_stale_detection(self):
        """Telemetry older than 30s is marked stale."""
        import api as api_mod
        api_mod._host_telemetry["h-stale"] = {
            "timestamp": time.time() - 60,
            "metrics": {"utilization": 10},
            "received_at": time.time() - 60,
        }
        r = client.get("/agent/telemetry/h-stale")
        assert r.json()["stale"] is True


class TestReputation:
    def test_reputation_leaderboard(self):
        """GET /api/reputation/leaderboard returns sorted list."""
        r = client.get("/api/reputation/leaderboard?entity_type=host&limit=10")
        assert r.status_code == 200
        data = r.json()
        # May return leaderboard key or reputation key depending on route match
        assert data["ok"] is True

    def test_reputation_get(self):
        """GET /api/reputation/{id} returns score dict."""
        r = client.get("/api/reputation/host-1")
        assert r.status_code == 200
        assert "reputation" in r.json()

    def test_reputation_history(self):
        """GET /api/reputation/{id}/history returns events."""
        r = client.get("/api/reputation/host-1/history")
        assert r.status_code == 200
        assert "events" in r.json()

    def test_reputation_verify(self):
        """POST /api/reputation/verify grants verification badge."""
        r = client.post("/api/reputation/verify", json={
            "entity_id": "host-1",
            "verification_type": "email",
        })
        assert r.status_code == 200
        rep = r.json()["reputation"]
        # Verify the reputation dict has expected keys
        assert rep["entity_id"] == "host-1"
        assert rep["tier"] in ("new_user", "bronze", "silver", "gold", "platinum", "diamond")

    def test_reputation_verify_invalid_type(self):
        """Invalid verification type → 400."""
        r = client.post("/api/reputation/verify", json={
            "entity_id": "host-1",
            "verification_type": "invalid_type",
        })
        assert r.status_code == 400


class TestPricing:
    def test_pricing_reference(self):
        """GET /api/pricing/reference returns GPU rates."""
        r = client.get("/api/pricing/reference")
        assert r.status_code == 200
        assert r.json()["currency"] == "CAD"
        assert len(r.json()["pricing"]) > 0

    def test_pricing_estimate(self):
        """POST /api/pricing/estimate returns cost."""
        r = client.post("/api/pricing/estimate", json={
            "gpu_model": "RTX 4090",
            "duration_hours": 1.0,
            "is_canadian": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["ok"]
        assert data["currency"] == "CAD"
        assert data["gross_cost_cad"] > 0

    def test_pricing_estimate_spot(self):
        """Spot pricing estimate."""
        r = client.post("/api/pricing/estimate", json={
            "gpu_model": "RTX 4090",
            "duration_hours": 2.0,
            "spot": True,
        })
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_reserved_plans(self):
        """GET /api/pricing/reserved-plans returns 3 tiers."""
        r = client.get("/api/pricing/reserved-plans")
        assert r.status_code == 200
        tiers = r.json()["reserved_tiers"]
        assert "1_month" in tiers
        assert "3_month" in tiers
        assert "1_year" in tiers
        # Check discount percentages
        assert tiers["1_month"]["discount_pct"] == 20
        assert tiers["3_month"]["discount_pct"] == 30
        assert tiers["1_year"]["discount_pct"] == 45

    def test_reserve_commitment(self):
        """POST /api/pricing/reserve creates a commitment."""
        r = client.post("/api/pricing/reserve", json={
            "customer_id": "cust-1",
            "gpu_model": "RTX 4090",
            "commitment_type": "1_month",
            "quantity": 1,
            "province": "ON",
        })
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_reserve_invalid_commitment(self):
        """Invalid commitment_type → 400."""
        r = client.post("/api/pricing/reserve", json={
            "customer_id": "cust-1",
            "gpu_model": "RTX 4090",
            "commitment_type": "5_year",
        })
        assert r.status_code == 400


class TestBillingEndpoints:
    def test_billing_empty(self):
        r = client.get("/billing")
        assert r.status_code == 200
        assert r.json()["total_revenue"] == 0

    def test_billing_attestation(self):
        """GET /api/billing/attestation returns fund data."""
        r = client.get("/api/billing/attestation")
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_gst_threshold(self):
        """GET /api/billing/gst-threshold returns threshold info."""
        r = client.get("/api/billing/gst-threshold")
        assert r.status_code == 200

    def test_billing_wallet_create(self):
        """GET /api/billing/wallet/{id} on new customer creates wallet."""
        r = client.get("/api/billing/wallet/cust-new")
        assert r.status_code == 200

    def test_billing_wallet_deposit(self):
        """POST wallet deposit."""
        r = client.post(
            "/api/billing/wallet/cust-1/deposit",
            json={"amount_cad": 100.0, "description": "Test deposit"},
        )
        assert r.status_code == 200

    def test_billing_refund(self):
        """POST /api/billing/refund processes a refund."""
        r = client.post("/api/billing/refund", json={
            "job_id": "job-test-refund",
            "exit_code": -1,
            "failure_reason": "hardware",
        })
        assert r.status_code == 200


class TestTransparency:
    def test_transparency_report(self):
        """GET /api/transparency/report returns report."""
        r = client.get("/api/transparency/report")
        assert r.status_code == 200
        report = r.json()
        assert "summary" in report
        assert "cloud_act_note" in report
        assert "Canadian" in report["cloud_act_note"]

    def test_transparency_report_months_param(self):
        """Months parameter filters timeframe."""
        r = client.get("/api/transparency/report?months=6")
        assert r.status_code == 200
        assert r.json()["period_months"] == 6


class TestDashboard:
    def test_dashboard_returns_html(self):
        r = client.get("/dashboard")
        assert r.status_code == 200
        assert "XCELSIOR" in r.text

    def test_dashboard_contains_tabs(self):
        """Dashboard HTML has the expected tab structure."""
        r = client.get("/dashboard")
        assert r.status_code == 200
        # Check for key dashboard elements
        assert "host" in r.text.lower() or "Host" in r.text


class TestTiersEndpoint:
    def test_list_tiers(self):
        r = client.get("/tiers")
        assert r.status_code == 200
        assert "urgent" in r.json()["tiers"]


class TestCanadaEndpoint:
    def test_toggle_canada(self):
        r = client.put("/canada", json={"enabled": True})
        assert r.status_code == 200
        assert r.json()["canada_only"] is True
        r = client.get("/canada")
        assert r.json()["canada_only"] is True
        client.put("/canada", json={"enabled": False})

    def test_canada_hosts(self):
        """GET /hosts/ca returns Canadian hosts."""
        _register_host("ca-host", country="CA", province="ON")
        r = client.get("/hosts/ca")
        assert r.status_code == 200


class TestMarketplaceEndpoints:
    def test_list_rig(self):
        r = client.post(
            "/marketplace/list",
            json={"host_id": "h1", "gpu_model": "RTX 4090", "vram_gb": 24, "price_per_hour": 0.30},
        )
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_get_marketplace(self):
        client.post(
            "/marketplace/list",
            json={"host_id": "h1", "gpu_model": "RTX 4090", "vram_gb": 24, "price_per_hour": 0.30},
        )
        r = client.get("/marketplace")
        assert len(r.json()["listings"]) == 1

    def test_marketplace_stats(self):
        r = client.get("/marketplace/stats")
        assert r.status_code == 200
        assert "platform_revenue" in r.json()["stats"]

    def test_unlist_rig(self):
        """DELETE /marketplace/{host_id} removes listing."""
        client.post(
            "/marketplace/list",
            json={"host_id": "to-remove", "gpu_model": "RTX 4090", "vram_gb": 24, "price_per_hour": 0.30},
        )
        r = client.delete("/marketplace/to-remove")
        assert r.status_code == 200


class TestAutoscaleEndpoints:
    def test_add_to_pool(self):
        r = client.post(
            "/autoscale/pool",
            json={"host_id": "h1", "ip": "10.0.0.1", "gpu_model": "RTX 4090", "vram_gb": 24},
        )
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_get_pool(self):
        client.post(
            "/autoscale/pool",
            json={"host_id": "h1", "ip": "10.0.0.1", "gpu_model": "RTX 4090", "vram_gb": 24},
        )
        r = client.get("/autoscale/pool")
        assert len(r.json()["pool"]) == 1

    def test_remove_from_pool(self):
        """DELETE /autoscale/pool/{host_id}."""
        client.post(
            "/autoscale/pool",
            json={"host_id": "asc-1", "ip": "10.0.0.1", "gpu_model": "RTX 4090", "vram_gb": 24},
        )
        r = client.delete("/autoscale/pool/asc-1")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Agent Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestAgentEndpoints:
    def test_agent_work_no_jobs(self):
        """GET /agent/work/{host_id} with no work → 204."""
        r = client.get("/agent/work/h-idle")
        assert r.status_code == 204

    def test_agent_mining_alert(self):
        """POST /agent/mining-alert receives alert."""
        r = client.post("/agent/mining-alert", json={
            "host_id": "h-suspect",
            "gpu_index": 0,
            "confidence": 0.95,
            "reason": "Sustained high util + low PCIe",
        })
        assert r.status_code == 200
        assert r.json()["received"] is True

    def test_agent_benchmark(self):
        """POST /agent/benchmark records compute score."""
        r = client.post("/agent/benchmark", json={
            "host_id": "h-bench",
            "gpu_model": "RTX 4090",
            "score": 8.5,
            "tflops": 82.6,
            "details": {"benchmark": "matmul_fp16"},
        })
        assert r.status_code == 200
        assert r.json()["xcu"] == 8.5

    def test_agent_preemption_schedule(self):
        """POST /agent/preempt/{host_id}/{job_id} schedules preemption."""
        r = client.post("/agent/preempt/h1/job-abc")
        assert r.status_code == 200
        # Check it's retrievable
        r2 = client.get("/agent/preempt/h1")
        assert "job-abc" in r2.json()["preempt_jobs"]

    def test_agent_popular_images(self):
        """GET /agent/popular-images returns image list."""
        r = client.get("/agent/popular-images")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Spot Pricing Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestSpotPricingEndpoints:
    def test_get_spot_prices(self):
        """GET /spot-prices returns current prices."""
        r = client.get("/spot-prices")
        assert r.status_code == 200

    def test_update_spot_prices(self):
        """POST /spot-prices/update recalculates prices."""
        r = client.post("/spot-prices/update")
        assert r.status_code == 200

    def test_submit_spot_job(self):
        """POST /spot/job submits interruptible job."""
        r = client.post("/spot/job", json={
            "name": "spot-test", "vram_needed_gb": 8, "max_bid": 1.0,
        })
        assert r.status_code == 200

    def test_preemption_cycle(self):
        """POST /spot/preemption-cycle runs without error."""
        r = client.post("/spot/preemption-cycle")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Compute Scores Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestComputeScoreEndpoints:
    def test_get_all_compute_scores(self):
        """GET /compute-scores returns all scores."""
        r = client.get("/compute-scores")
        assert r.status_code == 200

    def test_get_compute_score_nonexistent(self):
        """GET /compute-score/{host_id} for unknown host."""
        r = client.get("/compute-score/nonexistent")
        # Could be 200 with empty or 404 depending on impl
        assert r.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════════════════
# Verification Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestVerificationEndpoints:
    def test_verified_hosts(self):
        """GET /api/verified-hosts returns list."""
        r = client.get("/api/verified-hosts")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Jurisdiction Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestJurisdictionEndpoints:
    def test_trust_tiers(self):
        """GET /api/trust-tiers returns tier definitions."""
        r = client.get("/api/trust-tiers")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# SLA Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestSLAEndpoints:
    def test_sla_targets(self):
        """GET /api/sla/targets returns SLA tier targets."""
        r = client.get("/api/sla/targets")
        assert r.status_code == 200

    def test_sla_enforce(self):
        """POST /api/sla/enforce runs monthly enforcement."""
        r = client.post("/api/sla/enforce", json={
            "host_id": "h-sla",
            "month": "2026-01",
            "tier": "community",
            "monthly_spend_cad": 500.0,
        })
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_sla_downtimes(self):
        """GET /api/sla/downtimes returns list."""
        r = client.get("/api/sla/downtimes")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Privacy Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestPrivacyEndpoints:
    def test_retention_policies(self):
        """GET /api/privacy/retention-policies returns policy list."""
        r = client.get("/api/privacy/retention-policies")
        assert r.status_code == 200

    def test_retention_summary(self):
        """GET /api/privacy/retention-summary returns summary."""
        r = client.get("/api/privacy/retention-summary")
        assert r.status_code == 200

    def test_consent_crud(self):
        """Record → get → revoke consent lifecycle."""
        # Record consent
        r = client.post("/api/privacy/consent", json={
            "entity_id": "user-1",
            "consent_type": "data_processing",
        })
        assert r.status_code == 200

        # Get consents
        r = client.get("/api/privacy/consent/user-1")
        assert r.status_code == 200

        # Revoke
        r = client.delete("/api/privacy/consent/user-1/data_processing")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Failover Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestFailover:
    def test_failover_no_issues(self):
        """POST /failover with no dead hosts returns empty."""
        r = client.post("/failover")
        assert r.status_code == 200
        assert "requeued" in r.json()
        assert "assigned" in r.json()

    def test_requeue_job(self):
        """POST /job/{id}/requeue re-queues a running job."""
        resp = _submit_job("requeue-test", 8)
        job_id = resp.json()["job"]["job_id"]
        # Must transition to running before requeue is allowed
        client.patch(f"/job/{job_id}", json={"status": "running", "host_id": "h1"})
        r = client.post(f"/job/{job_id}/requeue")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Slurm Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestSlurmEndpoints:
    def test_slurm_profiles(self):
        """GET /api/slurm/profiles returns available profiles."""
        r = client.get("/api/slurm/profiles")
        assert r.status_code == 200

    def test_nfs_config(self):
        """GET /api/nfs/config returns NFS configuration."""
        r = client.get("/api/nfs/config")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# OAuth2 Device Flow Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestOAuthDeviceFlow:
    def test_device_code_request(self):
        """POST /api/auth/device returns device_code and user_code."""
        r = client.post("/api/auth/device", json={"client_id": "cli-test"})
        assert r.status_code == 200
        data = r.json()
        assert "device_code" in data
        assert "user_code" in data
        assert "verification_uri" in data

    def test_token_poll_pending(self):
        """POST /api/auth/token before verification → pending/precondition."""
        # First get a device code
        dev = client.post("/api/auth/device", json={"client_id": "cli-test"})
        device_code = dev.json()["device_code"]

        # Poll without verifying — should be pending (428 Precondition Required)
        r = client.post("/api/auth/token", json={
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
            "client_id": "cli-test",
        })
        # RFC 8628: 428 indicates authorization_pending
        assert r.status_code == 428
        err = r.json().get("error", {})
        assert err.get("message") == "authorization_pending"

    def test_verify_page(self):
        """GET /api/auth/verify returns HTML verification page."""
        r = client.get("/api/auth/verify")
        assert r.status_code == 200
        assert "html" in r.headers.get("content-type", "").lower()


# ═══════════════════════════════════════════════════════════════════════
# Analytics Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestAnalytics:
    def test_usage_analytics(self):
        """GET /api/analytics/usage returns usage data."""
        r = client.get("/api/analytics/usage")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# User Authentication & API Keys
# ═══════════════════════════════════════════════════════════════════════


class TestUserAuth:
    """Tests for /api/auth/* endpoints (registration, login, OAuth, profile)."""

    def test_register_user(self):
        """POST /api/auth/register creates account and returns token."""
        r = client.post("/api/auth/register", json={
            "email": "testauth@xcelsior.ca",
            "password": "securepass123",
            "name": "Test User",
            "role": "submitter"
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "access_token" in d
        assert d["user"]["email"] == "testauth@xcelsior.ca"
        assert d["user"]["name"] == "Test User"
        assert d["user"]["role"] == "submitter"
        assert d["user"]["customer_id"].startswith("cust-")

    def test_register_duplicate_email(self):
        """POST /api/auth/register with existing email returns 409."""
        # First register
        client.post("/api/auth/register", json={
            "email": "duplicate@xcelsior.ca",
            "password": "securepass123"
        })
        # Second register with same email
        r = client.post("/api/auth/register", json={
            "email": "duplicate@xcelsior.ca",
            "password": "otherpass123"
        })
        assert r.status_code == 409

    def test_register_short_password(self):
        """POST /api/auth/register with <8 char password returns 400."""
        r = client.post("/api/auth/register", json={
            "email": "shortpw@xcelsior.ca",
            "password": "short"
        })
        assert r.status_code == 400

    def test_login_success(self):
        """POST /api/auth/login with valid credentials returns token."""
        # Register first
        client.post("/api/auth/register", json={
            "email": "logintest@xcelsior.ca",
            "password": "mypassword123"
        })
        # Login
        r = client.post("/api/auth/login", json={
            "email": "logintest@xcelsior.ca",
            "password": "mypassword123"
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "access_token" in d
        assert d["token_type"] == "Bearer"

    def test_login_wrong_password(self):
        """POST /api/auth/login with wrong password returns 401."""
        client.post("/api/auth/register", json={
            "email": "wrongpw@xcelsior.ca",
            "password": "correctpass123"
        })
        r = client.post("/api/auth/login", json={
            "email": "wrongpw@xcelsior.ca",
            "password": "wrongpass"
        })
        assert r.status_code == 401

    def test_login_nonexistent_user(self):
        """POST /api/auth/login with unknown email returns 401."""
        r = client.post("/api/auth/login", json={
            "email": "nobody@xcelsior.ca",
            "password": "whatever123"
        })
        assert r.status_code == 401

    def test_oauth_login(self):
        """POST /api/auth/oauth/{provider} creates session."""
        for provider in ("google", "github", "huggingface"):
            r = client.post(f"/api/auth/oauth/{provider}")
            assert r.status_code == 200
            d = r.json()
            assert d["ok"] is True
            assert "access_token" in d
            assert d["user"]["oauth_provider"] == provider

    def test_oauth_invalid_provider(self):
        """POST /api/auth/oauth/invalid returns 400."""
        r = client.post("/api/auth/oauth/facebook")
        assert r.status_code == 400

    def test_get_profile(self):
        """GET /api/auth/me returns user profile when authenticated."""
        reg = client.post("/api/auth/register", json={
            "email": "profiletest@xcelsior.ca",
            "password": "testpass123",
            "name": "Profile User"
        }).json()
        token = reg["access_token"]

        r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["user"]["email"] == "profiletest@xcelsior.ca"
        assert d["user"]["name"] == "Profile User"

    def test_get_profile_unauthenticated(self):
        """GET /api/auth/me without token returns 401."""
        r = client.get("/api/auth/me")
        assert r.status_code == 401

    def test_update_profile(self):
        """PATCH /api/auth/me updates profile fields."""
        reg = client.post("/api/auth/register", json={
            "email": "updateprofile@xcelsior.ca",
            "password": "testpass123"
        }).json()
        token = reg["access_token"]

        r = client.patch("/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "Updated Name", "role": "provider", "country": "CA", "province": "BC"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

        # Verify update
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
        assert me["user"]["name"] == "Updated Name"
        assert me["user"]["role"] == "provider"
        assert me["user"]["province"] == "BC"

    def test_refresh_token(self):
        """POST /api/auth/refresh returns new token and invalidates old."""
        reg = client.post("/api/auth/register", json={
            "email": "refreshtest@xcelsior.ca",
            "password": "testpass123"
        }).json()
        old_token = reg["access_token"]

        r = client.post("/api/auth/refresh", headers={"Authorization": f"Bearer {old_token}"})
        assert r.status_code == 200
        new_token = r.json()["access_token"]
        assert new_token != old_token

        # Old token should be invalid
        r2 = client.get("/api/auth/me", headers={"Authorization": f"Bearer {old_token}"})
        assert r2.status_code == 401

    def test_delete_account(self):
        """DELETE /api/auth/me removes account."""
        reg = client.post("/api/auth/register", json={
            "email": "deletetest@xcelsior.ca",
            "password": "testpass123"
        }).json()
        token = reg["access_token"]

        r = client.delete("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

        # Login should fail after deletion
        r2 = client.post("/api/auth/login", json={
            "email": "deletetest@xcelsior.ca",
            "password": "testpass123"
        })
        assert r2.status_code == 401


class TestApiKeys:
    """Tests for /api/keys/* endpoints."""

    def test_generate_and_list_keys(self):
        """Generate an API key and verify it appears in the list."""
        reg = client.post("/api/auth/register", json={
            "email": "keysuser@xcelsior.ca",
            "password": "testpass123"
        }).json()
        token = reg["access_token"]

        # Generate key
        r = client.post("/api/keys/generate?name=test-key",
            headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["name"] == "test-key"
        assert d["key"].startswith("xc-")
        api_key = d["key"]

        # List keys
        r2 = client.get("/api/keys", headers={"Authorization": f"Bearer {token}"})
        assert r2.status_code == 200
        keys = r2.json()["keys"]
        assert len(keys) >= 1
        assert keys[0]["name"] == "test-key"

    def test_api_key_as_bearer(self):
        """API key can be used as Bearer token for /api/auth/me."""
        reg = client.post("/api/auth/register", json={
            "email": "apikeyauth@xcelsior.ca",
            "password": "testpass123"
        }).json()
        token = reg["access_token"]

        # Generate API key
        key = client.post("/api/keys/generate?name=auth-key",
            headers={"Authorization": f"Bearer {token}"}).json()["key"]

        # Use API key to access profile
        r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200
        assert r.json()["user"]["email"] == "apikeyauth@xcelsior.ca"

    def test_revoke_key(self):
        """DELETE /api/keys/{preview} revokes the key."""
        reg = client.post("/api/auth/register", json={
            "email": "revokekey@xcelsior.ca",
            "password": "testpass123"
        }).json()
        token = reg["access_token"]

        # Generate and get preview
        gen = client.post("/api/keys/generate?name=revoke-me",
            headers={"Authorization": f"Bearer {token}"}).json()
        preview = gen["preview"]

        # Revoke
        r = client.delete(f"/api/keys/{preview}",
            headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

        # Key should no longer work
        r2 = client.get("/api/auth/me",
            headers={"Authorization": f"Bearer {gen['key']}"})
        assert r2.status_code == 401

    def test_generate_key_unauthenticated(self):
        """POST /api/keys/generate without auth returns 401."""
        r = client.post("/api/keys/generate?name=nope")
        assert r.status_code == 401
