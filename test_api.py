"""Tests for Xcelsior API endpoints."""

import logging
import os
import tempfile

import pytest

# Use a TemporaryDirectory (auto-cleaned) for all scheduler data files
_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_api_test_")
_tmpdir = _tmp_ctx.name
os.environ["XCELSIOR_API_TOKEN"] = ""

import scheduler

# Patch file paths to use temp directory
scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")

# Reconfigure the logger so the FileHandler writes to the temp dir, not the repo
for _h in scheduler.log.handlers[:]:
    if isinstance(_h, logging.FileHandler):
        scheduler.log.removeHandler(_h)
        _h.close()
_fh = logging.FileHandler(scheduler.LOG_FILE)
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
scheduler.log.addHandler(_fh)

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_data():
    for f in (scheduler.HOSTS_FILE, scheduler.JOBS_FILE, scheduler.BILLING_FILE,
              scheduler.MARKETPLACE_FILE, scheduler.AUTOSCALE_POOL_FILE):
        if os.path.exists(f):
            os.remove(f)
    yield


class TestHealthEndpoint:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["name"] == "Xcelsior"


class TestHostEndpoints:
    def test_register_host(self):
        r = client.put("/host", json={
            "host_id": "h1", "ip": "10.0.0.1", "gpu_model": "RTX 4090",
            "total_vram_gb": 24, "free_vram_gb": 24
        })
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_list_hosts(self):
        client.put("/host", json={
            "host_id": "h1", "ip": "10.0.0.1", "gpu_model": "RTX 4090",
            "total_vram_gb": 24, "free_vram_gb": 24
        })
        r = client.get("/hosts")
        assert r.status_code == 200
        assert len(r.json()["hosts"]) == 1

    def test_remove_host(self):
        client.put("/host", json={
            "host_id": "h1", "ip": "10.0.0.1", "gpu_model": "RTX 4090",
            "total_vram_gb": 24, "free_vram_gb": 24
        })
        r = client.delete("/host/h1")
        assert r.status_code == 200

    def test_remove_nonexistent_host(self):
        r = client.delete("/host/nonexistent")
        assert r.status_code == 404


class TestJobEndpoints:
    def test_submit_job(self):
        r = client.post("/job", json={"name": "llama3", "vram_needed_gb": 16})
        assert r.status_code == 200
        assert r.json()["ok"]
        assert r.json()["job"]["status"] == "queued"

    def test_list_jobs(self):
        client.post("/job", json={"name": "llama3", "vram_needed_gb": 16})
        r = client.get("/jobs")
        assert len(r.json()["jobs"]) == 1

    def test_get_job(self):
        resp = client.post("/job", json={"name": "llama3", "vram_needed_gb": 16})
        job_id = resp.json()["job"]["job_id"]
        r = client.get(f"/job/{job_id}")
        assert r.status_code == 200
        assert r.json()["job"]["name"] == "llama3"

    def test_get_nonexistent_job(self):
        r = client.get("/job/nonexistent")
        assert r.status_code == 404

    def test_process_queue(self):
        client.put("/host", json={
            "host_id": "h1", "ip": "10.0.0.1", "gpu_model": "RTX 4090",
            "total_vram_gb": 24, "free_vram_gb": 24
        })
        client.post("/job", json={"name": "llama3", "vram_needed_gb": 16})
        r = client.post("/queue/process")
        assert r.status_code == 200
        assert len(r.json()["assigned"]) == 1


class TestBillingEndpoints:
    def test_billing_empty(self):
        r = client.get("/billing")
        assert r.status_code == 200
        assert r.json()["total_revenue"] == 0


class TestDashboard:
    def test_dashboard_returns_html(self):
        r = client.get("/dashboard")
        assert r.status_code == 200
        assert "XCELSIOR" in r.text


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


class TestMarketplaceEndpoints:
    def test_list_rig(self):
        r = client.post("/marketplace/list", json={
            "host_id": "h1", "gpu_model": "RTX 4090",
            "vram_gb": 24, "price_per_hour": 0.30
        })
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_get_marketplace(self):
        client.post("/marketplace/list", json={
            "host_id": "h1", "gpu_model": "RTX 4090",
            "vram_gb": 24, "price_per_hour": 0.30
        })
        r = client.get("/marketplace")
        assert len(r.json()["listings"]) == 1

    def test_marketplace_stats(self):
        r = client.get("/marketplace/stats")
        assert r.status_code == 200
        assert "platform_revenue" in r.json()["stats"]


class TestAutoscaleEndpoints:
    def test_add_to_pool(self):
        r = client.post("/autoscale/pool", json={
            "host_id": "h1", "ip": "10.0.0.1",
            "gpu_model": "RTX 4090", "vram_gb": 24
        })
        assert r.status_code == 200
        assert r.json()["ok"]

    def test_get_pool(self):
        client.post("/autoscale/pool", json={
            "host_id": "h1", "ip": "10.0.0.1",
            "gpu_model": "RTX 4090", "vram_gb": 24
        })
        r = client.get("/autoscale/pool")
        assert len(r.json()["pool"]) == 1


class TestJobStatusApiValidation:
    def test_invalid_status_returns_400(self):
        """PATCH /job/{id} with invalid status should return 400, not 200."""
        resp = client.post("/job", json={"name": "test", "vram_needed_gb": 8})
        job_id = resp.json()["job"]["job_id"]
        r = client.patch(f"/job/{job_id}", json={"status": "cancelled"})
        assert r.status_code == 400
        assert "Invalid status" in r.json()["detail"]
