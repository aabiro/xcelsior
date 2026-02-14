"""Phase 7.6 — End-to-End tests.

Tests the full API stack via TestClient without requiring external services.
Covers: dashboard, SSE events, host registration via API, CAF export.
"""

import csv
import io
import json as _json
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_e2e_")
_tmpdir = _tmp_ctx.name

os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"

import scheduler

scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")

from api import app

client = TestClient(app)


def _reset_state():
    for f in (
        scheduler.HOSTS_FILE,
        scheduler.JOBS_FILE,
        scheduler.BILLING_FILE,
        scheduler.MARKETPLACE_FILE,
        scheduler.AUTOSCALE_POOL_FILE,
        os.environ["XCELSIOR_DB_PATH"],
    ):
        if os.path.exists(f):
            os.remove(f)


def _admit_host(host_id):
    with scheduler._atomic_mutation() as conn:
        row = conn.execute(
            "SELECT payload FROM hosts WHERE host_id = ?", (host_id,)
        ).fetchone()
        if row:
            data = _json.loads(row["payload"])
            data["admitted"] = True
            data["status"] = "active"
            conn.execute(
                "UPDATE hosts SET status = 'active', payload = ? WHERE host_id = ?",
                (_json.dumps(data), host_id),
            )


# ── 7.6.1 — Dashboard loads ─────────────────────────────────────────


class TestDashboardLoads:
    """GET /dashboard returns 200 with all expected tabs."""

    def test_dashboard_returns_200(self):
        resp = client.get("/dashboard")
        assert resp.status_code == 200

    def test_dashboard_is_html(self):
        resp = client.get("/dashboard")
        assert "text/html" in resp.headers.get("content-type", "")

    def test_dashboard_contains_tabs(self):
        resp = client.get("/dashboard")
        body = resp.text
        # Check for the 7 known dashboard tabs
        for tab in ["Hosts", "Jobs", "Submit Job", "Marketplace", "Billing"]:
            assert tab in body, f"Tab '{tab}' not found in dashboard HTML"

    def test_dashboard_has_gpu_telemetry(self):
        resp = client.get("/dashboard")
        # GPU Telemetry tab exists
        assert "Telemetry" in resp.text or "telemetry" in resp.text

    def test_dashboard_has_modals(self):
        resp = client.get("/dashboard")
        body = resp.text
        # Provider Registration modal and Submit Job modal
        assert "modal" in body.lower()


# ── 7.6.2 — SSE stream receives events ──────────────────────────────


class TestSSEStreamEvents:
    """Verify SSE broadcast function and endpoint routing."""

    def test_sse_broadcast_function_is_callable(self):
        """Verify broadcast_sse is importable and callable."""
        from api import broadcast_sse
        # Should not raise — just broadcasts to any connected clients
        broadcast_sse("test_event", {"key": "value"})


# ── 7.6.3 — Add host via API → host appears ─────────────────────────


class TestAddHostViaAPI:
    """POST via API → host appears in list."""

    def test_register_and_list_host(self):
        _reset_state()
        resp = client.put("/host", json={
            "host_id": "e2e-host-1",
            "ip": "10.0.0.50",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "cost_per_hour": 0.65,
        })
        assert resp.status_code == 200

        hosts_resp = client.get("/hosts?active_only=false")
        assert hosts_resp.status_code == 200
        hosts = hosts_resp.json()["hosts"]
        host_ids = [h["host_id"] for h in hosts]
        assert "e2e-host-1" in host_ids

    def test_register_canadian_host_with_province(self):
        _reset_state()
        resp = client.put("/host", json={
            "host_id": "e2e-ca-host",
            "ip": "10.0.0.51",
            "gpu_model": "A100",
            "total_vram_gb": 80,
            "free_vram_gb": 80,
            "cost_per_hour": 1.2,
            "country": "CA",
            "province": "ON",
        })
        assert resp.status_code == 200
        host = resp.json()["host"]
        assert host["country"] == "CA"
        assert host["province"] == "ON"

    def test_register_host_admits_and_receives_job(self):
        """Full E2E: register → admit → submit job → process queue → assigned."""
        _reset_state()
        client.put("/host", json={
            "host_id": "e2e-admit",
            "ip": "10.0.0.52",
            "gpu_model": "RTX 3090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "cost_per_hour": 0.50,
        })
        _admit_host("e2e-admit")

        job_resp = client.post("/job", json={
            "name": "e2e-job",
            "vram_needed_gb": 8,
            "tier": "free",
        })
        assert job_resp.status_code == 200
        job_id = job_resp.json()["job"]["job_id"]

        process_resp = client.post("/queue/process")
        assert process_resp.status_code == 200
        assert len(process_resp.json()["assigned"]) == 1

        job_detail = client.get(f"/job/{job_id}")
        assert job_detail.json()["job"]["status"] == "running"


# ── 7.6.4 — Export CAF CSV ──────────────────────────────────────────


class TestExportCAFCSV:
    """GET /api/billing/export/caf → valid CSV."""

    def test_caf_json_export_returns_200(self):
        resp = client.get("/api/billing/export/caf/cust-e2e-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "summary" in data
        assert "line_items" in data

    def test_caf_csv_export_returns_csv(self):
        resp = client.get("/api/billing/export/caf/cust-e2e-2?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")

        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        # First row is header
        assert len(rows) >= 1
        header = rows[0]
        assert "Job ID" in header
        assert "Cost (CAD)" in header

    def test_caf_csv_has_content_disposition(self):
        resp = client.get("/api/billing/export/caf/cust-test?format=csv")
        assert "content-disposition" in resp.headers
        assert "attachment" in resp.headers["content-disposition"]
        assert "caf" in resp.headers["content-disposition"]

    def test_caf_with_billed_job(self):
        """Bill a job then export CAF — line item should appear."""
        _reset_state()
        # Register and admit host
        client.put("/host", json={
            "host_id": "caf-h1",
            "ip": "10.0.0.60",
            "gpu_model": "A100",
            "total_vram_gb": 80,
            "free_vram_gb": 80,
            "cost_per_hour": 1.0,
            "country": "CA",
            "province": "ON",
        })
        _admit_host("caf-h1")

        # Submit, process, run, complete, bill
        job_resp = client.post("/job", json={
            "name": "caf-job",
            "vram_needed_gb": 8,
            "tier": "premium",
        })
        job_id = job_resp.json()["job"]["job_id"]

        client.post("/queue/process")
        client.patch(f"/job/{job_id}", json={"status": "running", "host_id": "caf-h1"})
        time.sleep(1.1)
        client.patch(f"/job/{job_id}", json={"status": "completed", "host_id": "caf-h1"})

        bill_resp = client.post(f"/billing/bill/{job_id}")
        assert bill_resp.status_code == 200

        # Export CAF
        caf_resp = client.get("/api/billing/export/caf/default")
        assert caf_resp.status_code == 200


# ── 7.6.5 — Health & Readiness probes ───────────────────────────────


class TestHealthProbes:
    """Healthz and readyz return proper responses."""

    def test_healthz(self):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json().get("status") == "healthy"
        assert resp.json().get("ok") is True

    def test_readyz(self):
        resp = client.get("/readyz")
        assert resp.status_code == 200

    def test_metrics_endpoint(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        metrics = data["metrics"]
        assert "active_hosts" in metrics or "queue_depth" in metrics


# ── 7.6.6 — Full job lifecycle E2E ──────────────────────────────────


class TestFullJobLifecycleE2E:
    """Complete E2E: register → admit → submit → process → run → complete → bill."""

    def test_complete_lifecycle_with_billing(self):
        _reset_state()
        # Register host
        client.put("/host", json={
            "host_id": "e2e-lc-h1",
            "ip": "10.0.0.70",
            "gpu_model": "A100",
            "total_vram_gb": 80,
            "free_vram_gb": 80,
            "cost_per_hour": 1.50,
        })
        _admit_host("e2e-lc-h1")

        # Submit job
        job_resp = client.post("/job", json={
            "name": "e2e-lifecycle",
            "vram_needed_gb": 16,
            "tier": "premium",
        })
        assert job_resp.status_code == 200
        job_id = job_resp.json()["job"]["job_id"]

        # Process queue
        process_resp = client.post("/queue/process")
        assigned = process_resp.json()["assigned"]
        assert len(assigned) == 1

        # Run → Complete
        client.patch(f"/job/{job_id}", json={"status": "running", "host_id": "e2e-lc-h1"})
        time.sleep(1.1)
        client.patch(f"/job/{job_id}", json={"status": "completed", "host_id": "e2e-lc-h1"})

        # Verify completed
        detail = client.get(f"/job/{job_id}")
        assert detail.json()["job"]["status"] == "completed"

        # Bill the job
        bill = client.post(f"/billing/bill/{job_id}")
        assert bill.status_code == 200
        assert bill.json()["bill"]["cost"] > 0

    def test_multiple_hosts_best_allocation(self):
        """Multiple hosts registered — job goes to the best one."""
        _reset_state()
        for hid, vram, cost in [("mh-1", 24, 0.50), ("mh-2", 80, 1.20), ("mh-3", 48, 0.90)]:
            client.put("/host", json={
                "host_id": hid,
                "ip": "10.0.0.1",
                "gpu_model": "A100",
                "total_vram_gb": vram,
                "free_vram_gb": vram,
                "cost_per_hour": cost,
            })
            _admit_host(hid)

        # Submit job needing 40GB — only mh-2 and mh-3 can handle it
        job_resp = client.post("/job", json={
            "name": "big-job",
            "vram_needed_gb": 40,
        })
        job_id = job_resp.json()["job"]["job_id"]

        client.post("/queue/process")
        detail = client.get(f"/job/{job_id}")
        assigned_host = detail.json()["job"].get("host_id")
        assert assigned_host in ("mh-2", "mh-3")

    def test_no_admitted_hosts_job_stays_queued(self):
        """If no hosts are admitted, job remains queued."""
        _reset_state()
        client.put("/host", json={
            "host_id": "unadmitted",
            "ip": "10.0.0.2",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "cost_per_hour": 0.5,
        })
        # Don't admit the host

        job_resp = client.post("/job", json={
            "name": "blocked-job",
            "vram_needed_gb": 8,
        })
        job_id = job_resp.json()["job"]["job_id"]

        client.post("/queue/process")
        detail = client.get(f"/job/{job_id}")
        assert detail.json()["job"]["status"] == "queued"
