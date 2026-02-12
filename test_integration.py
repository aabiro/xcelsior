"""Integration coverage for API + scheduler lifecycle interactions."""

import os
import tempfile
import time

from fastapi.testclient import TestClient

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_integration_")
_tmpdir = _tmp_ctx.name

os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"

import scheduler
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


def test_job_lifecycle_and_billing_via_api():
    _reset_state()
    client.put(
        "/host",
        json={
            "host_id": "h-int-1",
            "ip": "10.0.0.9",
            "gpu_model": "A100",
            "total_vram_gb": 80,
            "free_vram_gb": 80,
            "cost_per_hour": 1.0,
        },
    )

    create = client.post("/job", json={"name": "job-int", "vram_needed_gb": 8, "tier": "premium"})
    job_id = create.json()["job"]["job_id"]

    process = client.post("/queue/process")
    assert process.status_code == 200
    assert len(process.json()["assigned"]) == 1

    client.patch(f"/job/{job_id}", json={"status": "running", "host_id": "h-int-1"})
    time.sleep(1.1)
    client.patch(f"/job/{job_id}", json={"status": "completed", "host_id": "h-int-1"})

    billed = client.post(f"/billing/bill/{job_id}")
    assert billed.status_code == 200
    assert billed.json()["bill"]["cost"] > 0


def test_marketplace_stats_with_mixed_platform_cuts():
    _reset_state()
    scheduler.list_rig("m1", "RTX 4090", 24, 1.0, owner="alice")
    scheduler.list_rig("m2", "RTX 3090", 24, 1.0, owner="bob")

    listings = scheduler.load_marketplace()
    for listing in listings:
        listing["platform_cut"] = 0.1 if listing["host_id"] == "m1" else 0.35
    scheduler.save_marketplace(listings)

    j1 = scheduler.submit_job("mk-a", 4)
    scheduler.update_job_status(j1["job_id"], "running", host_id="m1")
    time.sleep(1.1)
    scheduler.update_job_status(j1["job_id"], "completed")
    scheduler.marketplace_bill(j1["job_id"])

    j2 = scheduler.submit_job("mk-b", 4)
    scheduler.update_job_status(j2["job_id"], "running", host_id="m2")
    time.sleep(1.1)
    scheduler.update_job_status(j2["job_id"], "completed")
    scheduler.marketplace_bill(j2["job_id"])

    stats_resp = client.get("/marketplace/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()["stats"]
    assert stats["total_jobs_completed"] == 2
    assert stats["platform_revenue"] > 0
