"""Phase 3 — spot capacity gating, preemption lifecycle, host defaults."""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_spot_cap_")
_tmpdir = _tmp_ctx.name

os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"

import scheduler


scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")

for h in scheduler.log.handlers[:]:
    if isinstance(h, logging.FileHandler):
        scheduler.log.removeHandler(h)
        h.close()


def _admit(host_id: str, **extra):
    scheduler._set_host_fields(host_id, admitted=True, **extra)


def _host(host_id: str, *, spot_enabled: bool = True, spot_gpu_slots: int = 1, gpu_count: int = 1):
    scheduler.register_host(host_id, "10.0.0.1", "RTX 4090", 24, 24, cost_per_hour=0.55)
    _admit(host_id, spot_enabled=spot_enabled, spot_gpu_slots=spot_gpu_slots, gpu_count=gpu_count)


@pytest.fixture(autouse=True)
def clean_data():
    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM hosts")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM state")
    for f in (
        scheduler.HOSTS_FILE,
        scheduler.JOBS_FILE,
        scheduler.BILLING_FILE,
        scheduler.MARKETPLACE_FILE,
        os.environ["XCELSIOR_DB_PATH"],
    ):
        if os.path.exists(f):
            os.remove(f)
    yield


class TestHostSpotDefaults:
    def test_register_host_sets_spot_fields(self):
        entry = scheduler.register_host("d1", "10.0.0.2", "RTX 4090", 24, 24)
        assert entry.get("spot_enabled") is True
        assert entry.get("spot_gpu_slots") == 1
        assert entry.get("gpu_count") == 1


class TestSpotCapacityGating:
    def test_spot_disabled_host_skipped(self):
        _host("spot-off", spot_enabled=False)
        _host("spot-on", spot_enabled=True)
        job = scheduler.submit_job("spot-job", 8, pricing_mode="spot", gpu_model="RTX 4090")
        hosts = scheduler.list_hosts()
        picked = scheduler.allocate(job, hosts)
        assert picked is not None
        assert picked["host_id"] == "spot-on"

    def test_spot_pool_exhausted_skips_host(self):
        _host("h1", spot_gpu_slots=1, gpu_count=1)
        running = scheduler.submit_job("occupier", 8, pricing_mode="spot")
        scheduler.update_job_status(running["job_id"], "running", host_id="h1")
        queued = scheduler.submit_job("waiter", 8, pricing_mode="spot", gpu_model="RTX 4090")
        picked = scheduler.allocate(queued, scheduler.list_hosts())
        assert picked is None

    def test_on_demand_unaffected_by_spot_pool(self):
        _host("h1", spot_gpu_slots=0, gpu_count=1)
        job = scheduler.submit_job("od-job", 8, pricing_mode="on_demand", gpu_model="RTX 4090")
        picked = scheduler.allocate(job, scheduler.list_hosts())
        assert picked is not None
        assert picked["host_id"] == "h1"

    def test_process_queue_respects_spot_capacity(self):
        _host("good")
        _host("bad", spot_enabled=False)
        job = scheduler.submit_job("pq-spot", 8, pricing_mode="spot", gpu_model="RTX 4090")
        assigned = scheduler.process_queue()
        assert len(assigned) == 1
        assert assigned[0][0]["job_id"] == job["job_id"]
        assert assigned[0][1]["host_id"] == "good"
        refreshed = scheduler.get_job(job["job_id"])
        assert refreshed.get("spot_rate_cad") is not None
        assert refreshed.get("pricing_mode") == "spot"


class TestPreemptionLifecycle:
    def test_preempt_increments_count_and_requeues(self):
        _host("pre-h")
        job = scheduler.submit_job("preempt-me", 8, pricing_mode="spot")
        scheduler.update_job_status(job["job_id"], "running", host_id="pre-h")
        result = scheduler.preempt_job(job["job_id"])
        assert result is not None
        assert result["status"] == "queued"
        assert result.get("preempted_at") is not None
        assert result.get("preemption_count") == 1
        assert result.get("pricing_mode") == "spot"
        assert result.get("preemptible") is True