"""Phase 7 — capacity-based spot preemption (on-demand contention)."""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_preempt_cap_")
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


class TestCapacityPreemption:
    def test_on_demand_preempts_running_spot_on_single_gpu(self):
        _host("h1", gpu_count=1, spot_gpu_slots=1)
        spot = scheduler.submit_job("spot-runner", 8, pricing_mode="spot", gpu_model="RTX 4090")
        scheduler.update_job_status(spot["job_id"], "running", host_id="h1")

        od = scheduler.submit_job("od-waiter", 8, pricing_mode="on_demand", gpu_model="RTX 4090")
        assigned = scheduler.process_queue()

        assert len(assigned) == 1
        assert assigned[0][0]["job_id"] == od["job_id"]
        assert assigned[0][1]["host_id"] == "h1"

        spot_refreshed = scheduler.get_job(spot["job_id"])
        assert spot_refreshed["status"] == "queued"
        assert spot_refreshed.get("preemption_count") == 1

    def test_spot_never_preempts_on_demand(self):
        _host("h1", gpu_count=1, spot_gpu_slots=1)
        od = scheduler.submit_job("od-runner", 8, pricing_mode="on_demand", gpu_model="RTX 4090")
        scheduler.update_job_status(od["job_id"], "running", host_id="h1")

        spot = scheduler.submit_job("spot-waiter", 8, pricing_mode="spot", gpu_model="RTX 4090")
        host, preempted = scheduler.allocate_with_preemption(spot, scheduler.list_hosts())

        assert host is None
        assert preempted == []
        assert scheduler.get_job(od["job_id"])["status"] == "running"

    def test_lowest_priority_spot_preempted_first(self):
        _host("h1", gpu_count=2, spot_gpu_slots=2)
        low = scheduler.submit_job("spot-low", 8, pricing_mode="spot", gpu_model="RTX 4090")
        high = scheduler.submit_job("spot-high", 8, pricing_mode="spot", gpu_model="RTX 4090")
        scheduler._set_job_fields(low["job_id"], priority=0)
        scheduler._set_job_fields(high["job_id"], priority=2)
        scheduler.update_job_status(low["job_id"], "running", host_id="h1")
        scheduler.update_job_status(high["job_id"], "running", host_id="h1")

        od = scheduler.submit_job("od-need", 8, pricing_mode="on_demand", gpu_model="RTX 4090")
        host, preempted = scheduler.allocate_with_preemption(od, scheduler.list_hosts())

        assert host is not None
        assert host["host_id"] == "h1"
        assert len(preempted) == 1
        assert preempted[0]["job_id"] == low["job_id"]
        assert scheduler.get_job(high["job_id"])["status"] == "running"

    def test_identify_preemption_candidates_respects_shortfall(self):
        _host("h1", gpu_count=1, spot_gpu_slots=1)
        spot = scheduler.submit_job("spot-only", 8, pricing_mode="spot")
        scheduler.update_job_status(spot["job_id"], "running", host_id="h1")

        jobs = scheduler.list_jobs()
        victims = scheduler.identify_preemption_candidates("h1", 1, jobs)
        assert len(victims) == 1
        assert victims[0]["job_id"] == spot["job_id"]

        no_victims = scheduler.identify_preemption_candidates("h1", 2, jobs)
        assert no_victims == []

    def test_identify_preemptible_jobs_finds_contention(self):
        _host("h1", gpu_count=1, spot_gpu_slots=1)
        spot = scheduler.submit_job("spot-block", 8, pricing_mode="spot", gpu_model="RTX 4090")
        scheduler.update_job_status(spot["job_id"], "running", host_id="h1")
        scheduler.submit_job("od-blocked", 8, pricing_mode="on_demand", gpu_model="RTX 4090")

        preemptible = scheduler.identify_preemptible_jobs()
        assert len(preemptible) == 1
        assert preemptible[0][0]["job_id"] == spot["job_id"]

    def test_drain_preempts_spot_before_on_demand(self):
        _host("h1", gpu_count=2, spot_gpu_slots=2)
        spot = scheduler.submit_job("drain-spot", 8, pricing_mode="spot")
        od = scheduler.submit_job("drain-od", 8, pricing_mode="on_demand")
        scheduler.update_job_status(spot["job_id"], "running", host_id="h1")
        scheduler.update_job_status(od["job_id"], "running", host_id="h1")

        scheduler.set_host_draining("h1", draining=True)
        victims = scheduler.identify_drain_preemption_candidates("h1")
        assert [v["job_id"] for v in victims] == [spot["job_id"], od["job_id"]]