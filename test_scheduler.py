"""Tests for Xcelsior scheduler — core logic, all phases."""

import json
import os
import tempfile
import time

import pytest

# Redirect data files to temp directory before importing scheduler
_tmpdir = tempfile.mkdtemp(prefix="xcelsior_test_")

os.environ.setdefault("XCELSIOR_API_TOKEN", "")

import scheduler

# Patch file paths to use temp directory
scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")


@pytest.fixture(autouse=True)
def clean_data():
    """Remove data files before each test for isolation."""
    for f in (scheduler.HOSTS_FILE, scheduler.JOBS_FILE, scheduler.BILLING_FILE,
              scheduler.MARKETPLACE_FILE, scheduler.AUTOSCALE_POOL_FILE):
        if os.path.exists(f):
            os.remove(f)
    yield


# ── Phase 1: allocate ────────────────────────────────────────────────

class TestAllocate:
    def test_no_hosts_returns_none(self):
        assert scheduler.allocate({"vram_needed_gb": 8}, []) is None

    def test_no_fit_returns_none(self):
        hosts = [{"host_id": "h1", "free_vram_gb": 4, "latency_ms": 10, "cost_per_hour": 0.20}]
        assert scheduler.allocate({"vram_needed_gb": 8}, hosts) is None

    def test_picks_best_by_vram(self):
        hosts = [
            {"host_id": "h1", "free_vram_gb": 16, "latency_ms": 10, "gpu_model": "A", "cost_per_hour": 0.20},
            {"host_id": "h2", "free_vram_gb": 24, "latency_ms": 10, "gpu_model": "B", "cost_per_hour": 0.20},
        ]
        result = scheduler.allocate({"name": "job1", "vram_needed_gb": 8}, hosts)
        assert result["host_id"] == "h2"

    def test_picks_lower_latency_when_vram_equal(self):
        hosts = [
            {"host_id": "h1", "free_vram_gb": 24, "latency_ms": 50, "gpu_model": "A", "cost_per_hour": 0.20},
            {"host_id": "h2", "free_vram_gb": 24, "latency_ms": 5, "gpu_model": "B", "cost_per_hour": 0.20},
        ]
        result = scheduler.allocate({"name": "job1", "vram_needed_gb": 8}, hosts)
        assert result["host_id"] == "h2"

    def test_picks_lower_cost_when_vram_and_latency_equal(self):
        hosts = [
            {"host_id": "h1", "free_vram_gb": 24, "latency_ms": 10, "gpu_model": "A", "cost_per_hour": 0.50},
            {"host_id": "h2", "free_vram_gb": 24, "latency_ms": 10, "gpu_model": "B", "cost_per_hour": 0.20},
        ]
        result = scheduler.allocate({"name": "job1", "vram_needed_gb": 8}, hosts)
        assert result["host_id"] == "h2"


# ── Phase 2: Host Registry ──────────────────────────────────────────

class TestHostRegistry:
    def test_register_and_list(self):
        scheduler.register_host("h1", "10.0.0.1", "RTX 4090", 24, 24)
        hosts = scheduler.list_hosts()
        assert len(hosts) == 1
        assert hosts[0]["host_id"] == "h1"
        assert hosts[0]["gpu_model"] == "RTX 4090"
        assert hosts[0]["status"] == "active"

    def test_register_updates_existing(self):
        scheduler.register_host("h1", "10.0.0.1", "RTX 4090", 24, 24)
        scheduler.register_host("h1", "10.0.0.2", "RTX 4090", 24, 20)
        hosts = scheduler.list_hosts()
        assert len(hosts) == 1
        assert hosts[0]["ip"] == "10.0.0.2"
        assert hosts[0]["free_vram_gb"] == 20

    def test_remove_host(self):
        scheduler.register_host("h1", "10.0.0.1", "RTX 4090", 24, 24)
        scheduler.remove_host("h1")
        hosts = scheduler.list_hosts(active_only=False)
        assert len(hosts) == 0

    def test_list_active_only(self):
        scheduler.register_host("h1", "10.0.0.1", "RTX 4090", 24, 24)
        hosts = scheduler.load_hosts(active_only=False)
        hosts[0]["status"] = "dead"
        scheduler.save_hosts(hosts)
        assert len(scheduler.list_hosts(active_only=True)) == 0
        assert len(scheduler.list_hosts(active_only=False)) == 1


# ── Phase 3: Job Queue ──────────────────────────────────────────────

class TestJobQueue:
    def test_submit_and_list(self):
        job = scheduler.submit_job("llama3", 16, priority=1)
        assert job["name"] == "llama3"
        assert job["status"] == "queued"
        assert job["vram_needed_gb"] == 16
        jobs = scheduler.list_jobs()
        assert len(jobs) == 1

    def test_fifo_within_same_priority(self):
        scheduler.submit_job("job-a", 8, priority=0)
        time.sleep(0.01)
        scheduler.submit_job("job-b", 8, priority=0)
        nxt = scheduler.get_next_job()
        assert nxt["name"] == "job-a"

    def test_higher_priority_goes_first(self):
        scheduler.submit_job("low", 8, priority=0)
        scheduler.submit_job("high", 8, priority=2)
        nxt = scheduler.get_next_job()
        assert nxt["name"] == "high"

    def test_tier_overrides_priority(self):
        job = scheduler.submit_job("test", 8, priority=0, tier="urgent")
        assert job["priority"] == 3
        assert job["tier"] == "urgent"

    def test_invalid_tier_defaults_to_free(self):
        job = scheduler.submit_job("test", 8, tier="nonexistent")
        assert job["tier"] == "free"
        assert job["priority"] == 0


# ── Phase 3+1: Process Queue ────────────────────────────────────────

class TestProcessQueue:
    def test_process_assigns_job(self):
        scheduler.register_host("h1", "127.0.0.1", "RTX 4090", 24, 24)
        scheduler.submit_job("llama3", 16)
        assigned = scheduler.process_queue()
        assert len(assigned) == 1
        assert assigned[0][0]["name"] == "llama3"
        assert assigned[0][1]["host_id"] == "h1"

    def test_process_skips_when_no_vram(self):
        scheduler.register_host("h1", "127.0.0.1", "RTX 3060", 12, 8)
        scheduler.submit_job("big-model", 24)
        assigned = scheduler.process_queue()
        assert len(assigned) == 0

    def test_process_multiple_jobs(self):
        scheduler.register_host("h1", "127.0.0.1", "RTX 4090", 24, 24)
        scheduler.register_host("h2", "127.0.0.2", "A100", 80, 80)
        scheduler.submit_job("job-a", 16)
        scheduler.submit_job("job-b", 8)
        assigned = scheduler.process_queue()
        assert len(assigned) == 2


# ── Phase 7: Job status updates ──────────────────────────────────────

class TestJobStatus:
    def test_update_status(self):
        job = scheduler.submit_job("test", 8)
        scheduler.update_job_status(job["job_id"], "running", host_id="h1")
        jobs = scheduler.list_jobs(status="running")
        assert len(jobs) == 1
        assert jobs[0]["host_id"] == "h1"
        assert jobs[0]["started_at"] is not None

    def test_update_invalid_status_rejected(self):
        job = scheduler.submit_job("test", 8)
        with pytest.raises(ValueError):
            scheduler.update_job_status(job["job_id"], "invalid_status")
        jobs = scheduler.list_jobs()
        assert jobs[0]["status"] == "queued"

    def test_completed_sets_completed_at(self):
        job = scheduler.submit_job("test", 8)
        scheduler.update_job_status(job["job_id"], "running")
        scheduler.update_job_status(job["job_id"], "completed")
        jobs = scheduler.list_jobs(status="completed")
        assert len(jobs) == 1
        assert jobs[0]["completed_at"] is not None


# ── Phase 8: Billing ────────────────────────────────────────────────

class TestBilling:
    def test_bill_completed_job(self):
        job = scheduler.submit_job("test", 8)
        scheduler.update_job_status(job["job_id"], "running", host_id="h1")
        time.sleep(0.01)
        scheduler.update_job_status(job["job_id"], "completed")
        record = scheduler.bill_job(job["job_id"])
        assert record is not None
        assert record["job_id"] == job["job_id"]
        assert record["cost"] >= 0

    def test_bill_job_missing_timestamps(self):
        job = scheduler.submit_job("test", 8)
        assert scheduler.bill_job(job["job_id"]) is None

    def test_no_duplicate_billing(self):
        job = scheduler.submit_job("test", 8)
        scheduler.update_job_status(job["job_id"], "running", host_id="h1")
        time.sleep(0.01)
        scheduler.update_job_status(job["job_id"], "completed")
        first = scheduler.bill_job(job["job_id"])
        assert first is not None
        second = scheduler.bill_job(job["job_id"])
        assert second is None  # Already billed

    def test_bill_all_completed(self):
        j1 = scheduler.submit_job("a", 8)
        j2 = scheduler.submit_job("b", 8)
        for j in (j1, j2):
            scheduler.update_job_status(j["job_id"], "running")
            time.sleep(0.01)
            scheduler.update_job_status(j["job_id"], "completed")
        bills = scheduler.bill_all_completed()
        assert len(bills) == 2

    def test_tier_multiplier_applied(self):
        scheduler.register_host("h1", "127.0.0.1", "RTX 4090", 24, 24, cost_per_hour=1.0)
        job = scheduler.submit_job("test", 8, tier="urgent")
        scheduler.update_job_status(job["job_id"], "running", host_id="h1")
        time.sleep(0.01)
        scheduler.update_job_status(job["job_id"], "completed")
        record = scheduler.bill_job(job["job_id"])
        assert record["tier_multiplier"] == 2.0

    def test_total_revenue(self):
        job = scheduler.submit_job("test", 8)
        scheduler.update_job_status(job["job_id"], "running")
        time.sleep(0.01)
        scheduler.update_job_status(job["job_id"], "completed")
        scheduler.bill_job(job["job_id"])
        revenue = scheduler.get_total_revenue()
        assert revenue >= 0


# ── Phase 14: Failover ───────────────────────────────────────────────

class TestFailover:
    def test_requeue_running_job(self):
        job = scheduler.submit_job("test", 8)
        scheduler.update_job_status(job["job_id"], "running", host_id="h1")
        result = scheduler.requeue_job(job["job_id"])
        assert result is not None
        assert result["status"] == "queued"
        assert result["retries"] == 1
        assert result["host_id"] is None

    def test_requeue_failed_job(self):
        job = scheduler.submit_job("test", 8)
        scheduler.update_job_status(job["job_id"], "failed")
        result = scheduler.requeue_job(job["job_id"])
        assert result is not None
        assert result["status"] == "queued"

    def test_requeue_completed_job_rejected(self):
        job = scheduler.submit_job("test", 8)
        scheduler.update_job_status(job["job_id"], "running")
        scheduler.update_job_status(job["job_id"], "completed")
        result = scheduler.requeue_job(job["job_id"])
        assert result is None

    def test_requeue_queued_job_rejected(self):
        job = scheduler.submit_job("test", 8)
        result = scheduler.requeue_job(job["job_id"])
        assert result is None

    def test_max_retries_exceeded(self):
        job = scheduler.submit_job("test", 8)
        # Exhaust retries (max_retries=3)
        for _ in range(4):
            scheduler.update_job_status(job["job_id"], "running")
            result = scheduler.requeue_job(job["job_id"])
            if result is None:
                break
        # After 3 retries, 4th should fail permanently
        jobs = scheduler.list_jobs()
        target = [j for j in jobs if j["job_id"] == job["job_id"]][0]
        assert target["status"] == "failed"


# ── Phase 15: Priority Tiers ────────────────────────────────────────

class TestPriorityTiers:
    def test_list_tiers(self):
        tiers = scheduler.list_tiers()
        assert "free" in tiers
        assert "urgent" in tiers
        assert tiers["urgent"]["priority"] == 3
        assert tiers["urgent"]["multiplier"] == 2.0

    def test_get_tier_info(self):
        info = scheduler.get_tier_info("premium")
        assert info["priority"] == 2
        assert info["multiplier"] == 1.5

    def test_get_tier_by_priority(self):
        assert scheduler.get_tier_by_priority(0) == "free"
        assert scheduler.get_tier_by_priority(3) == "urgent"


# ── Phase 17: Marketplace ───────────────────────────────────────────

class TestMarketplace:
    def test_list_rig(self):
        listing = scheduler.list_rig("h1", "RTX 4090", 24, 0.30, owner="alice")
        assert listing["host_id"] == "h1"
        assert listing["active"]

    def test_unlist_rig(self):
        scheduler.list_rig("h1", "RTX 4090", 24, 0.30)
        assert scheduler.unlist_rig("h1")
        listings = scheduler.get_marketplace(active_only=True)
        assert len(listings) == 0

    def test_marketplace_stats_includes_platform_revenue(self):
        scheduler.list_rig("h1", "RTX 4090", 24, 0.30, owner="alice")
        stats = scheduler.marketplace_stats()
        assert "platform_revenue" in stats
        assert stats["platform_cut_pct"] == scheduler.PLATFORM_CUT


# ── Phase 18: Canada-Only Toggle ────────────────────────────────────

class TestCanadaOnly:
    def test_toggle(self):
        scheduler.set_canada_only(True)
        assert scheduler.CANADA_ONLY is True
        scheduler.set_canada_only(False)
        assert scheduler.CANADA_ONLY is False

    def test_register_host_ca(self):
        entry = scheduler.register_host_ca("h1", "10.0.0.1", "RTX 4090", 24, 24, country="CA")
        assert entry["country"] == "CA"

    def test_list_hosts_filtered_canada(self):
        scheduler.register_host_ca("h1", "10.0.0.1", "RTX 4090", 24, 24, country="CA")
        scheduler.register_host_ca("h2", "10.0.0.2", "RTX 3090", 24, 24, country="US")
        ca_hosts = scheduler.list_hosts_filtered(canada_only=True)
        assert len(ca_hosts) == 1
        assert ca_hosts[0]["host_id"] == "h1"


# ── Phase 19: Auto-Scaling ──────────────────────────────────────────

class TestAutoscale:
    def test_add_to_pool(self):
        entry = scheduler.add_to_pool("h1", "10.0.0.1", "RTX 4090", 24)
        assert entry["host_id"] == "h1"
        assert entry["provisioned"] is False
        pool = scheduler.load_autoscale_pool()
        assert len(pool) == 1

    def test_remove_from_pool(self):
        scheduler.add_to_pool("h1", "10.0.0.1", "RTX 4090", 24)
        scheduler.remove_from_pool("h1")
        pool = scheduler.load_autoscale_pool()
        assert len(pool) == 0

    def test_provision_host(self):
        pool_entry = scheduler.add_to_pool("h1", "10.0.0.1", "RTX 4090", 24)
        scheduler.provision_host(pool_entry)
        hosts = scheduler.list_hosts(active_only=True)
        assert len(hosts) == 1
        assert hosts[0]["host_id"] == "h1"
        pool = scheduler.load_autoscale_pool()
        assert pool[0]["provisioned"] is True

    def test_deprovision_host(self):
        pool_entry = scheduler.add_to_pool("h1", "10.0.0.1", "RTX 4090", 24)
        scheduler.provision_host(pool_entry)
        scheduler.deprovision_host("h1")
        hosts = scheduler.list_hosts(active_only=False)
        assert len(hosts) == 0
        pool = scheduler.load_autoscale_pool()
        assert pool[0]["provisioned"] is False

    def test_autoscale_up(self):
        scheduler.add_to_pool("h1", "10.0.0.1", "RTX 4090", 24)
        scheduler.submit_job("big-job", 16)
        provisioned = scheduler.autoscale_up()
        assert len(provisioned) == 1

    def test_autoscale_down_idle(self):
        pool_entry = scheduler.add_to_pool("h1", "10.0.0.1", "RTX 4090", 24)
        scheduler.provision_host(pool_entry)
        # Mark as autoscaled
        hosts = scheduler.load_hosts(active_only=False)
        hosts[0]["autoscaled"] = True
        scheduler.save_hosts(hosts)
        deprovisioned = scheduler.autoscale_down()
        assert "h1" in deprovisioned


# ── Input Validation ─────────────────────────────────────────────────

class TestValidation:
    def test_validate_name_accepts_valid(self):
        scheduler._validate_name("xcelsior/llama3:latest", "image")
        scheduler._validate_name("xcl-abc123", "container")
        scheduler._validate_name("my-model_v2.0", "model")

    def test_validate_name_rejects_shell_injection(self):
        with pytest.raises(ValueError):
            scheduler._validate_name("model; rm -rf /", "image")
        with pytest.raises(ValueError):
            scheduler._validate_name("model$(whoami)", "image")
        with pytest.raises(ValueError):
            scheduler._validate_name("", "image")


# ── Docker Image Builder (Phase 16) ─────────────────────────────────

class TestDockerBuilder:
    def test_generate_dockerfile(self):
        df = scheduler.generate_dockerfile("llama3")
        assert "FROM python:3.11-slim" in df
        assert 'model="llama3"' in df

    def test_generate_dockerfile_with_quantize(self):
        df = scheduler.generate_dockerfile("llama3", quantize="gguf")
        assert "llama-cpp-python" in df

    def test_list_builds_empty(self):
        assert scheduler.list_builds() == []


# ── Concurrency and Locking ─────────────────────────────────────────

class TestSaveJsonLocking:
    def test_save_json_creates_file(self):
        """_save_json should create the file if it doesn't exist."""
        path = os.path.join(_tmpdir, "lock_test.json")
        if os.path.exists(path):
            os.remove(path)
        scheduler._save_json(path, [{"a": 1}])
        assert os.path.exists(path)
        data = scheduler._load_json(path)
        assert data == [{"a": 1}]

    def test_save_json_overwrites(self):
        """_save_json should fully replace the file contents."""
        path = os.path.join(_tmpdir, "lock_test2.json")
        scheduler._save_json(path, [{"a": 1}])
        scheduler._save_json(path, [{"b": 2}])
        data = scheduler._load_json(path)
        assert data == [{"b": 2}]

    def test_save_json_no_leftover_bytes(self):
        """Writing shorter data should not leave old bytes behind."""
        path = os.path.join(_tmpdir, "lock_test3.json")
        scheduler._save_json(path, list(range(1000)))
        scheduler._save_json(path, [1])
        data = scheduler._load_json(path)
        assert data == [1]


class TestBillJobAtomic:
    def test_bill_job_atomic_dedup(self):
        """bill_job should not produce duplicate records on sequential calls."""
        job = scheduler.submit_job("test", 8)
        scheduler.update_job_status(job["job_id"], "running", host_id="h1")
        time.sleep(0.01)
        scheduler.update_job_status(job["job_id"], "completed")

        first = scheduler.bill_job(job["job_id"])
        assert first is not None
        second = scheduler.bill_job(job["job_id"])
        assert second is None
        records = scheduler.load_billing()
        assert sum(1 for r in records if r["job_id"] == job["job_id"]) == 1


class TestMarketplaceStatsPerListingCut:
    def test_per_listing_platform_revenue(self):
        """marketplace_stats should use each listing's own platform_cut."""
        # Create two listings with different platform_cut values
        listings = scheduler.load_marketplace()
        listings.append({
            "host_id": "h-low",
            "gpu_model": "RTX 3060",
            "vram_gb": 12,
            "price_per_hour": 0.10,
            "description": "",
            "owner": "alice",
            "platform_cut": 0.10,   # 10%
            "listed_at": time.time(),
            "updated_at": time.time(),
            "active": True,
            "total_jobs": 1,
            "total_earned": 9.0,    # host payout
        })
        listings.append({
            "host_id": "h-high",
            "gpu_model": "A100",
            "vram_gb": 80,
            "price_per_hour": 1.00,
            "description": "",
            "owner": "bob",
            "platform_cut": 0.30,   # 30%
            "listed_at": time.time(),
            "updated_at": time.time(),
            "active": True,
            "total_jobs": 2,
            "total_earned": 7.0,    # host payout
        })
        scheduler.save_marketplace(listings)

        stats = scheduler.marketplace_stats()
        # h-low: 9.0 * 0.10 / 0.90 = 1.0
        # h-high: 7.0 * 0.30 / 0.70 = 3.0
        assert stats["platform_revenue"] == 4.0
        assert stats["total_host_payouts"] == 16.0
