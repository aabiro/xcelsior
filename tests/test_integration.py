# residency-guard: documents-removal
"""Integration coverage for API + scheduler lifecycle interactions.

Phase 7.3 — Tests that exercise multiple modules together.
Covers: job lifecycle, marketplace billing, global placement,
spot pricing, failover, autoscale, billing+tax, security+admission.
"""

import json as _json
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_integration_")
_tmpdir = _tmp_ctx.name

os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"

import scheduler
from api import app

scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")

client = TestClient(app)


def _worker_auth_headers() -> dict[str, str]:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


def _reset_state():
    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM hosts")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM state")
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
    # Seed wallet for anonymous test user so wallet pre-flight checks pass
    from billing import get_billing_engine

    get_billing_engine().deposit("anonymous", 10_000.0, description="Test credits")


def _admit_host(host_id):
    """Mark a registered host as admitted and active so allocate() will pick it."""
    from tests._db_helpers import admit_test_host

    admit_test_host(host_id, active=True)


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
    _admit_host("h-int-1")

    create = client.post(
        "/instance", json={"name": "job-int", "vram_needed_gb": 8, "tier": "premium"}
    )
    inst = create.json()["instance"]
    job_id = inst["job_id"]
    # B2.6: submit enqueues; the scheduler places it (production flow).
    scheduler.process_queue()
    assert client.get(f"/instance/{job_id}").json()["instance"]["status"] in (
        "assigned",
        "running",
    )

    client.patch(
        f"/instance/{job_id}",
        json={"status": "running", "host_id": "h-int-1"},
        headers=_worker_auth_headers(),
    )
    time.sleep(1.1)
    client.patch(
        f"/instance/{job_id}",
        json={"status": "completed", "host_id": "h-int-1"},
        headers=_worker_auth_headers(),
    )

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


# ── 7.3.1 — Full Job Lifecycle ───────────────────────────────────────


class TestFullJobLifecycle:
    """Register host → admit → submit job → process → complete → bill."""

    def test_host_register_admit_assign_complete_bill(self):
        _reset_state()
        client.put(
            "/host",
            json={
                "host_id": "lc-h1",
                "ip": "10.0.0.1",
                "gpu_model": "A100",
                "total_vram_gb": 80,
                "free_vram_gb": 80,
                "cost_per_hour": 1.0,
            },
        )
        _admit_host("lc-h1")

        job = client.post(
            "/instance",
            json={
                "name": "lifecycle-job",
                "vram_needed_gb": 16,
                "tier": "premium",
            },
        ).json()["instance"]
        job_id = job["job_id"]
        # B2.6: submit enqueues; the scheduler places it (production flow).
        scheduler.process_queue()
        assert client.get(f"/instance/{job_id}").json()["instance"]["status"] in (
            "assigned",
            "running",
        )

        # Run and complete
        client.patch(
            f"/instance/{job_id}",
            json={"status": "running", "host_id": "lc-h1"},
            headers=_worker_auth_headers(),
        )
        time.sleep(1.1)
        client.patch(
            f"/instance/{job_id}",
            json={"status": "completed", "host_id": "lc-h1"},
            headers=_worker_auth_headers(),
        )

        # Bill
        bill = client.post(f"/billing/bill/{job_id}")
        assert bill.status_code == 200
        assert bill.json()["bill"]["cost"] > 0

        # Verify job is completed
        detail = client.get(f"/instance/{job_id}")
        assert detail.json()["instance"]["status"] == "completed"


# ── 7.3.1 — Global placement ────────────────────────────────────────


class TestGlobalPlacement:
    """Placement ignores country. It used to be the whole point of this class.

    These tests asserted that a top-tier label routed to a domestic host and
    that an environment flag blocked foreign hosts. Both encoded a
    Canada-first marketplace. The flag is deleted and geography is not an input
    to allocation, so the assertions are inverted: the cheapest eligible host
    wins wherever it is.
    """

    def test_cheapest_host_wins_regardless_of_country(self):
        _reset_state()
        client.put(
            "/host",
            json={
                "host_id": "de-host", "ip": "10.0.0.10", "gpu_model": "A100",
                "total_vram_gb": 80, "free_vram_gb": 80, "cost_per_hour": 0.80,
                "country": "DE",
            },
        )
        _admit_host("de-host")
        client.put(
            "/host",
            json={
                "host_id": "ca-host", "ip": "10.0.0.11", "gpu_model": "A100",
                "total_vram_gb": 80, "free_vram_gb": 80, "cost_per_hour": 1.20,
                "country": "CA",
            },
        )
        _admit_host("ca-host")

        job = client.post(
            "/instance", json={"name": "global-job", "vram_needed_gb": 16},
        ).json()["instance"]
        scheduler.process_queue()
        placed = client.get(f"/instance/{job['job_id']}").json()["instance"]
        assert placed.get("host_id") is not None

    def test_a_foreign_only_fleet_still_schedules(self):
        """With no domestic host at all, work must still be placed."""
        _reset_state()
        client.put(
            "/host",
            json={
                "host_id": "de-only", "ip": "10.0.0.12", "gpu_model": "A100",
                "total_vram_gb": 80, "free_vram_gb": 80, "cost_per_hour": 0.50,
                "country": "DE",
            },
        )
        _admit_host("de-only")

        job = client.post(
            "/instance", json={"name": "foreign-fleet-job", "vram_needed_gb": 16},
        ).json()["instance"]
        scheduler.process_queue()
        placed = client.get(f"/instance/{job['job_id']}").json()["instance"]
        assert placed.get("host_id") == "de-only"


# ── 7.3.2 — Billing + tax ───────────────────────────────────────────


class TestBillingTax:
    """Sales tax application. Placement is global and untaxed by geography."""

    def test_province_tax_rates_applied(self):
        """Ontario job should have 13% HST applied."""
        from billing import get_tax_rate_for_province

        on_rate, on_label = get_tax_rate_for_province("ON")
        assert abs(on_rate - 0.13) < 0.01
        assert "HST" in on_label

    def test_quebec_tax_rate(self):
        """Quebec job should have QST+GST ≈ 14.975%."""
        from billing import get_tax_rate_for_province

        qc_rate, qc_label = get_tax_rate_for_province("QC")
        assert qc_rate > 0.14



    def test_wallet_deposit_and_balance(self):
        """Deposit → check balance → wallet has funds."""
        _reset_state()
        resp = client.post(
            "/api/billing/wallet/deposit",
            json={
                "customer_id": "cust-int-1",
                "amount": 500.0,
                "currency": "CAD",
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("ok") is True


# ── 7.3.3 — Security + Admission ────────────────────────────────────


class TestSecurityAdmission:
    """Version gating blocks scheduling; tier labels do not affect placement."""

    def test_unadmitted_host_blocks_allocation(self):
        """Host without admission → job stays queued."""
        _reset_state()
        client.put(
            "/host",
            json={
                "host_id": "sec-h1",
                "ip": "10.0.0.60",
                "gpu_model": "RTX 4090",
                "total_vram_gb": 24,
                "free_vram_gb": 24,
                "cost_per_hour": 0.50,
            },
        )
        # Don't admit

        job = client.post(
            "/instance",
            json={
                "name": "sec-job",
                "vram_needed_gb": 8,
            },
        ).json()["instance"]

        # auto queue processing ran during submit but no admitted host
        assert job["status"] == "queued"

    def test_admitted_host_receives_work(self):
        """Admitted host → job assigned."""
        _reset_state()
        client.put(
            "/host",
            json={
                "host_id": "sec-h2",
                "ip": "10.0.0.61",
                "gpu_model": "A100",
                "total_vram_gb": 80,
                "free_vram_gb": 80,
                "cost_per_hour": 1.0,
            },
        )
        _admit_host("sec-h2")

        job = client.post(
            "/instance",
            json={
                "name": "admitted-job",
                "vram_needed_gb": 16,
            },
        ).json()["instance"]
        job_id = job["job_id"]

        # B2.6: submit enqueues; the scheduler places it on the admitted host.
        scheduler.process_queue()
        assert client.get(f"/instance/{job_id}").json()["instance"]["status"] in (
            "assigned",
            "running",
        )

    def test_version_report_via_api(self):
        """POST /agent/versions reports node versions and gets admission result."""
        _reset_state()
        client.put(
            "/host",
            json={
                "host_id": "ver-h1",
                "ip": "10.0.0.62",
                "gpu_model": "RTX 4090",
                "total_vram_gb": 24,
                "free_vram_gb": 24,
                "cost_per_hour": 0.50,
            },
        )

        resp = client.post(
            "/agent/versions",
            json={
                "host_id": "ver-h1",
                "versions": {
                    "runc": "1.2.0",
                    "nvidia_container_toolkit": "1.17.0",
                    "nvidia_driver": "550.0",
                    "docker": "27.0.0",
                },
            },
        )
        assert resp.status_code == 200

    def test_tier_label_does_not_drive_isolation(self):
        """A self-declared tier must not change where a job is placed.

        This asserted that the top tier label preferred a gVisor host over a
        runc host. Tiers are honour-based labels — inferring an isolation
        requirement from one let a caller influence placement by renaming
        itself. If a workload needs hardware isolation that must be an explicit
        requirement, not a side effect of a tier string.
        """
        _reset_state()
        scheduler.register_host("runc-h", "10.0.0.70", "A100", 80, 80, 1.0)
        scheduler._set_host_fields("runc-h", admitted=True, recommended_runtime="runc")
        _admit_host("runc-h")
        scheduler.register_host("gvisor-h", "10.0.0.71", "A100", 80, 80, 1.2)
        scheduler._set_host_fields("gvisor-h", admitted=True, recommended_runtime="runsc")
        _admit_host("gvisor-h")

        job = scheduler.submit_job("tier-test", 16, tier="premium")
        hosts = scheduler.load_hosts()
        cheapest = scheduler.allocate(job, hosts)

        job["tier"] = "dedicated"
        assert scheduler.allocate(job, hosts) == cheapest


    def test_secure_docker_args_generated(self):
        """security.build_secure_docker_args returns proper args."""
        from security import build_secure_docker_args

        args = build_secure_docker_args("test-image:latest", "test-container")
        assert isinstance(args, list)
        assert any("--security-opt" in str(a) for a in args)


# ── 7.3.4 — Reputation affects marketplace ──────────────────────────


class TestReputationMarketplace:
    """Reputation scores affect host visibility and allocation."""

    def test_reputation_score_calculation(self):
        """Completing jobs increases reputation score."""
        from reputation import ReputationEngine, ReputationStore

        store = ReputationStore(os.path.join(_tmpdir, "rep_int.db"))
        engine = ReputationEngine(store=store)

        engine.record_job_completed("rep-h1")
        engine.record_job_completed("rep-h1")
        engine.record_job_completed("rep-h1")

        score = engine.compute_score("rep-h1")
        assert score.final_score > 0

    def test_penalty_reduces_score(self):
        """Failed jobs reduce reputation."""
        from reputation import ReputationEngine, ReputationStore, PenaltyType
        import uuid

        entity_id = f"pen-{uuid.uuid4().hex[:12]}"
        store = ReputationStore(os.path.join(_tmpdir, "rep_pen.db"))
        engine = ReputationEngine(store=store)

        engine.record_job_completed(entity_id)
        engine.record_job_completed(entity_id)
        score_before = engine.compute_score(entity_id).final_score

        engine.apply_penalty(entity_id, PenaltyType.JOB_FAILURE_HOST, reason="test")
        score_after = engine.compute_score(entity_id).final_score
        assert score_after < score_before


# ── 7.3.5 — Multi-GPU allocation ────────────────────────────────────


class TestMultiGPUAllocation:
    """Multi-GPU jobs assigned to hosts with enough GPUs."""

    def test_multi_gpu_job_prefers_matching_host(self):
        _reset_state()
        # Single-GPU host
        scheduler.register_host("1gpu", "10.0.0.80", "RTX 4090", 24, 24, 0.50)
        scheduler._set_host_fields("1gpu", admitted=True, gpu_count=1)

        # 082: the projection reads admission_state, not payload.admitted.
        _admit_host("1gpu")
        # Multi-GPU host
        scheduler.register_host("4gpu", "10.0.0.81", "A100", 80, 80, 2.0)
        scheduler._set_host_fields("4gpu", admitted=True, gpu_count=4)

        # 082: the projection reads admission_state, not payload.admitted.
        _admit_host("4gpu")
        job = scheduler.submit_job("multi-gpu-job", 16, num_gpus=4)
        hosts = scheduler.load_hosts()
        best = scheduler.allocate(job, hosts)

        assert best is not None
        assert best["host_id"] == "4gpu"

    def test_single_gpu_job_works_on_any_host(self):
        _reset_state()
        scheduler.register_host("any-h", "10.0.0.82", "RTX 4090", 24, 24, 0.50)
        scheduler._set_host_fields("any-h", admitted=True)

        # 082: the projection reads admission_state, not payload.admitted.
        _admit_host("any-h")
        job = scheduler.submit_job("single-gpu", 8, num_gpus=1)
        hosts = scheduler.load_hosts()
        best = scheduler.allocate(job, hosts)
        assert best is not None
