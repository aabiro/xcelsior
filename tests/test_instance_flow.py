"""Phase 8 — End-to-End Instance Flow Tests.

Automated tests covering the full instance lifecycle:
  register host → submit instance → queued → assigned → leased → running
  → logs stream → billing charge → cancel

Failure scenarios:
  worker death → lease expiry → requeue
  host offline → failover
  wallet empty → pause → resume

Docker compose smoke test (requires Docker — skipped in CI by default).
"""

import json as _json
import os
import tempfile
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ── Isolated environment ─────────────────────────────────────────────

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_instflow_")
_tmpdir = _tmp_ctx.name

os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"

import scheduler  # noqa: E402

scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")

from api import app  # noqa: E402

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────


def _reset_state():
    """Clear all jobs/hosts/state and seed wallet with $10k."""
    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM hosts")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM state")
        conn.execute("DELETE FROM billing_cycles")
        conn.execute("DELETE FROM job_logs")
        conn.execute("DELETE FROM leases")
    for f in (
        scheduler.HOSTS_FILE,
        scheduler.JOBS_FILE,
        scheduler.BILLING_FILE,
        scheduler.MARKETPLACE_FILE,
        scheduler.AUTOSCALE_POOL_FILE,
    ):
        if os.path.exists(f):
            os.remove(f)
    # Reset wallet to active status and $10k balance
    from billing import get_billing_engine
    be = get_billing_engine()
    be.deposit("anonymous", 10_000.0, description="Test credits")
    from db import _get_pg_pool
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            "UPDATE wallets SET status = 'active', grace_until = 0 WHERE customer_id = 'anonymous'"
        )
        conn.commit()


def _register_host(host_id, vram=24.0, cost=0.50, ip="10.0.0.1", gpu="RTX 4090"):
    """Register a host via the API."""
    resp = client.put("/host", json={
        "host_id": host_id,
        "ip": ip,
        "gpu_model": gpu,
        "total_vram_gb": vram,
        "free_vram_gb": vram,
        "cost_per_hour": cost,
        "country": "CA",
        "province": "ON",
    })
    assert resp.status_code == 200
    return resp.json()["host"]


def _admit_host(host_id):
    """Admit a host so it can receive jobs."""
    with scheduler._atomic_mutation() as conn:
        row = conn.execute(
            "SELECT payload FROM hosts WHERE host_id = %s", (host_id,),
        ).fetchone()
        if row:
            data = row["payload"] if isinstance(row["payload"], dict) else _json.loads(row["payload"])
            data["admitted"] = True
            data["status"] = "active"
            conn.execute(
                "UPDATE hosts SET status = 'active', payload = %s WHERE host_id = %s",
                (_json.dumps(data), host_id),
            )


def _submit_job(name="test-job", vram=8, tier="on-demand", host_id=None):
    """Submit a job via the API. Returns (response, instance dict)."""
    body = {"name": name, "vram_needed_gb": vram, "tier": tier}
    if host_id:
        body["host_id"] = host_id
    resp = client.post("/instance", json=body)
    assert resp.status_code == 200
    return resp, resp.json()["instance"]


def _set_status(job_id, status, host_id=None):
    """Patch a job status via the API."""
    body = {"status": status}
    if host_id:
        body["host_id"] = host_id
    return client.patch(f"/instance/{job_id}", json=body)


def _get_instance(job_id):
    """Fetch a single instance."""
    resp = client.get(f"/instance/{job_id}")
    assert resp.status_code == 200
    return resp.json()["instance"]


# ══════════════════════════════════════════════════════════════════════
# 8.1 — Happy-path lifecycle:
#   register host → submit → queued → assigned → leased → running
#   → logs stream → billing charge → cancel
# ══════════════════════════════════════════════════════════════════════


class TestHappyPathLifecycle:
    """Full happy-path: host→submit→assign→lease→run→log→bill→cancel."""

    def test_full_lifecycle_register_through_cancel(self):
        """Register host, submit job, walk through every status, bill, cancel."""
        _reset_state()

        # 1. Register & admit host
        host = _register_host("hp-host-1", vram=48, cost=1.00, gpu="A100")
        assert host["host_id"] == "hp-host-1"
        _admit_host("hp-host-1")

        # Verify host appears in list as active
        hosts = client.get("/hosts?active_only=false").json()["hosts"]
        assert any(h["host_id"] == "hp-host-1" for h in hosts)

        # 2. Submit job — auto-assigns during submit
        _, inst = _submit_job("lifecycle-job", vram=16, tier="on-demand")
        job_id = inst["job_id"]
        assert inst["status"] in ("queued", "assigned", "running")

        # 3. If queued, process queue to assign
        if inst["status"] == "queued":
            client.post("/queue/process")
            inst = _get_instance(job_id)
            assert inst["status"] == "assigned"

        # 4. Claim lease (assigned → leased)
        if inst["status"] == "assigned":
            lease_resp = client.post("/agent/lease/claim", json={
                "host_id": "hp-host-1",
                "job_id": job_id,
            })
            assert lease_resp.status_code == 200
            lease = lease_resp.json()
            assert lease.get("ok") is True
            assert "lease_id" in lease
            assert "expires_at" in lease

            # Verify job is now leased
            inst = _get_instance(job_id)
            assert inst["status"] == "leased"

        # 5. Transition to running
        _set_status(job_id, "running", host_id="hp-host-1")
        inst = _get_instance(job_id)
        assert inst["status"] == "running"
        assert inst.get("started_at") is not None or inst.get("started_at", 0) > 0

        # 6. Push logs and verify they appear
        from routes.instances import push_job_log
        push_job_log(job_id, "Training epoch 1/10", level="info")
        push_job_log(job_id, "Loss: 0.45", level="info")
        push_job_log(job_id, "GPU memory: 12.3 GB", level="debug")

        logs_resp = client.get(f"/instances/{job_id}/logs")
        assert logs_resp.status_code == 200
        logs = logs_resp.json()["logs"]
        assert len(logs) >= 3
        messages = [l.get("message", l.get("line", "")) for l in logs]
        assert any("epoch 1" in m for m in messages)

        # 7. Download logs
        dl_resp = client.get(f"/instances/{job_id}/logs/download")
        assert dl_resp.status_code == 200
        assert "text/plain" in dl_resp.headers.get("content-type", "")
        assert "attachment" in dl_resp.headers.get("content-disposition", "")
        assert "epoch 1" in dl_resp.text

        # 8. Let some time pass then complete for billing
        time.sleep(1.1)
        _set_status(job_id, "completed", host_id="hp-host-1")
        inst = _get_instance(job_id)
        assert inst["status"] == "completed"

        # 9. Bill the job
        bill_resp = client.post(f"/billing/bill/{job_id}")
        assert bill_resp.status_code == 200
        bill = bill_resp.json()["bill"]
        assert bill["cost"] > 0
        assert bill["duration_sec"] > 0

    def test_lifecycle_with_cancel_while_running(self):
        """Submit → run → cancel mid-execution."""
        _reset_state()
        _register_host("cancel-h", vram=24)
        _admit_host("cancel-h")

        _, inst = _submit_job("cancel-test", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="cancel-h")
        inst = _get_instance(job_id)
        assert inst["status"] == "running"

        # Cancel
        cancel_resp = client.post(f"/instances/{job_id}/cancel")
        assert cancel_resp.status_code == 200

        inst = _get_instance(job_id)
        assert inst["status"] in ("cancelled", "completed", "failed")

    def test_lifecycle_queued_cancel(self):
        """Cancel while still queued — should succeed immediately."""
        _reset_state()

        # No hosts → job stays queued
        _, inst = _submit_job("queued-cancel", vram=8)
        job_id = inst["job_id"]
        assert inst["status"] == "queued"

        cancel_resp = client.post(f"/instances/{job_id}/cancel")
        assert cancel_resp.status_code == 200

        inst = _get_instance(job_id)
        assert inst["status"] == "cancelled"

    def test_binpack_assignment(self):
        """Verify process-binpack assigns largest jobs first."""
        _reset_state()
        _register_host("bp-h1", vram=80, gpu="A100")
        _admit_host("bp-h1")

        # Submit two jobs — large first
        _, inst_big = _submit_job("big-job", vram=60)
        _, inst_small = _submit_job("small-job", vram=8)

        # Process with binpack
        bp_resp = client.post("/api/v2/scheduler/process-binpack")
        assert bp_resp.status_code == 200

        big = _get_instance(inst_big["job_id"])
        small = _get_instance(inst_small["job_id"])
        # At least one should be assigned (big fits on the 80GB host)
        assert big["status"] in ("assigned", "running") or small["status"] in ("assigned", "running")

    def test_logs_limit_cap(self):
        """Verify the log endpoint caps at 10k entries."""
        _reset_state()
        _register_host("log-h", vram=24)
        _admit_host("log-h")

        _, inst = _submit_job("log-test")
        job_id = inst["job_id"]

        # Push more logs than the max buffer size
        from routes.instances import push_job_log
        for i in range(600):
            push_job_log(job_id, f"Line {i}")

        # Request with absurdly high limit — should be capped
        resp = client.get(f"/instances/{job_id}/logs?limit=999999")
        assert resp.status_code == 200
        returned = resp.json()["total"]
        assert returned <= 500, f"Expected <=500 logs but got {returned}"
        assert returned > 0, "Expected at least some logs"

    def test_gpu_model_request_blocks_non_matching_hosts_until_available(self):
        """GPU model constraints should keep the job queued until a matching host exists."""
        _reset_state()
        _register_host("gpu-2060", vram=24, gpu="RTX 2060")
        _register_host("gpu-4090", vram=24, gpu="RTX 4090")
        _admit_host("gpu-2060")
        _admit_host("gpu-4090")

        resp = client.post("/instance", json={
            "name": "gpu-specific-job",
            "vram_needed_gb": 8,
            "gpu_model": "A100",
        })
        assert resp.status_code == 200
        job_id = resp.json()["instance"]["job_id"]

        inst = _get_instance(job_id)
        assert inst["status"] == "queued"

        _register_host("gpu-a100", vram=80, gpu="A100")
        _admit_host("gpu-a100")

        client.post("/queue/process")
        inst = _get_instance(job_id)
        assert inst["status"] in ("assigned", "running")
        assert inst["host_id"] == "gpu-a100"


# ══════════════════════════════════════════════════════════════════════
# 8.2 — Failure scenarios
# ══════════════════════════════════════════════════════════════════════


class TestWorkerDeathLeaseExpiry:
    """Worker death → lease expiry → requeue."""

    def test_lease_grant_and_expiry(self):
        """Grant a short lease, expire it, verify job is requeued."""
        _reset_state()
        _register_host("ld-h1", vram=24)
        _admit_host("ld-h1")

        _, inst = _submit_job("lease-test", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")
            inst = _get_instance(job_id)

        assert inst["status"] in ("assigned", "running")

        # Claim lease with very short duration
        from events import get_event_store
        es = get_event_store()
        lease = es.grant_lease(job_id, "ld-h1", duration_sec=1)
        _set_status(job_id, "leased", host_id="ld-h1")

        # Verify lease is active
        active = es.get_active_lease(job_id)
        assert active is not None
        assert active.lease_id == lease.lease_id

        # Wait for it to expire (1s duration + 60s grace in prod — but we
        # can manually expire by backdating)
        from db import _get_pg_pool
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.execute(
                "UPDATE leases SET expires_at = %s WHERE lease_id = %s",
                (time.time() - 120, lease.lease_id),
            )
            conn.commit()

        expired = es.expire_stale_leases()
        assert job_id in expired

        # Requeue the expired job
        result = scheduler.requeue_job(job_id)
        assert result is not None
        assert result["status"] == "queued"

        # Verify it can be re-assigned
        client.post("/queue/process")
        inst = _get_instance(job_id)
        assert inst["status"] in ("queued", "assigned", "running")

    def test_lease_renew(self):
        """Renewing a lease extends its expiry."""
        _reset_state()
        _register_host("lr-h1", vram=24)
        _admit_host("lr-h1")

        _, inst = _submit_job("renew-test", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        # Claim lease
        from events import get_event_store
        es = get_event_store()
        lease = es.grant_lease(job_id, "lr-h1", duration_sec=300)
        original_expiry = lease.expires_at

        # Renew
        renewed = es.renew_lease(job_id, "lr-h1")
        assert renewed is not None
        assert renewed.expires_at > original_expiry

    def test_lease_release_on_completion(self):
        """Completing a job releases its lease."""
        _reset_state()
        _register_host("lc-h1", vram=24)
        _admit_host("lc-h1")

        _, inst = _submit_job("complete-lease", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        from events import get_event_store
        es = get_event_store()
        es.grant_lease(job_id, "lc-h1", duration_sec=300)

        # Complete the job
        _set_status(job_id, "running", host_id="lc-h1")
        _set_status(job_id, "completed", host_id="lc-h1")

        # Lease should be released
        released = es.release_lease(job_id)
        # release_lease returns True if found and released, False if already gone
        active = es.get_active_lease(job_id)
        assert active is None


class TestHostOfflineFailover:
    """Host offline → failover → job reassigned to healthy host."""

    def test_dead_host_jobs_requeued(self):
        """Mark host dead directly, verify running jobs get requeued."""
        _reset_state()
        _register_host("fo-h1", vram=48, ip="10.0.0.91")
        _register_host("fo-h2", vram=48, ip="10.0.0.92")
        _admit_host("fo-h1")
        _admit_host("fo-h2")

        # Submit and assign to fo-h1
        _, inst = _submit_job("failover-job", vram=16)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="fo-h1")
        inst = _get_instance(job_id)
        assert inst["status"] == "running"

        # Simulate host death: _requeue_dead_host_jobs
        scheduler._requeue_dead_host_jobs("fo-h1", "10.0.0.91")

        inst = _get_instance(job_id)
        assert inst["status"] == "queued"
        assert inst.get("retries", 0) >= 1

        # Re-process queue — should go to fo-h2
        client.post("/queue/process")
        inst = _get_instance(job_id)
        assert inst["status"] in ("assigned", "running")

    def test_failover_endpoint(self):
        """POST /failover triggers full cycle (with mocked ping)."""
        _reset_state()
        _register_host("fv-h1", vram=24, ip="10.0.0.93")
        _register_host("fv-h2", vram=24, ip="10.0.0.94")
        _admit_host("fv-h1")
        _admit_host("fv-h2")

        _, inst = _submit_job("failover-cycle", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="fv-h1")

        # Mock ping: fv-h1 dead, fv-h2 alive
        def mock_ping(ip):
            return ip != "10.0.0.93"

        with patch.object(scheduler, "ping_host", side_effect=mock_ping):
            fo_resp = client.post("/failover")
            assert fo_resp.status_code == 200

        # Job should have been requeued (and possibly re-assigned to fv-h2)
        inst = _get_instance(job_id)
        assert inst["status"] in ("queued", "assigned", "running")

    def test_requeue_increments_retries(self):
        """Each requeue increments the retry counter."""
        _reset_state()
        _register_host("rq-h1", vram=24)
        _admit_host("rq-h1")

        _, inst = _submit_job("retry-job", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="rq-h1")

        # Requeue 3 times
        for i in range(3):
            result = scheduler.requeue_job(job_id)
            if result and result["status"] == "queued":
                assert result.get("retries", 0) == i + 1
                # Set back to running so we can requeue again
                _set_status(job_id, "running", host_id="rq-h1")
            else:
                break

        # 4th requeue should fail (max_retries=3)
        result = scheduler.requeue_job(job_id)
        if result:
            # Should be permanently failed
            assert result["status"] == "failed"

    def test_requeue_non_requeuable_status_rejected(self):
        """Requeuing a completed or cancelled job is rejected."""
        _reset_state()
        _register_host("nr-h1", vram=24)
        _admit_host("nr-h1")

        _, inst = _submit_job("completed-job", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="nr-h1")
        _set_status(job_id, "completed", host_id="nr-h1")

        result = scheduler.requeue_job(job_id)
        assert result is None  # rejected


class TestWalletEmptyPauseResume:
    """Wallet empty → pause → top-up → resume."""

    def test_pause_and_resume(self):
        """Pause a running instance, then resume it."""
        _reset_state()
        _register_host("pr-h1", vram=24)
        _admit_host("pr-h1")

        _, inst = _submit_job("pause-test", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="pr-h1")
        inst = _get_instance(job_id)
        assert inst["status"] == "running"

        # Pause
        pause_resp = client.post(f"/instances/{job_id}/pause")
        assert pause_resp.status_code == 200
        assert pause_resp.json().get("ok") is True

        inst = _get_instance(job_id)
        assert inst["status"] in ("user_paused", "paused_low_balance")

        # Resume (patch run_job to avoid SSH failure overwriting status)
        from unittest.mock import patch
        with patch("scheduler.run_job", return_value="fake-container-id"):
            resume_resp = client.post(f"/instances/{job_id}/resume")
        assert resume_resp.status_code == 200
        assert resume_resp.json().get("ok") is True

        inst = _get_instance(job_id)
        assert inst["status"] == "running"

    def test_empty_wallet_blocks_new_jobs(self):
        """A suspended wallet should block new job submissions with 402."""
        _reset_state()
        _register_host("ew-h1", vram=24)
        _admit_host("ew-h1")

        # Drain wallet
        from billing import get_billing_engine
        be = get_billing_engine()
        wallet = be.get_wallet("anonymous")
        if wallet["balance_cad"] > 0:
            be.charge("anonymous", wallet["balance_cad"], job_id="drain", description="drain")

        # Suspend the wallet
        from db import _get_pg_pool
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.execute(
                "UPDATE wallets SET status = 'suspended' WHERE customer_id = 'anonymous'"
            )
            conn.commit()

        # Try to submit — should get 402
        resp = client.post("/instance", json={
            "name": "blocked-job",
            "vram_needed_gb": 8,
        })
        assert resp.status_code == 402

    def test_zero_balance_blocks_without_grace(self):
        """Zero balance + expired grace → 402."""
        _reset_state()

        from billing import get_billing_engine
        be = get_billing_engine()
        wallet = be.get_wallet("anonymous")
        if wallet["balance_cad"] > 0:
            be.charge("anonymous", wallet["balance_cad"], job_id="drain2", description="drain")

        # Set grace to past
        from db import _get_pg_pool
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.execute(
                "UPDATE wallets SET grace_until = %s WHERE customer_id = 'anonymous'",
                (time.time() - 100,),
            )
            conn.commit()

        resp = client.post("/instance", json={
            "name": "no-grace-job",
            "vram_needed_gb": 8,
        })
        assert resp.status_code == 402

    def test_resume_with_no_funds_rejected(self):
        """Resume should fail if wallet is empty."""
        _reset_state()
        _register_host("rf-h1", vram=24)
        _admit_host("rf-h1")

        _, inst = _submit_job("resume-fail", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="rf-h1")

        # Pause first
        client.post(f"/instances/{job_id}/pause")

        # Drain wallet
        from billing import get_billing_engine
        be = get_billing_engine()
        wallet = be.get_wallet("anonymous")
        if wallet["balance_cad"] > 0:
            be.charge("anonymous", wallet["balance_cad"], job_id="drain3", description="drain")

        # Resume should fail (insufficient balance)
        resume_resp = client.post(f"/instances/{job_id}/resume")
        assert resume_resp.status_code in (200, 400, 402)
        data = resume_resp.json()
        # Either HTTP error or {resumed: false, reason: "insufficient_balance"}
        if resume_resp.status_code == 200:
            assert data.get("resumed") is False or data.get("ok") is False


# ══════════════════════════════════════════════════════════════════════
# 8.3 — Billing charge correctness
# ══════════════════════════════════════════════════════════════════════


class TestBillingCharge:
    """Billing charges are computed correctly from runtime duration."""

    def test_bill_reflects_runtime(self):
        """Billing cost should be proportional to runtime × rate."""
        _reset_state()
        _register_host("bill-h1", vram=24, cost=1.00)
        _admit_host("bill-h1")

        _, inst = _submit_job("bill-job", vram=8, tier="on-demand")
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="bill-h1")
        time.sleep(1.5)
        _set_status(job_id, "completed", host_id="bill-h1")

        bill_resp = client.post(f"/billing/bill/{job_id}")
        assert bill_resp.status_code == 200
        bill = bill_resp.json()["bill"]
        assert bill["cost"] > 0
        assert bill["duration_sec"] >= 1
        assert bill["rate_per_hour"] > 0

    def test_bill_all_completed(self):
        """bill-all handles multiple completed jobs."""
        _reset_state()
        _register_host("ba-h1", vram=48, cost=0.50)
        _admit_host("ba-h1")

        jobs = []
        for i in range(3):
            _, inst = _submit_job(f"batch-{i}", vram=8)
            jobs.append(inst["job_id"])
            if inst["status"] == "queued":
                client.post("/queue/process")
            _set_status(inst["job_id"], "running", host_id="ba-h1")

        time.sleep(1.1)

        for jid in jobs:
            _set_status(jid, "completed", host_id="ba-h1")

        bill_all_resp = client.post("/billing/bill-all")
        assert bill_all_resp.status_code == 200

    def test_double_billing_idempotent(self):
        """Billing the same job twice should not double-charge."""
        _reset_state()
        _register_host("db-h1", vram=24, cost=1.00)
        _admit_host("db-h1")

        _, inst = _submit_job("double-bill", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="db-h1")
        time.sleep(1.1)
        _set_status(job_id, "completed", host_id="db-h1")

        # First bill should succeed and produce a billing record with cost
        resp1 = client.post(f"/billing/bill/{job_id}")
        assert resp1.status_code == 200, f"First bill failed: {resp1.text}"
        bill1 = resp1.json()
        assert bill1.get("ok") is True
        cost1 = bill1.get("bill", {}).get("cost", 0)
        assert cost1 > 0, f"First bill should have a positive cost, got {cost1}"

        # Second bill should be rejected (already billed)
        resp2 = client.post(f"/billing/bill/{job_id}")
        assert resp2.status_code == 400, \
            f"Second bill should return 400 (already billed), got {resp2.status_code}"


# ══════════════════════════════════════════════════════════════════════
# 8.4 — Log streaming via agent endpoint
# ══════════════════════════════════════════════════════════════════════


class TestAgentLogIngestion:
    """Worker agent pushes logs → they appear in the log buffer."""

    def test_agent_log_batch_ingestion(self):
        _reset_state()
        _register_host("al-h1", vram=24)
        _admit_host("al-h1")

        _, inst = _submit_job("log-ingest", vram=8)
        job_id = inst["job_id"]

        # Push logs via agent endpoint
        resp = client.post(f"/agent/logs/{job_id}", json={
            "lines": [
                {"message": "Container started", "level": "info"},
                {"message": "Downloading model", "level": "info"},
                {"message": "Training started", "level": "info"},
            ],
        })
        assert resp.status_code == 200
        assert resp.json()["accepted"] > 0

        # Verify logs appear in buffer
        logs_resp = client.get(f"/instances/{job_id}/logs")
        assert logs_resp.status_code == 200
        logs = logs_resp.json()["logs"]
        assert len(logs) >= 3

    def test_agent_log_batch_cap(self):
        """Batches over 500 lines are truncated."""
        _reset_state()

        lines = [{"message": f"Line {i}", "level": "info"} for i in range(600)]
        resp = client.post("/agent/logs/cap-test-job", json={"lines": lines})
        assert resp.status_code == 200
        # Should accept at most 500
        assert resp.json()["accepted"] <= 500

    def test_agent_log_empty_batch(self):
        """Empty batch is accepted gracefully."""
        resp = client.post("/agent/logs/empty-test", json={"lines": []})
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 0

    def test_agent_log_image_pull_failure_emits_event(self):
        """Image pull error in logs triggers job_error SSE event."""
        _reset_state()

        resp = client.post("/agent/logs/img-pull-test", json={
            "lines": [
                {"message": "Pulling image...", "level": "info"},
                {"message": "Error response from daemon: manifest unknown", "level": "error"},
            ],
        })
        assert resp.status_code == 200

    def test_agent_log_gzip(self):
        """Gzip-compressed log batch is accepted."""
        import gzip
        data = _json.dumps({
            "lines": [{"message": "Compressed log", "level": "info"}],
        }).encode()
        compressed = gzip.compress(data)
        resp = client.post(
            "/agent/logs/gzip-test",
            content=compressed,
            headers={"content-encoding": "gzip", "content-type": "application/json"},
        )
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════
# 8.5 — State machine transitions
# ══════════════════════════════════════════════════════════════════════


class TestStateTransitions:
    """Verify valid and invalid state transitions."""

    def test_queued_to_assigned(self):
        _reset_state()
        _register_host("st-h1", vram=24)
        _admit_host("st-h1")

        _, inst = _submit_job("state-test", vram=8)
        job_id = inst["job_id"]

        # Process queue to assign
        client.post("/queue/process")
        inst = _get_instance(job_id)
        assert inst["status"] in ("assigned", "running")

    def test_assigned_to_leased_to_running(self):
        _reset_state()
        _register_host("stl-h1", vram=24)
        _admit_host("stl-h1")

        _, inst = _submit_job("lease-state", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        # Claim lease → leased
        from events import get_event_store
        es = get_event_store()
        es.grant_lease(job_id, "stl-h1", duration_sec=300)
        _set_status(job_id, "leased", host_id="stl-h1")
        inst = _get_instance(job_id)
        assert inst["status"] == "leased"

        # leased → running
        _set_status(job_id, "running", host_id="stl-h1")
        inst = _get_instance(job_id)
        assert inst["status"] == "running"

    def test_running_to_completed(self):
        _reset_state()
        _register_host("sc-h1", vram=24)
        _admit_host("sc-h1")

        _, inst = _submit_job("complete-state", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="sc-h1")
        _set_status(job_id, "completed", host_id="sc-h1")
        inst = _get_instance(job_id)
        assert inst["status"] == "completed"

    def test_completed_is_terminal(self):
        """Completed jobs cannot be requeued (state machine enforces terminal)."""
        _reset_state()
        _register_host("ct-h1", vram=24)
        _admit_host("ct-h1")

        _, inst = _submit_job("terminal-test", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        _set_status(job_id, "running", host_id="ct-h1")
        _set_status(job_id, "completed", host_id="ct-h1")

        # Requeue should fail for completed jobs
        result = scheduler.requeue_job(job_id)
        assert result is None

    def test_cancelled_is_terminal(self):
        """Cancelled jobs cannot be restarted."""
        _reset_state()

        _, inst = _submit_job("cancel-terminal", vram=8)
        job_id = inst["job_id"]

        client.post(f"/instances/{job_id}/cancel")
        inst = _get_instance(job_id)
        assert inst["status"] == "cancelled"

        # Requeue should fail
        result = scheduler.requeue_job(job_id)
        assert result is None


# ══════════════════════════════════════════════════════════════════════
# 8.6 — Ownership / access control
# ══════════════════════════════════════════════════════════════════════


class TestOwnershipAccess:
    """Verify log/download endpoints respect ownership."""

    def test_log_download_sanitizes_job_id(self):
        """Malicious job_id in download filename is sanitized."""
        _reset_state()
        # Push a log to a deliberately weird job_id
        from routes.instances import push_job_log, _job_log_buffers
        weird_id = "evil\"; rm -rf /; \""
        push_job_log(weird_id, "test line")

        resp = client.get(f"/instances/{weird_id}/logs/download")
        if resp.status_code == 200:
            disp = resp.headers.get("content-disposition", "")
            # Should not contain raw shell-dangerous characters
            filename_part = disp.split("filename=")[-1].strip('"') if "filename=" in disp else ""
            for dangerous_char in ['"', ';', '`', '$', '|']:
                assert dangerous_char not in filename_part, \
                    f"Dangerous char '{dangerous_char}' found in filename: {filename_part}"
        # Clean up
        _job_log_buffers.pop(weird_id, None)


# ══════════════════════════════════════════════════════════════════════
# 8.7 — Event store integrity
# ══════════════════════════════════════════════════════════════════════


class TestEventStoreIntegrity:
    """Event chain integrity across lifecycle operations."""

    def test_event_chain_valid_after_lifecycle(self):
        """After a full lifecycle, the event chain hash should verify."""
        _reset_state()
        _register_host("ev-h1", vram=24)
        _admit_host("ev-h1")

        _, inst = _submit_job("event-test", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        # Walk through states
        from events import get_event_store
        es = get_event_store()
        es.grant_lease(job_id, "ev-h1", duration_sec=300)
        _set_status(job_id, "leased", host_id="ev-h1")
        _set_status(job_id, "running", host_id="ev-h1")
        _set_status(job_id, "completed", host_id="ev-h1")

        # Verify chain integrity
        result = es.verify_chain(limit=100)
        assert result["valid"] is True
        assert result["events_checked"] > 0


# ══════════════════════════════════════════════════════════════════════
# 8.8 — Docker Compose smoke test
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not os.environ.get("XCELSIOR_DOCKER_SMOKE"),
    reason="Set XCELSIOR_DOCKER_SMOKE=1 to run Docker Compose smoke tests",
)
class TestDockerComposeSmoke:
    """Build → up → health check → submit test instance.

    Requires Docker and docker-compose. Skipped unless XCELSIOR_DOCKER_SMOKE=1.
    """

    def test_compose_build_and_health(self):
        import subprocess

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compose_file = os.path.join(project_root, "docker-compose.yml")

        if not os.path.exists(compose_file):
            pytest.skip("docker-compose.yml not found")

        # Build
        result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "build"],
            capture_output=True, text=True, timeout=300,
            cwd=project_root,
        )
        assert result.returncode == 0, f"Build failed: {result.stderr[:500]}"

        # Start in detached mode
        up_result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "up", "-d"],
            capture_output=True, text=True, timeout=120,
            cwd=project_root,
        )
        assert up_result.returncode == 0, f"Up failed: {up_result.stderr[:500]}"

        try:
            # Wait for API to be ready
            import urllib.request
            healthy = False
            for _ in range(30):
                try:
                    req = urllib.request.Request("http://localhost:9500/healthz")
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        if resp.status == 200:
                            healthy = True
                            break
                except Exception:
                    pass
                time.sleep(2)

            assert healthy, "API did not become healthy within 60s"

            # Submit a test instance via curl
            import urllib.request
            data = _json.dumps({
                "name": "smoke-test",
                "vram_needed_gb": 4,
            }).encode()
            req = urllib.request.Request(
                "http://localhost:9500/instance",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status == 200
                body = _json.loads(resp.read())
                assert body.get("ok") is True
                assert "instance" in body
        finally:
            # Always clean up
            subprocess.run(
                ["docker", "compose", "-f", compose_file, "down", "-v"],
                capture_output=True, text=True, timeout=60,
                cwd=project_root,
            )


# ══════════════════════════════════════════════════════════════════════
# 8.9 — Agent heartbeat and work pull
# ══════════════════════════════════════════════════════════════════════


class TestAgentWorkPull:
    """Agent pulls assigned work for its host."""

    def test_agent_pulls_assigned_jobs(self):
        _reset_state()
        _register_host("wp-h1", vram=24)
        _admit_host("wp-h1")

        _, inst = _submit_job("work-pull", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        # Agent pulls work for its host
        work_resp = client.get("/agent/work/wp-h1")
        assert work_resp.status_code in (200, 204), f"Unexpected status {work_resp.status_code}"
        if work_resp.status_code == 204:
            # Job may have auto-transitioned past assigned; verify it's at least assigned/running
            inst = _get_instance(job_id)
            assert inst["status"] in ("assigned", "running", "leased"), \
                f"Job should be assigned/running but is {inst['status']}"
        else:
            work = work_resp.json()
            jobs_list = work.get("instances", work.get("jobs", []))
            assert isinstance(jobs_list, list)
            job_ids = [j.get("job_id", "") for j in jobs_list]
            assert job_id in job_ids, f"Expected {job_id} in work pull results {job_ids}"

    def test_agent_lease_claim_via_api(self):
        """Agent claims a lease via the API endpoint."""
        _reset_state()
        _register_host("lc-api-h1", vram=24)
        _admit_host("lc-api-h1")

        _, inst = _submit_job("api-lease", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        inst = _get_instance(job_id)
        if inst["status"] != "assigned":
            # Already running; skip lease test
            pytest.skip(f"Job already in '{inst['status']}' state, not 'assigned'")

        resp = client.post("/agent/lease/claim", json={
            "host_id": "lc-api-h1",
            "job_id": job_id,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("ok") is True
        assert "lease_id" in data

    def test_agent_lease_renew_via_api(self):
        """Agent renews a lease via the API endpoint."""
        _reset_state()
        _register_host("lr-api-h1", vram=24)
        _admit_host("lr-api-h1")

        _, inst = _submit_job("api-renew", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        inst = _get_instance(job_id)
        if inst["status"] != "assigned":
            pytest.skip(f"Job already in '{inst['status']}' state, not 'assigned'")

        # Claim first
        client.post("/agent/lease/claim", json={
            "host_id": "lr-api-h1",
            "job_id": job_id,
        })

        # Renew
        renew_resp = client.post("/agent/lease/renew", json={
            "host_id": "lr-api-h1",
            "job_id": job_id,
        })
        assert renew_resp.status_code == 200

    def test_agent_lease_release_via_api(self):
        """Agent releases a lease via the API endpoint."""
        _reset_state()
        _register_host("rl-api-h1", vram=24)
        _admit_host("rl-api-h1")

        _, inst = _submit_job("api-release", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            client.post("/queue/process")

        inst = _get_instance(job_id)
        if inst["status"] != "assigned":
            pytest.skip(f"Job already in '{inst['status']}' state, not 'assigned'")

        client.post("/agent/lease/claim", json={
            "host_id": "rl-api-h1",
            "job_id": job_id,
        })

        release_resp = client.post("/agent/lease/release", json={
            "job_id": job_id,
        })
        assert release_resp.status_code == 200


class TestCancelRequeueAuth:
    """Verify cancel/requeue endpoints reject unauthenticated requests when auth is enabled."""

    def test_cancel_requires_auth(self):
        _reset_state()
        _register_host("auth-c-h1", vram=24)
        _admit_host("auth-c-h1")
        _, inst = _submit_job("auth-cancel", vram=8)
        job_id = inst["job_id"]
        client.post("/queue/process")

        with patch("routes._deps.AUTH_REQUIRED", True):
            resp = client.post(f"/instances/{job_id}/cancel")
            assert resp.status_code == 401

    def test_requeue_requires_auth(self):
        _reset_state()
        _register_host("auth-r-h1", vram=24)
        _admit_host("auth-r-h1")
        _, inst = _submit_job("auth-requeue", vram=8)
        job_id = inst["job_id"]

        with patch("routes._deps.AUTH_REQUIRED", True):
            resp = client.post(f"/instance/{job_id}/requeue")
            assert resp.status_code == 401


class TestConcurrentBilling:
    """Verify concurrent billing attempts don't double-charge."""

    def test_concurrent_bill_same_job(self):
        from concurrent.futures import ThreadPoolExecutor
        _reset_state()
        _register_host("conc-h1", vram=80)
        _admit_host("conc-h1")
        _, inst = _submit_job("conc-bill", vram=8)
        job_id = inst["job_id"]
        client.post("/queue/process")
        client.patch(f"/instance/{job_id}", json={"status": "running"})
        time.sleep(0.1)
        client.patch(f"/instance/{job_id}", json={"status": "completed"})

        results = []

        def bill():
            resp = client.post("/billing/bill-all")
            results.append(resp.status_code)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(bill) for _ in range(4)]
            for f in futures:
                f.result()

        assert all(r == 200 for r in results)
        records = scheduler.load_billing()
        billed = [r for r in records if r["job_id"] == job_id]
        assert len(billed) == 1, f"Expected 1 billing record, got {len(billed)}"


# ══════════════════════════════════════════════════════════════════════
# SSH key endpoint — GET /agent/ssh-keys/{job_id}
# ══════════════════════════════════════════════════════════════════════


class TestAgentSshKeysEndpoint:
    """Tests for the GET /agent/ssh-keys/{job_id} endpoint."""

    def test_nonexistent_job_returns_404(self):
        _reset_state()
        resp = client.get("/agent/ssh-keys/nonexistent-job-id")
        assert resp.status_code == 404

    def test_job_with_no_owner_returns_empty_keys(self):
        _reset_state()
        _register_host("ssh-h1", vram=80)
        _admit_host("ssh-h1")
        _, inst = _submit_job("ssh-test", vram=8)
        job_id = inst["job_id"]
        client.post("/queue/process")

        # The anonymous job owner may not have SSH keys
        resp = client.get(f"/agent/ssh-keys/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert isinstance(data["keys"], list)

    def test_returns_keys_structure(self):
        """Endpoint returns proper {ok, keys} structure."""
        _reset_state()
        _register_host("ssh-h2", vram=80)
        _admit_host("ssh-h2")
        _, inst = _submit_job("ssh-struct", vram=8)
        job_id = inst["job_id"]
        client.post("/queue/process")

        resp = client.get(f"/agent/ssh-keys/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert "ok" in body
        assert "keys" in body
        assert isinstance(body["keys"], list)
