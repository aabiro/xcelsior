"""Phase 8 — End-to-End Instance Flow Tests.

Automated tests covering the full instance lifecycle:
  register host → submit instance → queued → assigned → leased → running
  → logs stream → billing charge → cancel

Failure scenarios:
  worker death → lease expiry → requeue
  host offline → failover
  wallet empty → stop → start

Docker Compose smoke builds a disposable network-isolated application stack.
"""

import json as _json
import os
import tempfile
import time
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401

# ── Isolated environment ─────────────────────────────────────────────

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_instflow_")
_tmpdir = _tmp_ctx.name

# Preserve conftest's synthetic worker token. Clearing it during collection
# invalidates the API's already-imported credential and every worker PATCH.
os.environ.setdefault("XCELSIOR_API_TOKEN", "test-token-not-for-production")
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
        # Commands outlive the hosts and jobs deleted above, so a fixed fixture
        # host id would otherwise claim every earlier run's start_attempt.
        conn.execute("DELETE FROM agent_commands")
        conn.execute("DELETE FROM state")
        conn.execute("DELETE FROM billing_cycles")
        conn.execute("DELETE FROM job_logs")
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


def _register_host(host_id, vram=24.0, cost=0.50, ip="10.0.0.1", gpu="RTX 4090", gpu_count=1):
    """Register a host via the API."""
    resp = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": ip,
            "gpu_model": gpu,
            "gpu_count": gpu_count,
            "total_vram_gb": vram,
            "free_vram_gb": vram,
            "cost_per_hour": cost,
            "country": "CA",
            "province": "ON",
        },
    )
    assert resp.status_code == 200
    from db import _get_pg_pool
    from control_plane.physical_inventory import sync_physical_inventory

    with _get_pg_pool().connection() as conn:
        sync_physical_inventory(
            conn,
            host_id=host_id,
            inventory=[
                dict(
                    gpu_uuid=f"GPU-{uuid.uuid5(uuid.NAMESPACE_URL, host_id)}-{i}",
                    device_index=i,
                    model=gpu,
                    total_vram_mb=int(vram * 1024),
                )
                for i in range(gpu_count)
            ],
        )
    return resp.json()["host"]


def _admit_host(host_id):
    """Admit a host so it can receive jobs."""
    from tests._db_helpers import admit_test_host

    admit_test_host(host_id, active=True, report_inventory=True)


def _submit_job(name="test-job", vram=8, tier="on-demand", host_id=None):
    """Submit a job via the API. Returns (response, instance dict)."""
    body = {"name": name, "vram_needed_gb": vram, "tier": tier}
    if host_id:
        body["host_id"] = host_id
    resp = client.post("/instance", json=body)
    assert resp.status_code == 200
    return resp, resp.json()["instance"]


def _worker_auth_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


def _process_queue():
    """Run the actual scheduler separately from the HTTP admission request."""
    from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
    from control_plane.scheduler.service import SchedulerService

    return SchedulerService(
        SchedulerConfig(mode=SchedulerMode.ACTIVE, replica_id="instance-flow")
    ).tick()


def _set_status(job_id, status, host_id=None):
    """Claim and report the real native authority through the worker API."""
    from tests.native_jobs import report_native_status

    return report_native_status(
        client,
        job_id,
        status,
        headers=_worker_auth_headers(),
        host_id=host_id,
        session="instance-flow",
    )


def _start_native_attempt(job_id, host_id):
    """Exercise real reservation, lease claim and settlement; no Docker is implied."""
    from db import _get_pg_pool
    from control_plane.db import run_transaction
    from control_plane.scheduler.claim import claim_next_job
    from control_plane.scheduler.reservation import reserve_and_bind
    from control_plane.leases import claim_lease
    from control_plane.attempts import report_attempt_status

    replica = f"instance-flow-{uuid.uuid4().hex[:8]}"

    def place(conn):
        claim = claim_next_job(conn, replica_id=replica)
        assert claim and claim.job_id == job_id
        return reserve_and_bind(
            conn,
            job_id=job_id,
            claim_token=claim.claim_token,
            replica_id=replica,
            host_id=host_id,
            requested_vram_mb=8192,
        )

    reserved = run_transaction(place)
    authority = {
        key: getattr(reserved, key) for key in ("job_id", "attempt_id", "host_id", "fencing_token")
    }
    run_transaction(
        lambda conn: claim_lease(
            conn, **authority, lease_id=reserved.lease_id, worker_session_id=replica
        )
    )
    run_transaction(lambda conn: report_attempt_status(conn, **authority, status="running"))
    return authority


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

    def test_host_pinned_launch_waits_for_transactional_placement(self, monkeypatch):
        _reset_state()
        _register_host("pin-launch-host", vram=24)
        _admit_host("pin-launch-host")
        monkeypatch.setattr("security.probe_image_exists", lambda image: None)
        response = client.post(
            "/instance",
            json={
                "name": "pinned-launch",
                "host_id": "pin-launch-host",
                "image": "nvidia/cuda:12.6.3-runtime-ubuntu22.04",
                "vram_needed_gb": 8,
            },
        )
        assert response.status_code == 200, response.text
        job = response.json()["instance"]
        assert job["status"] == "queued"
        assert job.get("host_id") is None
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn:
            row = conn.execute(
                "SELECT active_attempt_id,payload->>'preferred_host_id' FROM jobs WHERE job_id=%s",
                (job["job_id"],),
            ).fetchone()
            assert row == (None, "pin-launch-host")
            assert (
                conn.execute(
                    "SELECT count(*) FROM agent_commands WHERE job_id=%s", (job["job_id"],)
                ).fetchone()[0]
                == 0
            )

    def test_register_assign_logs_and_billing(self):
        """Register, queue, project status, stream logs, and bill without a v1 lease."""
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
            _process_queue()
            inst = _get_instance(job_id)
            assert inst["status"] == "assigned"

        # The old lease protocol is permanently retired, even outside active mode.
        lease_resp = client.post(
            "/agent/lease/claim", json={"host_id": "hp-host-1", "job_id": job_id}
        )
        assert lease_resp.status_code == 410
        assert _get_instance(job_id)["status"] == "assigned"

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

    def test_cancel_refuses_running_job_without_fenced_authority(self):
        """An old running row cannot authorize a destructive cancellation."""
        _reset_state()
        _register_host("cancel-h", vram=24)
        _admit_host("cancel-h")

        _, inst = _submit_job("cancel-test", vram=8)
        job_id = inst["job_id"]

        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE jobs SET status='running',phase='running',host_id='cancel-h' WHERE job_id=%s",
                (job_id,),
            )
        inst = _get_instance(job_id)
        assert inst["status"] == "running"

        # Cancel
        cancel_resp = client.post(f"/instances/{job_id}/cancel")
        assert cancel_resp.status_code == 409

        inst = _get_instance(job_id)
        assert inst["status"] == "running"

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

    def test_retired_binpack_cannot_assign_jobs(self):
        """Only the native scheduler may reserve an admitted job."""
        _reset_state()
        _register_host("bp-h1", vram=80, gpu="A100")
        _admit_host("bp-h1")

        # Submit two jobs — large first
        _, inst_big = _submit_job("big-job", vram=60)
        _, inst_small = _submit_job("small-job", vram=8)

        response = client.post("/api/v2/scheduler/process-binpack")
        assert response.status_code == 410
        assert response.json()["error"]["code"] == "scheduler_service_required"
        assert _get_instance(inst_big["job_id"])["status"] == "queued"
        assert _get_instance(inst_small["job_id"])["status"] == "queued"
        report = _process_queue()
        assert len(report.placements) == 1
        assert _get_instance(report.placements[0].job_id)["status"] == "assigned"

    def test_logs_limit_cap(self):
        """Verify the log endpoint caps at 10k entries."""
        _reset_state()
        _register_host("log-h", vram=24)
        _admit_host("log-h")

        _, inst = _submit_job("log-test")
        job_id = inst["job_id"]

        # Seed more durable logs than both the process buffer and public cap.
        from routes.instances import push_job_log
        from db import _get_pg_pool

        push_job_log(job_id, "Stale process-buffer entry")
        with _get_pg_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO job_logs(job_id,ts,level,line) VALUES(%s,%s,'info',%s)",
                    [(job_id, time.time() + i, f"Durable line {i}") for i in range(10_005)],
                )

        # Request with absurdly high limit — should be capped
        resp = client.get(f"/instances/{job_id}/logs?limit=999999")
        assert resp.status_code == 200
        returned = resp.json()["total"]
        assert returned == 10_000
        assert resp.json()["logs"][0]["message"] == "Durable line 5"
        assert resp.json()["logs"][-1]["message"] == "Durable line 10004"

    def test_gpu_model_request_blocks_non_matching_hosts_until_available(self):
        """GPU model constraints should keep the job queued until a matching host exists."""
        _reset_state()
        _register_host("gpu-2060", vram=24, gpu="RTX 2060")
        _register_host("gpu-4090", vram=24, gpu="RTX 4090")
        _admit_host("gpu-2060")
        _admit_host("gpu-4090")

        resp = client.post(
            "/instance",
            json={
                "name": "gpu-specific-job",
                "vram_needed_gb": 8,
                "gpu_model": "A100",
            },
        )
        assert resp.status_code == 200
        job_id = resp.json()["instance"]["job_id"]

        inst = _get_instance(job_id)
        assert inst["status"] == "queued"

        _register_host("gpu-a100", vram=80, gpu="A100")
        _admit_host("gpu-a100")

        _process_queue()
        inst = _get_instance(job_id)
        assert inst["status"] in ("assigned", "running")
        assert inst["host_id"] == "gpu-a100"


# ══════════════════════════════════════════════════════════════════════
# 8.2 — Failure scenarios
# ══════════════════════════════════════════════════════════════════════


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
            _process_queue()

        chosen_host = _get_instance(job_id)["host_id"]
        _set_status(job_id, "running", host_id=chosen_host)
        inst = _get_instance(job_id)
        assert inst["status"] == "running"

        # Simulate host death: _requeue_dead_host_jobs
        scheduler._requeue_dead_host_jobs(chosen_host, "10.0.0.91")

        assert _get_instance(job_id)["status"] == "running"
        from db import _get_pg_pool
        from control_plane.leases import expire_stale_leases

        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE hosts SET status='dead',availability_state='not_ready' WHERE host_id=%s",
                (chosen_host,),
            )
            conn.execute(
                "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 hour' WHERE job_id=%s AND status='active'",
                (job_id,),
            )
            expired = expire_stale_leases(conn, current_job_id=job_id)
            assert len(expired) == 1
            # Expiry alone holds the requeue until the old attempt is proven
            # stopped, so a silent host cannot double-run it. Disabling the dead
            # host is that proof.
            conn.execute(
                "UPDATE hosts SET administrative_state='disabled' WHERE host_id=%s",
                (chosen_host,),
            )
            from control_plane.leases import release_stopped_attempts

            assert len(release_stopped_attempts(conn)) == 1
        assert _get_instance(job_id)["status"] == "queued"

        # Re-process queue — should go to fo-h2
        _process_queue()
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
            _process_queue()

        _set_status(job_id, "running", host_id=_get_instance(job_id)["host_id"])

        # Mock ping: fv-h1 dead, fv-h2 alive
        def mock_ping(ip):
            return ip != "10.0.0.93"

        with patch.object(scheduler, "ping_host", side_effect=mock_ping):
            fo_resp = client.post("/failover")
            assert fo_resp.status_code == 200

        # Job should have been requeued (and possibly re-assigned to fv-h2)
        inst = _get_instance(job_id)
        assert inst["status"] in ("queued", "assigned", "running")

    def test_raw_requeue_never_discards_native_authority(self):
        """Repeated raw retries cannot bypass lease settlement."""
        _reset_state()
        _register_host("rq-h1", vram=24)
        _admit_host("rq-h1")
        _, inst = _submit_job("retry-job", vram=8)
        job_id = inst["job_id"]
        _process_queue()
        _set_status(job_id, "running", host_id="rq-h1")
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn:
            before = conn.execute(
                "SELECT active_attempt_id,generation FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        for _ in range(4):
            assert scheduler.requeue_job(job_id) is None
        with _get_pg_pool().connection() as conn:
            assert (
                conn.execute(
                    "SELECT active_attempt_id,generation FROM jobs WHERE job_id=%s", (job_id,)
                ).fetchone()
                == before
            )
        assert _get_instance(job_id)["status"] == "running"

    def test_requeue_non_requeuable_status_rejected(self):
        """Requeuing a completed or cancelled job is rejected."""
        _reset_state()
        _register_host("nr-h1", vram=24)
        _admit_host("nr-h1")

        _, inst = _submit_job("completed-job", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            _process_queue()

        _set_status(job_id, "running", host_id="nr-h1")
        _set_status(job_id, "completed", host_id="nr-h1")

        result = scheduler.requeue_job(job_id)
        assert result is None  # rejected


class TestWalletEmptyStopStart:
    """Wallet empty → stop → top-up → start (pause/resume routes were removed)."""

    def test_stop_and_start(self, monkeypatch):
        """Stop a running instance, then start it again."""
        monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "active")
        _reset_state()
        _register_host("pr-h1", vram=24)
        _admit_host("pr-h1")

        _, inst = _submit_job("pause-test", vram=8)
        job_id = inst["job_id"]

        first = _start_native_attempt(job_id, "pr-h1")
        inst = _get_instance(job_id)
        assert inst["status"] == "running"

        stop_resp = client.post(f"/instances/{job_id}/stop")
        assert stop_resp.status_code == 200
        assert stop_resp.json().get("ok") is True

        assert _get_instance(job_id)["status"] == "stopping"
        from control_plane.db import run_transaction
        from control_plane.attempts import report_attempt_status

        run_transaction(lambda conn: report_attempt_status(conn, **first, status="stopped"))
        assert _get_instance(job_id)["status"] == "stopped"

        start_resp = client.post(f"/instances/{job_id}/start")
        assert start_resp.status_code == 200
        assert start_resp.json().get("ok") is True

        assert _get_instance(job_id)["status"] == "queued"
        second = _start_native_attempt(job_id, "pr-h1")
        assert second["attempt_id"] != first["attempt_id"]
        assert second["fencing_token"] > first["fencing_token"]
        assert _get_instance(job_id)["status"] == "running"

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
            conn.execute("UPDATE wallets SET status = 'suspended' WHERE customer_id = 'anonymous'")
            conn.commit()

        # Try to submit — should get 402
        resp = client.post(
            "/instance",
            json={
                "name": "blocked-job",
                "vram_needed_gb": 8,
            },
        )
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

        resp = client.post(
            "/instance",
            json={
                "name": "no-grace-job",
                "vram_needed_gb": 8,
            },
        )
        assert resp.status_code == 402

    def test_start_with_no_funds_rejected(self, monkeypatch):
        """Start should fail if wallet is empty."""
        monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "active")
        _reset_state()
        _register_host("rf-h1", vram=24)
        _admit_host("rf-h1")

        _, inst = _submit_job("resume-fail", vram=8)
        job_id = inst["job_id"]

        authority = _start_native_attempt(job_id, "rf-h1")
        stop = client.post(f"/instances/{job_id}/stop")
        assert stop.status_code == 200, stop.text
        from control_plane.db import run_transaction
        from control_plane.attempts import report_attempt_status

        run_transaction(lambda conn: report_attempt_status(conn, **authority, status="stopped"))

        # Drain wallet
        from billing import get_billing_engine

        be = get_billing_engine()
        wallet = be.get_wallet("anonymous")
        if wallet["balance_cad"] > 0:
            be.charge("anonymous", wallet["balance_cad"], job_id="drain3", description="drain")

        start_resp = client.post(f"/instances/{job_id}/start")
        assert start_resp.status_code == 402


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
            _process_queue()

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
                _process_queue()
            _set_status(inst["job_id"], "running", host_id="ba-h1")
            time.sleep(0.02)
            _set_status(inst["job_id"], "completed", host_id="ba-h1")

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
            _process_queue()

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
        assert (
            resp2.status_code == 400
        ), f"Second bill should return 400 (already billed), got {resp2.status_code}"


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
        resp = client.post(
            f"/agent/logs/{job_id}",
            json={
                "lines": [
                    {"message": "Container started", "level": "info"},
                    {"message": "Downloading model", "level": "info"},
                    {"message": "Training started", "level": "info"},
                ],
            },
        )
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

        resp = client.post(
            "/agent/logs/img-pull-test",
            json={
                "lines": [
                    {"message": "Pulling image...", "level": "info"},
                    {"message": "Error response from daemon: manifest unknown", "level": "error"},
                ],
            },
        )
        assert resp.status_code == 200

    def test_agent_log_gzip(self):
        """Gzip-compressed log batch is accepted."""
        import gzip

        data = _json.dumps(
            {
                "lines": [{"message": "Compressed log", "level": "info"}],
            }
        ).encode()
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
        _process_queue()
        inst = _get_instance(job_id)
        assert inst["status"] in ("assigned", "running")

    def test_assigned_to_leased_to_running(self):
        _reset_state()
        _register_host("stl-h1", vram=24)
        _admit_host("stl-h1")

        _, inst = _submit_job("lease-state", vram=8)
        job_id = inst["job_id"]

        if inst["status"] == "queued":
            _process_queue()

        # Compatibility status projection does not acquire a legacy lease.
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
            _process_queue()

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
            _process_queue()

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

        weird_id = 'evil"; rm -rf /; "'
        push_job_log(weird_id, "test line")

        resp = client.get(f"/instances/{weird_id}/logs/download")
        if resp.status_code == 200:
            disp = resp.headers.get("content-disposition", "")
            # Should not contain raw shell-dangerous characters
            filename_part = disp.split("filename=")[-1].strip('"') if "filename=" in disp else ""
            for dangerous_char in ['"', ";", "`", "$", "|"]:
                assert (
                    dangerous_char not in filename_part
                ), f"Dangerous char '{dangerous_char}' found in filename: {filename_part}"
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
            _process_queue()

        # Walk through states
        from events import get_event_store

        es = get_event_store()
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


class TestDockerComposeSmoke:
    """Build → isolated Compose → readiness → durable native admission."""

    @pytest.mark.timeout(600)
    def test_compose_build_and_health(self, empty_db, tmp_path):
        from pathlib import Path
        from tests.fixtures.compose_smoke import run_compose_smoke
        from tests.test_migration_from_production_snapshot import _alembic
        from psycopg.conninfo import conninfo_to_dict

        # Compose may mount only the security runner's disposable socket, never a
        # TCP DSN. That runner fails any run with a skip, so this cannot skip there.
        if not Path(conninfo_to_dict(empty_db).get("host") or "").parent.name.startswith(
            "xcl-sec-pg-"
        ):
            pytest.skip("runs only under api/scripts/run_security_gates.py")
        _alembic(empty_db, "upgrade", "head")
        run_compose_smoke(Path(__file__).resolve().parents[1], tmp_path, empty_db)


# ══════════════════════════════════════════════════════════════════════
# 8.9 — Agent heartbeat and work pull
# ══════════════════════════════════════════════════════════════════════


class TestAgentWorkPull:
    def test_native_claim_replaces_retired_work_poll(self):
        _reset_state()
        _register_host("wp-h1", vram=24)
        _admit_host("wp-h1")
        _, inst = _submit_job("work-pull", vram=8)
        job_id = inst["job_id"]
        _process_queue()
        assert client.get("/agent/work/wp-h1").status_code == 410
        response = client.post(
            "/agent/v2/commands/claim",
            json={"host_id": "wp-h1", "worker_session_id": "native-work-pull"},
            headers=_worker_auth_headers(),
        )
        assert response.status_code == 200, response.text
        commands = response.json()["commands"]
        assert len(commands) == 1
        assert commands[0]["job_id"] == job_id
        assert commands[0]["command"] == "start_attempt"
        assert commands[0]["attempt_id"] and commands[0]["fencing_token"] > 0


class TestCancelRequeueAuth:
    """Verify cancel/requeue endpoints reject unauthenticated requests when auth is enabled."""

    def test_cancel_requires_auth(self):
        _reset_state()
        _register_host("auth-c-h1", vram=24)
        _admit_host("auth-c-h1")
        _, inst = _submit_job("auth-cancel", vram=8)
        job_id = inst["job_id"]
        _process_queue()

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
        _process_queue()
        _set_status(job_id, "running")
        time.sleep(0.1)
        _set_status(job_id, "completed")

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
        _process_queue()

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
        _process_queue()

        resp = client.get(f"/agent/ssh-keys/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert "ok" in body
        assert "keys" in body
        assert isinstance(body["keys"], list)
