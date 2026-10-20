"""Phase 5 — /agent/v2 fenced worker protocol over real HTTP + PostgreSQL.

Drives the full authority lifecycle through the API surface the worker
agent uses: negotiation gate, claim+ACK work delivery (and its exclusive
partition against the v1 destructive drain), the hard lease-claim gate,
fenced status reporting through terminal settlement, and definitive
fence-loss signals.
"""

import json
import os
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _migrated = (
            _c.execute(
                "SELECT tgname FROM pg_trigger " "WHERE tgname='trg_jobs_control_plane_projection'"
            ).fetchone()
            is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 059")

from fastapi.testclient import TestClient

from api import app
from control_plane.commands import enqueue_current_attempt_command
from control_plane.db import run_transaction
from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.service import SchedulerService

client = TestClient(app)


def _worker_headers() -> dict:
    import os

    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def fleet():
    ids = {"jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _place_one(fleet, marker):
    """Seed host+job and place through the real reservation pipeline.

    Returns (job_id, host_id, command_row) where command_row carries the
    attempt/lease/fence refs the worker receives.
    """
    model = f"RTX-{marker}"
    host_id = f"h-{marker}-1"
    job_id = f"j-{marker}-run"
    host_payload = {
        "host_id": host_id,
        "gpu_model": model,
        "gpu_count": 2,
        "free_vram_gb": 48.0,
        "total_vram_gb": 48.0,
        "cost_per_hour": 1.0,
        "admitted": True,
        "capabilities": {"egress_isolation": True},
        "last_seen": time.time(),
    }
    job_payload = {
        "job_id": job_id,
        "name": job_id,
        "gpu_model": model,
        "num_gpus": 1,
        "vram_needed_gb": 8.0,
        "status": "queued",
    }
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload, admission_state) "
            "VALUES (%s, 'active', %s, %s, 'admitted')",
            (host_id, time.time(), json.dumps(host_payload)),
        )
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, payload) "
            "VALUES (%s, 'queued', 0, %s, %s)",
            (job_id, time.time(), json.dumps(job_payload)),
        )
        conn.commit()
    fleet["hosts"].append(host_id)
    fleet["jobs"].append(job_id)

    cfg = SchedulerConfig(
        mode=SchedulerMode.CANARY,
        replica_id=f"v2test-{marker}",
        canary_gpu_models=frozenset({model.lower()}),
        canary_host_ids=frozenset({host_id}),
        # Long offer TTL: full-suite runs exceed the 60s default and a
        # rival test's lease-expiry sweep would cancel our start command.
        lease_claim_ttl_sec=600,
    )
    report = SchedulerService(cfg).tick()
    assert len(report.placements) == 1, report
    resv = report.placements[0]

    with _pool.connection() as conn:
        row = conn.execute(
            """SELECT command_id, command, args, attempt_id, fencing_token
                 FROM agent_commands WHERE command_id=%s""",
            (resv.command_id,),
        ).fetchone()
    cmd = (
        row
        if isinstance(row, dict)
        else {
            "command_id": row[0],
            "command": row[1],
            "args": row[2],
            "attempt_id": row[3],
            "fencing_token": row[4],
        }
    )
    return (
        job_id,
        host_id,
        {
            "command_id": str(cmd["command_id"]),
            "attempt_id": str(cmd["attempt_id"]),
            "fencing_token": int(cmd["fencing_token"]),
            "lease_id": cmd["args"]["lease_id"],
        },
    )


def _auth(job_id, host_id, refs, **overrides):
    body = {
        "lease_id": refs["lease_id"],
        "job_id": job_id,
        "attempt_id": refs["attempt_id"],
        "host_id": host_id,
        "fencing_token": refs["fencing_token"],
        "worker_session_id": "sess-test",
    }
    body.update(overrides)
    return body


class TestNegotiation:
    @pytest.mark.parametrize("mode,legacy", [("active", False), ("canary", True)])
    def test_execution_policy_is_explicit(self, monkeypatch, mode, legacy):
        monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", mode)
        monkeypatch.setenv("XCELSIOR_AGENT_V2_HOSTS", "*")
        response = client.get("/agent/v2/negotiate/policy-host")
        assert response.json()["legacy_execution_allowed"] is legacy

    def test_default_denies_v2(self, monkeypatch):
        # This asserts the *default* posture, so it must establish it: the
        # test environment pins XCELSIOR_AGENT_V2_HOSTS=* for the suites that
        # need v2 reachable, and ambient config must not decide the outcome.
        monkeypatch.delenv("XCELSIOR_AGENT_V2_HOSTS", raising=False)
        r = client.get("/agent/v2/negotiate/some-host")
        assert r.status_code == 200
        assert r.json()["v2"] is False and r.json()["protocol"] == 1

    def test_allowlist_enables_v2(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_AGENT_V2_HOSTS", "Host-A, host-b")
        r = client.get("/agent/v2/negotiate/host-a")
        assert r.json()["v2"] is True and r.json()["protocol"] == 2
        assert "leases.fenced" in r.json()["features"]
        assert client.get("/agent/v2/negotiate/host-c").json()["v2"] is False

    def test_wildcard(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_AGENT_V2_HOSTS", "*")
        assert client.get("/agent/v2/negotiate/anything").json()["v2"] is True


class TestCommandDeliveryPartition:
    def test_cached_fenced_work_is_not_delivered(self, fleet):
        from routes import agent

        job_id, host_id, refs = _place_one(fleet, uuid.uuid4().hex[:8])
        hint = {"job_id": job_id, "status": "assigned", "host_id": host_id}
        with agent._agent_lock:
            agent._agent_work[host_id] = [hint]
        assert client.get("/agent/work/" + host_id).status_code == 204

    def test_failed_ownership_read_keeps_cached_work(self, fleet, monkeypatch):
        from routes import agent

        job_id, host_id, refs = _place_one(fleet, uuid.uuid4().hex[:8])
        hint = {"job_id": job_id, "status": "assigned", "host_id": host_id}
        with agent._agent_lock:
            agent._agent_work[host_id] = [hint]
        monkeypatch.setattr(agent, "eligible_work_ids", lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))
        try:
            assert client.get("/agent/work/" + host_id).status_code == 503
            assert agent._agent_work[host_id] == [hint]
        finally:
            agent._agent_work.pop(host_id, None)

    @pytest.mark.parametrize("operation", ["grant", "renew", "release"])
    def test_event_store_cannot_mutate_a_legacy_lease_for_fenced_history(self, fleet, operation):
        from events import get_event_store, Lease
        from control_plane.legacy_execution import LegacyExecutionRejected

        job_id, host_id, refs = _place_one(fleet, uuid.uuid4().hex[:8])
        lease = Lease(job_id=job_id, host_id=host_id)
        with _pool.connection() as conn:
            conn.execute("INSERT INTO leases (lease_id,job_id,host_id,granted_at,expires_at,"
                         "last_renewed,duration_sec,status) VALUES (%s,%s,%s,%s,%s,%s,%s,'active')",
                         (lease.lease_id,job_id,host_id,lease.granted_at,lease.expires_at,
                          lease.last_renewed,lease.duration_sec))
        try:
            store = get_event_store()
            with pytest.raises(LegacyExecutionRejected):
                if operation == "grant":
                    store.grant_lease(job_id, host_id)
                elif operation == "renew":
                    store.renew_lease(job_id, host_id)
                else:
                    store.release_lease(job_id)
            with _pool.connection() as conn:
                row = conn.execute("SELECT status,expires_at FROM leases WHERE lease_id=%s", (lease.lease_id,)).fetchone()
                assert row == ("active", lease.expires_at)
        finally:
            with _pool.connection() as conn:
                conn.execute("DELETE FROM leases WHERE job_id=%s", (job_id,))

    def test_compatibility_write_waits_for_the_placement_job_lock(self, fleet):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from events import get_event_store
        from control_plane.legacy_execution import LegacyExecutionRejected

        job_id, host_id, refs = _place_one(fleet, uuid.uuid4().hex[:8])
        started = threading.Event()
        def grant():
            started.set()
            return get_event_store().grant_lease(job_id, host_id)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with _pool.connection() as conn:
                conn.execute("SELECT job_id FROM jobs WHERE job_id=%s FOR UPDATE", (job_id,))
                future = executor.submit(grant)
                assert started.wait(2)
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.1)
            with pytest.raises(LegacyExecutionRejected):
                future.result(timeout=5)

    def test_legacy_execution_usage_is_exported(self, monkeypatch):
        from prometheus_client import REGISTRY, generate_latest
        monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "active")
        labels = {"path": "work_poll", "outcome": "requested"}
        before = REGISTRY.get_sample_value("xcelsior_legacy_execution_total", labels) or 0
        assert client.get("/agent/work/retired-probe").status_code == 410
        assert REGISTRY.get_sample_value("xcelsior_legacy_execution_total", labels) == before + 1
        assert REGISTRY.get_sample_value("xcelsior_legacy_execution_enabled") == 0
        assert b'xcelsior_legacy_execution_total{' in generate_latest(REGISTRY)

    @pytest.mark.parametrize("path", ["/agent/work/", "/agent/v2/work/"])
    def test_work_poll_never_returns_fenced_placement(self, fleet, path):
        job_id, host_id, refs = _place_one(fleet, uuid.uuid4().hex[:8])
        response = client.get(path + host_id)
        assert response.status_code == 204
        with _pool.connection() as conn:
            assert conn.execute("SELECT status FROM agent_commands WHERE command_id=%s",
                                (refs["command_id"],)).fetchone()[0] == "pending"

    @pytest.mark.parametrize("operation", ["claim", "renew", "release"])
    def test_legacy_lease_route_cannot_write_fenced_job(self, fleet, operation):
        job_id, host_id, refs = _place_one(fleet, uuid.uuid4().hex[:8])
        body = {"job_id": job_id, "host_id": host_id}
        if operation == "release":
            body["reason"] = "completed"
        response = client.post(f"/agent/lease/{operation}", json=body)
        assert response.status_code == 409
        with _pool.connection() as conn:
            assert conn.execute("SELECT count(*) FROM leases WHERE job_id=%s", (job_id,)).fetchone()[0] == 0
            assert conn.execute("SELECT status FROM placement_leases WHERE lease_id=%s",
                                (refs["lease_id"],)).fetchone()[0] == "offered"

    def test_fenced_history_cannot_be_reclaimed_as_legacy(self, fleet):
        job_id, host_id, refs = _place_one(fleet, uuid.uuid4().hex[:8])
        with _pool.connection() as conn:
            conn.execute("UPDATE jobs SET active_attempt_id=NULL WHERE job_id=%s", (job_id,))
        response = client.post("/agent/lease/claim", json={"job_id": job_id, "host_id": host_id})
        assert response.status_code == 409

    def test_v1_drain_never_touches_attempt_commands(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)

        # v1 destructive drain: must NOT return or delete the start command.
        r = client.get(f"/agent/commands/{host_id}")
        assert r.status_code == 200
        assert all(c["command"] != "start_attempt" for c in r.json()["commands"])
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT status FROM agent_commands WHERE command_id=%s",
                (refs["command_id"],),
            ).fetchone()
        assert row is not None, "v1 drain destroyed the v2 start command"

        # v2 claim: returns it with full authority refs.
        r = client.post(
            "/agent/v2/commands/claim",
            json={"host_id": host_id, "worker_session_id": "sess-test"},
        )
        assert r.status_code == 200
        cmds = [c for c in r.json()["commands"] if c["job_id"] == job_id]
        assert len(cmds) == 1
        got = cmds[0]
        assert got["command"] == "start_attempt"
        assert got["attempt_id"] == refs["attempt_id"]
        assert got["fencing_token"] == refs["fencing_token"]
        assert got["args"]["spec"]["job_id"] == job_id

    def test_ack_is_once_only_and_replayed(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        client.post(
            "/agent/v2/commands/claim",
            json={"host_id": host_id, "worker_session_id": "sess-test"},
        )
        r1 = client.post(
            f"/agent/v2/commands/{refs['command_id']}/ack",
            json={"host_id": host_id, "result": {"started": True}},
        )
        assert r1.status_code == 200 and r1.json()["duplicate"] is False
        r2 = client.post(
            f"/agent/v2/commands/{refs['command_id']}/ack",
            json={"host_id": host_id, "result": {"started": True}},
        )
        assert r2.status_code == 200 and r2.json()["duplicate"] is True
        assert r2.json()["result"] == {"started": True}

    def test_nack_nonretryable_dead_letters(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        client.post(
            "/agent/v2/commands/claim",
            json={"host_id": host_id, "worker_session_id": "sess-test"},
        )
        r = client.post(
            f"/agent/v2/commands/{refs['command_id']}/nack",
            json={
                "host_id": host_id,
                "error_code": "lease_claim_rejected",
                "retryable": False,
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "dead_letter"


class TestFencedLeaseLifecycle:
    def test_full_lifecycle_to_success(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        auth = _auth(job_id, host_id, refs)

        # Hard gate: claim succeeds exactly once.
        r = client.post("/agent/v2/leases/claim", json=auth)
        assert r.status_code == 200
        assert r.json()["fencing_token"] == refs["fencing_token"]
        r2 = client.post("/agent/v2/leases/claim", json=auth)
        assert r2.status_code == 409  # not 'offered' anymore

        # Renewal works under the exact tuple.
        assert client.post("/agent/v2/leases/renew", json=auth).status_code == 200

        # Fenced status progression.
        for status in ("lease_claimed", "starting", "running"):
            r = client.post(
                "/agent/v2/attempts/status",
                json={
                    "job_id": job_id,
                    "attempt_id": refs["attempt_id"],
                    "host_id": host_id,
                    "fencing_token": refs["fencing_token"],
                    "status": status,
                },
            )
            assert r.status_code == 200, (status, r.text)
        with _pool.connection() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE job_id=%s", (job_id,)).fetchone()
        assert (row[0] if not isinstance(row, dict) else row["status"]) == "running"

        # Terminal success settles everything atomically.
        r = client.post(
            "/agent/v2/attempts/status",
            json={
                "job_id": job_id,
                "attempt_id": refs["attempt_id"],
                "host_id": host_id,
                "fencing_token": refs["fencing_token"],
                "status": "succeeded",
            },
        )
        assert r.status_code == 200 and r.json()["terminal"] is True
        with _pool.connection() as conn:
            job = conn.execute(
                "SELECT status, active_attempt_id FROM jobs WHERE job_id=%s",
                (job_id,),
            ).fetchone()
            allocs = conn.execute(
                "SELECT count(*) FROM gpu_device_allocations "
                "WHERE attempt_id=%s AND status='active'",
                (refs["attempt_id"],),
            ).fetchone()
            lease = conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s",
                (refs["lease_id"],),
            ).fetchone()
        job_status = job[0] if not isinstance(job, dict) else job["status"]
        active_attempt = job[1] if not isinstance(job, dict) else job["active_attempt_id"]
        assert job_status == "completed" and active_attempt is None
        assert (allocs[0] if not isinstance(allocs, dict) else allocs["count"]) == 0
        assert (lease[0] if not isinstance(lease, dict) else lease["status"]) == "released"

        # After terminal: any further report is a definitive fence loss.
        r = client.post(
            "/agent/v2/attempts/status",
            json={
                "job_id": job_id,
                "attempt_id": refs["attempt_id"],
                "host_id": host_id,
                "fencing_token": refs["fencing_token"],
                "status": "running",
            },
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "fencing_violation"

    def test_wrong_fence_rejected_everywhere(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        bad = _auth(job_id, host_id, refs, fencing_token=refs["fencing_token"] + 999)

        assert client.post("/agent/v2/leases/claim", json=bad).status_code == 409
        r = client.post(
            "/agent/v2/attempts/status",
            json={
                "job_id": job_id,
                "attempt_id": refs["attempt_id"],
                "host_id": host_id,
                "fencing_token": refs["fencing_token"] + 999,
                "status": "running",
            },
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "fencing_violation"

    def test_renewal_rejected_after_release(self, fleet):
        """The worker's definitive fence-loss signal on the renew path."""
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        auth = _auth(job_id, host_id, refs)
        assert client.post("/agent/v2/leases/claim", json=auth).status_code == 200
        assert client.post("/agent/v2/leases/release", json=auth).json()["released"]
        r = client.post("/agent/v2/leases/renew", json=auth)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "lease_renew_rejected"

    def test_out_of_order_status_rejected(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        auth = _auth(job_id, host_id, refs)
        assert client.post("/agent/v2/leases/claim", json=auth).status_code == 200
        base = {
            "job_id": job_id,
            "attempt_id": refs["attempt_id"],
            "host_id": host_id,
            "fencing_token": refs["fencing_token"],
        }
        assert (
            client.post("/agent/v2/attempts/status", json={**base, "status": "running"}).status_code
            == 200
        )
        r = client.post("/agent/v2/attempts/status", json={**base, "status": "starting"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "out_of_order"

    def test_failed_attempt_records_failure(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        auth = _auth(job_id, host_id, refs)
        assert client.post("/agent/v2/leases/claim", json=auth).status_code == 200
        r = client.post(
            "/agent/v2/attempts/status",
            json={
                "job_id": job_id,
                "attempt_id": refs["attempt_id"],
                "host_id": host_id,
                "fencing_token": refs["fencing_token"],
                "status": "failed",
                "failure_code": "container_failed",
                "detail": {"error": "OOM"},
            },
        )
        assert r.status_code == 200 and r.json()["terminal"] is True
        with _pool.connection() as conn:
            job = conn.execute(
                "SELECT status, reason_code FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
            attempt = conn.execute(
                "SELECT status, failure_code FROM job_attempts WHERE attempt_id=%s",
                (refs["attempt_id"],),
            ).fetchone()
        j = job if isinstance(job, dict) else {"status": job[0], "reason_code": job[1]}
        a = (
            attempt
            if isinstance(attempt, dict)
            else {
                "status": attempt[0],
                "failure_code": attempt[1],
            }
        )
        assert j["status"] == "failed" and j["reason_code"] == "container_failed"
        assert a["status"] == "failed" and a["failure_code"] == "container_failed"

    def test_fenced_stop_command_is_idempotent_and_settles_attempt(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)

        # Consume the reservation's start command, then establish active
        # worker authority and move the attempt to running.
        claimed = client.post(
            "/agent/v2/commands/claim",
            json={"host_id": host_id, "worker_session_id": "sess-test"},
        ).json()["commands"]
        start = next(c for c in claimed if c["command_id"] == refs["command_id"])
        assert start["command"] == "start_attempt"
        assert (
            client.post(
                f"/agent/v2/commands/{refs['command_id']}/ack",
                json={"host_id": host_id, "result": {"lease_claimed": True}},
            ).status_code
            == 200
        )
        auth = _auth(job_id, host_id, refs)
        assert client.post("/agent/v2/leases/claim", json=auth).status_code == 200
        assert (
            client.post(
                "/agent/v2/attempts/status",
                json={**auth, "status": "running"},
            ).status_code
            == 200
        )

        def enqueue(conn):
            return enqueue_current_attempt_command(
                conn,
                job_id=job_id,
                command="stop_attempt",
                args={"preserve": True},
                created_by="test_fenced_stop",
            )

        first = run_transaction(enqueue)
        duplicate = run_transaction(enqueue)
        assert duplicate.command_id == first.command_id
        assert first.attempt_id == refs["attempt_id"]
        assert first.fencing_token == refs["fencing_token"]

        stop_claims = client.post(
            "/agent/v2/commands/claim",
            json={"host_id": host_id, "worker_session_id": "sess-test"},
        ).json()["commands"]
        stop = next(c for c in stop_claims if c["command_id"] == first.command_id)
        assert stop["command"] == "stop_attempt"
        assert stop["attempt_id"] == refs["attempt_id"]
        assert stop["fencing_token"] == refs["fencing_token"]

        stopped = client.post(
            "/agent/v2/attempts/status",
            json={**auth, "status": "stopped"},
        )
        assert stopped.status_code == 200 and stopped.json()["terminal"] is True
        with _pool.connection() as conn:
            job = conn.execute(
                "SELECT status, active_attempt_id, payload FROM jobs WHERE job_id=%s",
                (job_id,),
            ).fetchone()
            attempt = conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id=%s",
                (refs["attempt_id"],),
            ).fetchone()
            lease = conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s",
                (refs["lease_id"],),
            ).fetchone()
            active_allocations = conn.execute(
                "SELECT count(*) FROM gpu_device_allocations "
                "WHERE attempt_id=%s AND status='active'",
                (refs["attempt_id"],),
            ).fetchone()[0]
        assert job[0] == "stopped" and job[1] is None
        assert job[2]["status"] == "stopped"
        assert isinstance(job[2]["stopped_at"], (int, float))
        assert attempt[0] == "cancelled"
        assert lease[0] == "released"
        assert active_allocations == 0


class TestLegacyPatchFence:
    def test_scheduler_raw_writer_is_fenced_for_attempt_owned_job(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        from scheduler import update_job_status

        with pytest.raises(RuntimeError, match="raw lifecycle mutation is fenced out"):
            update_job_status(job_id, "running", host_id=host_id)
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT status, active_attempt_id FROM jobs WHERE job_id=%s",
                (job_id,),
            ).fetchone()
        assert row[0] == "assigned"
        assert str(row[1]) == refs["attempt_id"]

    def test_wrong_host_patch_fenced_out(self, fleet):
        """Split-brain guard: a worker on another host cannot v1-report
        status for an attempt-owned job."""
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        r = client.patch(
            f"/instance/{job_id}",
            json={
                "status": "running",
                "host_id": f"h-{marker}-imposter",
                "attempt_id": refs["attempt_id"],
                "fencing_token": refs["fencing_token"],
            },
            headers=_worker_headers(),
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "fencing_violation"
        with _pool.connection() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE job_id=%s", (job_id,)).fetchone()
        assert (row[0] if not isinstance(row, dict) else row["status"]) == "assigned"

    def test_authority_host_patch_allowed(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        r = client.patch(
            f"/instance/{job_id}",
            json={
                "status": "starting",
                "host_id": host_id,
                "attempt_id": refs["attempt_id"],
                "fencing_token": refs["fencing_token"],
            },
            headers=_worker_headers(),
        )
        assert r.status_code == 200

    def test_patch_without_authority_tuple_is_fenced(self, fleet):
        """Attempt-owned compatibility writes fail closed when unattributed."""
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        r = client.patch(
            f"/instance/{job_id}",
            json={"status": "starting"},
            headers=_worker_headers(),
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "fencing_violation"

    def test_same_host_wrong_fence_is_rejected(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        r = client.patch(
            f"/instance/{job_id}",
            json={
                "status": "running",
                "host_id": host_id,
                "attempt_id": refs["attempt_id"],
                "fencing_token": refs["fencing_token"] + 1,
            },
            headers=_worker_headers(),
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "fencing_violation"

    def test_non_attempt_job_unaffected(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id = f"j-{marker}-legacy"
        with _pool.connection() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, priority, submitted_at, payload) "
                "VALUES (%s, 'assigned', 0, %s, %s)",
                (job_id, time.time(), json.dumps({"job_id": job_id, "name": job_id})),
            )
            conn.commit()
        fleet["jobs"].append(job_id)
        r = client.patch(
            f"/instance/{job_id}",
            json={"status": "starting", "host_id": "any-host"},
            headers=_worker_headers(),
        )
        assert r.status_code == 200


class TestCompleteWorkerCutover:
    def test_host_commands_use_claim_ack_without_destructive_drain(self, fleet):
        jid, hid, refs = _place_one(fleet, 'host-control-' + uuid.uuid4().hex[:8])
        with _pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO agent_commands(host_id,job_id,command,args,created_at,expires_at,status) "
                "VALUES(%s,%s,'reverify','{}',%s,%s,'pending') RETURNING command_id",
                (hid, jid, time.time(), time.time() + 300),
            ).fetchone()
            command_id = str(row[0])
        response = client.post('/agent/v2/commands/claim', json={
            'host_id': hid, 'worker_session_id': 'complete-cutover', 'host_commands': True,
        })
        assert response.status_code == 200, response.text
        assert {item['command_id'] for item in response.json()['commands']} == {command_id, refs['command_id']}
        assert client.post('/agent/v2/commands/claim', json={
            'host_id': hid, 'worker_session_id': 'other-session', 'host_commands': True,
        }).json()['commands'] == []
        assert client.post(f'/agent/v2/commands/{command_id}/ack', json={'host_id': hid, 'result': {'dispatched': True}}).status_code == 200
        duplicate = client.post(f'/agent/v2/commands/{command_id}/ack', json={'host_id': hid})
        assert duplicate.json()['duplicate'] is True
        with _pool.connection() as conn:
            assert conn.execute('SELECT status FROM agent_commands WHERE command_id=%s', (command_id,)).fetchone()[0] == 'acknowledged'

    def test_runtime_connection_metadata_is_fenced_and_atomic(self, fleet):
        jid, hid, refs = _place_one(fleet, 'runtime-meta-' + uuid.uuid4().hex[:8])
        assert client.post('/agent/v2/leases/claim', json=_auth(jid, hid, refs)).status_code == 200
        payload = {key: value for key, value in _auth(jid, hid, refs).items() if key not in ('lease_id', 'worker_session_id')}
        payload.update(status='running', runtime_metadata={
            'container_id': 'exact-container', 'container_name': 'xcl-exact-attempt',
            'public_ssh_port': 12345, 'interactive': True,
        })
        response = client.post('/agent/v2/attempts/status', json=payload)
        assert response.status_code == 200, response.text
        # Same lifecycle status can receive a later connection observation.
        payload['runtime_metadata']['public_ssh_port'] = 12346
        assert client.post('/agent/v2/attempts/status', json=payload).status_code == 200
        stale = {**payload, 'fencing_token': refs['fencing_token'] + 1, 'runtime_metadata': {'public_ssh_port': 12347}}
        assert client.post('/agent/v2/attempts/status', json=stale).status_code == 409
        invalid = {**payload, 'runtime_metadata': {'public_ssh_port': 22}}
        assert client.post('/agent/v2/attempts/status', json=invalid).status_code == 422
        assert client.post('/agent/v2/attempts/status', json={**payload, 'status': 'starting', 'runtime_metadata': {'public_ssh_port': 12348}}).status_code == 400
        with _pool.connection() as conn:
            row = conn.execute("SELECT status,payload->>'public_ssh_port',payload->>'container_name' FROM jobs WHERE job_id=%s", (jid,)).fetchone()
            assert row == ('running', '12346', 'xcl-exact-attempt')
        assert client.post('/agent/v2/attempts/status', json={**payload, 'status': 'succeeded'}).status_code == 200
        with _pool.connection() as conn:
            assert conn.execute('SELECT status,active_attempt_id FROM jobs WHERE job_id=%s', (jid,)).fetchone() == ('completed', None)

    @pytest.mark.parametrize('method,path,body', [
        ('GET', '/agent/work/cutover-host', None),
        ('GET', '/agent/commands/cutover-host', None),
        ('GET', '/agent/preempt/cutover-host', None),
        ('POST', '/agent/lease/claim', {'host_id': 'cutover-host', 'job_id': 'cutover-job'}),
        ('POST', '/agent/lease/renew', {'host_id': 'cutover-host', 'job_id': 'cutover-job'}),
        ('POST', '/agent/lease/release', {'job_id': 'cutover-job'}),
        ('PATCH', '/instance/cutover-job', {'status': 'running', 'host_id': 'cutover-host'}),
    ])
    def test_active_mode_closes_legacy_worker_mutations(self, monkeypatch, method, path, body):
        monkeypatch.setenv('XCELSIOR_SCHEDULER_MODE', 'active')
        response = client.request(method, path, json=body, headers={"Authorization": "Bearer " + os.environ["XCELSIOR_API_TOKEN"]})
        assert response.status_code == 410, response.text

    def test_active_mode_rejects_legacy_admin_preemption_before_queue_effect(self, monkeypatch):
        import agent_preempt
        import routes.agent as agent_routes
        from unittest.mock import Mock

        monkeypatch.setenv('XCELSIOR_SCHEDULER_MODE', 'active')
        monkeypatch.setattr(agent_routes, '_require_user_grant', lambda *a, **k: {'is_admin': True})
        enqueue = Mock(side_effect=AssertionError('legacy preemption queued'))
        notify = Mock(side_effect=AssertionError('preemption announced without delivery'))
        monkeypatch.setattr(agent_preempt, 'schedule_host_preemptions', enqueue)
        monkeypatch.setattr(agent_routes, 'broadcast_sse', notify)
        response = client.post('/agent/preempt/cutover-host/cutover-job')
        assert response.status_code == 410, response.text
        assert response.json()['detail']['error']['code'] == 'v2_protocol_required'
        enqueue.assert_not_called()
        notify.assert_not_called()

    def test_new_operational_metrics_read_durable_command_states(self, fleet):
        from control_plane.operational_metrics import collect_operational_snapshot
        jid, hid, refs = _place_one(fleet, 'durable-metrics-' + uuid.uuid4().hex[:8])
        with _pool.connection() as conn:
            before = collect_operational_snapshot(conn)
            for state in ['pending', 'claimed', 'dead_letter']:
                row = conn.execute("INSERT INTO agent_commands(host_id,job_id,command,args,status,created_at,expires_at) VALUES(%s,%s,'reverify','{}',%s,%s,%s) RETURNING command_id", (hid, jid, 'pending' if state == 'claimed' else state, time.time(), time.time()+300)).fetchone()
                if state == 'claimed':
                    conn.execute("UPDATE agent_commands SET status='claimed',claim_owner=%s,claim_session='metric-session',claim_expires_at=clock_timestamp()+interval '60 seconds',attempt_count=1 WHERE command_id=%s", (hid,row[0]))
            after = collect_operational_snapshot(conn)
            assert after['commands_pending'] == before['commands_pending'] + 1
            assert after['commands_claimed'] == before['commands_claimed'] + 1
            assert after['commands_dead_lettered'] == before['commands_dead_lettered'] + 1
            assert after['active_attempts'] >= 1
            assert after['active_gpu_allocations'] >= 1

    def test_real_worker_health_probe_claims_and_acks_without_gpu_work(self, fleet, monkeypatch, tmp_path):
        import worker_agent as worker
        jid, hid, refs = _place_one(fleet, 'worker-probe-' + uuid.uuid4().hex[:8])
        with _pool.connection() as conn:
            conn.execute("UPDATE agent_commands SET status='cancelled' WHERE command_id=%s", (refs['command_id'],))
            row = conn.execute("INSERT INTO agent_commands(host_id,job_id,command,args,status,created_at,expires_at) VALUES(%s,%s,'health_probe','{}','pending',%s,%s) RETURNING command_id", (hid,jid,time.time(),time.time()+300)).fetchone()
            command_id = str(row[0])
        monkeypatch.setattr(worker, 'HOST_ID', hid)
        monkeypatch.setattr(worker, '_v2_enabled', True)
        monkeypatch.setattr(worker, '_v2_attempts', {})
        monkeypatch.setattr(worker, '_v2_completed_commands', {})
        monkeypatch.setattr(worker, '_V2_JOURNAL_PATH', str(tmp_path/'probe-journal.json'))
        def post(path, payload, timeout=10):
            response = client.post(path, json=payload)
            return response.status_code, response.json()
        monkeypatch.setattr(worker, '_v2_post', post)
        monkeypatch.setattr(worker.subprocess, 'run', lambda *a,**k: pytest.fail('probe must not launch a process'))
        assert worker.drain_v2_commands() == 1
        with _pool.connection() as conn:
            state, result = conn.execute('SELECT status,ack_result FROM agent_commands WHERE command_id=%s', (command_id,)).fetchone()
            assert state == 'acknowledged'
            assert result['protocol'] == 2 and result['worker_session_id'] == worker.WORKER_SESSION_ID
