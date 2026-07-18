"""Phase 5 — /agent/v2 fenced worker protocol over real HTTP + PostgreSQL.

Drives the full authority lifecycle through the API surface the worker
agent uses: negotiation gate, claim+ACK work delivery (and its exclusive
partition against the v1 destructive drain), the hard lease-claim gate,
fenced status reporting through terminal settlement, and definitive
fence-loss signals.
"""

import json
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
                "SELECT tgname FROM pg_trigger "
                "WHERE tgname='trg_jobs_control_plane_projection'"
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
        "host_id": host_id, "gpu_model": model, "gpu_count": 2,
        "free_vram_gb": 48.0, "total_vram_gb": 48.0, "cost_per_hour": 1.0,
        "admitted": True, "last_seen": time.time(),
    }
    job_payload = {
        "job_id": job_id, "name": job_id, "gpu_model": model,
        "num_gpus": 1, "vram_needed_gb": 8.0, "status": "queued",
    }
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload) "
            "VALUES (%s, 'active', %s, %s)",
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
    cmd = row if isinstance(row, dict) else {
        "command_id": row[0], "command": row[1], "args": row[2],
        "attempt_id": row[3], "fencing_token": row[4],
    }
    return job_id, host_id, {
        "command_id": str(cmd["command_id"]),
        "attempt_id": str(cmd["attempt_id"]),
        "fencing_token": int(cmd["fencing_token"]),
        "lease_id": cmd["args"]["lease_id"],
    }


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
    def test_default_denies_v2(self):
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
                "host_id": host_id, "error_code": "lease_claim_rejected",
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
                    "job_id": job_id, "attempt_id": refs["attempt_id"],
                    "host_id": host_id, "fencing_token": refs["fencing_token"],
                    "status": status,
                },
            )
            assert r.status_code == 200, (status, r.text)
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        assert (row[0] if not isinstance(row, dict) else row["status"]) == "running"

        # Terminal success settles everything atomically.
        r = client.post(
            "/agent/v2/attempts/status",
            json={
                "job_id": job_id, "attempt_id": refs["attempt_id"],
                "host_id": host_id, "fencing_token": refs["fencing_token"],
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
                "job_id": job_id, "attempt_id": refs["attempt_id"],
                "host_id": host_id, "fencing_token": refs["fencing_token"],
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
                "job_id": job_id, "attempt_id": refs["attempt_id"],
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
            "job_id": job_id, "attempt_id": refs["attempt_id"],
            "host_id": host_id, "fencing_token": refs["fencing_token"],
        }
        assert (
            client.post("/agent/v2/attempts/status", json={**base, "status": "running"})
            .status_code == 200
        )
        r = client.post(
            "/agent/v2/attempts/status", json={**base, "status": "starting"}
        )
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
                "job_id": job_id, "attempt_id": refs["attempt_id"],
                "host_id": host_id, "fencing_token": refs["fencing_token"],
                "status": "failed", "failure_code": "container_failed",
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
        a = attempt if isinstance(attempt, dict) else {
            "status": attempt[0], "failure_code": attempt[1],
        }
        assert j["status"] == "failed" and j["reason_code"] == "container_failed"
        assert a["status"] == "failed" and a["failure_code"] == "container_failed"


class TestLegacyPatchFence:
    def test_wrong_host_patch_fenced_out(self, fleet):
        """Split-brain guard: a worker on another host cannot v1-report
        status for an attempt-owned job."""
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        r = client.patch(
            f"/instance/{job_id}",
            json={"status": "running", "host_id": f"h-{marker}-imposter"},
            headers=_worker_headers(),
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "fencing_violation"
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        assert (row[0] if not isinstance(row, dict) else row["status"]) == "assigned"

    def test_authority_host_patch_allowed(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        r = client.patch(
            f"/instance/{job_id}",
            json={"status": "starting", "host_id": host_id},
            headers=_worker_headers(),
        )
        assert r.status_code == 200

    def test_patch_without_host_id_passes(self, fleet):
        """Unattributable reports pass (v2 mirror is authoritative for
        enrolled hosts) — the fence is best-effort by design."""
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, refs = _place_one(fleet, marker)
        r = client.patch(
            f"/instance/{job_id}", json={"status": "starting"},
            headers=_worker_headers(),
        )
        assert r.status_code == 200

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
