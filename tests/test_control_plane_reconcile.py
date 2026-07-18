"""Phase 6 — observation ingest + desired-vs-observed reconciliation.

Real PostgreSQL + real HTTP route; report-only findings with dedupe and
auto-resolution, and the PK-coalesced queue lifecycle.
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
            _c.execute("SELECT to_regclass('host_observations')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 057")

from fastapi.testclient import TestClient

from api import app
from control_plane.db import run_transaction
from control_plane.observations import (
    ingest_observation,
    latest_observation,
    prune_observations,
)
from control_plane.reconcile import process_due, reconcile_host
from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.service import SchedulerService

client = TestClient(app)


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
            conn.execute(
                "DELETE FROM reconciliation_findings WHERE resource_id=%s "
                "OR (resource_type='host' AND resource_id=%s)",
                (hid, hid),
            )
            conn.execute(
                "DELETE FROM reconciliation_queue WHERE resource_id=%s", (hid,)
            )
            # Reconciler-enqueued stop commands are host-scoped (no job_id).
            conn.execute(
                "DELETE FROM agent_commands WHERE host_id=%s "
                "AND created_by='reconciler'",
                (hid,),
            )
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mk_host(fleet, host_id, *, gpu_model):
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload) "
            "VALUES (%s, 'active', %s, %s)",
            (host_id, time.time(), json.dumps({
                "host_id": host_id, "gpu_model": gpu_model, "gpu_count": 2,
                "free_vram_gb": 48.0, "total_vram_gb": 48.0,
                "cost_per_hour": 1.0, "admitted": True,
                "last_seen": time.time(),
            })),
        )
        conn.commit()
    fleet["hosts"].append(host_id)


def _place(fleet, marker):
    """Place a job through the real pipeline; force attempt to running."""
    model = f"RTX-{marker}"
    host_id = f"h-{marker}-1"
    job_id = f"j-{marker}-run"
    _mk_host(fleet, host_id, gpu_model=model)
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, payload) "
            "VALUES (%s, 'queued', 0, %s, %s)",
            (job_id, time.time(), json.dumps({
                "job_id": job_id, "name": job_id, "gpu_model": model,
                "num_gpus": 1, "vram_needed_gb": 8.0, "status": "queued",
            })),
        )
        conn.commit()
    fleet["jobs"].append(job_id)
    cfg = SchedulerConfig(
        mode=SchedulerMode.CANARY, replica_id=f"rec-{marker}",
        canary_gpu_models=frozenset({model.lower()}),
        canary_host_ids=frozenset({host_id}),
        # Long offer TTL: full-suite runs exceed the 60s default and a
        # rival test's lease-expiry sweep would cancel our start command.
        lease_claim_ttl_sec=600,
    )
    report = SchedulerService(cfg).tick()
    assert len(report.placements) == 1
    resv = report.placements[0]
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET status='running', "
            "lease_claimed_at=clock_timestamp(), started_at=clock_timestamp() "
            "WHERE attempt_id=%s",
            (resv.attempt_id,),
        )
        conn.commit()
    return job_id, host_id, resv


def _ingest(host_id, session, gen, workloads):
    return run_transaction(
        lambda c: ingest_observation(
            c, host_id=host_id, session_id=session,
            observation_generation=gen, workloads=workloads,
            agent_version="test", worker_reported_at=time.time(),
        ),
        what="test_ingest",
    )


def _open_findings(resource_id):
    with _pool.connection() as conn:
        rows = conn.execute(
            "SELECT finding_type, severity FROM reconciliation_findings "
            "WHERE resource_id=%s AND resolved_at IS NULL",
            (resource_id,),
        ).fetchall()
    return [
        (r[0], r[1]) if not isinstance(r, dict)
        else (r["finding_type"], r["severity"])
        for r in rows
    ]


class TestIngest:
    def test_ingest_and_duplicate(self, fleet):
        marker = uuid.uuid4().hex[:8]
        host_id = f"h-{marker}-obs"
        _mk_host(fleet, host_id, gpu_model=f"m-{marker}")
        r1 = _ingest(host_id, "sess-1", 100, [
            {"job_id": "j-x", "container_name": "xcl-j-x", "state": "running"},
        ])
        assert not r1.duplicate and r1.workloads == 1
        r2 = _ingest(host_id, "sess-1", 100, [])
        assert r2.duplicate

        latest = run_transaction(
            lambda c: latest_observation(c, host_id), what="t"
        )
        assert latest is not None
        assert latest["observation_generation"] == 100
        assert latest["workloads"][0]["job_id"] == "j-x"

        with _pool.connection() as conn:
            q = conn.execute(
                "SELECT reason FROM reconciliation_queue "
                "WHERE resource_type='host' AND resource_id=%s",
                (host_id,),
            ).fetchone()
            seen = conn.execute(
                "SELECT last_observed_at FROM hosts WHERE host_id=%s", (host_id,)
            ).fetchone()
        assert q is not None
        assert (seen[0] if not isinstance(seen, dict) else seen["last_observed_at"]) is not None

    def test_route_ingests(self, fleet):
        marker = uuid.uuid4().hex[:8]
        host_id = f"h-{marker}-rt"
        _mk_host(fleet, host_id, gpu_model=f"m-{marker}")
        r = client.post(
            "/agent/v2/observations",
            json={
                "host_id": host_id, "worker_session_id": "sess-rt",
                "observation_generation": 1,
                "workloads": [{"job_id": "j-y", "container_name": "xcl-j-y",
                               "state": "running"}],
                "agent_version": "2.1.0",
                "worker_reported_at": time.time(),
            },
        )
        assert r.status_code == 200 and r.json()["duplicate"] is False
        assert r.json()["workloads"] == 1


class TestReconcileFindings:
    def test_missing_container_opens_then_resolves(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, resv = _place(fleet, marker)

        # Observation without the attempt's container → finding.
        _ingest(host_id, "sess-a", 1, [])
        report = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=0),
            what="t",
        )
        assert "attempt_container_missing" in report.findings_opened
        assert ("attempt_container_missing", "warning") in _open_findings(
            resv.attempt_id
        )

        # Re-running does not duplicate the open finding.
        again = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=0),
            what="t",
        )
        assert again.findings_opened == []

        # A healthy observation resolves it.
        _ingest(host_id, "sess-a", 2, [{
            "job_id": job_id, "attempt_id": resv.attempt_id,
            "fencing_token": resv.fencing_token,
            "container_name": f"xcl-{job_id}", "state": "running",
        }])
        healthy = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=0),
            what="t",
        )
        assert healthy.findings_opened == []
        assert healthy.findings_resolved >= 1
        assert _open_findings(resv.attempt_id) == []

    def test_stale_fence_container_flagged(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, resv = _place(fleet, marker)
        _ingest(host_id, "sess-b", 1, [{
            "job_id": job_id, "attempt_id": str(uuid.uuid4()),
            "fencing_token": resv.fencing_token + 999,
            "container_name": f"xcl-{job_id}", "state": "running",
        }])
        report = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=3600),
            what="t",
        )
        assert "stale_fence_container" in report.findings_opened
        assert ("stale_fence_container", "error") in _open_findings(host_id)

    def test_unmanaged_workload_flagged(self, fleet):
        marker = uuid.uuid4().hex[:8]
        host_id = f"h-{marker}-um"
        _mk_host(fleet, host_id, gpu_model=f"m-{marker}")
        _ingest(host_id, "sess-c", 1, [{
            "job_id": f"ghost-{marker}", "container_name": f"xcl-ghost-{marker}",
            "state": "running",
        }])
        report = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=0), what="t"
        )
        assert "unmanaged_workload" in report.findings_opened
        assert ("unmanaged_workload", "info") in _open_findings(host_id)

    def test_young_attempt_gets_grace(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, resv = _place(fleet, marker)
        _ingest(host_id, "sess-d", 1, [])
        report = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=3600),
            what="t",
        )
        assert report.findings_opened == []  # inside the grace window


def _stop_commands(host_id):
    with _pool.connection() as conn:
        rows = conn.execute(
            "SELECT args, status FROM agent_commands "
            "WHERE host_id=%s AND command='stop_container' "
            "AND created_by='reconciler'",
            (host_id,),
        ).fetchall()
    return [
        (r[0], r[1]) if not isinstance(r, dict) else (r["args"], r["status"])
        for r in rows
    ]


def _activate_lease(attempt_id):
    """Move the reservation's offered lease to active (as a worker would)."""
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE placement_leases "
            "   SET status='active', claimed_at=clock_timestamp(), "
            "       expires_at=clock_timestamp() + interval '5 minutes' "
            " WHERE attempt_id=%s AND status='offered'",
            (attempt_id,),
        )
        conn.commit()


def _scalar(row):
    if row is None:
        return None
    return row[0] if not isinstance(row, dict) else next(iter(row.values()))


class TestReconcileActions:
    def _stale_fence_obs(self, host_id, job_id, resv, session):
        _ingest(host_id, session, 1, [{
            "job_id": job_id, "attempt_id": str(uuid.uuid4()),
            "fencing_token": resv.fencing_token + 999,
            "container_name": f"xcl-{job_id}", "state": "running",
        }])

    def test_policy_resolution(self, monkeypatch):
        from control_plane.reconcile import ActionPolicy, action_policy_for

        # Non-enforceable types are always report-only, even if env says otherwise.
        monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_UNMANAGED_WORKLOAD", "enforce")
        assert action_policy_for("unmanaged_workload") is ActionPolicy.REPORT_ONLY
        # Enforceable type defaults to report-only.
        monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_STALE_FENCE_CONTAINER",
                           raising=False)
        assert action_policy_for("stale_fence_container") is ActionPolicy.REPORT_ONLY
        # Opt-in.
        monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_STALE_FENCE_CONTAINER", "enforce")
        assert action_policy_for("stale_fence_container") is ActionPolicy.ENFORCE
        # Malformed fails safe.
        monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_STALE_FENCE_CONTAINER", "yes-please")
        assert action_policy_for("stale_fence_container") is ActionPolicy.REPORT_ONLY

    def test_report_only_default_takes_no_action(self, fleet, monkeypatch):
        monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_STALE_FENCE_CONTAINER",
                           raising=False)
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, resv = _place(fleet, marker)
        self._stale_fence_obs(host_id, job_id, resv, "sess-ro")
        report = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=3600), what="t"
        )
        assert "stale_fence_container" in report.findings_opened
        assert report.actions_taken == []
        assert _stop_commands(host_id) == []

    def test_enforce_enqueues_stop_command_once(self, fleet, monkeypatch):
        monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_STALE_FENCE_CONTAINER", "enforce")
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, resv = _place(fleet, marker)
        self._stale_fence_obs(host_id, job_id, resv, "sess-en")

        report = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=3600), what="t"
        )
        assert report.actions_taken == ["stale_fence_container"]
        cmds = _stop_commands(host_id)
        assert len(cmds) == 1
        args, status = cmds[0]
        assert args["container_name"] == f"xcl-{job_id}"
        assert status == "pending"

        # Finding records the action.
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT action_taken, action_result FROM reconciliation_findings "
                "WHERE resource_id=%s AND finding_type='stale_fence_container' "
                "AND resolved_at IS NULL",
                (host_id,),
            ).fetchone()
        action_taken = row[0] if not isinstance(row, dict) else row["action_taken"]
        result = row[1] if not isinstance(row, dict) else row["action_result"]
        assert action_taken == "stop_container"
        assert result["enqueued"] is True

        # Re-running the same still-open finding does NOT re-enqueue.
        run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=3600), what="t"
        )
        assert len(_stop_commands(host_id)) == 1

    def test_stop_command_visible_to_v1_drain(self, fleet, monkeypatch):
        """The reconciler stop is attempt_id NULL, so the v1 drain (not the
        v2 fenced path) delivers it — a stale fence has no valid authority."""
        monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_STALE_FENCE_CONTAINER", "enforce")
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, resv = _place(fleet, marker)
        self._stale_fence_obs(host_id, job_id, resv, "sess-drain")
        run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=3600), what="t"
        )
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT attempt_id FROM agent_commands "
                "WHERE host_id=%s AND command='stop_container' "
                "AND created_by='reconciler'",
                (host_id,),
            ).fetchone()
        attempt_id = row[0] if not isinstance(row, dict) else row["attempt_id"]
        assert attempt_id is None  # v1-drain visible, not v2-claimable

    def test_missing_container_report_only_default(self, fleet, monkeypatch):
        monkeypatch.delenv(
            "XCELSIOR_RECONCILE_ACTION_ATTEMPT_CONTAINER_MISSING", raising=False
        )
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, resv = _place(fleet, marker)
        _activate_lease(resv.attempt_id)
        _ingest(host_id, "sess-mc-ro", 1, [])  # observation with no container
        report = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=0), what="t"
        )
        assert "attempt_container_missing" in report.findings_opened
        assert report.actions_taken == []
        # Lease untouched, still active with a future expiry.
        with _pool.connection() as conn:
            future = _scalar(conn.execute(
                "SELECT expires_at > clock_timestamp() FROM placement_leases "
                "WHERE attempt_id=%s AND status='active'", (resv.attempt_id,)
            ).fetchone())
        assert future is True

    def test_missing_container_enforce_expedites_lease_and_settles(
        self, fleet, monkeypatch
    ):
        monkeypatch.setenv(
            "XCELSIOR_RECONCILE_ACTION_ATTEMPT_CONTAINER_MISSING", "enforce"
        )
        marker = uuid.uuid4().hex[:8]
        job_id, host_id, resv = _place(fleet, marker)
        _activate_lease(resv.attempt_id)
        _ingest(host_id, "sess-mc-en", 1, [])
        report = run_transaction(
            lambda c: reconcile_host(c, host_id, missing_grace_sec=0), what="t"
        )
        assert report.actions_taken == ["attempt_container_missing"]

        # The action recorded on the finding + lease now expired-in-the-past.
        with _pool.connection() as conn:
            action = _scalar(conn.execute(
                "SELECT action_taken FROM reconciliation_findings "
                "WHERE resource_type='attempt' AND resource_id=%s "
                "AND resolved_at IS NULL", (resv.attempt_id,)
            ).fetchone())
            past = _scalar(conn.execute(
                "SELECT expires_at < clock_timestamp() FROM placement_leases "
                "WHERE attempt_id=%s AND status='active'", (resv.attempt_id,)
            ).fetchone())
        assert action == "expire_lease"
        assert past is True

        # The lease controller (not the reconciler) then does the terminal
        # settlement uniformly: attempt lost, job requeued, allocations freed.
        from control_plane.leases import expire_stale_leases

        run_transaction(lambda c: expire_stale_leases(c, grace_sec=0), what="t")
        with _pool.connection() as conn:
            att_status = _scalar(conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id=%s",
                (resv.attempt_id,)
            ).fetchone())
            job_status = _scalar(conn.execute(
                "SELECT status FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone())
            active_allocs = _scalar(conn.execute(
                "SELECT count(*) FROM gpu_device_allocations "
                "WHERE attempt_id=%s AND status='active'", (resv.attempt_id,)
            ).fetchone())
        assert att_status == "lost"
        assert job_status == "queued"
        assert active_allocs == 0


class TestQueueProcessing:
    def test_process_due_settles_entry(self, fleet):
        marker = uuid.uuid4().hex[:8]
        host_id = f"h-{marker}-q"
        _mk_host(fleet, host_id, gpu_model=f"m-{marker}")
        _ingest(host_id, "sess-q", 1, [])
        stats = run_transaction(
            lambda c: process_due(c, worker_id="test-rec", missing_grace_sec=0),
            what="t",
        )
        assert stats["reconciled"] >= 1 and stats["failed"] == 0
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM reconciliation_queue "
                "WHERE resource_type='host' AND resource_id=%s",
                (host_id,),
            ).fetchone()
        assert row is None  # settled


class TestRetention:
    def test_prune_old_observations(self, fleet):
        marker = uuid.uuid4().hex[:8]
        host_id = f"h-{marker}-pr"
        _mk_host(fleet, host_id, gpu_model=f"m-{marker}")
        result = _ingest(host_id, "sess-p", 1, [])
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE host_observations SET received_at = "
                "clock_timestamp() - interval '10 days' WHERE observation_id=%s",
                (result.observation_id,),
            )
            conn.commit()
        deleted = run_transaction(
            lambda c: prune_observations(c, retention_days=3), what="t"
        )
        assert deleted >= 1
        latest = run_transaction(lambda c: latest_observation(c, host_id), what="t")
        assert latest is None


class TestWorkerCollection:
    def test_collect_local_workloads_parses_docker_ps(self):
        import os as _os
        import sys as _sys
        from pathlib import Path as _Path
        from types import SimpleNamespace
        from unittest import mock

        _os.environ.setdefault("XCELSIOR_HOST_ID", "test-host-obs")
        _os.environ.setdefault("XCELSIOR_API_URL", "http://localhost:9500")
        root = str(_Path(__file__).resolve().parents[1])
        if root not in _sys.path:
            _sys.path.insert(0, root)
        import worker_agent

        stdout = (
            "abc123\txcl-job-1\trunning\tjob-1\tatt-1\t7\n"
            "def456\txcl-job-2\texited\t\t\t\n"
        )
        fake = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        with mock.patch.object(
            worker_agent.subprocess, "run", return_value=fake
        ):
            got = worker_agent._collect_local_workloads()
        assert len(got) == 2
        assert got[0] == {
            "job_id": "job-1", "attempt_id": "att-1", "fencing_token": 7,
            "container_id": "abc123", "container_name": "xcl-job-1",
            "spec_hash": None,
            "state": "running",
        }
        assert got[1]["job_id"] == "job-2"  # from container name
        assert got[1]["attempt_id"] is None
        assert got[1]["state"] == "exited"
