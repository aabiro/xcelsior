"""Real placed attempts and ingested observations exercise job authority checks."""

from __future__ import annotations

import time
import uuid

import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from control_plane.attempts import report_attempt_status
from control_plane.job_reconcile import reconcile_job
from control_plane.leases import claim_lease
from control_plane.observations import ingest_observation
from control_plane.state import job_phase
from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.service import SchedulerService
from db import _get_pg_pool

pytestmark = pytest.mark.needs_db


@pytest.fixture
def placed(request):
    pool = _get_pg_pool()
    inputs = dict(getattr(request, "param", {}))
    if "native_admission_fields" in request.fixturenames:
        inputs.update(request.getfixturevalue("native_admission_fields"))
    billable = inputs.pop("billable", False)
    marker = uuid.uuid4().hex
    host_id, job_id, model = f"h-job-rec-{marker}", f"j-job-rec-{marker}", f"GPU-{marker}"
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload, admission_state, administrative_state, availability_state, capabilities) VALUES (%s, 'active', %s, %s, 'admitted', 'admitted', 'ready', '{"egress_isolation":true}')""",
            (
                host_id,
                time.time(),
                Jsonb(
                    {
                        "host_id": host_id,
                        "gpu_model": model,
                        "gpu_count": inputs.get("num_gpus", 1),
                        "free_vram_gb": 24,
                        "total_vram_gb": 24,
                        "cost_per_hour": inputs.get("cost_per_hour", 1),
                        "admitted": True,
                        "last_seen": time.time(),
                        "capabilities": {"egress_isolation": True},
                    }
                ),
            ),
        )
        fixture_submitted_at = time.time()
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload, phase, desired_state, effective_priority, queued_at) VALUES (%s, 'queued', 0, %s, %s, 'pending', 'running', 0, to_timestamp(%s))""",
            (
                job_id,
                fixture_submitted_at,
                Jsonb(
                    {
                        "job_id": job_id,
                        "name": job_id,
                        **({"owner": job_id} if billable else {}),
                        "status": "queued",
                        "gpu_model": model,
                        "num_gpus": 1,
                        "vram_needed_gb": 8,
                        **{k: v for k, v in inputs.items() if k != "cost_per_hour"},
                    }
                ),
                fixture_submitted_at,
            ),
        )
        if billable:
            conn.execute("UPDATE jobs SET owner_id=%s WHERE job_id=%s", (job_id, job_id))
        from tests.native_jobs import admit_fixture_job

        admit_fixture_job(conn, job_id)
    try:
        report = SchedulerService(
            SchedulerConfig(
                mode=SchedulerMode.CANARY,
                replica_id=f"job-rec-{marker}",
                canary_gpu_models=frozenset({model.lower()}),
                canary_host_ids=frozenset({host_id}),
                lease_claim_ttl_sec=600,
                lease_renewal_ttl_sec=600,
            )
        ).tick()
        assert len(report.placements) == 1
        reservation = report.placements[0]
        with pool.connection() as conn:
            claim_lease(
                conn,
                lease_id=reservation.lease_id,
                job_id=job_id,
                attempt_id=reservation.attempt_id,
                host_id=host_id,
                fencing_token=reservation.fencing_token,
                worker_session_id="test-session",
            )
            report_attempt_status(
                conn,
                job_id=job_id,
                attempt_id=reservation.attempt_id,
                host_id=host_id,
                fencing_token=reservation.fencing_token,
                status="running",
            )
            spec_hash = conn.execute(
                "SELECT spec_hash FROM job_attempts WHERE attempt_id = %s",
                (reservation.attempt_id,),
            ).fetchone()[0]
        yield {
            "pool": pool,
            "job_id": job_id,
            "host_id": host_id,
            "attempt_id": reservation.attempt_id,
            "fencing_token": reservation.fencing_token,
            "spec_hash": spec_hash,
        }
    finally:
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM reconciliation_findings WHERE resource_id = ANY(%s) "
                "OR resource_id IN (SELECT attempt_id::text FROM job_attempts WHERE job_id = %s)",
                ([job_id, host_id], job_id),
            )
            conn.execute(
                "DELETE FROM reconciliation_queue WHERE resource_id = ANY(%s)", ([job_id, host_id],)
            )
            conn.execute("DELETE FROM usage_meters WHERE job_id = %s", (job_id,))
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id = %s", (job_id,))
            conn.execute("DELETE FROM agent_commands WHERE job_id = %s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
            conn.execute("DELETE FROM hosts WHERE host_id = %s", (host_id,))


def _observe(placed, *, state="running", workloads=None, **changes):
    workload = {key: placed[key] for key in ("job_id", "attempt_id", "fencing_token", "spec_hash")}
    workload.update(state=state, container_name=f"xcl-{placed['job_id']}", **changes)
    with placed["pool"].connection() as conn:
        return ingest_observation(
            conn,
            host_id=placed["host_id"],
            session_id=uuid.uuid4().hex,
            observation_generation=1,
            workloads=[workload] if workloads is None else workloads,
            # Worker clock must not override API receipt freshness.
            worker_reported_at=time.time() + 86400,
        )


def _reconcile(placed, **kwargs):
    with placed["pool"].connection() as conn:
        return reconcile_job(conn, placed["job_id"], missing_grace_sec=0, **kwargs)


def _desire_stop(placed, *, status="stopping", intent=None):
    with placed["pool"].connection() as conn:
        conn.execute(
            "UPDATE jobs SET status=%s,phase=%s,desired_state='stopped',lifecycle_intent=%s WHERE job_id=%s",
            (status, job_phase(status), intent, placed["job_id"]),
        )


def _stop_commands(placed):
    with placed["pool"].connection() as conn:
        return conn.execute(
            "SELECT command_id, args, attempt_id::text, fencing_token, status FROM agent_commands "
            "WHERE job_id = %s AND command = 'stop_attempt'",
            (placed["job_id"],),
        ).fetchall()


@pytest.mark.parametrize("mode", [None, "report_only", "typo"])
def test_stop_repair_defaults_to_reporting(placed, monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_JOB_STOP_COMMAND_MISSING", raising=False)
    else:
        monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_JOB_STOP_COMMAND_MISSING", mode)
    _desire_stop(placed)
    report = _reconcile(placed)
    assert "job_stop_command_missing" in report.findings_opened
    assert report.actions_taken == [] and _stop_commands(placed) == []


@pytest.mark.parametrize(
    "status,intent,preserve",
    [
        ("stopping", "stop", True),
        ("terminated", "terminate", False),
        ("cancelled", "cancel", False),
        ("stopping", "restart", False),
    ],
)
def test_stop_repair_after_report_only_preserves_intent_and_is_idempotent(
    placed,
    monkeypatch,
    status,
    intent,
    preserve,
):
    monkeypatch.delenv("XCELSIOR_RECONCILE_ACTION_JOB_STOP_COMMAND_MISSING", raising=False)
    _desire_stop(placed, status=status, intent=intent)
    assert "job_stop_command_missing" in _reconcile(placed).findings_opened
    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_JOB_STOP_COMMAND_MISSING", "enforce")
    report = _reconcile(placed)
    assert report.actions_taken == ["enqueue_fenced_stop"]
    assert "job_stop_command_missing" not in report.findings_opened
    assert _reconcile(placed).actions_taken == []
    commands = _stop_commands(placed)
    assert len(commands) == 1
    assert commands[0][1]["intent"] == intent and commands[0][1]["preserve"] is preserve
    assert commands[0][2:4] == (placed["attempt_id"], placed["fencing_token"])
    with placed["pool"].connection() as conn:
        assert conn.execute(
            "SELECT status FROM jobs WHERE job_id = %s", (placed["job_id"],)
        ).fetchone() == (status,)
        assert conn.execute(
            "SELECT action_taken, resolved_at IS NOT NULL FROM reconciliation_findings "
            "WHERE resource_id = %s AND finding_type = 'job_stop_command_missing'",
            (placed["job_id"],),
        ).fetchone() == ("enqueue_fenced_stop", True)


@pytest.mark.parametrize(
    "defect", ["expired_lease", "wrong_fence", "invalid_intent", "dead_letter", "expired_command"]
)
def test_stop_repair_refuses_invalid_authority_or_exhausted_command(placed, monkeypatch, defect):
    from control_plane.lifecycle import request_fenced_stop

    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_JOB_STOP_COMMAND_MISSING", "enforce")
    if defect in {"dead_letter", "expired_command"}:
        result = request_fenced_stop(job_id=placed["job_id"], created_by="test")
        assert result.ok
    if defect == "invalid_intent":
        from psycopg.errors import CheckViolation

        with pytest.raises(CheckViolation, match="ck_jobs_lifecycle_intent"):
            _desire_stop(placed, intent="resume")
        assert _stop_commands(placed) == []
        return
    _desire_stop(placed, intent="stop")
    with placed["pool"].connection() as conn:
        if defect == "expired_lease":
            conn.execute(
                "UPDATE placement_leases SET expires_at = clock_timestamp() - interval '1 hour' WHERE attempt_id = %s",
                (placed["attempt_id"],),
            )
        elif defect == "wrong_fence":
            conn.execute(
                "UPDATE placement_leases SET fencing_token = fencing_token + 1 WHERE attempt_id = %s",
                (placed["attempt_id"],),
            )
        elif defect == "dead_letter":
            conn.execute(
                "UPDATE agent_commands SET status = 'dead_letter' WHERE command_id = %s",
                (result.command_id,),
            )
        elif defect == "expired_command":
            conn.execute(
                "UPDATE agent_commands SET expires_at = 1 WHERE command_id = %s",
                (result.command_id,),
            )
    before = _stop_commands(placed)
    assert _reconcile(placed).actions_taken == []
    assert _stop_commands(placed) == before
    with placed["pool"].connection() as conn:
        row = conn.execute(
            "SELECT action_taken, action_result FROM reconciliation_findings "
            "WHERE resource_id = %s AND finding_type = 'job_stop_command_missing'",
            (placed["job_id"],),
        ).fetchone()
        assert row[0] == "fenced_stop_refused" and row[1]["ok"] is False


def test_old_stop_command_cannot_satisfy_new_termination_intent(placed, monkeypatch):
    from control_plane.lifecycle import request_fenced_stop

    assert request_fenced_stop(job_id=placed["job_id"], created_by="test").ok
    _desire_stop(placed, status="terminated", intent="terminate")
    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_JOB_STOP_COMMAND_MISSING", "enforce")
    assert _reconcile(placed).actions_taken == ["enqueue_fenced_stop"]
    assert sorted(c[1]["intent"] for c in _stop_commands(placed)) == ["stop", "terminate"]


@pytest.mark.parametrize(
    "status,intent,expected",
    [
        ("stopping", "stop", "stopped"),
        ("stopping", "terminate", "terminated"),
        ("stopping", "cancel", "cancelled"),
        ("stopping", "restart", "queued"),
        ("terminated", None, "terminated"),
        ("cancelled", None, "cancelled"),
        ("completed", None, "completed"),
        ("failed", None, "failed"),
    ],
)
def test_lease_expiry_cannot_undo_stop_or_terminal_state(placed, status, intent, expected):
    from control_plane.leases import expire_stale_leases

    _desire_stop(placed, status=status, intent=intent)
    with placed["pool"].connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET expires_at = clock_timestamp() - interval '1 hour' "
            "WHERE attempt_id = %s",
            (placed["attempt_id"],),
        )
        expired = expire_stale_leases(conn, grace_sec=0)
        assert any(e["attempt_id"] == placed["attempt_id"] for e in expired)
        assert conn.execute(
            "SELECT status, active_attempt_id, payload->>'status', lifecycle_intent "
            "FROM jobs WHERE job_id = %s",
            (placed["job_id"],),
        ).fetchone() == (expected, None, expected, None)
        assert conn.execute(
            "SELECT status FROM job_attempts WHERE attempt_id = %s", (placed["attempt_id"],)
        ).fetchone() == ("lost",)
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id = %s AND status = 'active'",
            (placed["attempt_id"],),
        ).fetchone() == (0,)
        assert not any(
            e["attempt_id"] == placed["attempt_id"] for e in expire_stale_leases(conn, grace_sec=0)
        )


def test_repair_and_evidence_rollback_together_on_queue_failure(placed, monkeypatch):
    from control_plane import job_reconcile, reconcile

    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_JOB_STOP_COMMAND_MISSING", "enforce")
    _desire_stop(placed)

    def fail_record(conn, *args, **kwargs):
        conn.execute("SELECT 1 / 0")

    monkeypatch.setattr(job_reconcile, "_record_action", fail_record)
    with placed["pool"].connection() as conn:
        conn.execute(
            "UPDATE reconciliation_queue SET priority = 100 WHERE resource_type = 'job' AND resource_id = %s",
            (placed["job_id"],),
        )
        result = reconcile.process_due(conn, worker_id="test", limit=1)
        assert result["failed"] == 1
        assert conn.execute(
            "SELECT lifecycle_intent FROM jobs WHERE job_id = %s", (placed["job_id"],)
        ).fetchone() == (None,)
        assert (
            conn.execute(
                "SELECT 1 FROM reconciliation_findings WHERE resource_id = %s", (placed["job_id"],)
            ).fetchone()
            is None
        )
    assert _stop_commands(placed) == []


def test_fresh_full_authority_updates_observed_generation_and_reports_missing_meter(placed):
    _observe(placed)
    report = _reconcile(placed)
    assert report.findings_opened == ["job_active_meter_missing"]
    assert report.actions_taken == [] and report.observation_id is not None
    with placed["pool"].connection() as conn:
        assert conn.execute(
            "SELECT status, observed_generation = generation FROM jobs WHERE job_id = %s",
            (placed["job_id"],),
        ).fetchone() == ("running", True)
    assert _reconcile(placed).findings_opened == []  # durable dedupe
    with placed["pool"].connection() as conn:
        conn.execute(
            "INSERT INTO usage_meters (meter_id, job_id, attempt_id, owner, started_at, completed_at) "
            "VALUES (%s, %s, %s, 'test-owner', %s, NULL)",
            (f"meter-{uuid.uuid4().hex}", placed["job_id"], placed["attempt_id"], time.time()),
        )
        conn.execute(
            "INSERT INTO reconciliation_findings "
            "(resource_type, resource_id, finding_type, severity, summary) "
            "VALUES ('job', %s, 'other_controller_finding', 'warning', 'not ours')",
            (placed["job_id"],),
        )
    repaired = _reconcile(placed)
    assert repaired.findings_opened == [] and repaired.findings_resolved == 1
    with placed["pool"].connection() as conn:
        assert conn.execute(
            "SELECT finding_type FROM reconciliation_findings "
            "WHERE resource_id = %s AND resolved_at IS NULL",
            (placed["job_id"],),
        ).fetchall() == [("other_controller_finding",)]


@pytest.mark.parametrize(
    "changes",
    [
        {"job_id": "different-job"},
        {"attempt_id": str(uuid.uuid4())},
        {"fencing_token": -1},
        {"spec_hash": "different-spec"},
        {"spec_hash": None},
    ],
)
def test_partial_or_wrong_authority_cannot_advance_observed_generation(placed, changes):
    _observe(placed, **changes)
    assert _reconcile(placed).findings_opened == ["job_observation_authority_mismatch"]
    with placed["pool"].connection() as conn:
        assert (
            conn.execute(
                "SELECT observed_generation FROM jobs WHERE job_id = %s", (placed["job_id"],)
            ).fetchone()[0]
            == 0
        )


def test_stale_api_receipt_cannot_be_freshened_by_worker_clock(placed):
    observed = _observe(placed)
    with placed["pool"].connection() as conn:
        conn.execute(
            "UPDATE host_observations SET received_at = clock_timestamp() - interval '1 hour' "
            "WHERE observation_id = %s",
            (observed.observation_id,),
        )
    assert _reconcile(placed).findings_opened == ["job_observation_stale"]


@pytest.mark.parametrize("state", ["paused", "unknown", "preparing"])
def test_non_running_observation_cannot_prove_running(placed, state):
    _observe(placed, state=state)
    assert _reconcile(placed).findings_opened == ["job_attempt_status_mismatch"]
    with placed["pool"].connection() as conn:
        assert (
            conn.execute(
                "SELECT observed_generation FROM jobs WHERE job_id = %s", (placed["job_id"],)
            ).fetchone()[0]
            == 0
        )


def test_missing_immutable_spec_on_both_sides_is_not_a_match(placed):
    with placed["pool"].connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET spec_hash = NULL WHERE attempt_id = %s",
            (placed["attempt_id"],),
        )
    _observe(placed, spec_hash=None)
    assert _reconcile(placed).findings_opened == ["job_observation_authority_mismatch"]


def test_missing_observation_is_reported_and_resolves_when_a_fresh_match_arrives(placed):
    assert _reconcile(placed).findings_opened == ["job_observation_missing"]
    _observe(placed)
    report = _reconcile(placed)
    assert report.findings_resolved == 1
    assert report.findings_opened == ["job_active_meter_missing"]


def test_exited_container_does_not_invent_success_or_release_authority(placed):
    _observe(placed, state="exited", details={"exit_code": 0})
    assert _reconcile(placed).findings_opened == ["job_terminal_observation"]
    with placed["pool"].connection() as conn:
        assert conn.execute(
            "SELECT status, active_attempt_id::text FROM jobs WHERE job_id = %s",
            (placed["job_id"],),
        ).fetchone() == ("running", placed["attempt_id"])
        assert conn.execute(
            "SELECT status FROM placement_leases WHERE attempt_id = %s", (placed["attempt_id"],)
        ).fetchone() == ("active",)


def test_desired_stopped_requires_an_actual_fenced_stop_command(placed):
    with placed["pool"].connection() as conn:
        conn.execute(
            """UPDATE jobs SET status = 'stopping', phase='running', desired_state='stopped' WHERE job_id = %s""",
            (placed["job_id"],),
        )
    _observe(placed)
    report = _reconcile(placed)
    assert "job_stop_command_missing" in report.findings_opened
    assert report.actions_taken == []
    from control_plane.lifecycle import request_fenced_stop

    assert request_fenced_stop(job_id=placed["job_id"], created_by="test").ok
    after_delivery = _reconcile(placed)
    assert "job_stop_command_missing" not in after_delivery.findings_opened
    assert "job_stop_not_observed" in after_delivery.findings_opened
    assert after_delivery.findings_resolved == 1  # delivery alone is not settlement


def test_expired_lease_is_reported_without_reviving_it(placed):
    with placed["pool"].connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET expires_at = clock_timestamp() - interval '1 hour' "
            "WHERE attempt_id = %s",
            (placed["attempt_id"],),
        )
    _observe(placed)
    assert "job_lease_authority_missing" in _reconcile(placed).findings_opened


def test_duplicate_current_authority_is_reported(placed):
    workload = {key: placed[key] for key in ("job_id", "attempt_id", "fencing_token", "spec_hash")}
    workload["state"] = "running"
    _observe(placed, workloads=[workload, workload])
    assert _reconcile(placed).findings_opened == ["job_observation_authority_mismatch"]


def test_dictionary_rows_are_supported_without_leaking_row_factory(placed):
    _observe(placed)
    with placed["pool"].connection() as conn:
        original = conn.row_factory
        try:
            conn.row_factory = dict_row
            report = reconcile_job(conn, placed["job_id"], missing_grace_sec=0)
            assert report.findings_opened == ["job_active_meter_missing"]
        finally:
            conn.row_factory = original
