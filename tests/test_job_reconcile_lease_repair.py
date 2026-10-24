"""Per-job lease repair delegates to the existing fenced expiry authority."""

import pytest
from psycopg.errors import LockNotAvailable

from control_plane.job_reconcile import reconcile_job
from tests.test_job_reconcile_observations import placed, _desire_stop  # noqa: F401

pytestmark = pytest.mark.needs_db
FLAG = "XCELSIOR_RECONCILE_ACTION_JOB_LEASE_AUTHORITY_MISSING"


def _expire(placed, *, seconds=60):
    with placed["pool"].connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET expires_at=clock_timestamp()-make_interval(secs => %s) "
            "WHERE attempt_id=%s",
            (seconds, placed["attempt_id"]),
        )


def _state(conn, placed):
    return conn.execute(
        "SELECT j.status, j.active_attempt_id::text, a.status, l.status "
        "FROM jobs j JOIN job_attempts a ON a.job_id=j.job_id "
        "JOIN placement_leases l ON l.attempt_id=a.attempt_id "
        "WHERE a.attempt_id=%s",
        (placed["attempt_id"],),
    ).fetchone()


@pytest.mark.parametrize("policy", [None, "report_only", "typo"])
def test_lease_repair_defaults_to_report_only(placed, monkeypatch, policy):
    if policy is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, policy)
    _expire(placed)
    with placed["pool"].connection() as conn:
        result = reconcile_job(conn, placed["job_id"])
        assert "job_lease_authority_missing" in result.findings_opened
        assert result.actions_taken == []
        assert _state(conn, placed) == ("running", placed["attempt_id"], "running", "active")


@pytest.mark.parametrize(
    "intent,expected",
    [(None, "queued"), ("stop", "stopped"), ("restart", "queued"), ("cancel", "cancelled")],
)
def test_enforced_expiry_preserves_intent_and_is_atomic_and_idempotent(
    placed, monkeypatch, intent, expected
):
    if intent:
        _desire_stop(
            placed, status="cancelled" if intent == "cancel" else "stopping", intent=intent
        )
    _expire(placed)
    # Existing report-only evidence must become actionable after opt-in.
    with placed["pool"].connection() as conn:
        reconcile_job(conn, placed["job_id"])
    monkeypatch.setenv(FLAG, "enforce")
    with placed["pool"].connection() as conn:
        result = reconcile_job(conn, placed["job_id"])
        assert result.actions_taken == ["expire_current_lease"]
        assert _state(conn, placed) == (expected, None, "lost", "expired")
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND status='active'",
            (placed["attempt_id"],),
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT action_taken, resolved_at IS NOT NULL FROM reconciliation_findings "
            "WHERE resource_id=%s AND finding_type='job_lease_authority_missing'",
            (placed["job_id"],),
        ).fetchone() == ("expire_current_lease", True)
        assert reconcile_job(conn, placed["job_id"]).actions_taken == []
        assert conn.execute(
            "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s AND event_type='job.v1.lease_expired'",
            (placed["job_id"],),
        ).fetchone() == (1,)


@pytest.mark.parametrize("defect", ["grace", "wrong_fence", "wrong_host", "released", "missing"])
def test_lease_repair_refuses_unproven_or_still_graced_authority(placed, monkeypatch, defect):
    monkeypatch.setenv(FLAG, "enforce")
    _expire(placed, seconds=1 if defect == "grace" else 60)
    with placed["pool"].connection() as conn:
        if defect == "wrong_fence":
            conn.execute(
                "UPDATE placement_leases SET fencing_token=fencing_token+1 WHERE attempt_id=%s",
                (placed["attempt_id"],),
            )
        elif defect == "wrong_host":
            conn.execute(
                "UPDATE placement_leases SET host_id='unrelated' WHERE attempt_id=%s",
                (placed["attempt_id"],),
            )
        elif defect == "released":
            conn.execute(
                "UPDATE placement_leases SET status='released' WHERE attempt_id=%s",
                (placed["attempt_id"],),
            )
        elif defect == "missing":
            conn.execute(
                "DELETE FROM placement_leases WHERE attempt_id=%s", (placed["attempt_id"],)
            )
        result = reconcile_job(conn, placed["job_id"])
        assert "job_lease_authority_missing" in result.findings_opened
        assert result.actions_taken == []
        assert conn.execute(
            "SELECT active_attempt_id::text FROM jobs WHERE job_id=%s", (placed["job_id"],)
        ).fetchone() == (placed["attempt_id"],)


def test_lease_repair_and_evidence_roll_back_together(placed, monkeypatch):
    monkeypatch.setenv(FLAG, "enforce")
    _expire(placed)
    with placed["pool"].connection() as conn:
        with pytest.raises(RuntimeError, match="abort"):
            with conn.transaction():
                assert reconcile_job(conn, placed["job_id"]).actions_taken == [
                    "expire_current_lease"
                ]
                raise RuntimeError("abort")
        assert _state(conn, placed) == ("running", placed["attempt_id"], "running", "active")
        assert conn.execute(
            "SELECT count(*) FROM reconciliation_findings WHERE resource_id=%s", (placed["job_id"],)
        ).fetchone() == (0,)


def test_lease_repair_does_not_wait_on_command_writer(placed, monkeypatch):
    monkeypatch.setenv(FLAG, "enforce")
    _expire(placed)
    with placed["pool"].connection() as writer:
        writer.execute(
            "SELECT 1 FROM agent_commands WHERE attempt_id=%s FOR UPDATE", (placed["attempt_id"],)
        )
        with placed["pool"].connection() as conn:
            conn.execute("SET LOCAL statement_timeout='500ms'")
            with pytest.raises(LockNotAvailable):
                with conn.transaction():
                    reconcile_job(conn, placed["job_id"])
            assert _state(conn, placed) == ("running", placed["attempt_id"], "running", "active")


def test_unclaimed_expired_offer_uses_failed_attempt_semantics(placed, monkeypatch):
    monkeypatch.setenv(FLAG, "enforce")
    with placed["pool"].connection() as conn:
        conn.execute(
            """UPDATE jobs SET status='leased', phase='scheduled', desired_state='running' WHERE job_id=%s""",
            (placed["job_id"],),
        )
        conn.execute(
            "UPDATE job_attempts SET status='lease_offered' WHERE attempt_id=%s",
            (placed["attempt_id"],),
        )
        conn.execute(
            "UPDATE placement_leases SET status='offered', claim_deadline=clock_timestamp()-interval '1 second' WHERE attempt_id=%s",
            (placed["attempt_id"],),
        )
        assert reconcile_job(conn, placed["job_id"]).actions_taken == ["expire_current_lease"]
        assert _state(conn, placed) == ("queued", None, "failed", "expired")


def test_job_repair_cannot_settle_another_jobs_earlier_expired_lease(placed, monkeypatch):
    monkeypatch.setenv(FLAG, "enforce")
    _expire(placed)
    peer = placed["job_id"] + "-peer"
    with placed["pool"].connection() as conn:
        conn.execute(
            """INSERT INTO jobs(job_id,status,host_id,payload,submitted_at,priority, phase, desired_state, effective_priority, queued_at) VALUES (%s,'running',%s,'{}',0,0, 'running', 'running', 0, NULL)""",
            (peer, placed["host_id"]),
        )
        aid, fence = conn.execute(
            "INSERT INTO job_attempts(job_id,attempt_number,status,host_id,fencing_token) "
            "VALUES (%s,1,'running',%s,nextval('placement_fencing_token_seq')) RETURNING attempt_id,fencing_token",
            (peer, placed["host_id"]),
        ).fetchone()
        conn.execute("UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s", (aid, peer))
        conn.execute(
            "INSERT INTO placement_leases(lease_id,job_id,attempt_id,host_id,fencing_token,status,claim_deadline,claimed_at,expires_at) "
            "VALUES ('00000000-0000-0000-0000-000000000001',%s,%s,%s,%s,'active',clock_timestamp(),clock_timestamp()-interval '1 hour',clock_timestamp()-interval '1 minute')",
            (peer, aid, placed["host_id"], fence),
        )
        assert reconcile_job(conn, placed["job_id"]).actions_taken == ["expire_current_lease"]
        assert conn.execute(
            "SELECT status,active_attempt_id FROM jobs WHERE job_id=%s", (peer,)
        ).fetchone() == ("running", aid)
        assert conn.execute(
            "SELECT status FROM placement_leases WHERE attempt_id=%s", (aid,)
        ).fetchone() == ("active",)
        conn.rollback()  # The peer belongs only to this transaction.
