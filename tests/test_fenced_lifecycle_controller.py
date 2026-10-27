"""Fenced terminate/cancel lifecycle controller.

Drives the shipped entry points (BillingEngine.terminate_instance and
control_plane.lifecycle.request_fenced_stop_remove for cancel) against
real Postgres control-plane tables. Asserts durable agent_commands rows
and intermediate job projection — not a mocked "success" string.
"""

from __future__ import annotations

from control_plane.state import initial_desired_state, job_phase
from tests.native_jobs import admit_fixture_job

import json
import time
import uuid

import pytest
from fastapi import HTTPException, Request

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_cp = _c.execute("SELECT to_regclass('public.job_attempts')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
    _has_cp = False
else:
    if not _has_cp:  # pragma: no cover
        pytestmark = pytest.mark.skip("control-plane tables missing")


@pytest.fixture
def cleanup_ids():
    ids = {"jobs": [], "hosts": [], "agent_hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM placement_leases WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mkhost(cleanup, host_id):
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload, admission_state, administrative_state, availability_state)
               VALUES (%s, 'online', %s, %s, 'admitted', 'admitted', 'unknown')
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, time.time(), json.dumps({"name": host_id})),
        )
        conn.commit()
    cleanup["hosts"].append(host_id)
    cleanup["agent_hosts"].append(host_id)


def _mkjob(cleanup, *, host_id, status="running", container_name=None):
    job_id = f"job-life-{uuid.uuid4().hex[:10]}"
    cname = container_name or f"xcl-{job_id}"
    payload = {
        "owner": "lifecycle@test",
        "name": job_id,
        "container_name": cname,
    }
    with _pool.connection() as conn:
        fixture_submitted_at = time.time()
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload, phase, desired_state, effective_priority, queued_at)
               VALUES (%s, %s, 0, %s, %s, %s, %s, %s, 0, CASE WHEN %s IN ('queued','preempted') THEN to_timestamp(%s) ELSE NULL END)""",
            (
                job_id,
                status,
                fixture_submitted_at,
                host_id,
                json.dumps(payload),
                job_phase(status),
                initial_desired_state(status),
                status,
                fixture_submitted_at,
            ),
        )
        admit_fixture_job(conn, job_id)
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id, cname


def _mk_active_attempt(job_id, host_id):
    attempt_id = str(uuid.uuid4())
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token)
               VALUES (%s, %s, 1, 'running', %s,
                       nextval('placement_fencing_token_seq'))""",
            (attempt_id, job_id, host_id),
        )
        fence = conn.execute(
            "SELECT fencing_token FROM job_attempts WHERE attempt_id=%s",
            (attempt_id,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
            (attempt_id, job_id),
        )
        conn.execute(
            """INSERT INTO placement_leases
                   (job_id, attempt_id, host_id, fencing_token, status,
                    claim_deadline, claimed_at, last_renewed_at, expires_at)
               VALUES (%s, %s, %s, %s, 'active',
                       clock_timestamp() + interval '10 minutes',
                       clock_timestamp(), clock_timestamp(),
                       clock_timestamp() + interval '10 minutes')""",
            (job_id, attempt_id, host_id, fence),
        )
        conn.commit()
    return attempt_id, fence


def test_terminate_entry_point_enqueues_fenced_remove(cleanup_ids):
    host_id = f"h-life-t-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, cname = _mkjob(cleanup_ids, host_id=host_id)
    attempt_id, fence = _mk_active_attempt(job_id, host_id)

    from billing import BillingEngine

    result = BillingEngine().terminate_instance(job_id)
    assert result["terminated"] is True
    assert result["status"] == "stopping"
    assert result["attempt_id"] == attempt_id
    assert "fenced_terminate_requires_controller" not in str(result)

    with _pool.connection() as conn:
        job = conn.execute(
            """SELECT status, lifecycle_intent,
                      active_attempt_id::text
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
        cmd = conn.execute(
            """SELECT command, attempt_id::text, fencing_token, args, status
                 FROM agent_commands WHERE job_id=%s""",
            (job_id,),
        ).fetchone()

    assert job[0] == "stopping"
    assert job[1] == "terminate"
    assert job[2] == attempt_id
    assert cmd[0] == "stop_attempt"
    assert cmd[1] == attempt_id
    assert cmd[2] == fence
    assert cmd[3]["preserve"] is False
    assert cmd[3]["intent"] == "terminate"
    assert cmd[3]["container_name"] == cname
    assert cmd[4] == "pending"


def test_cancel_entry_point_enqueues_fenced_remove(cleanup_ids):
    host_id = f"h-life-c-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _cname = _mkjob(cleanup_ids, host_id=host_id)
    attempt_id, fence = _mk_active_attempt(job_id, host_id)

    from control_plane.lifecycle import request_fenced_stop_remove

    result = request_fenced_stop_remove(
        job_id=job_id,
        intent="cancel",
        created_by="test_cancel",
        reason_tag="user_cancelled",
    )
    assert result.ok is True
    assert result.status == "stopping"
    assert result.attempt_id == attempt_id
    assert result.reason is None

    with _pool.connection() as conn:
        job = conn.execute(
            """SELECT status, lifecycle_intent
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
        cmd = conn.execute(
            """SELECT command, fencing_token, args
                 FROM agent_commands WHERE job_id=%s""",
            (job_id,),
        ).fetchone()

    assert job[0] == "stopping"
    assert job[1] == "cancel"
    assert cmd[0] == "stop_attempt"
    assert cmd[1] == fence
    assert cmd[2]["preserve"] is False
    assert cmd[2]["intent"] == "cancel"


@pytest.fixture
def cancel_route(monkeypatch):
    import routes.instances as routes

    monkeypatch.setattr(routes, "_require_auth", lambda request: {"is_admin": True})
    monkeypatch.setattr(routes, "_require_scope", lambda *args: None)
    monkeypatch.setattr(routes, "_check_job_write_access", lambda *args: None)
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    return lambda job_id: routes.api_cancel_instance(job_id, request)


@pytest.mark.parametrize("status", ["running", "assigned", "stopped", "queued"])
def test_cancel_refuses_unfenced_history_without_changing_it(cleanup_ids, cancel_route, status):
    host_id = f"h-cancel-history-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id, status=status)
    with _pool.connection() as conn:
        before = conn.execute(
            "SELECT to_jsonb(j) FROM jobs j WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
    with pytest.raises(HTTPException) as exc:
        cancel_route(job_id)
    assert exc.value.status_code == 409
    with _pool.connection() as conn:
        assert (
            conn.execute("SELECT to_jsonb(j) FROM jobs j WHERE job_id=%s", (job_id,)).fetchone()[0]
            == before
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM agent_commands WHERE job_id=%s", (job_id,)
            ).fetchone()[0]
            == 0
        )


def test_cancel_can_settle_an_unplaced_queued_job(cleanup_ids, cancel_route):
    job_id, _ = _mkjob(cleanup_ids, host_id=None, status="queued")
    assert cancel_route(job_id)["status"] == "cancelled"
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,active_attempt_id FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone() == ("cancelled", None)


def test_cancel_route_uses_fenced_stop_for_active_attempt(cleanup_ids, cancel_route, monkeypatch):
    from unittest.mock import Mock

    detach = Mock(side_effect=AssertionError("volume detached before worker stop confirmation"))
    monkeypatch.setattr("volumes.get_volume_engine", detach)
    host_id = f"h-cancel-active-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)
    attempt_id, fence = _mk_active_attempt(job_id, host_id)
    assert cancel_route(job_id)["status"] == "stopping"
    detach.assert_not_called()
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT command,attempt_id::text,fencing_token FROM agent_commands WHERE job_id=%s",
            (job_id,),
        ).fetchone() == ("stop_attempt", attempt_id, fence)


@pytest.mark.parametrize("race", ["assigned", "fenced", "host_only"])
def test_cancel_rechecks_authority_after_initial_read(cleanup_ids, cancel_route, monkeypatch, race):
    import routes.instances as routes

    host_id = f"h-cancel-race-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=None, status="queued")
    update = routes.update_job_status

    def advance_before_cancel(*args, **kwargs):
        if race == "fenced":
            _mk_active_attempt(job_id, host_id)
        elif race == "host_only":
            # A competing native controller wins the row lock; payload patches
            # are deliberately unable to assign hosts.
            with _pool.connection() as conn:
                conn.execute("UPDATE jobs SET host_id=%s WHERE job_id=%s", (host_id, job_id))
        else:
            update(job_id, "assigned", host_id=host_id)
        return update(*args, **kwargs)

    monkeypatch.setattr(routes, "update_job_status", advance_before_cancel)
    with pytest.raises(HTTPException) as exc:
        cancel_route(job_id)
    assert exc.value.status_code == 409
    with _pool.connection() as conn:
        assert (
            conn.execute("SELECT status FROM jobs WHERE job_id=%s", (job_id,)).fetchone()[0]
            != "cancelled"
        )


def test_stopped_report_projects_terminated_from_intent(cleanup_ids):
    """Worker ``stopped`` + lifecycle_intent=terminate → jobs.status terminated."""
    host_id = f"h-life-ack-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)
    attempt_id, fence = _mk_active_attempt(job_id, host_id)

    from control_plane.attempts import report_attempt_status
    from control_plane.lifecycle import request_fenced_stop_remove

    assert request_fenced_stop_remove(job_id=job_id, intent="terminate", created_by="test").ok

    with _pool.connection() as conn:
        report_attempt_status(
            conn,
            job_id=job_id,
            attempt_id=attempt_id,
            host_id=host_id,
            fencing_token=int(fence),
            status="stopped",
        )
        conn.commit()
        row = conn.execute(
            """SELECT status, active_attempt_id,
                      lifecycle_intent
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
        lease = conn.execute(
            "SELECT status FROM placement_leases WHERE attempt_id=%s",
            (attempt_id,),
        ).fetchone()[0]

    assert row[0] == "terminated"
    assert row[1] is None
    assert row[2] is None  # intent consumed on terminal settle
    assert lease == "released"


def test_stopped_report_projects_cancelled_from_intent(cleanup_ids):
    host_id = f"h-life-ackc-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)
    attempt_id, fence = _mk_active_attempt(job_id, host_id)

    from control_plane.attempts import report_attempt_status
    from control_plane.lifecycle import request_fenced_stop_remove

    assert request_fenced_stop_remove(job_id=job_id, intent="cancel", created_by="test").ok

    with _pool.connection() as conn:
        report_attempt_status(
            conn,
            job_id=job_id,
            attempt_id=attempt_id,
            host_id=host_id,
            fencing_token=int(fence),
            status="stopped",
        )
        conn.commit()
        status = conn.execute("SELECT status FROM jobs WHERE job_id=%s", (job_id,)).fetchone()[0]
    assert status == "cancelled"


def test_unfenced_running_termination_refuses_without_status_projection(cleanup_ids):
    host_id = f"h-life-leg-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)

    from billing import BillingEngine

    # A missing attempt pointer is not evidence that running work has stopped.
    result = BillingEngine().terminate_instance(job_id)
    assert result.get("terminated") is False
    assert result.get("reason") == "native_attempt_required"
    with _pool.connection() as conn:
        status = conn.execute("SELECT status FROM jobs WHERE job_id=%s", (job_id,)).fetchone()[0]
        cmds = conn.execute(
            "SELECT COUNT(*) FROM agent_commands WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
    assert status == "running"
    assert cmds == 0


def test_routes_no_longer_hardfail_only_on_controller_missing():
    """Static: terminate/cancel controllers are wired; fail-closed strings gone as sole path."""
    from pathlib import Path

    billing = Path("billing/billing.py").read_text(encoding="utf-8")
    instances = Path("api/routes/instances.py").read_text(encoding="utf-8")
    assert "request_fenced_stop_remove" in billing
    assert "request_fenced_stop_remove" in instances
    # Must not be the only return for attempt-owned terminate.
    assert 'reason": "fenced_terminate_requires_controller"' not in billing
    assert 'raise HTTPException(409, "fenced_cancel_requires_controller")' not in instances


def test_enqueue_refuse_does_not_leave_stopping_without_command(cleanup_ids):
    """Atomicity: no lease → ok=False, status still running, zero commands."""
    host_id = f"h-life-refuse-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)
    attempt_id = str(uuid.uuid4())
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token)
               VALUES (%s, %s, 1, 'running', %s,
                       nextval('placement_fencing_token_seq'))""",
            (attempt_id, job_id, host_id),
        )
        conn.execute(
            "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
            (attempt_id, job_id),
        )
        conn.commit()

    from control_plane.lifecycle import request_fenced_stop_remove

    result = request_fenced_stop_remove(job_id=job_id, intent="terminate", created_by="test")
    assert result.ok is False
    assert result.reason == "no_active_fenced_authority"
    assert result.command_created is False

    with _pool.connection() as conn:
        row = conn.execute(
            """SELECT status, lifecycle_intent
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
        n = conn.execute(
            "SELECT COUNT(*) FROM agent_commands WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
    assert row[0] == "running"
    assert row[1] is None
    assert n == 0


def test_terminate_billing_cycle_once_per_attempt(cleanup_ids):
    host_id = f"h-life-bc-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)
    attempt_id, _ = _mk_active_attempt(job_id, host_id)

    from billing import BillingEngine

    assert BillingEngine().terminate_instance(job_id)["terminated"] is True
    assert BillingEngine().terminate_instance(job_id)["terminated"] is True
    with _pool.connection() as conn:
        n_bc = conn.execute(
            """SELECT COUNT(*) FROM billing_cycles
                WHERE job_id=%s AND cycle_id = %s""",
            (job_id, f"BC-term-{attempt_id}"),
        ).fetchone()[0]
        n_cmd = conn.execute(
            "SELECT COUNT(*) FROM agent_commands WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
    assert n_bc == 1
    assert n_cmd == 1


@pytest.mark.parametrize("dead", ["dead_letter", "cancelled", "expired"])
def test_a_repeated_stop_rearms_a_command_that_died_undelivered(cleanup_ids, dead):
    """The {intent}:{attempt} key used to return the dead command: nothing was delivered."""
    from control_plane.commands import claim_commands
    from control_plane.lifecycle import request_fenced_stop

    host_id = f"h-life-r-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)
    _mk_active_attempt(job_id, host_id)

    first = request_fenced_stop(job_id=job_id, created_by="user")
    assert first.ok and first.command_created and not first.command_redelivered
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE agent_commands SET status=%s, attempt_count=5, "
            "expires_at=CASE WHEN %s THEN EXTRACT(EPOCH FROM NOW())-1 ELSE expires_at END "
            "WHERE command_id=%s",
            ("pending" if dead == "expired" else dead, dead == "expired", first.command_id),
        )
        conn.commit()

    again = request_fenced_stop(job_id=job_id, created_by="user")
    assert again.ok and again.command_id == first.command_id
    assert again.command_redelivered and not again.command_created
    with _pool.connection() as conn:
        delivered = claim_commands(conn, host_id=host_id, worker_session_id="rearm-test")
        conn.commit()
    assert [c.command_id for c in delivered] == [first.command_id]


def test_a_repeated_stop_leaves_an_in_flight_command_alone(cleanup_ids):
    from control_plane.lifecycle import request_fenced_stop

    host_id = f"h-life-f-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)
    _mk_active_attempt(job_id, host_id)
    first = request_fenced_stop(job_id=job_id, created_by="user")
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE agent_commands SET status='claimed', attempt_count=1, claim_owner=%s, "
            "claim_expires_at=clock_timestamp()+interval '1 minute' WHERE command_id=%s",
            (host_id, first.command_id),
        )
        conn.commit()
    again = request_fenced_stop(job_id=job_id, created_by="user")
    assert again.ok and not again.command_redelivered
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status, attempt_count FROM agent_commands WHERE command_id=%s",
            (first.command_id,),
        ).fetchone() == ("claimed", 1)
