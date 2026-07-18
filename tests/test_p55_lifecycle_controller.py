"""P5.5 — fenced terminate/cancel lifecycle controller.

Drives the shipped entry points (BillingEngine.terminate_instance and
control_plane.lifecycle.request_fenced_stop_remove for cancel) against
real Postgres control-plane tables. Asserts durable agent_commands rows
and intermediate job projection — not a mocked "success" string.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_cp = _c.execute(
            "SELECT to_regclass('public.job_attempts')"
        ).fetchone()[0] is not None
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
            conn.execute(
                "DELETE FROM placement_leases WHERE job_id=%s", (jid,)
            )
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mkhost(cleanup, host_id):
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'online', %s, %s)
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, time.time(), json.dumps({"name": host_id})),
        )
        conn.commit()
    cleanup["hosts"].append(host_id)
    cleanup["agent_hosts"].append(host_id)


def _mkjob(cleanup, *, host_id, status="running", container_name=None):
    job_id = f"job-p55-{uuid.uuid4().hex[:10]}"
    cname = container_name or f"xcl-{job_id}"
    payload = {
        "owner": "p55@test",
        "name": job_id,
        "container_name": cname,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, %s, 0, %s, %s, %s)""",
            (job_id, status, time.time(), host_id, json.dumps(payload)),
        )
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
    host_id = f"h-p55-t-{uuid.uuid4().hex[:6]}"
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
            """SELECT status, payload->>'lifecycle_intent',
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
    host_id = f"h-p55-c-{uuid.uuid4().hex[:6]}"
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
            """SELECT status, payload->>'lifecycle_intent'
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


def test_stopped_report_projects_terminated_from_intent(cleanup_ids):
    """Worker ``stopped`` + lifecycle_intent=terminate → jobs.status terminated."""
    host_id = f"h-p55-ack-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)
    attempt_id, fence = _mk_active_attempt(job_id, host_id)

    from control_plane.attempts import report_attempt_status
    from control_plane.lifecycle import request_fenced_stop_remove

    assert request_fenced_stop_remove(
        job_id=job_id, intent="terminate", created_by="test"
    ).ok

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
                      payload->>'lifecycle_intent'
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
        lease = conn.execute(
            "SELECT status FROM placement_leases WHERE attempt_id=%s",
            (attempt_id,),
        ).fetchone()[0]

    assert row[0] == "terminated"
    assert row[1] is None
    assert row[2] == "terminate"
    assert lease == "released"


def test_stopped_report_projects_cancelled_from_intent(cleanup_ids):
    host_id = f"h-p55-ackc-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)
    attempt_id, fence = _mk_active_attempt(job_id, host_id)

    from control_plane.attempts import report_attempt_status
    from control_plane.lifecycle import request_fenced_stop_remove

    assert request_fenced_stop_remove(
        job_id=job_id, intent="cancel", created_by="test"
    ).ok

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
        status = conn.execute(
            "SELECT status FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
    assert status == "cancelled"


def test_legacy_terminate_still_marks_terminal_without_attempt(cleanup_ids):
    host_id = f"h-p55-leg-{uuid.uuid4().hex[:6]}"
    _mkhost(cleanup_ids, host_id)
    job_id, _ = _mkjob(cleanup_ids, host_id=host_id)

    from billing import BillingEngine

    # No host SSH — terminate still marks terminal in PG; kill may warn.
    result = BillingEngine().terminate_instance(job_id)
    assert result.get("terminated") is True
    assert result.get("status") == "terminated"
    with _pool.connection() as conn:
        status = conn.execute(
            "SELECT status FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
        cmds = conn.execute(
            "SELECT COUNT(*) FROM agent_commands WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
    assert status == "terminated"
    assert cmds == 0


def test_routes_no_longer_hardfail_only_on_controller_missing():
    """Static: terminate/cancel controllers are wired; fail-closed strings gone as sole path."""
    from pathlib import Path

    billing = Path("billing.py").read_text(encoding="utf-8")
    instances = Path("routes/instances.py").read_text(encoding="utf-8")
    assert "request_fenced_stop_remove" in billing
    assert "request_fenced_stop_remove" in instances
    # Must not be the only return for attempt-owned terminate.
    assert 'reason": "fenced_terminate_requires_controller"' not in billing
    assert 'raise HTTPException(409, "fenced_cancel_requires_controller")' not in instances


def test_enqueue_refuse_does_not_leave_stopping_without_command(cleanup_ids):
    """Atomicity: no lease → ok=False, status still running, zero commands."""
    host_id = f"h-p55-refuse-{uuid.uuid4().hex[:6]}"
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

    result = request_fenced_stop_remove(
        job_id=job_id, intent="terminate", created_by="test"
    )
    assert result.ok is False
    assert result.reason == "no_active_fenced_authority"
    assert result.command_created is False

    with _pool.connection() as conn:
        row = conn.execute(
            """SELECT status, payload->>'lifecycle_intent'
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
    host_id = f"h-p55-bc-{uuid.uuid4().hex[:6]}"
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
