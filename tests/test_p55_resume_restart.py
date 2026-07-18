"""P5.5 — fresh-attempt resume/restart for fenced (v2) jobs.

Drives BillingEngine.start_instance / restart_instance against real
control-plane rows. Asserts requeue projection and absence of unfenced
start_container / pause_container for fenced cases.
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
        _has = _c.execute(
            "SELECT to_regclass('public.job_attempts')"
        ).fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
    _has = False
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("control-plane tables missing")


@pytest.fixture
def cleanup_ids():
    ids = {"jobs": [], "hosts": [], "wallets": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM placement_leases WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        for cid in ids["wallets"]:
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (cid,))
        conn.commit()


@pytest.fixture
def captured_enqueue(monkeypatch):
    calls = []

    def _fake(host_id, command, args, *, created_by, expires_in=900):
        calls.append(
            {
                "host_id": host_id,
                "command": command,
                "args": dict(args) if args else {},
                "created_by": created_by,
            }
        )
        return f"cmd-fake-{len(calls)}"

    import routes.agent as agent_mod

    monkeypatch.setattr(agent_mod, "enqueue_agent_command", _fake)
    return calls


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


def _mkwallet(cleanup, customer_id, balance=50.0):
    now = time.time()
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO wallets
                   (customer_id, balance_cad, total_deposited_cad, total_spent_cad,
                    total_refunded_cad, status, created_at, updated_at)
               VALUES (%s, %s, %s, 0, 0, 'active', %s, %s)
               ON CONFLICT (customer_id) DO UPDATE SET balance_cad=EXCLUDED.balance_cad""",
            (customer_id, balance, balance, now, now),
        )
        conn.commit()
    cleanup["wallets"].append(customer_id)


def _mkjob(cleanup, *, owner, host_id, status, container_name=None):
    job_id = f"job-rr-{uuid.uuid4().hex[:10]}"
    cname = container_name or f"xcl-{job_id}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, %s, 0, %s, %s, %s)""",
            (
                job_id,
                status,
                time.time(),
                host_id,
                json.dumps(
                    {"owner": owner, "name": job_id, "container_name": cname}
                ),
            ),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _mk_attempt(job_id, host_id, *, active=False, status="cancelled"):
    attempt_id = str(uuid.uuid4())
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, ended_at)
               VALUES (%s, %s, 1, %s, %s,
                       nextval('placement_fencing_token_seq'),
                       CASE WHEN %s THEN NULL ELSE clock_timestamp() END)""",
            (
                attempt_id,
                job_id,
                "running" if active else status,
                host_id,
                active,
            ),
        )
        if active:
            conn.execute(
                "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
                (attempt_id, job_id),
            )
        conn.commit()
    return attempt_id


def _mk_lease(job_id, host_id, attempt_id):
    with _pool.connection() as conn:
        fence = conn.execute(
            "SELECT fencing_token FROM job_attempts WHERE attempt_id=%s",
            (attempt_id,),
        ).fetchone()[0]
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
    return fence


def test_start_stopped_fenced_requeues(cleanup_ids, captured_enqueue):
    owner = "rr-start@test"
    host = f"h-rr-s-{uuid.uuid4().hex[:6]}"
    _mkwallet(cleanup_ids, owner)
    _mkhost(cleanup_ids, host)
    job_id = _mkjob(
        cleanup_ids, owner=owner, host_id=host, status="stopped"
    )
    _mk_attempt(job_id, host, active=False)

    from billing import BillingEngine

    result = BillingEngine().start_instance(job_id)
    assert result["started"] is True
    assert result["status"] == "queued"
    assert "fenced_resume_requires_new_attempt" not in str(result)
    assert captured_enqueue == []

    with _pool.connection() as conn:
        row = conn.execute(
            """SELECT status, host_id, phase, desired_state,
                      payload->>'lifecycle_intent',
                      payload->>'resume_kind'
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
    assert row[0] == "queued"
    assert row[1] is None
    # 059 trigger derives phase/desired from status.
    assert row[2] == "pending"
    assert row[3] == "running"
    assert row[4] is None  # no sticky lifecycle_intent
    assert row[5] == "resume"


def test_start_idempotent_when_already_queued(cleanup_ids, captured_enqueue):
    owner = "rr-idem@test"
    host = f"h-rr-i-{uuid.uuid4().hex[:6]}"
    _mkwallet(cleanup_ids, owner)
    _mkhost(cleanup_ids, host)
    job_id = _mkjob(cleanup_ids, owner=owner, host_id=host, status="stopped")
    _mk_attempt(job_id, host, active=False)

    from billing import BillingEngine

    assert BillingEngine().start_instance(job_id)["started"]
    second = BillingEngine().start_instance(job_id)
    assert second["started"] is True
    assert second["status"] == "queued"
    assert captured_enqueue == []


def test_restart_running_fenced_enqueues_stop_attempt(
    cleanup_ids, captured_enqueue
):
    owner = "rr-run@test"
    host = f"h-rr-r-{uuid.uuid4().hex[:6]}"
    _mkwallet(cleanup_ids, owner)
    _mkhost(cleanup_ids, host)
    job_id = _mkjob(cleanup_ids, owner=owner, host_id=host, status="running")
    attempt_id = _mk_attempt(job_id, host, active=True)
    fence = _mk_lease(job_id, host, attempt_id)

    from billing import BillingEngine

    result = BillingEngine().restart_instance(job_id)
    assert result["restarted"] is True
    assert result["status"] == "stopping"
    assert result["attempt_id"] == attempt_id
    assert captured_enqueue == []

    with _pool.connection() as conn:
        cmd = conn.execute(
            """SELECT command, attempt_id::text, fencing_token, args
                 FROM agent_commands WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
        intent = conn.execute(
            "SELECT payload->>'lifecycle_intent' FROM jobs WHERE job_id=%s",
            (job_id,),
        ).fetchone()[0]
    assert cmd[0] == "stop_attempt"
    assert cmd[1] == attempt_id
    assert cmd[2] == fence
    assert cmd[3]["intent"] == "restart"
    assert cmd[3]["preserve"] is False
    assert intent == "restart"


def test_restart_ack_projects_to_queued(cleanup_ids, captured_enqueue):
    """Worker stopped report + lifecycle_intent=restart → jobs.status queued.

    Also proves lifecycle_intent is *consumed* so a later normal stop on a
    new attempt settles to ``stopped`` (not sticky requeue).
    """
    owner = "rr-ack@test"
    host = f"h-rr-a-{uuid.uuid4().hex[:6]}"
    _mkwallet(cleanup_ids, owner)
    _mkhost(cleanup_ids, host)
    job_id = _mkjob(cleanup_ids, owner=owner, host_id=host, status="running")
    attempt_id = _mk_attempt(job_id, host, active=True)
    fence = _mk_lease(job_id, host, attempt_id)

    from billing import BillingEngine
    from control_plane.attempts import report_attempt_status

    assert BillingEngine().restart_instance(job_id)["restarted"]

    with _pool.connection() as conn:
        report_attempt_status(
            conn,
            job_id=job_id,
            attempt_id=attempt_id,
            host_id=host,
            fencing_token=int(fence),
            status="stopped",
        )
        conn.commit()
        row = conn.execute(
            """SELECT status, host_id, active_attempt_id, phase, desired_state,
                      payload->>'lifecycle_intent'
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
        lease = conn.execute(
            "SELECT status FROM placement_leases WHERE attempt_id=%s",
            (attempt_id,),
        ).fetchone()[0]
    assert row[0] == "queued"
    assert row[1] is None
    assert row[2] is None
    assert row[3] == "pending"
    assert row[4] == "running"
    assert row[5] is None  # intent consumed — not sticky
    assert lease == "released"
    assert captured_enqueue == []

    # New placement attempt then user stop (no lifecycle_intent) → stopped.
    attempt2 = str(uuid.uuid4())
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token)
               VALUES (%s, %s, 2, 'running', %s,
                       nextval('placement_fencing_token_seq'))""",
            (attempt2, job_id, host),
        )
        fence2 = conn.execute(
            "SELECT fencing_token FROM job_attempts WHERE attempt_id=%s",
            (attempt2,),
        ).fetchone()[0]
        conn.execute(
            """UPDATE jobs
                  SET status = 'running',
                      host_id = %s,
                      active_attempt_id = %s
                WHERE job_id = %s""",
            (host, attempt2, job_id),
        )
        conn.execute(
            """INSERT INTO placement_leases
                   (job_id, attempt_id, host_id, fencing_token, status,
                    claim_deadline, claimed_at, last_renewed_at, expires_at)
               VALUES (%s, %s, %s, %s, 'active',
                       clock_timestamp() + interval '10 minutes',
                       clock_timestamp(), clock_timestamp(),
                       clock_timestamp() + interval '10 minutes')""",
            (job_id, attempt2, host, fence2),
        )
        conn.commit()
        # Normal user-stop path: intermediate stopping without restart intent.
        conn.execute(
            """UPDATE jobs SET status = 'stopping',
                   payload = jsonb_set(
                       COALESCE(payload, '{}'::jsonb) - 'lifecycle_intent',
                       '{status}', '"stopping"'::jsonb, true
                   )
             WHERE job_id = %s""",
            (job_id,),
        )
        report_attempt_status(
            conn,
            job_id=job_id,
            attempt_id=attempt2,
            host_id=host,
            fencing_token=int(fence2),
            status="stopped",
        )
        conn.commit()
        after = conn.execute(
            """SELECT status, payload->>'lifecycle_intent'
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
    assert after[0] == "stopped", (
        f"sticky lifecycle_intent would requeue; got status={after[0]!r} intent={after[1]!r}"
    )
    assert after[1] is None


def test_restart_stopped_fenced_requeues(cleanup_ids, captured_enqueue):
    owner = "rr-st@test"
    host = f"h-rr-st-{uuid.uuid4().hex[:6]}"
    _mkwallet(cleanup_ids, owner)
    _mkhost(cleanup_ids, host)
    job_id = _mkjob(cleanup_ids, owner=owner, host_id=host, status="stopped")
    _mk_attempt(job_id, host, active=False)

    from billing import BillingEngine

    result = BillingEngine().restart_instance(job_id)
    assert result["restarted"] is True
    assert result["status"] == "queued"
    assert captured_enqueue == []


def test_legacy_start_still_uses_start_container(cleanup_ids, captured_enqueue):
    owner = "rr-leg@test"
    host = f"h-rr-l-{uuid.uuid4().hex[:6]}"
    _mkwallet(cleanup_ids, owner)
    _mkhost(cleanup_ids, host)
    job_id = _mkjob(cleanup_ids, owner=owner, host_id=host, status="stopped")
    # No job_attempts rows → legacy path.

    from billing import BillingEngine

    result = BillingEngine().start_instance(job_id)
    assert result.get("started") is True, result
    assert any(c["command"] == "start_container" for c in captured_enqueue)


def test_static_controller_reachable():
    from pathlib import Path

    billing = Path("billing.py").read_text(encoding="utf-8")
    assert "request_fresh_attempt_resume" in billing
    assert "intent=\"restart\"" in billing or "intent='restart'" in billing
    assert 'reason": "fenced_resume_requires_new_attempt"' not in billing
    assert 'reason": "fenced_restart_requires_controller"' not in billing
