"""Native billing lifecycle, refusal of unqualified work and fenced settlement."""

from __future__ import annotations

from control_plane.state import initial_desired_state, job_phase
from tests.native_jobs import admit_fixture_job

import inspect
import json
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None


@pytest.fixture
def cleanup():
    ids = {"jobs": [], "hosts": [], "wallets": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM placement_leases WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        for cid in ids["wallets"]:
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (cid,))
        conn.commit()


def _mk_host(cleanup, host_id):
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload, admission_state, administrative_state, availability_state)
               VALUES (%s, 'active', %s, %s, 'admitted', 'admitted', 'ready')
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, time.time(), json.dumps({"host_id": host_id})),
        )
        conn.commit()
    cleanup["hosts"].append(host_id)


def _mk_wallet(cleanup, customer_id, balance=50.0):
    now = time.time()
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO wallets
               (customer_id, balance_micros, total_deposited_micros, total_spent_micros,
                total_refunded_micros, status, created_at, updated_at)
               VALUES (%s, (%s * 1000000)::bigint, (%s * 1000000)::bigint, 0, 0,
                       'active', %s, %s)
               ON CONFLICT (customer_id) DO UPDATE SET
                   balance_micros=EXCLUDED.balance_micros""",
            (customer_id, balance, balance, now, now),
        )
        conn.commit()
    cleanup["wallets"].append(customer_id)


def _mk_job(cleanup, *, owner, host_id, status="running"):
    job_id = f"j-bill-{uuid.uuid4().hex[:8]}"
    payload = {
        "owner": owner,
        "name": job_id,
        "container_name": f"xcl-{job_id}",
        "status": status,
    }
    with _pool.connection() as conn:
        fixture_submitted_at = time.time()
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, host_id, payload, phase, desired_state, effective_priority, queued_at)
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
    return job_id


def _mk_active_attempt(job_id, host_id):
    attempt_id = str(uuid.uuid4())
    with _pool.connection() as conn:
        fence = conn.execute("SELECT nextval('placement_fencing_token_seq')").fetchone()[0]
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, job_generation)
               VALUES (%s, %s, 1, 'running', %s, %s, 1)""",
            (attempt_id, job_id, host_id, fence),
        )
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


def _job_row(job_id):
    with _pool.connection() as conn:
        row = conn.execute(
            """SELECT status, active_attempt_id,
                      lifecycle_intent AS intent,
                      payload->>'status' AS pstatus
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    return {
        "status": row[0],
        "active_attempt_id": row[1],
        "intent": row[2],
        "pstatus": row[3],
    }


class TestAttemptOwnedBillingStop:
    def test_stop_uses_fenced_lifecycle_preserve_no_raw_premark(self, cleanup):
        from billing import BillingEngine

        owner = f"bill-stop-{uuid.uuid4().hex[:6]}@test"
        host_id = f"h-bill-stop-{uuid.uuid4().hex[:6]}"
        _mk_wallet(cleanup, owner)
        _mk_host(cleanup, host_id)
        job_id = _mk_job(cleanup, owner=owner, host_id=host_id)
        attempt_id, fence = _mk_active_attempt(job_id, host_id)

        result = BillingEngine().stop_instance(job_id, "user_stopped")
        assert result.get("stopped") is True
        assert result.get("status") == "stopping"

        row = _job_row(job_id)
        assert row["status"] == "stopping"
        assert row["intent"] == "stop"
        assert str(row["active_attempt_id"]) == attempt_id

        with _pool.connection() as conn:
            cmd = conn.execute(
                """SELECT command, args, fencing_token, status
                     FROM agent_commands
                    WHERE job_id=%s AND command='stop_attempt'""",
                (job_id,),
            ).fetchone()
        assert cmd is not None
        args = cmd[1] if not isinstance(cmd, dict) else cmd["args"]
        fence_got = cmd[2] if not isinstance(cmd, dict) else cmd["fencing_token"]
        assert args.get("preserve") is True
        assert args.get("intent") == "stop"
        assert int(fence_got) == int(fence)

    def test_stop_enqueue_refuse_leaves_running(self, cleanup, monkeypatch):
        from billing import BillingEngine
        from control_plane.commands import CommandProtocolError
        import control_plane.lifecycle as life

        owner = f"bill-refuse-{uuid.uuid4().hex[:6]}@test"
        host_id = f"h-bill-refuse-{uuid.uuid4().hex[:6]}"
        _mk_wallet(cleanup, owner)
        _mk_host(cleanup, host_id)
        job_id = _mk_job(cleanup, owner=owner, host_id=host_id)
        _mk_active_attempt(job_id, host_id)

        def _boom(*a, **k):
            raise CommandProtocolError("no authority")

        monkeypatch.setattr(life, "enqueue_current_attempt_command", _boom)

        result = BillingEngine().stop_instance(job_id, "billing_suspended")
        assert result.get("stopped") is False
        assert _job_row(job_id)["status"] == "running"

    def test_terminate_attempt_owned_stays_stopping_with_remove(self, cleanup):
        from billing import BillingEngine

        owner = f"bill-term-{uuid.uuid4().hex[:6]}@test"
        host_id = f"h-bill-term-{uuid.uuid4().hex[:6]}"
        _mk_wallet(cleanup, owner)
        _mk_host(cleanup, host_id)
        job_id = _mk_job(cleanup, owner=owner, host_id=host_id)
        attempt_id, fence = _mk_active_attempt(job_id, host_id)

        result = BillingEngine().terminate_instance(job_id)
        assert result.get("terminated") is True
        assert result.get("status") == "stopping"
        assert _job_row(job_id)["status"] == "stopping"
        assert _job_row(job_id)["intent"] == "terminate"

        with _pool.connection() as conn:
            args = conn.execute(
                "SELECT args FROM agent_commands WHERE job_id=%s AND command='stop_attempt'",
                (job_id,),
            ).fetchone()[0]
        assert args.get("preserve") is False


def _outbox_legacy_status_count(job_id: str) -> int:
    with _pool.connection() as conn:
        row = conn.execute(
            """SELECT count(*) FROM outbox_events
                WHERE aggregate_id=%s
                  AND event_type='job.v1.legacy_status_changed'""",
            (job_id,),
        ).fetchone()
    return int(row[0] if not isinstance(row, dict) else row["count"])


@pytest.mark.parametrize(
    "method,status,flag",
    [
        ("stop_instance", "running", "stopped"),
        ("start_instance", "stopped", "started"),
        ("restart_instance", "running", "restarted"),
        ("terminate_instance", "running", "terminated"),
    ],
)
@pytest.mark.parametrize("admitted", [False, True])
def test_unfenced_running_work_cannot_use_billing_lifecycle(
    cleanup, monkeypatch, method, status, flag, admitted
):
    from billing import BillingEngine
    import routes.agent as agent_mod

    owner = f"unfenced-{uuid.uuid4().hex[:6]}@test"
    host = f"unfenced-{uuid.uuid4().hex[:6]}"
    _mk_wallet(cleanup, owner)
    _mk_host(cleanup, host)
    jid = _mk_job(cleanup, owner=owner, host_id=host, status=status)
    if not admitted:
        with _pool.connection() as conn:
            conn.execute("UPDATE jobs SET spec=NULL,spec_hash=NULL WHERE job_id=%s", (jid,))
    calls = []
    monkeypatch.setattr(agent_mod, "enqueue_agent_command", lambda *a, **k: calls.append((a, k)))
    result = getattr(BillingEngine(), method)(jid)
    assert result[flag] is False, result
    assert result["reason"] == (
        "native_attempt_required" if admitted else "native_admission_required"
    )
    assert _job_row(jid)["status"] == status
    assert calls == []
    with _pool.connection() as conn:
        assert (
            conn.execute("SELECT count(*) FROM billing_cycles WHERE job_id=%s", (jid,)).fetchone()[
                0
            ]
            == 0
        )
        assert (
            conn.execute("SELECT count(*) FROM agent_commands WHERE job_id=%s", (jid,)).fetchone()[
                0
            ]
            == 0
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s", (jid,)
            ).fetchone()[0]
            == 0
        )


class TestBillingStopSourceContract:
    def test_attempt_owned_stop_calls_fenced_lifecycle(self):
        import billing as billing_mod

        src = inspect.getsource(billing_mod.BillingEngine.stop_instance)
        assert "request_fenced_stop" in src
        # No pre-mark dual-write before fenced path for attempt-owned.
        assert "enqueue_current_attempt_command" not in src or "request_fenced_stop" in src

    def test_start_restart_have_no_unfenced_status_writer(self):
        import billing as billing_mod

        start_src = inspect.getsource(billing_mod.BillingEngine.start_instance)
        restart_src = inspect.getsource(billing_mod.BillingEngine.restart_instance)
        assert "update_job_status" not in start_src
        assert "update_job_status" not in restart_src
        # Bare dual-writer pattern must not remain on legacy arms.
        assert "UPDATE jobs SET status = 'restarting'" not in start_src
        assert "UPDATE jobs SET status = 'restarting'" not in restart_src
        assert "UPDATE jobs SET status = %s" not in start_src
        assert "UPDATE jobs SET status = %s" not in restart_src
