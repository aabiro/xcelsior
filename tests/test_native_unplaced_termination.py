"""Cancellation before placement must preserve authority and financial atomicity."""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from tests.test_billing_lifecycle_domain import (
    _mk_job,
    _mk_host,
    _mk_wallet,
    _mk_active_attempt,
    _pool,
    cleanup,
)
from control_plane.lifecycle import request_unplaced_termination


@pytest.fixture
def waiting_job(cleanup):
    owner = f"waiting-{uuid.uuid4().hex[:8]}@test"
    host = f"waiting-{uuid.uuid4().hex[:8]}"
    _mk_wallet(cleanup, owner)
    _mk_host(cleanup, host)
    jid = _mk_job(cleanup, owner=owner, host_id=host, status="queued")
    hold = str(uuid.uuid4())
    now = time.time()
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO wallet_holds
            (hold_id,customer_id,amount_micros,status,job_id,created_at,updated_at,expires_at)
            VALUES(%s,%s,1000000,'held',%s,%s,%s,%s)""",
            (hold, owner, jid, now, now, now + 600),
        )
        conn.execute("UPDATE jobs SET wallet_hold_id=%s WHERE job_id=%s", (hold, jid))
    yield jid, owner, host, hold
    with _pool.connection() as conn:
        conn.execute("UPDATE jobs SET wallet_hold_id=NULL WHERE job_id=%s", (jid,))
        conn.execute("DELETE FROM wallet_holds WHERE hold_id=%s", (hold,))


def snapshot(jid, hold):
    with _pool.connection() as conn:
        job = conn.execute(
            "SELECT status,phase,desired_state,wallet_hold_id,active_attempt_id FROM jobs WHERE job_id=%s",
            (jid,),
        ).fetchone()
        status = conn.execute(
            "SELECT status FROM wallet_holds WHERE hold_id=%s", (hold,)
        ).fetchone()[0]
        events = conn.execute(
            "SELECT event_type,payload FROM outbox_events WHERE aggregate_id=%s", (jid,)
        ).fetchall()
        commands = conn.execute(
            "SELECT count(*) FROM agent_commands WHERE job_id=%s", (jid,)
        ).fetchone()[0]
        wallet = conn.execute(
            "SELECT balance_micros,total_spent_micros FROM wallets WHERE customer_id=(SELECT customer_id FROM wallet_holds WHERE hold_id=%s)",
            (hold,),
        ).fetchone()
    return job, status, events, commands, wallet


def test_unplaced_termination_is_atomic_and_idempotent(waiting_job, monkeypatch):
    from billing import BillingEngine
    import routes.agent

    jid, owner, host, hold = waiting_job
    host_io = Mock(side_effect=AssertionError("unfenced host action"))
    monkeypatch.setattr(routes.agent, "enqueue_agent_command", host_io)
    first = BillingEngine().terminate_instance(jid)
    assert first["terminated"] and first["status"] == "terminated", first
    second = request_unplaced_termination(job_id=jid, created_by="test-retry")
    assert second.ok
    job, status, events, commands, wallet = snapshot(jid, hold)
    assert job == ("terminated", "stopped", "stopped", None, None)
    assert status == "released"
    assert len(events) == 1 and events[0][0] == "job.v1.terminated"
    assert commands == 0 and wallet == (50000000, 0)
    host_io.assert_not_called()


def test_notification_failure_rolls_back_job_and_hold(waiting_job, monkeypatch):
    import control_plane.outbox

    jid, _, _, hold = waiting_job
    before = snapshot(jid, hold)
    monkeypatch.setattr(
        control_plane.outbox, "append_event", Mock(side_effect=RuntimeError("outbox unavailable"))
    )
    with pytest.raises(RuntimeError, match="outbox unavailable"):
        request_unplaced_termination(job_id=jid, created_by="test")
    assert snapshot(jid, hold) == before


@pytest.mark.parametrize(
    "authority", ["attempt_pointer", "orphaned_lease", "orphaned_attempt", "orphaned_allocation"]
)
def test_live_authority_is_never_optimistically_terminated(waiting_job, authority):
    jid, _, host, hold = waiting_job
    aid, _ = _mk_active_attempt(jid, host)
    if authority != "attempt_pointer":
        with _pool.connection() as conn:
            conn.execute("UPDATE jobs SET active_attempt_id=NULL WHERE job_id=%s", (jid,))
            if authority == "orphaned_lease":
                conn.execute(
                    "UPDATE job_attempts SET status='cancelled',ended_at=clock_timestamp() WHERE attempt_id=%s",
                    (aid,),
                )
            else:
                conn.execute(
                    "UPDATE placement_leases SET status='released',released_at=clock_timestamp() WHERE attempt_id=%s",
                    (aid,),
                )
    if authority == "orphaned_allocation":
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE job_attempts SET status='cancelled',ended_at=clock_timestamp() WHERE attempt_id=%s",
                (aid,),
            )
            device = conn.execute(
                "INSERT INTO host_gpu_devices(host_id,gpu_uuid) VALUES(%s,%s) RETURNING gpu_device_id",
                (host, str(uuid.uuid4())),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO gpu_device_allocations(attempt_id,job_id,host_id,gpu_device_id) VALUES(%s,%s,%s,%s)",
                (aid, jid, host, device),
            )
    before = snapshot(jid, hold)
    result = request_unplaced_termination(job_id=jid, created_by="test")
    assert not result.ok
    assert result.reason == (
        "still_attempt_owned" if authority == "attempt_pointer" else "execution_authority_remains"
    )
    assert snapshot(jid, hold) == before


def test_cancellation_waits_for_placement_lock_and_rereads_authority(waiting_job):
    jid, _, host, hold = waiting_job
    started = Event()

    def cancel():
        started.set()
        return request_unplaced_termination(job_id=jid, created_by="concurrent-cancel")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with _pool.connection() as conn:
            conn.execute("SELECT job_id FROM jobs WHERE job_id=%s FOR UPDATE", (jid,))
            future = executor.submit(cancel)
            assert started.wait(5)
            aid = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO job_attempts(attempt_id,job_id,attempt_number,status,host_id,fencing_token)
                VALUES(%s,%s,1,'reserved',%s,nextval('placement_fencing_token_seq'))""",
                (aid, jid, host),
            )
            conn.execute(
                "UPDATE jobs SET active_attempt_id=%s,phase='scheduled',status='assigned' WHERE job_id=%s",
                (aid, jid),
            )
        result = future.result(timeout=10)
    assert not result.ok and result.reason == "still_attempt_owned"
    job, status, events, commands, _ = snapshot(jid, hold)
    assert job[0] == "assigned" and str(job[4]) == aid
    assert status == "held" and events == [] and commands == 0


@pytest.mark.parametrize("problem", ["missing", "corrupt", "history"])
def test_unplaced_termination_refuses_unqualified_evidence(waiting_job, problem):
    jid, _, _, hold = waiting_job
    with _pool.connection() as conn:
        if problem == "missing":
            conn.execute("UPDATE jobs SET spec=NULL,spec_hash=NULL WHERE job_id=%s", (jid,))
        elif problem == "corrupt":
            conn.execute("UPDATE jobs SET spec_hash='invalid' WHERE job_id=%s", (jid,))
        else:
            # A retained job is terminal by schema; use a real disposition migration fixture elsewhere.
            from control_plane.lifecycle import native_admission_problem

            assert (
                native_admission_problem({"history_disposition": "retained_recovered"})
                == "historical_job_requires_new_admission"
            )
            return
    before = snapshot(jid, hold)
    result = request_unplaced_termination(job_id=jid, created_by="test")
    assert not result.ok and result.reason == "native_admission_required"
    assert snapshot(jid, hold) == before


def test_billing_stop_uses_admitted_owner_after_observation_drift(cleanup):
    from billing import BillingEngine

    owner = f"admitted-{uuid.uuid4().hex[:6]}@test"
    host = f"admitted-{uuid.uuid4().hex[:6]}"
    _mk_wallet(cleanup, owner)
    _mk_host(cleanup, host)
    jid = _mk_job(cleanup, owner=owner, host_id=host)
    _mk_active_attempt(jid, host)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=jsonb_set(payload,'{owner}',to_jsonb(%s::text)) WHERE job_id=%s",
            ("forged@test", jid),
        )
    result = BillingEngine().stop_instance(jid)
    assert result["stopped"], result
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT customer_id FROM billing_cycles WHERE job_id=%s", (jid,)
        ).fetchall() == [(owner,)]
