"""Runtime changes wake the owning job atomically, without renewal churn."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from psycopg.errors import LockNotAvailable

from control_plane.commands import (
    ack_command,
    claim_commands,
    nack_command,
    redeliver_expired_claims,
)
from control_plane.leases import LeaseRenewRejected, release_lease, renew_lease
from control_plane.lifecycle import reconcile_fenced_stop
from control_plane.reconcile import _expedite_lease_expiry, process_due
from tests.test_job_reconcile_observations import placed  # noqa: F401
from tests.test_agent_v2 import fleet  # noqa: F401

pytestmark = pytest.mark.needs_db


def _clear(conn, job):
    conn.execute(
        "DELETE FROM reconciliation_queue WHERE resource_type = 'job' AND resource_id = %s",
        (job["job_id"],),
    )


def _request(conn, job):
    return conn.execute(
        "SELECT reason, due_at <= clock_timestamp(), claim_owner FROM reconciliation_queue "
        "WHERE resource_type = 'job' AND resource_id = %s",
        (job["job_id"],),
    ).fetchone()


def _lease(conn, job):
    return str(
        conn.execute(
            "SELECT lease_id FROM placement_leases WHERE attempt_id = %s AND status = 'active'",
            (job["attempt_id"],),
        ).fetchone()[0]
    )


def _claim(conn, job):
    commands = claim_commands(conn, host_id=job["host_id"], worker_session_id="events-test")
    assert len(commands) == 1
    _clear(conn, job)
    return commands[0].command_id


@pytest.mark.parametrize("retryable", [True, False])
def test_nack_wakes_current_job_in_same_transaction(placed, retryable):
    with placed["pool"].connection() as conn:
        command_id = _claim(conn, placed)
        status = nack_command(
            conn,
            command_id=command_id,
            host_id=placed["host_id"],
            error_code="runtime_failure",
            retryable=retryable,
        )
        assert status == ("pending" if retryable else "dead_letter")
        assert _request(conn, placed) == ("command_changed", True, None)


def test_ack_wakes_job_but_duplicate_ack_does_not(placed):
    with placed["pool"].connection() as conn:
        command_id = _claim(conn, placed)
        assert not ack_command(conn, command_id=command_id, host_id=placed["host_id"]).duplicate
        assert _request(conn, placed) == ("command_changed", True, None)
        _clear(conn, placed)
        assert ack_command(conn, command_id=command_id, host_id=placed["host_id"]).duplicate
        assert _request(conn, placed) is None


@pytest.mark.parametrize("exhausted", [False, True])
def test_expired_delivery_claim_wakes_job_even_without_worker_report(placed, exhausted):
    with placed["pool"].connection() as conn:
        command_id = _claim(conn, placed)
        conn.execute(
            "UPDATE agent_commands SET claim_expires_at = clock_timestamp() - interval '1 second', "
            "attempt_count = CASE WHEN %s THEN max_attempts ELSE 1 END WHERE command_id = %s",
            (exhausted, command_id),
        )
        _clear(conn, placed)
        assert redeliver_expired_claims(conn) == 1
        assert _request(conn, placed) == ("command_changed", True, None)


def test_command_failure_and_request_both_roll_back(placed):
    with placed["pool"].connection() as conn:
        command_id = _claim(conn, placed)
        with pytest.raises(RuntimeError, match="rollback"):
            with conn.transaction():
                nack_command(
                    conn, command_id=command_id, host_id=placed["host_id"], error_code="failed"
                )
                assert _request(conn, placed) is not None
                raise RuntimeError("rollback")
        assert _request(conn, placed) is None
        assert conn.execute(
            "SELECT status FROM agent_commands WHERE command_id = %s",
            (command_id,),
        ).fetchone() == ("claimed",)


def test_lease_release_wakes_job_but_replay_does_not(placed):
    with placed["pool"].connection() as conn:
        kwargs = dict(
            lease_id=_lease(conn, placed),
            attempt_id=placed["attempt_id"],
            host_id=placed["host_id"],
            fencing_token=placed["fencing_token"],
        )
        _clear(conn, placed)
        assert release_lease(conn, **kwargs)
        assert _request(conn, placed) == ("lease_changed", True, None)
        _clear(conn, placed)
        assert not release_lease(conn, **kwargs)
        assert _request(conn, placed) is None


def test_healthy_renewals_do_not_wake_or_postpone_job(placed):
    with placed["pool"].connection() as conn:
        lease_id = _lease(conn, placed)
        _clear(conn, placed)
        for _ in range(2):
            renew_lease(
                conn,
                lease_id=lease_id,
                attempt_id=placed["attempt_id"],
                host_id=placed["host_id"],
                fencing_token=placed["fencing_token"],
                worker_session_id="events-test",
            )
        assert _request(conn, placed) is None


def test_expired_renewal_is_refused_and_keeps_expiry_request(placed):
    with placed["pool"].connection() as conn:
        lease_id = _lease(conn, placed)
        _expedite_lease_expiry(conn, attempt_id=placed["attempt_id"])
        with pytest.raises(LeaseRenewRejected):
            renew_lease(
                conn,
                lease_id=lease_id,
                attempt_id=placed["attempt_id"],
                host_id=placed["host_id"],
                fencing_token=placed["fencing_token"],
                worker_session_id="events-test",
            )
        assert _request(conn, placed) == ("lease_changed", True, None)


def test_shortened_offer_and_claim_both_wake_job(fleet):
    import uuid
    from db import _get_pg_pool
    from control_plane.leases import claim_lease
    from tests.test_agent_v2 import _place_one, _auth

    jid, hid, refs = _place_one(fleet, uuid.uuid4().hex[:8])
    job = {"job_id": jid}
    with _get_pg_pool().connection() as conn:
        lease_id = conn.execute(
            "SELECT lease_id FROM placement_leases WHERE job_id=%s AND status='offered'", (jid,)
        ).fetchone()[0]
        _clear(conn, job)
        conn.execute(
            "UPDATE placement_leases SET claim_deadline=clock_timestamp()+interval '5 seconds' WHERE lease_id=%s",
            (lease_id,),
        )
        assert _request(conn, job) == ("lease_changed", True, None)
        _clear(conn, job)
        claim_lease(conn, **_auth(jid, hid, refs))
        assert _request(conn, job) is not None
        assert _request(conn, job)[1:] == (True, None)
        assert conn.execute(
            "SELECT j.status,a.status,l.status FROM jobs j JOIN job_attempts a ON a.attempt_id=j.active_attempt_id JOIN placement_leases l ON l.attempt_id=a.attempt_id WHERE j.job_id=%s",
            (jid,),
        ).fetchone() == ("leased", "lease_claimed", "active")


@pytest.mark.parametrize("field,value", [("host_id", "unrelated-host"), ("fencing_token", -1)])
def test_untrusted_command_binding_does_not_route_a_request(placed, field, value):
    from psycopg import sql

    with placed["pool"].connection() as conn:
        command_id = _claim(conn, placed)
        conn.execute(
            sql.SQL("UPDATE agent_commands SET {} = %s WHERE command_id = %s").format(
                sql.Identifier(field)
            ),
            (value, command_id),
        )
        _clear(conn, placed)
        conn.execute(
            "UPDATE agent_commands SET status = 'failed' WHERE command_id = %s", (command_id,)
        )
        assert _request(conn, placed) is None


def test_shortened_lease_wakes_job_preserving_claim_and_rollback(placed):
    with placed["pool"].connection() as conn:
        conn.execute(
            "UPDATE reconciliation_queue SET due_at = clock_timestamp() + interval '1 hour', "
            "claim_owner = 'other-replica' WHERE resource_type = 'job' AND resource_id = %s",
            (placed["job_id"],),
        )
        with pytest.raises(RuntimeError, match="rollback"):
            with conn.transaction():
                _expedite_lease_expiry(conn, attempt_id=placed["attempt_id"])
                assert _request(conn, placed) == ("lease_changed", True, "other-replica")
                raise RuntimeError("rollback")
        assert _request(conn, placed)[1:] == (False, "other-replica")
        _expedite_lease_expiry(conn, attempt_id=placed["attempt_id"])
        assert _request(conn, placed) == ("lease_changed", True, "other-replica")


def test_removed_current_lease_wakes_job_but_deleted_job_stays_deleted(placed):
    with placed["pool"].connection() as conn:
        _clear(conn, placed)
        conn.execute("DELETE FROM placement_leases WHERE attempt_id = %s", (placed["attempt_id"],))
        assert _request(conn, placed) == ("lease_changed", True, None)
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (placed["job_id"],))
        assert _request(conn, placed) is None


def test_stale_attempt_and_unbound_admin_command_cannot_wake_current_job(placed):
    with placed["pool"].connection() as conn:
        command_id = _claim(conn, placed)
        conn.execute(
            "UPDATE jobs SET active_attempt_id = NULL WHERE job_id = %s", (placed["job_id"],)
        )
        _clear(conn, placed)
        nack_command(conn, command_id=command_id, host_id=placed["host_id"], error_code="stale")
        conn.execute(
            "UPDATE placement_leases SET status = 'fenced' WHERE attempt_id = %s",
            (placed["attempt_id"],),
        )
        conn.execute(
            "INSERT INTO agent_commands (host_id, command, args, status, job_id) "
            "VALUES (%s, 'stop_container', '{}'::jsonb, 'failed', %s)",
            (placed["host_id"], placed["job_id"]),
        )
        assert _request(conn, placed) is None


@pytest.mark.parametrize("locked_resource", ["lease", "command"])
def test_stop_repair_never_waits_on_runtime_writer(placed, locked_resource):
    with placed["pool"].connection() as conn:
        conn.execute(
            """UPDATE jobs SET status = 'stopping', phase='running', desired_state='stopped' WHERE job_id = %s""",
            (placed["job_id"],),
        )
        if locked_resource == "command":
            repaired = reconcile_fenced_stop(
                conn, job_id=placed["job_id"], created_by="events-test"
            )
            assert repaired.command_created
    with placed["pool"].connection() as writer:
        if locked_resource == "lease":
            writer.execute(
                "SELECT 1 FROM placement_leases WHERE attempt_id = %s FOR UPDATE",
                (placed["attempt_id"],),
            )
        else:
            writer.execute(
                "SELECT 1 FROM agent_commands WHERE command_id = %s FOR UPDATE",
                (repaired.command_id,),
            )
        with placed["pool"].connection() as conn:
            conn.execute("SET LOCAL statement_timeout = '500ms'")
            with pytest.raises(LockNotAvailable):
                with conn.transaction():
                    reconcile_fenced_stop(conn, job_id=placed["job_id"], created_by="events-test")


def test_host_expiry_repair_never_waits_on_lease_writer(placed):
    with placed["pool"].connection() as writer:
        writer.execute(
            "SELECT 1 FROM placement_leases WHERE attempt_id = %s FOR UPDATE",
            (placed["attempt_id"],),
        )
        with placed["pool"].connection() as conn:
            conn.execute("SET LOCAL statement_timeout = '500ms'")
            with pytest.raises(LockNotAvailable):
                with conn.transaction():
                    _expedite_lease_expiry(conn, attempt_id=placed["attempt_id"])


def test_lease_writer_and_claimed_reconcile_commit_without_deadlock_or_lost_wakeup(
    placed, monkeypatch
):
    monkeypatch.setenv("XCELSIOR_RECONCILE_ACTION_JOB_STOP_COMMAND_MISSING", "enforce")
    with placed["pool"].connection() as conn:
        conn.execute(
            """UPDATE jobs SET status = 'stopping', phase='running', desired_state='stopped' WHERE job_id = %s""",
            (placed["job_id"],),
        )
        conn.execute(
            "UPDATE reconciliation_queue SET priority = 9999 WHERE resource_id = %s",
            (placed["job_id"],),
        )
    locked = Event()

    def release():
        with placed["pool"].connection() as writer:
            writer.execute("SET LOCAL statement_timeout = '3s'")
            lease_id = _lease(writer, placed)
            writer.execute(
                "SELECT 1 FROM placement_leases WHERE lease_id = %s FOR UPDATE", (lease_id,)
            )
            locked.set()
            release_lease(
                writer,
                lease_id=lease_id,
                attempt_id=placed["attempt_id"],
                host_id=placed["host_id"],
                fencing_token=placed["fencing_token"],
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        with placed["pool"].connection() as conn:
            conn.execute("SET LOCAL statement_timeout = '2s'")
            conn.execute(
                "SELECT 1 FROM reconciliation_queue WHERE resource_type = 'job' "
                "AND resource_id = %s FOR UPDATE",
                (placed["job_id"],),
            )
            writer = executor.submit(release)
            assert locked.wait(2)
            stats = process_due(conn, worker_id="events-test", limit=1)
            assert stats["failed"] == 1 and stats["actions_taken"] == 0
            error = conn.execute(
                "SELECT last_error FROM reconciliation_queue WHERE resource_id = %s",
                (placed["job_id"],),
            ).fetchone()[0]
            assert "could not obtain lock" in error
        writer.result(timeout=4)
    with placed["pool"].connection() as conn:
        assert _request(conn, placed) == ("lease_changed", True, None)
        assert conn.execute(
            "SELECT status FROM jobs WHERE job_id = %s", (placed["job_id"],)
        ).fetchone() == ("stopping",)
