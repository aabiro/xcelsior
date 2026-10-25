"""Lease claim and queue release commit their native projection plus notification."""

import uuid

import pytest
from psycopg.errors import UndefinedTable

from control_plane.db import run_transaction
from control_plane.leases import claim_lease
from control_plane.scheduler.claim import claim_next_job, release_claim
from tests.test_agent_v2 import _place_one, _auth, fleet  # noqa: F401


def snapshot(pool, job):
    with pool.connection() as conn:
        return [
            conn.execute(query, (job,)).fetchall()
            for query in [
                "SELECT to_jsonb(j) FROM jobs j WHERE job_id=%s",
                "SELECT to_jsonb(a) FROM job_attempts a WHERE job_id=%s",
                "SELECT to_jsonb(l) FROM placement_leases l WHERE job_id=%s",
                "SELECT to_jsonb(e) FROM outbox_events e WHERE aggregate_id=%s ORDER BY event_id",
            ]
        ]


def test_claim_projects_actual_leased_state_and_one_status_event(fleet):
    from db import _get_pg_pool

    job, host, refs = _place_one(fleet, uuid.uuid4().hex[:8])
    run_transaction(lambda conn: claim_lease(conn, **_auth(job, host, refs)))
    with _get_pg_pool().connection() as conn:
        assert conn.execute(
            "SELECT j.status,j.phase,a.status,a.lease_claimed_at IS NOT NULL,l.status "
            "FROM jobs j JOIN job_attempts a ON a.attempt_id=j.active_attempt_id "
            "JOIN placement_leases l ON l.attempt_id=a.attempt_id WHERE j.job_id=%s",
            (job,),
        ).fetchone() == ("leased", "scheduled", "lease_claimed", True, "active")
        events = conn.execute(
            "SELECT payload FROM outbox_events WHERE aggregate_id=%s AND event_type='job.v1.attempt_status_changed'",
            (job,),
        ).fetchall()
        assert len(events) == 1
        assert events[0][0]["status"] == "lease_claimed"


def test_claim_outbox_sql_failure_rolls_back_lease_attempt_and_job(fleet, monkeypatch):
    from db import _get_pg_pool
    from control_plane import outbox

    job, host, refs = _place_one(fleet, uuid.uuid4().hex[:8])
    before = snapshot(_get_pg_pool(), job)

    def fail(conn, **kwargs):
        conn.execute("SELECT 1 FROM missing_native_claim_outbox")

    monkeypatch.setattr(outbox, "append_event", fail)
    with pytest.raises(UndefinedTable):
        run_transaction(lambda conn: claim_lease(conn, **_auth(job, host, refs)))
    assert snapshot(_get_pg_pool(), job) == before


@pytest.mark.parametrize("outbox_failure", [False, True])
def test_queue_release_notification_is_atomic_and_idempotent(fleet, monkeypatch, outbox_failure):
    from db import _get_pg_pool
    from control_plane import outbox
    from scheduler import submit_job

    model = "queue-outbox-" + uuid.uuid4().hex
    job = submit_job("no-capacity", 8, gpu_model=model)["job_id"]
    fleet["jobs"].append(job)
    pool = _get_pg_pool()
    claimed = run_transaction(
        lambda conn: claim_next_job(conn, replica_id=model, scope_gpu_models=[model])
    )
    assert claimed and claimed.job_id == job
    before = snapshot(pool, job)
    if outbox_failure:

        def fail(conn, **kwargs):
            conn.execute("SELECT 1 FROM missing_native_release_outbox")

        monkeypatch.setattr(outbox, "append_event", fail)
        with pytest.raises(UndefinedTable):
            run_transaction(
                lambda conn: release_claim(
                    conn, job, claimed.claim_token, reason_code="no_capacity"
                )
            )
        assert snapshot(pool, job) == before
        return
    assert run_transaction(
        lambda conn: release_claim(conn, job, claimed.claim_token, reason_code="no_capacity")
    )
    assert not run_transaction(
        lambda conn: release_claim(
            conn, job, claimed.claim_token, reason_code="different_stale_reason"
        )
    )
    again = run_transaction(
        lambda conn: claim_next_job(conn, replica_id=model, scope_gpu_models=[model])
    )
    assert again and again.job_id == job
    assert run_transaction(
        lambda conn: release_claim(conn, job, again.claim_token, reason_code="no_capacity")
    )
    with pool.connection() as conn:
        assert conn.execute(
            "SELECT reason_code,schedule_claim_token FROM jobs WHERE job_id=%s", (job,)
        ).fetchone() == ("no_capacity", None)
        assert conn.execute(
            "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s AND event_type='job.v1.queue_blocked'",
            (job,),
        ).fetchone() == (1,)
