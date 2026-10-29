"""Lease loss, replacement and callbacks against real native PostgreSQL state."""

from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from psycopg.errors import LockNotAvailable
from psycopg.types.json import Jsonb

from billing import get_billing_engine
from control_plane.attempts import report_attempt_status
from control_plane.db import run_transaction
from control_plane.leases import claim_lease, expire_stale_leases, FencingViolation
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import reserve_and_bind
from tests.test_serverless_native_lifecycle import native, start  # noqa: F401


def expire(x):
    with x.pool.connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET claim_deadline=clock_timestamp()-interval '10 minutes',"
            "expires_at=clock_timestamp()-interval '10 minutes' WHERE lease_id=%s",
            (x.r.lease_id,),
        )
    return run_transaction(lambda c: expire_stale_leases(c, grace_sec=0, current_job_id=x.jid))


@pytest.mark.parametrize("claimed", [False, True])
def test_expiry_closes_worker_and_requeues_requests_without_asserting_physical_exit(
    native, claimed
):
    if claimed:
        start(native)
    else:
        native.reserve()
    request = native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {"task": 1})
    if claimed:
        assert native.svc.worker_claim_job(native.wid)
    assert len(expire(native)) == 1
    w = native.repo.get_worker(native.wid)
    assert w["state"] == "error" and w["current_concurrency"] == 0
    assert w["released_at"] is not None
    job = native.repo.get_job(request["job_id"])
    assert job["status"] == "IN_QUEUE" and job["worker_id"] is None
    with native.pool.connection() as conn:
        row = conn.execute(
            "SELECT j.status,j.active_attempt_id,a.status,EXTRACT(EPOCH FROM a.ended_at)::float8 "
            "FROM jobs j JOIN job_attempts a ON a.job_id=j.job_id WHERE a.attempt_id=%s",
            (native.r.attempt_id,),
        ).fetchone()
        assert row == ("failed", None, "lost" if claimed else "failed", w["released_at"])
        # A never-claimed offer frees its devices at once. A claimed one may
        # still have a container, so its devices wait for proof of stop; the
        # worker and its requests are released without that proof.
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND status='active'",
            (native.r.attempt_id,),
        ).fetchone()[0] == (1 if claimed else 0)
        assert (
            conn.execute(
                "SELECT count(*) FROM agent_commands WHERE job_id=%s AND command='stop_attempt' "
                "AND status='acknowledged'",
                (native.jid,),
            ).fetchone()[0]
            == 0
        )
    assert native.svc.worker_heartbeat(native.wid) is None
    assert native.svc.worker_claim_job(native.wid) is None
    assert (
        native.svc.worker_complete_job(native.wid, request["job_id"], output={"late": True}) is None
    )
    assert native.svc.worker_append_event(native.wid, request["job_id"], "result", {}) is None
    assert native.svc.worker_get_job(native.wid, request["job_id"]) is None
    assert (
        run_transaction(lambda c: expire_stale_leases(c, grace_sec=0, current_job_id=native.jid))
        == []
    )


def test_expiry_projection_failure_rolls_back_lease_attempt_worker_and_request(native, monkeypatch):
    start(native)
    request = native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {})
    assert native.svc.worker_claim_job(native.wid)
    import control_plane.serverless_workers as projection

    original = projection.sync_serverless_attempt

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("after expiry projection")

    monkeypatch.setattr(projection, "sync_serverless_attempt", fail)
    with pytest.raises(RuntimeError, match="after expiry projection"):
        expire(native)
    assert native.repo.get_worker(native.wid)["released_at"] is None
    assert native.repo.get_worker(native.wid)["current_concurrency"] == 1
    assert native.repo.get_job(request["job_id"])["status"] == "IN_PROGRESS"
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s", (native.r.lease_id,)
            ).fetchone()[0]
            == "active"
        )
        assert (
            conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id=%s", (native.r.attempt_id,)
            ).fetchone()[0]
            == "running"
        )


def test_single_requeued_request_gets_fresh_worker_job_and_fence(native):
    start(native)
    old = native.r
    request = native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {"task": 1})
    assert native.svc.worker_claim_job(native.wid)
    expire(native)
    # The host reports no workload, which proves the old attempt stopped and
    # releases the device it held for the replacement below.
    from control_plane.leases import release_stopped_attempts
    from tests._db_helpers import report_test_host_inventory

    report_test_host_inventory(native.hid)
    assert run_transaction(release_stopped_attempts)
    get_billing_engine().deposit(native.owner, 50)
    import scheduler

    original = scheduler.submit_job

    def submit(**kwargs):
        # Isolate scheduler selection from other tests' queued jobs while using
        # the real submission, claim, reservation, lease and status functions.
        kwargs["gpu_model"] = native.jid
        return original(**kwargs)

    with mock.patch.object(scheduler, "submit_job", side_effect=submit):
        result = native.svc.reconcile_endpoint(native.ep["endpoint_id"])
    assert result["scaled_up"] == 1
    workers = native.repo.list_workers(native.ep["endpoint_id"])
    new = next(w for w in workers if w["worker_id"] != native.wid)
    jid = new["scheduler_job_id"]
    try:
        assert jid != native.jid and new["state"] == "booting" and new["attempt_id"] is None

        def reserve(conn):
            claim = claim_next_job(conn, replica_id=jid, scope_gpu_models=[native.jid])
            assert claim.job_id == jid
            return reserve_and_bind(
                conn, job_id=jid, claim_token=claim.claim_token, replica_id=jid, host_id=native.hid
            )

        r = run_transaction(reserve)
        assert r.fencing_token > old.fencing_token and r.attempt_id != old.attempt_id
        run_transaction(
            lambda c: claim_lease(
                c,
                lease_id=r.lease_id,
                job_id=jid,
                attempt_id=r.attempt_id,
                host_id=r.host_id,
                fencing_token=r.fencing_token,
                worker_session_id="replacement",
            )
        )
        run_transaction(
            lambda c: report_attempt_status(
                c,
                job_id=jid,
                attempt_id=r.attempt_id,
                host_id=r.host_id,
                fencing_token=r.fencing_token,
                status="running",
            )
        )
        assert native.svc.worker_claim_job(new["worker_id"])["job_id"] == request["job_id"]
        assert (
            native.svc.worker_complete_job(native.wid, request["job_id"], output={"late": 1})
            is None
        )
        assert native.repo.get_job(request["job_id"])["worker_id"] == new["worker_id"]
        with pytest.raises(FencingViolation):
            run_transaction(
                lambda c: report_attempt_status(
                    c,
                    job_id=old.job_id,
                    attempt_id=old.attempt_id,
                    host_id=old.host_id,
                    fencing_token=old.fencing_token,
                    status="running",
                )
            )
    finally:
        with native.pool.connection() as conn:
            conn.execute(
                "UPDATE serverless_jobs SET worker_id=NULL WHERE worker_id=%s", (new["worker_id"],)
            )
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM serverless_workers WHERE worker_id=%s", (new["worker_id"],))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))


def test_concurrent_claims_reserve_only_available_worker_slots(native):
    start(native)
    native.repo.patch_endpoint(native.ep["endpoint_id"], native.owner, {"max_concurrency": 2})
    for i in range(8):
        native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {"task": i})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: native.repo.claim_next_job(native.ep["endpoint_id"], native.wid),
                range(12),
            )
        )
    assert sum(r is not None for r in results) == 2
    assert native.repo.get_worker(native.wid)["current_concurrency"] == 2
    assert native.repo.queue_depth(native.ep["endpoint_id"]) == 6


@pytest.mark.parametrize("failure_point", ["record_token_usage_idempotency", "append_stream_event"])
def test_completion_failure_rolls_back_request_slot_tokens_totals_and_events(
    native, monkeypatch, failure_point
):
    start(native)
    job = native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {})
    native.svc.worker_claim_job(native.wid)
    monkeypatch.setattr(
        native.repo,
        failure_point,
        mock.Mock(side_effect=RuntimeError("token ledger failed")),
    )
    with pytest.raises(RuntimeError, match="token ledger failed"):
        native.svc.worker_complete_job(
            native.wid, job["job_id"], output={"ok": 1}, input_tokens=1000
        )
    assert native.repo.get_job(job["job_id"])["status"] == "IN_PROGRESS"
    assert native.repo.get_worker(native.wid)["current_concurrency"] == 1
    assert native.repo.peek_endpoint_token_cost(native.ep["endpoint_id"]) == 0
    assert native.repo.get_endpoint(native.ep["endpoint_id"])["total_requests"] == 0
    assert native.repo.list_stream_events(job["job_id"]) == []
    assert native.repo.list_token_ledger(native.ep["endpoint_id"]) == []


def test_concurrent_completion_applies_tokens_slot_totals_and_result_once(native):
    start(native)
    job = native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {})
    native.svc.worker_claim_job(native.wid)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(
                lambda _: native.svc.worker_complete_job(
                    native.wid, job["job_id"], output={"ok": 1}, input_tokens=1000
                ),
                range(3),
            )
        )
    assert all(r["status"] == "COMPLETED" for r in results)
    assert native.repo.get_worker(native.wid)["current_concurrency"] == 0
    assert native.repo.get_endpoint(native.ep["endpoint_id"])["total_requests"] == 1
    assert len(native.repo.list_token_ledger(native.ep["endpoint_id"])) == 1
    assert len(native.repo.list_stream_events(job["job_id"])) == 1


def test_slot_update_failure_rolls_back_the_request_claim(native, monkeypatch):
    start(native)
    job = native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {})
    original = native.repo._conn

    @contextmanager
    def faulty_connection():
        with original() as conn:

            def execute(query, params=()):
                if "SET current_concurrency=current_concurrency+1" in query:
                    raise RuntimeError("slot update failed")
                return conn.execute(query, params)

            yield SimpleNamespace(execute=execute)

    monkeypatch.setattr(native.repo, "_conn", faulty_connection)
    with pytest.raises(RuntimeError, match="slot update failed"):
        native.repo.claim_next_job(native.ep["endpoint_id"], native.wid)
    row = native.repo.get_job(job["job_id"])
    assert row["status"] == "IN_QUEUE" and row["worker_id"] is None
    assert native.repo.get_worker(native.wid)["current_concurrency"] == 0


def test_reconcile_expiry_fails_fast_on_a_worker_callback_lock(native):
    start(native)
    with native.pool.connection() as conn:
        conn.execute(
            "SELECT worker_id FROM serverless_workers WHERE worker_id=%s FOR UPDATE", (native.wid,)
        )
        conn.execute(
            "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '10 minutes' WHERE lease_id=%s",
            (native.r.lease_id,),
        )
        # Make the expired deadline visible while retaining the worker lock on
        # this connection, as a callback transaction would do.
        conn.commit()
        conn.execute(
            "SELECT worker_id FROM serverless_workers WHERE worker_id=%s FOR UPDATE", (native.wid,)
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                run_transaction,
                lambda c: expire_stale_leases(
                    c, grace_sec=0, current_job_id=native.jid, lock_nowait=True
                ),
            )
            with pytest.raises(LockNotAvailable):
                future.result(timeout=3)
    assert native.repo.get_worker(native.wid)["released_at"] is None


@pytest.mark.parametrize("native", [{"billable": True}], indirect=True)
def test_expiry_settles_the_bound_usage_meter_at_the_native_authority_end(native):
    start(native)
    expire(native)
    with native.pool.connection() as conn:
        row = conn.execute(
            "SELECT completed_at FROM usage_meters WHERE attempt_id=%s", (native.r.attempt_id,)
        ).fetchone()
    assert row and row[0] == native.repo.get_worker(native.wid)["released_at"]


def test_concurrent_worker_events_have_distinct_sequences(native):
    start(native)
    job = native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {})
    native.svc.worker_claim_job(native.wid)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda n: native.svc.worker_append_event(
                    native.wid, job["job_id"], "chunk", {"n": n}
                ),
                range(8),
            )
        )
    assert len({r["seq_no"] for r in results}) == 8
    assert len(native.repo.list_stream_events(job["job_id"])) == 8


def test_cancel_and_complete_release_one_slot_without_touching_another_request(native):
    start(native)
    native.repo.patch_endpoint(native.ep["endpoint_id"], native.owner, {"max_concurrency": 2})
    jobs = [native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {}) for _ in range(2)]
    for _ in jobs:
        assert native.svc.worker_claim_job(native.wid)
    jid = jobs[0]["job_id"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        cancel = pool.submit(native.svc.cancel_inflight_job, jid, native.ep["endpoint_id"])
        complete = pool.submit(native.svc.worker_complete_job, native.wid, jid, output={"ok": 1})
        duplicate = pool.submit(native.svc.cancel_inflight_job, jid, native.ep["endpoint_id"])
        assert cancel.result()["status"] in ("COMPLETED", "CANCELLED")
        assert complete.result()["status"] in ("COMPLETED", "CANCELLED")
        assert duplicate.result()["status"] in ("COMPLETED", "CANCELLED")
    assert native.repo.get_worker(native.wid)["current_concurrency"] == 1
    assert native.repo.get_job(jobs[1]["job_id"])["status"] == "IN_PROGRESS"
    assert len(native.repo.list_stream_events(jid)) == 1


def test_queued_failure_cannot_overwrite_claimed_or_finished_work(native):
    start(native)
    job = native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {})
    native.svc.worker_claim_job(native.wid)
    assert (
        native.svc.fail_job_without_worker(job["job_id"], error="stale queue timeout")["status"]
        == "IN_PROGRESS"
    )
    native.svc.worker_complete_job(native.wid, job["job_id"], output={"ok": 1})
    assert (
        native.svc.fail_job_without_worker(job["job_id"], error="late timeout")["status"]
        == "COMPLETED"
    )
    assert len(native.repo.list_stream_events(job["job_id"])) == 1
