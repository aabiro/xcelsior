"""Failure and replica races discovered in the final cutover audit."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db import DatabaseOps
from control_plane.db import run_transaction
from control_plane.host_lifecycle import set_draining
from tests.test_control_plane_reservation import fleet, _mkhost, _pool, _reserve  # noqa: F401
from tests.test_durable_fairness import policies  # noqa: F401
from tests.test_durable_preemption import setup, plan, settle
from tests.test_serverless_native_lifecycle import native  # noqa: F401
from tests._db_helpers import PRICED_GPU, fund_test_owner


def test_native_submission_cannot_commit_without_its_durable_event(native, monkeypatch):
    import scheduler

    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
    emit = mock.Mock()
    monkeypatch.setattr(scheduler, "emit_event", emit)
    with mock.patch("control_plane.outbox.append_event", side_effect=RuntimeError("outbox down")):
        with pytest.raises(RuntimeError, match="outbox"):
            scheduler.submit_job(
                "atomic",
                0,
                gpu_model=PRICED_GPU,
                owner=fund_test_owner(native.owner),
                job_id=native.jid,
            )
    with native.pool.connection() as conn:
        assert conn.execute("SELECT 1 FROM jobs WHERE job_id=%s", (native.jid,)).fetchone() is None
    emit.assert_not_called()


def test_exact_submission_retry_repairs_event_from_admission_not_observations(native):
    import scheduler

    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
    args = dict(
        name="original",
        vram_needed_gb=0,
        gpu_model=PRICED_GPU,
        owner=fund_test_owner(native.owner),
        job_id=native.jid,
    )
    scheduler.submit_job(**args)
    with native.pool.connection() as conn:
        original = conn.execute(
            "SELECT payload FROM outbox_events WHERE aggregate_id=%s AND event_type='job.v1.submitted'",
            (native.jid,),
        ).fetchone()[0]
        conn.execute(
            "DELETE FROM outbox_events WHERE aggregate_id=%s AND event_type='job.v1.submitted'",
            (native.jid,),
        )
        conn.execute(
            "UPDATE jobs SET payload=payload||%s, status='running',phase='running' WHERE job_id=%s",
            (Jsonb(dict(name="mutable observation", tier="urgent")), native.jid),
        )
    for _ in range(2):
        assert scheduler.submit_job(**args)["status"] == "running"
    with native.pool.connection() as conn:
        assert conn.execute(
            "SELECT payload FROM outbox_events WHERE aggregate_id=%s AND event_type='job.v1.submitted'",
            (native.jid,),
        ).fetchall() == [(original,)]


@pytest.mark.parametrize("administrative", ["draining", "disabled"])
def test_stale_host_observation_cannot_reopen_operator_placement(fleet, administrative):
    host = _mkhost(fleet)
    stale = dict(host_id=host, status="active", registered_at=1, free_vram_gb=24)
    drained = run_transaction(lambda c: set_draining(c, host_id=host, draining=True))
    if administrative == "disabled":
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE hosts SET status='disabled',administrative_state='disabled' WHERE host_id=%s",
                (host,),
            )
    with _pool.connection() as conn:
        DatabaseOps.upsert_host(conn, stale, backend="postgres")
        row = conn.execute(
            "SELECT status,administrative_state,version,payload->>'status' FROM hosts WHERE host_id=%s",
            (host,),
        ).fetchone()
        assert row == (administrative, administrative, drained["version"], administrative)


def test_stale_preemption_sweep_cannot_cancel_a_placed_plan(fleet, policies, monkeypatch):
    import control_plane.scheduler.preemption as preemption

    host, active, nominee, _ = setup(fleet, policies)
    pid = plan(nominee, host)
    observed, resume = threading.Event(), threading.Event()

    def pause_first_read(cur):
        make_row = dict_row(cur)

        def row(values):
            result = make_row(values)
            if (
                threading.current_thread().name.startswith("stale-sweep")
                and "nominated_job_id" in result
                and not observed.is_set()
            ):
                observed.set()
                assert resume.wait(10), "concurrent placement did not finish"
            return result

        return row

    monkeypatch.setattr(preemption, "dict_row", pause_first_read)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stale-sweep") as executor:
        stale = executor.submit(run_transaction, lambda c: preemption.reconcile_plan(c, pid))
        try:
            assert observed.wait(10)
            settle(active)
            assert run_transaction(lambda c: preemption.reconcile_plan(c, pid)) == "ready"
            placed = _reserve(nominee, host)
        finally:
            resume.set()
        assert stale.result(timeout=10) == "placed"
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT state FROM preemption_plans WHERE plan_id=%s", (pid,)
        ).fetchone() == ("placed",)
        assert conn.execute(
            "SELECT active_attempt_id::text FROM jobs WHERE job_id=%s", (nominee.job_id,)
        ).fetchone() == (placed.attempt_id,)


@pytest.mark.parametrize("config", [None, {}])
def test_unbound_serverless_submission_cannot_start_unpriced_work(native, config):
    import scheduler

    with pytest.raises(ValueError, match="server-selected binding"):
        scheduler.submit_job("unpriced", 0, job_type="serverless_worker", serverless_config=config)
