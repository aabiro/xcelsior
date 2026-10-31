"""A lost attempt is billed until its lease expired, not until the sweep noticed.

Periodic billing stops at the lease's `expires_at`: past it the host has proved
nothing. The lease sweep records `ended_at` when it runs — at least the grace
period later, and arbitrarily later if the sweep is behind — and the final
slice, the meter and serverless execution all billed to that `ended_at`. The
customer paid for time after the host went silent.

For `lost` and `fenced` attempts every billing bound is now
`min(ended_at, lease expires_at)`.
"""

from __future__ import annotations

import uuid

import pytest

from billing import get_billing_engine
from control_plane.db import run_transaction
from control_plane.leases import expire_stale_leases
from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance

#: How long the host had been silent when the sweep finally ran.
SILENT_SEC = 1800


def _expire_lease_in_the_past(job_id: str) -> float:
    with _get_pg_pool().connection() as conn:
        expiry = conn.execute(
            "UPDATE placement_leases SET expires_at=clock_timestamp()-make_interval(secs=>%s) "
            "WHERE job_id=%s AND status='active' "
            "RETURNING EXTRACT(EPOCH FROM expires_at)::float8",
            (SILENT_SEC, job_id),
        ).fetchone()[0]
    expired = run_transaction(
        lambda conn: expire_stale_leases(conn, grace_sec=30, current_job_id=job_id)
    )
    assert expired and expired[0]["job_id"] == job_id
    return expiry


def test_a_lost_gpu_attempt_meter_and_final_slice_stop_at_lease_expiry():
    owner = f"lost-gpu-{uuid.uuid4().hex[:10]}"
    engine = get_billing_engine()
    engine.deposit(owner, 100)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=3600) as native:
        job_id = native["job_id"]
        with _get_pg_pool().connection() as conn:
            started = conn.execute(
                "SELECT EXTRACT(EPOCH FROM started_at)::float8 FROM job_attempts WHERE job_id=%s",
                (job_id,),
            ).fetchone()[0]
            conn.execute("UPDATE usage_meters SET started_at=%s WHERE job_id=%s", (started, job_id))
        expiry = _expire_lease_in_the_past(job_id)
        with _get_pg_pool().connection() as conn:
            status, ended = conn.execute(
                "SELECT status,EXTRACT(EPOCH FROM ended_at)::float8 FROM job_attempts "
                "WHERE job_id=%s",
                (job_id,),
            ).fetchone()
            assert status == "lost" and ended - expiry >= SILENT_SEC - 5
            completed, meter_cost = conn.execute(
                "SELECT completed_at,total_cost_micros FROM usage_meters WHERE job_id=%s",
                (job_id,),
            ).fetchone()
        assert completed == pytest.approx(expiry, abs=1e-6), "the meter closes at lease expiry"
        # 0.001 CAD/s for the ~1800 s the lease covered, not the 3600 s to the sweep.
        assert meter_cost == pytest.approx(round((expiry - started) * 0.001, 4) * 1e6)

        before = engine.get_wallet(owner)["balance_micros"]
        assert engine.auto_billing_cycle()["errors"] == 0
        charged = before - engine.get_wallet(owner)["balance_micros"]
        assert charged == meter_cost
        with _get_pg_pool().connection() as conn:
            final_end = conn.execute(
                "SELECT max(period_end) FROM billing_cycles WHERE job_id=%s AND status='charged'",
                (job_id,),
            ).fetchone()[0]
        assert final_end == pytest.approx(expiry, abs=1e-6)

        # Settled: the attempt does not stay pending and is not billed again.
        assert engine.auto_billing_cycle()["errors"] == 0
        assert before - engine.get_wallet(owner)["balance_micros"] == charged
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT count(*) FROM billing_cycles WHERE job_id=%s", (job_id,)
            ).fetchone() == (1,)


def test_a_lost_serverless_attempt_bills_execution_only_to_lease_expiry():
    from serverless.metering import bill_active_serverless_workers
    from serverless.repo import EndpointCreate, ServerlessRepo
    from tests.fixtures.native_serverless import running_serverless

    owner = f"lost-slvr-{uuid.uuid4().hex[:10]}"
    engine = get_billing_engine()
    engine.deposit(owner, 100)
    repo = ServerlessRepo()
    endpoint = repo.create_endpoint(
        EndpointCreate(owner_id=owner, name="lost-slvr", gpu_tier="RTX 4090", min_workers=0)
    )
    try:
        with running_serverless(repo, endpoint["endpoint_id"], age_sec=3600) as worker:
            job_id = worker["scheduler_job_id"]
            expiry = _expire_lease_in_the_past(job_id)
            released = repo.get_worker(worker["worker_id"])["released_at"]
            assert released - expiry >= SILENT_SEC - 5
            bill_active_serverless_workers(engine)
            with _get_pg_pool().connection() as conn:
                final_end = conn.execute(
                    "SELECT max(period_end) FROM billing_cycles WHERE job_id=%s "
                    "AND resource_type='serverless_gpu' AND status='charged'",
                    (job_id,),
                ).fetchone()[0]
            assert final_end == pytest.approx(
                expiry, abs=1e-6
            ), "serverless execution was billed past the lease expiry"
            before = engine.get_wallet(owner)["balance_micros"]
            bill_active_serverless_workers(engine)
            assert engine.get_wallet(owner)["balance_micros"] == before
            with _get_pg_pool().connection() as conn:
                assert conn.execute(
                    "SELECT count(*) FROM billing_cycles WHERE job_id=%s "
                    "AND resource_type='serverless_gpu'",
                    (job_id,),
                ).fetchone() == (1,)
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "DELETE FROM serverless_endpoints WHERE endpoint_id=%s", (endpoint["endpoint_id"],)
            )
