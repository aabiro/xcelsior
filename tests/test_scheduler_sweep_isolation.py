"""One bad row in a maintenance sweep must not stop scheduling.

The tick runs its lease/preemption/eviction sweeps before placement. Before
the fix a sweep exception propagated out of ``tick()`` (no placement at all),
and inside the lease sweep one failing row rolled back every other row in its
batch — on every tick, because the batch order is deterministic.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from psycopg.errors import LockNotAvailable

from tests.native_jobs import admit_fixture_job, insert_schedulable_host

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None

from control_plane import leases
from control_plane.db import run_transaction
from control_plane.scheduler import service as service_module
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.reservation import reserve_and_bind
from control_plane.scheduler.service import SchedulerService
from control_plane.sweep_isolation import SweepQuarantine


@pytest.fixture
def fleet():
    ids = {"jobs": [], "hosts": []}
    yield ids
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _job(fleet, model):
    job_id = f"j-sweep-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 phase, desired_state, queued_at, effective_priority)
               VALUES (%s, 'queued', 0, %s, %s, 'pending', 'running', now(), 0)""",
            (job_id, time.time(), json.dumps({"name": job_id, "gpu_model": model})),
        )
        admit_fixture_job(conn, job_id)
        conn.commit()
    fleet["jobs"].append(job_id)
    return job_id


def _host(fleet, model, gpus=1):
    host_id = f"h-sweep-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        insert_schedulable_host(conn, host_id, model=model, gpus=gpus, payload={"gpu_model": model})
        conn.commit()
    fleet["hosts"].append(host_id)
    return host_id


def _offer(job_id, model, host_id):
    def place(conn):
        claimed = claim_next_job(conn, replica_id="sweep-test", scope_gpu_models=[model])
        assert claimed is not None and claimed.job_id == job_id
        return reserve_and_bind(
            conn,
            job_id=job_id,
            claim_token=claimed.claim_token,
            replica_id="sweep-test",
            host_id=host_id,
        )

    return run_transaction(place)


def test_a_failing_sweep_does_not_stop_placement(fleet, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    model = f"RTX 4090 sweep-{marker}"
    host_id = _host(fleet, model)
    job_id = _job(fleet, model.lower())

    def broken_sweep(conn):
        raise RuntimeError("deterministic sweep failure")

    monkeypatch.setattr(service_module, "expire_stale_leases", broken_sweep)
    cfg = SchedulerConfig(
        mode=SchedulerMode.CANARY,
        replica_id=f"sweep-{marker}",
        canary_gpu_models=frozenset({model.lower()}),
        canary_host_ids=frozenset({host_id.lower()}),
    )
    report = SchedulerService(cfg).tick()
    assert report.sweep_errors == 1
    assert [(r.job_id, r.host_id) for r in report.placements] == [(job_id, host_id)]


def test_one_failing_lease_does_not_roll_back_the_rest_of_the_batch(fleet, monkeypatch):
    marker = uuid.uuid4().hex[:8]
    first_model = f"RTX 4090 a-{marker}"
    second_model = f"RTX 4090 b-{marker}"
    stuck = _offer(_job(fleet, first_model.lower()), first_model.lower(), _host(fleet, first_model))
    healthy = _offer(
        _job(fleet, second_model.lower()), second_model.lower(), _host(fleet, second_model)
    )
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET claim_deadline=clock_timestamp()-interval '1 minute' "
            "WHERE lease_id=ANY(%s::uuid[])",
            ([stuck.lease_id, healthy.lease_id],),
        )
        conn.commit()

    from control_plane import billing_effects

    real = billing_effects.serialize_native_period

    def settling(conn, job_id, *, wait=True):
        if job_id == stuck.job_id:
            raise LockNotAvailable("billing period is settling")
        return real(conn, job_id, wait=wait)

    monkeypatch.setattr(billing_effects, "serialize_native_period", settling)
    monkeypatch.setattr(leases, "_EXPIRY_QUARANTINE", SweepQuarantine("lease_expiry"))

    expired = run_transaction(lambda conn: leases.expire_stale_leases(conn, limit=500))
    expired_leases = {e["lease_id"] for e in expired}
    assert healthy.lease_id in expired_leases
    assert stuck.lease_id not in expired_leases
    with _pool.connection() as conn:
        states = dict(
            conn.execute(
                "SELECT lease_id::text,status FROM placement_leases WHERE lease_id=ANY(%s::uuid[])",
                ([stuck.lease_id, healthy.lease_id],),
            ).fetchall()
        )
    assert states == {stuck.lease_id: "offered", healthy.lease_id: "expired"}

    # The failing row backs off instead of failing first on the next tick.
    assert leases._EXPIRY_QUARANTINE.failures(stuck.lease_id) == 1
    assert stuck.lease_id in leases._EXPIRY_QUARANTINE.skipped()
    again = run_transaction(lambda conn: leases.expire_stale_leases(conn, limit=500))
    assert stuck.lease_id not in {e["lease_id"] for e in again}


def test_quarantine_backs_off_exponentially_and_forgets_success():
    now = [100.0]
    quarantine = SweepQuarantine(
        "unit", base_backoff_sec=5, max_backoff_sec=12, clock=lambda: now[0]
    )

    class Conn:
        def transaction(self):
            from contextlib import nullcontext

            return nullcontext()

    def boom():
        raise ValueError("bad row")

    assert quarantine.run(Conn(), "row", boom) == (False, None)
    assert quarantine.skipped() == ["row"]
    now[0] += 5.1
    assert quarantine.skipped() == []
    assert quarantine.run(Conn(), "row", boom) == (False, None)
    now[0] += 9.9
    assert quarantine.skipped() == ["row"]  # 10s after the second failure
    now[0] += 3
    assert quarantine.run(Conn(), "row", lambda: "ok") == (True, "ok")
    assert quarantine.failures("row") == 0
