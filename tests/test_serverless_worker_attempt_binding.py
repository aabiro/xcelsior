"""Track B B3.1 — a serverless worker binds to its fenced attempt (§15.3/§5.6).

When the scheduler reserves a fenced attempt for a serverless_worker job, the
worker row must point at that attempt — so serverless capacity is attempt-scoped
exactly like compute metering: one worker → one fenced attempt → one allocation
set. This drives the real reservation authority (`reserve_and_bind`) end to end.
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
        _has = (
            _c.execute("SELECT to_regclass('serverless_workers')").fetchone()[0] is not None
            and _c.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name='serverless_workers' AND column_name='attempt_id'"
            ).fetchone()
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("serverless_workers.attempt_id missing — upgrade to >= 070")

from control_plane.db import run_transaction
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import reserve_and_bind
from serverless.repo import EndpointCreate, ServerlessRepo


@pytest.fixture
def scratch():
    ids = {"jobs": [], "hosts": [], "workers": [], "endpoints": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM gpu_device_allocations WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM placement_leases WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
        for wid in ids["workers"]:
            conn.execute("DELETE FROM serverless_workers WHERE worker_id=%s", (wid,))
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM host_gpu_devices WHERE host_id=%s", (hid,))
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        for eid in ids["endpoints"]:
            conn.execute("DELETE FROM serverless_endpoints WHERE endpoint_id=%s", (eid,))
        conn.commit()


def _mkhost(scratch, *, gpus=1):
    host_id = f"host-swk-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload,
                                  administrative_state, availability_state, inventory_generation)
               VALUES (%s, 'active', %s, %s, 'admitted', 'ready', 1)""",
            (host_id, time.time(), json.dumps({"admitted": True})),
        )
        for i in range(gpus):
            conn.execute(
                """INSERT INTO host_gpu_devices
                       (host_id, gpu_uuid, device_index, model,
                        total_vram_mb, allocatable_vram_mb, health)
                   VALUES (%s, %s, %s, 'RTX 4090', 24576, 24576, 'healthy')""",
                (host_id, f"GPU-{host_id}-{i}", i),
            )
        conn.commit()
    scratch["hosts"].append(host_id)
    return host_id


def _mk_serverless_job(scratch, worker_id: str, *, num_gpus=1):
    """A serverless_worker job that carries its worker id in the payload, as
    provision_worker's _set_job_fields does."""
    job_id = f"job-swk-{uuid.uuid4().hex[:10]}"
    payload = {
        "name": job_id,
        "num_gpus": num_gpus,
        "job_type": "serverless_worker",
        "serverless_worker_id": worker_id,
        "gpu_model": job_id,  # unique claim scope so we never claim foreign residue
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 phase, desired_state, queued_at)
               VALUES (%s, 'queued', 0, %s, %s, 'pending', 'running', now())""",
            (job_id, time.time(), json.dumps(payload)),
        )
        conn.commit()
    scratch["jobs"].append(job_id)
    return job_id


def _worker_row(worker_id: str) -> dict | None:
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT attempt_id, host_id FROM serverless_workers WHERE worker_id=%s",
            (worker_id,),
        ).fetchone()
    return {"attempt_id": row[0], "host_id": row[1]} if row else None


def test_scale_up_binds_worker_to_exactly_one_fenced_attempt(scratch):
    repo = ServerlessRepo()
    ep = repo.create_endpoint(
        EndpointCreate(owner_id=f"own-{uuid.uuid4().hex[:8]}", name="b31", mode="preset", model_ref="m", min_workers=0)
    )
    scratch["endpoints"].append(ep["endpoint_id"])

    # A worker + its serverless_worker job (as provision_worker wires them).
    job_id = None
    worker = repo.create_worker(ep["endpoint_id"], gpu_count=1)
    worker_id = worker["worker_id"]
    scratch["workers"].append(worker_id)
    job_id = _mk_serverless_job(scratch, worker_id, num_gpus=1)
    host_id = _mkhost(scratch, gpus=1)

    # Before placement the worker is unbound.
    assert _worker_row(worker_id)["attempt_id"] is None

    # Claim + reserve the fenced attempt (the scheduler's real path).
    with _pool.connection() as conn:
        claimed = claim_next_job(conn, replica_id="b31-test", scope_gpu_models=[job_id])
        conn.commit()
    assert claimed is not None and claimed.job_id == job_id
    reservation = run_transaction(
        lambda conn: reserve_and_bind(
            conn,
            job_id=job_id,
            claim_token=claimed.claim_token,
            replica_id="b31-test",
            host_id=host_id,
            num_gpus=1,
        )
    )
    attempt_id = str(reservation.attempt_id)

    # The worker now points at exactly that fenced attempt, on that host.
    bound = _worker_row(worker_id)
    assert str(bound["attempt_id"]) == attempt_id
    assert bound["host_id"] == host_id

    # Exactly one attempt and its allocation set exist for the job.
    with _pool.connection() as conn:
        n_attempts = conn.execute(
            "SELECT count(*) FROM job_attempts WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
        n_allocs = conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s", (attempt_id,)
        ).fetchone()[0]
        n_bound = conn.execute(
            "SELECT count(*) FROM serverless_workers WHERE attempt_id=%s", (attempt_id,)
        ).fetchone()[0]
    assert n_attempts == 1
    assert n_allocs == 1
    assert n_bound == 1  # one worker bound to the attempt, closed once when fenced


def test_non_serverless_job_leaves_workers_untouched(scratch):
    """A normal job reserving an attempt must not touch serverless_workers."""
    # Plain job: payload carries no serverless_worker_id.
    plain = f"job-plain-{uuid.uuid4().hex[:10]}"
    payload = {"name": plain, "num_gpus": 1, "gpu_model": plain}
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 phase, desired_state, queued_at)
               VALUES (%s, 'queued', 0, %s, %s, 'pending', 'running', now())""",
            (plain, time.time(), json.dumps(payload)),
        )
        conn.commit()
    scratch["jobs"].append(plain)
    host_id = _mkhost(scratch, gpus=1)

    with _pool.connection() as conn:
        claimed = claim_next_job(conn, replica_id="b31-test", scope_gpu_models=[plain])
        conn.commit()
    reservation = run_transaction(
        lambda conn: reserve_and_bind(
            conn, job_id=plain, claim_token=claimed.claim_token,
            replica_id="b31-test", host_id=host_id, num_gpus=1,
        )
    )
    with _pool.connection() as conn:
        n = conn.execute(
            "SELECT count(*) FROM serverless_workers WHERE attempt_id=%s",
            (str(reservation.attempt_id),),
        ).fetchone()[0]
    assert n == 0
