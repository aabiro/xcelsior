"""§26.2 — multi-process scheduler concurrency stress test.

Eight independent OS processes (real replicas: separate connections,
separate transactions) race to claim and reserve 30 jobs over a fleet
with only 8 exclusive GPU slots. The invariants under test are the
Track A hard SLOs (§25.4):

- zero devices with more than one active allocation;
- zero jobs with more than one active attempt;
- every scheduled job has exactly one attempt, lease, start command;
- exactly as many placements as the fleet has slots.
"""

import json
import multiprocessing as mp
import time
import uuid

import pytest

try:
    from db import _get_pg_pool, resolve_postgres_dsn

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _migrated = (
            _c.execute("SELECT to_regclass('outbox_events')").fetchone()[0] is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 056")

from tests._concurrency_worker import run_replica

N_REPLICAS = 8
N_JOBS = 30
N_HOSTS = 4
GPUS_PER_HOST = 2
TOTAL_SLOTS = N_HOSTS * GPUS_PER_HOST


@pytest.fixture
def stress_fleet():
    marker = uuid.uuid4().hex[:8]
    job_ids = [f"job-{marker}-{i:03d}" for i in range(N_JOBS)]
    host_ids = [f"host-{marker}-{i}" for i in range(N_HOSTS)]
    with _pool.connection() as conn:
        for host_id in host_ids:
            conn.execute(
                """INSERT INTO hosts (host_id, status, registered_at, payload,
                                      administrative_state, availability_state)
                   VALUES (%s, 'active', %s, '{}', 'admitted', 'ready')""",
                (host_id, time.time()),
            )
            for g in range(GPUS_PER_HOST):
                conn.execute(
                    """INSERT INTO host_gpu_devices
                           (host_id, gpu_uuid, device_index, model,
                            total_vram_mb, allocatable_vram_mb, health)
                       VALUES (%s, %s, %s, 'RTX 4090', 24576, 24576, 'healthy')""",
                    (host_id, f"GPU-{host_id}-{g}", g),
                )
        for i, job_id in enumerate(job_ids):
            conn.execute(
                """INSERT INTO jobs (job_id, status, priority, submitted_at,
                                     payload, phase, desired_state,
                                     effective_priority, queued_at)
                   VALUES (%s, 'queued', %s, %s, %s, 'pending', 'running', %s,
                           now())""",
                (job_id, i % 3, time.time(), json.dumps({"name": job_id}), i % 3),
            )
        conn.commit()
    yield marker, job_ids, host_ids
    with _pool.connection() as conn:
        for job_id in job_ids:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (job_id,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        for host_id in host_ids:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
        conn.commit()


class TestSchedulerConcurrency:
    def test_replicas_never_double_allocate(self, stress_fleet):
        marker, job_ids, host_ids = stress_fleet
        dsn = resolve_postgres_dsn()

        ctx = mp.get_context("spawn")
        with ctx.Pool(N_REPLICAS) as pool:
            results = pool.starmap(
                run_replica,
                [
                    (dsn, f"stress-replica-{i}", host_ids, marker)
                    for i in range(N_REPLICAS)
                ],
            )

        total_reserved = sum(r["reserved"] for r in results)
        assert total_reserved == TOTAL_SLOTS, results

        with _pool.connection() as conn:
            # Invariant: no device double-booked (would also be impossible
            # to insert thanks to uq_gpu_one_exclusive_allocation — this
            # asserts the code never even tried to lean on the constraint).
            double_booked = conn.execute(
                """SELECT a.gpu_device_id, count(*)
                     FROM gpu_device_allocations a
                     JOIN host_gpu_devices d ON d.gpu_device_id = a.gpu_device_id
                    WHERE d.host_id = ANY(%s) AND a.status = 'active'
                    GROUP BY a.gpu_device_id HAVING count(*) > 1""",
                (host_ids,),
            ).fetchall()
            assert double_booked == []

            multi_attempt = conn.execute(
                """SELECT job_id, count(*) FROM job_attempts
                    WHERE job_id = ANY(%s)
                    GROUP BY job_id HAVING count(*) > 1""",
                (job_ids,),
            ).fetchall()
            assert multi_attempt == []

            scheduled = conn.execute(
                """SELECT count(*) FROM jobs
                    WHERE job_id = ANY(%s) AND phase = 'scheduled'""",
                (job_ids,),
            ).fetchone()[0]
            assert scheduled == TOTAL_SLOTS

            # Every scheduled job carries its full bound chain.
            incomplete = conn.execute(
                """
                SELECT j.job_id
                  FROM jobs j
                 WHERE j.job_id = ANY(%s) AND j.phase = 'scheduled'
                   AND (
                       (SELECT count(*) FROM job_attempts a
                         WHERE a.job_id = j.job_id AND a.status = 'reserved') <> 1
                    OR (SELECT count(*) FROM placement_leases l
                         WHERE l.job_id = j.job_id AND l.status = 'offered') <> 1
                    OR (SELECT count(*) FROM agent_commands c
                         WHERE c.job_id = j.job_id AND c.status = 'pending') <> 1
                   )
                """,
                (job_ids,),
            ).fetchall()
            assert incomplete == []

            # Unplaced jobs stayed pending with a durable queue reason.
            unplaced = conn.execute(
                """SELECT count(*) FROM jobs
                    WHERE job_id = ANY(%s) AND phase = 'pending'
                      AND reason_code = 'no_capacity'""",
                (job_ids,),
            ).fetchone()[0]
            assert unplaced == N_JOBS - TOTAL_SLOTS
