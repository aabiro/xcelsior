"""Stage-E reservation transaction (§10.5) — atomicity and revalidation.

Real PostgreSQL. Each test builds a small fleet (job + host + devices),
claims through Stage B, then drives reserve_and_bind and asserts either
the complete bound state (attempt + allocations + lease + command +
outbox + job projection, all present) or a clean conflict with zero
residue — never anything in between.
"""

import json
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

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

from control_plane.db import run_transaction
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import (
    CapacityConflict,
    ClaimLost,
    HostNotEligible,
    InventoryChanged,
    JobNotSchedulable,
    reserve_and_bind,
)


@pytest.fixture
def fleet():
    """One pending job + one admitted host with 2 exclusive GPUs."""
    ids = {"jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mkjob(fleet, *, num_gpus=1):
    job_id = f"job-resv-{uuid.uuid4().hex[:10]}"
    # gpu_model doubles as a unique claim scope so these tests never
    # claim residue jobs left behind by other suites (and vice versa).
    payload = {
        "name": job_id, "num_gpus": num_gpus, "image": "pytorch:latest",
        "gpu_model": job_id,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 phase, desired_state, queued_at)
               VALUES (%s, 'queued', 0, %s, %s, 'pending', 'running', now())""",
            (job_id, time.time(), json.dumps(payload)),
        )
        conn.commit()
    fleet["jobs"].append(job_id)
    return job_id


def _mkhost(fleet, *, gpus=2, inventory_generation=1):
    host_id = f"host-resv-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload,
                                  administrative_state, availability_state,
                                  inventory_generation)
               VALUES (%s, 'active', %s, %s, 'admitted', 'ready', %s)""",
            (host_id, time.time(), json.dumps({"admitted": True}), inventory_generation),
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
    fleet["hosts"].append(host_id)
    return host_id


def _claim(job_id):
    with _pool.connection() as conn:
        claimed = claim_next_job(
            conn, replica_id="resv-test", scope_gpu_models=[job_id]
        )
        conn.commit()
    assert claimed is not None
    assert claimed.job_id == job_id
    return claimed


def _reserve(claimed, host_id, **kw):
    return run_transaction(
        lambda conn: reserve_and_bind(
            conn,
            job_id=claimed.job_id,
            claim_token=claimed.claim_token,
            replica_id="resv-test",
            host_id=host_id,
            **kw,
        )
    )


def _counts(job_id):
    with _pool.connection() as conn:
        return {
            "attempts": conn.execute(
                "SELECT count(*) FROM job_attempts WHERE job_id=%s", (job_id,)
            ).fetchone()[0],
            "allocations": conn.execute(
                "SELECT count(*) FROM gpu_device_allocations WHERE job_id=%s",
                (job_id,),
            ).fetchone()[0],
            "leases": conn.execute(
                "SELECT count(*) FROM placement_leases WHERE job_id=%s", (job_id,)
            ).fetchone()[0],
            "commands": conn.execute(
                "SELECT count(*) FROM agent_commands WHERE job_id=%s", (job_id,)
            ).fetchone()[0],
            "outbox": conn.execute(
                "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s",
                (job_id,),
            ).fetchone()[0],
        }


class TestHappyPath:
    def test_reservation_binds_everything_atomically(self, fleet):
        job_id = _mkjob(fleet)
        host_id = _mkhost(fleet)
        claimed = _claim(job_id)
        resv = _reserve(claimed, host_id)

        assert resv.fencing_token > 0
        assert len(resv.gpu_device_ids) == 1
        assert resv.attempt_number == 1
        assert resv.spec_hash.startswith("sha256:")

        counts = _counts(job_id)
        assert counts == {
            "attempts": 1, "allocations": 1, "leases": 1,
            "commands": 1, "outbox": 2,
        }
        with _pool.connection() as conn:
            job = conn.execute(
                """SELECT phase, status, host_id, active_attempt_id,
                          schedule_claim_token FROM jobs WHERE job_id=%s""",
                (job_id,),
            ).fetchone()
            lease = conn.execute(
                "SELECT status, fencing_token FROM placement_leases WHERE job_id=%s",
                (job_id,),
            ).fetchone()
            cmd = conn.execute(
                "SELECT command, status, idempotency_key, args FROM agent_commands "
                "WHERE job_id=%s",
                (job_id,),
            ).fetchone()
        assert job[0] == "scheduled" and job[1] == "assigned"
        assert job[2] == host_id and str(job[3]) == resv.attempt_id
        assert job[4] is None  # claim consumed
        assert lease[0] == "offered" and lease[1] == resv.fencing_token
        assert cmd[0] == "start_attempt" and cmd[1] == "pending"
        assert cmd[2] == f"start:{resv.attempt_id}"
        assert cmd[3]["fencing_token"] == resv.fencing_token

    def test_multi_gpu_reserves_all_devices(self, fleet):
        job_id = _mkjob(fleet, num_gpus=2)
        host_id = _mkhost(fleet, gpus=2)
        claimed = _claim(job_id)
        resv = _reserve(claimed, host_id, num_gpus=2)
        assert len(resv.gpu_device_ids) == 2
        assert _counts(job_id)["allocations"] == 2


class TestConflictsLeaveNoResidue:
    def test_wrong_claim_token(self, fleet):
        job_id = _mkjob(fleet)
        host_id = _mkhost(fleet)
        claimed = _claim(job_id)
        bogus = type(claimed)(**{**claimed.__dict__, "claim_token": str(uuid.uuid4())})
        with pytest.raises(ClaimLost):
            _reserve(bogus, host_id)
        assert _counts(job_id) == {
            "attempts": 0, "allocations": 0, "leases": 0, "commands": 0, "outbox": 0,
        }

    def test_job_no_longer_pending(self, fleet):
        job_id = _mkjob(fleet)
        host_id = _mkhost(fleet)
        claimed = _claim(job_id)
        with _pool.connection() as conn:
            # Cancel via status — the 059 trigger projects phase/desired
            # ('stopped'/'stopped'), exactly how production writers stop
            # a job out from under a claim.
            conn.execute(
                "UPDATE jobs SET status='cancelled' WHERE job_id=%s",
                (job_id,),
            )
            conn.commit()
        with pytest.raises(JobNotSchedulable):
            _reserve(claimed, host_id)
        assert _counts(job_id)["attempts"] == 0

    def test_host_started_draining(self, fleet):
        job_id = _mkjob(fleet)
        host_id = _mkhost(fleet)
        claimed = _claim(job_id)
        with _pool.connection() as conn:
            # Drain the legacy way (payload flag): the 059 projection
            # trigger derives administrative_state='draining' from it —
            # writing the column directly would just be recomputed.
            conn.execute(
                """UPDATE hosts
                      SET payload = jsonb_set(payload, '{draining}', 'true')
                    WHERE host_id=%s""",
                (host_id,),
            )
            conn.commit()
        with pytest.raises(HostNotEligible):
            _reserve(claimed, host_id)
        assert _counts(job_id)["attempts"] == 0

    def test_inventory_generation_moved(self, fleet):
        job_id = _mkjob(fleet)
        host_id = _mkhost(fleet, inventory_generation=1)
        claimed = _claim(job_id)
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE hosts SET inventory_generation=2 WHERE host_id=%s",
                (host_id,),
            )
            conn.commit()
        with pytest.raises(InventoryChanged):
            _reserve(claimed, host_id, expected_inventory_generation=1)
        assert _counts(job_id)["attempts"] == 0

    def test_multi_gpu_all_or_nothing(self, fleet):
        """2-GPU request on a host with 1 free GPU inserts zero rows."""
        blocker_job = _mkjob(fleet)
        job_id = _mkjob(fleet, num_gpus=2)
        host_id = _mkhost(fleet, gpus=2)
        # Blocker takes one GPU first.
        _reserve(_claim(blocker_job), host_id)
        claimed = _claim(job_id)
        with pytest.raises(CapacityConflict):
            _reserve(claimed, host_id, num_gpus=2)
        assert _counts(job_id) == {
            "attempts": 0, "allocations": 0, "leases": 0, "commands": 0, "outbox": 0,
        }

    def test_rival_reservation_wins_race(self, fleet):
        """Two jobs, one single-GPU host: exactly one reservation lands."""
        job_a = _mkjob(fleet)
        job_b = _mkjob(fleet)
        host_id = _mkhost(fleet, gpus=1)
        claimed_a = _claim(job_a)
        claimed_b = _claim(job_b)
        assert {claimed_a.job_id, claimed_b.job_id} == {job_a, job_b}
        _reserve(claimed_a, host_id)
        with pytest.raises(CapacityConflict):
            _reserve(claimed_b, host_id)
        with _pool.connection() as conn:
            active = conn.execute(
                """SELECT count(*) FROM gpu_device_allocations a
                     JOIN host_gpu_devices d ON d.gpu_device_id=a.gpu_device_id
                    WHERE d.host_id=%s AND a.status='active'""",
                (host_id,),
            ).fetchone()[0]
        assert active == 1

    def test_reserved_job_cannot_be_reserved_again(self, fleet):
        job_id = _mkjob(fleet)
        host_id = _mkhost(fleet)
        claimed = _claim(job_id)
        _reserve(claimed, host_id)
        # Same (now consumed) claim: the job is scheduled, claim gone.
        with pytest.raises((ClaimLost, JobNotSchedulable)):
            _reserve(claimed, host_id)
        assert _counts(job_id)["attempts"] == 1
