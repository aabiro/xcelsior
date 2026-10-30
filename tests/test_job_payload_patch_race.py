"""Rename/lock/unlock patches cannot undo a concurrent placement.

``_set_job_fields`` (behind instance rename, lock and unlock) used to read the
job without a lock, merge the patch, and write every column back through
``upsert_job``'s DO UPDATE — status, host_id, priority, submitted_at and the
whole payload. A placement committed between the read and the write was
silently reverted to the stale read.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

import scheduler
from tests._db_helpers import PRICED_GPU, fund_test_owner

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
    _native = scheduler._active_backend() == "postgres"
except Exception:  # pragma: no cover - skip path
    _pool, _native = None, False

pytestmark = pytest.mark.skipif(not _native, reason="native job rows need PostgreSQL")


@pytest.fixture
def job_id():
    job = scheduler.submit_job(
        name=f"patch-race-{uuid.uuid4().hex[:8]}",
        vram_needed_gb=1,
        gpu_model=PRICED_GPU,
        owner=fund_test_owner("patch-race-owner"),
    )
    yield job["job_id"]
    with _pool.connection() as conn:
        conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (job["job_id"],))
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (job["job_id"],))
        conn.commit()


def _row(job_id):
    with _pool.connection() as conn:
        return conn.execute(
            "SELECT status, host_id, priority, payload->>'name', payload->>'locked' "
            "FROM jobs WHERE job_id=%s",
            (job_id,),
        ).fetchone()


def test_a_rename_racing_a_placement_keeps_the_placement(job_id):
    import psycopg
    from db import resolve_postgres_dsn

    placer = psycopg.connect(resolve_postgres_dsn())
    try:
        placer.execute(
            "UPDATE jobs SET status='assigned', phase='scheduled', host_id='race-host', "
            "priority=7 WHERE job_id=%s",
            (job_id,),
        )
        renamed = threading.Thread(target=lambda: scheduler._set_job_fields(job_id, name="renamed"))
        renamed.start()
        # The rename has now read (or is blocked on) the row the placement holds.
        time.sleep(0.5)
        placer.commit()
        renamed.join(timeout=10)
        assert not renamed.is_alive()
    finally:
        placer.close()
    assert tuple(_row(job_id)) == ("assigned", "race-host", 7, "renamed", None)


def test_lock_and_unlock_only_touch_their_payload_key(job_id):
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='running', phase='running', host_id='lock-host' "
            "WHERE job_id=%s",
            (job_id,),
        )
        conn.commit()
    locked = scheduler._set_job_fields(job_id, locked=True)
    assert locked["locked"] is True and locked["status"] == "running"
    scheduler._set_job_fields(job_id, locked=False)
    status, host_id, _priority, _name, flag = _row(job_id)
    assert (status, host_id, flag) == ("running", "lock-host", "false")


@pytest.mark.parametrize(
    "field", ["status", "host_id", "priority", "submitted_at", "owner", "spec", "team_id"]
)
def test_a_payload_patch_refuses_authority_fields(job_id, field):
    before = _row(job_id)
    with pytest.raises(ValueError, match="authority"):
        scheduler._set_job_fields(job_id, **{field: "forged"})
    assert _row(job_id) == before


def test_patching_a_missing_job_returns_none():
    assert scheduler._set_job_fields("no-such-job-" + uuid.uuid4().hex, name="x") is None
