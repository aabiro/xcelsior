"""A job's priority, submit time, pricing mode and hold are read from their columns.

The queue orders jobs by the `priority` and `submitted_at` columns and
settlement charges by `pricing_mode`, but every reader handed callers the
payload's copies of the same three fields. The copies were kept in step by
each writer, so a writer that touched only the column (an operator raising a
priority in SQL, a migration) left a job that the queue ranked one way and the
scheduler, the API and the next save another. The next `upsert_job` then wrote
the stale copy back over the column.

Every reader now lays the columns over the payload, so the column is the one
value; the copies are written only for the release before 171 and are
stripped in the next contract step.
"""

from __future__ import annotations

import uuid

import pytest
from psycopg.types.json import Jsonb

from db import DatabaseOps, _get_pg_pool

pytestmark = pytest.mark.needs_db

#: What a stale payload copy says, against what the columns hold.
STALE = {"priority": 0, "submitted_at": 5.0, "pricing_mode": "on_demand"}
COLUMNS = {"priority": 7, "submitted_at": 100.0, "pricing_mode": "spot"}


@pytest.fixture
def drifted_job():
    job_id = "colsrc-" + uuid.uuid4().hex[:12]
    with _get_pg_pool().connection() as conn:
        DatabaseOps.upsert_job(
            conn,
            {"job_id": job_id, "status": "queued", "owner": "colsrc-owner", **COLUMNS},
            backend="postgres",
        )
        conn.execute(
            "UPDATE jobs SET payload = payload || %s WHERE job_id = %s", (Jsonb(STALE), job_id)
        )
    try:
        yield job_id
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


def _fields(job: dict) -> dict:
    return {key: job.get(key) for key in COLUMNS}


def test_every_reader_returns_the_columns(drifted_job):
    import scheduler

    with _get_pg_pool().connection() as conn:
        one = DatabaseOps.get_job(conn, drifted_job, backend="postgres")
        listed = [
            j
            for j in DatabaseOps.load_jobs(conn, status="queued", backend="postgres")
            if j["job_id"] == drifted_job
        ]

    with scheduler._db_connection() as conn:
        loaded = [j for j in scheduler._load_jobs_from_conn(conn) if j["job_id"] == drifted_job]

    assert _fields(one) == COLUMNS
    assert [_fields(j) for j in listed] == [COLUMNS]
    assert [_fields(j) for j in loaded] == [COLUMNS], "the scheduler's own loader"


def test_saving_a_job_keeps_its_columns(drifted_job):
    """A read-modify-write must not carry the stale copy back into the column."""
    with _get_pg_pool().connection() as conn:
        job = DatabaseOps.get_job(conn, drifted_job, backend="postgres")
        job["queue_reason"] = "waiting"
        DatabaseOps.upsert_job(conn, job, backend="postgres")
        row = conn.execute(
            "SELECT priority, submitted_at, pricing_mode FROM jobs WHERE job_id = %s",
            (drifted_job,),
        ).fetchone()
    assert dict(zip(COLUMNS, row)) == COLUMNS


def test_the_next_job_is_the_columns_job(drifted_job):
    import scheduler

    with _get_pg_pool().connection() as conn:
        # First in the queue whatever else the shared test database holds.
        top = conn.execute(
            "UPDATE jobs SET priority = (SELECT max(priority) + 1 FROM jobs "
            "WHERE status = 'queued') WHERE job_id = %s RETURNING priority",
            (drifted_job,),
        ).fetchone()[0]
    nxt = scheduler.get_next_job()
    assert nxt is not None and nxt["job_id"] == drifted_job
    assert _fields(nxt) == {**COLUMNS, "priority": top}
