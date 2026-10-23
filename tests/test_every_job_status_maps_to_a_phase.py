"""The real persistence writer gives every compatibility status a native phase.

The independent expected mapping protects phase-based consumers as writers
take over responsibility from the transitional database trigger. The separate
native-writer rehearsal runs these paths with both projection functions gone.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from db import DatabaseOps, _get_pg_pool

# Statuses written to `jobs` by `update_job_status` calls and by direct
# `UPDATE jobs SET status = ...` statements across the codebase.
WRITTEN_STATUSES = (
    "queued",
    "assigned",
    "leased",
    "running",
    "restarting",
    "stopping",
    "stopped",
    "completed",
    "failed",
    "cancelled",
)

# The phase each one must collapse to. Written out rather than derived, so that
# changing the trigger's mapping has to be a deliberate edit here too.
EXPECTED_PHASE = {
    "queued": "pending",
    "assigned": "scheduled",
    "leased": "scheduled",
    "running": "running",
    "restarting": "starting",
    "stopping": "running",
    "stopped": "stopped",
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "stopped",
}


@pytest.fixture
def inserted():
    made: list[str] = []
    yield made
    if made:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ANY(%s)", (made,))
            conn.commit()


@pytest.mark.parametrize("status", WRITTEN_STATUSES)
def test_the_writer_gives_every_written_status_a_phase(status, inserted):
    job_id = f"j-phase-{uuid.uuid4().hex[:8]}"
    with _get_pg_pool().connection() as conn:
        DatabaseOps.upsert_job(
            conn,
            dict(job_id=job_id, status=status, priority=0, submitted_at=1),
            backend="postgres",
        )
        inserted.append(job_id)
        conn.commit()
        phase = conn.execute("SELECT phase FROM jobs WHERE job_id = %s", (job_id,)).fetchone()[0]

    assert (
        phase is not None
    ), f"status {status!r} has no native phase and is invisible to phase-based queries"
    assert (
        phase == EXPECTED_PHASE[status]
    ), f"status {status!r} now maps to phase {phase!r}, not {EXPECTED_PHASE[status]!r}"
