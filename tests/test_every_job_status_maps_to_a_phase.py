"""Every job status the code writes must map to a phase, not to NULL.

`jobs.phase` is not written by the application. A BEFORE INSERT/UPDATE trigger,
`control_plane_project_job`, derives it from `status` through a CASE whose last
arm is `ELSE NULL`. So a status the CASE does not list does not fail loudly —
the row lands with `phase = NULL` and drops out of *every* phase-based query at
once: the control plane's own reads, the admin dashboard's job failure rate, and
anything downstream that asks what a job's outcome was.

The mapping is also not one-to-one, which is the reason the column exists:
`completed` becomes `succeeded`, and `cancelled`, `stopped`, `paused` and
`terminated` all collapse to `stopped`. Any consumer counting raw `status`
strings has to reproduce that by hand, and will drift from it.

This inserts one row per status the code actually writes and asserts the trigger
produced a phase. Adding a new status to `update_job_status` without extending
the CASE fails here, at the migration that should have carried it, rather than
as a number that quietly stops adding up.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from db import _get_pg_pool

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
def test_the_trigger_gives_every_written_status_a_phase(status, inserted):
    job_id = f"j-phase-{uuid.uuid4().hex[:8]}"
    with _get_pg_pool().connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload)
               VALUES (%s, %s, 0, 0, %s)""",
            (job_id, status, json.dumps({"job_id": job_id})),
        )
        inserted.append(job_id)
        conn.commit()
        phase = conn.execute("SELECT phase FROM jobs WHERE job_id = %s", (job_id,)).fetchone()[0]

    assert phase is not None, (
        f"status {status!r} fell through the CASE in control_plane_project_job to "
        "ELSE NULL — jobs in this state are invisible to every phase-based query"
    )
    assert phase == EXPECTED_PHASE[status], (
        f"status {status!r} now maps to phase {phase!r}, not {EXPECTED_PHASE[status]!r}"
    )
