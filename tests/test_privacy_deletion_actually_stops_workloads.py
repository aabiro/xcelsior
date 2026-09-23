"""A deletion request must actually stop the subject's running workloads.

`jobs.desired_state` is not a writable column. `control_plane_project_job` is a
BEFORE INSERT OR UPDATE trigger that recomputes it from `status` on every write:

    NEW.desired_state := CASE
        WHEN NEW.status IN ('stopping','stopped','paused','cancelled','terminated')
            THEN 'stopped' ELSE 'running' END

`_stop_subject_workloads` set `desired_state = 'stopped'` and left `status`
alone. The trigger then read the unchanged `status` — `'running'` — and wrote
`desired_state` straight back to `'running'`. The UPDATE reported a rowcount, and
`reason_code = 'privacy_deletion_requested'` did persist, so the request looked
honoured from every angle except the one the control plane acts on.

Nothing stopped. `delete_authoritative_subject` then returns `"pending"` with
`retry_after_sec=30` and "waiting for active workloads to stop", and retries on
that cadence until the job ends by itself — which for GPU training is days,
against a statutory deadline.

The fix follows what the rest of the control plane already does: write `status`
and let the trigger derive the projection. `control_plane/leases.py` and
`scheduler/reservation.py` both set `phase` *alongside* the `status` that
produces it, so their redundant write lands on the same value. This was the one
site that wrote a derived column without it.

`status = 'stopping'` is the correct state: the trigger maps it to
`phase = 'running'` (it has not stopped yet) and `desired_state = 'stopped'`
(it should), which is exactly the reconciler's cue.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from db import _get_pg_pool
from privacy_sinks import _stop_subject_workloads


@pytest.fixture
def running_job_for_subject():
    customer_id = f"cust-priv-{uuid.uuid4().hex[:8]}"
    job_id = f"j-priv-{uuid.uuid4().hex[:8]}"
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, owner_id, payload)
               VALUES (%s, 'running', 0, %s, %s, %s)""",
            (job_id, time.time(), customer_id, json.dumps({"job_id": job_id})),
        )
        conn.commit()
    yield job_id, customer_id
    with pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
        conn.commit()


def _state(job_id: str) -> tuple:
    with _get_pg_pool().connection() as conn:
        return conn.execute(
            "SELECT status, phase, desired_state, reason_code FROM jobs WHERE job_id = %s",
            (job_id,),
        ).fetchone()


def test_the_stop_survives_the_projection_trigger(running_job_for_subject):
    job_id, customer_id = running_job_for_subject
    assert _state(job_id)[2] == "running"

    with _get_pg_pool().connection() as conn:
        _stop_subject_workloads(conn, [customer_id])
        conn.commit()

    status, phase, desired_state, reason_code = _state(job_id)
    assert desired_state == "stopped", (
        "the trigger recomputed desired_state from an unchanged status — the "
        "deletion recorded its intent but nothing will stop the workload"
    )
    # Still running, but wanted stopped: that pairing is what the reconciler acts on.
    assert phase == "running"
    assert status == "stopping"
    assert reason_code == "privacy_deletion_requested"


def test_terminal_jobs_are_left_alone(running_job_for_subject):
    """A finished job is not restarted or re-marked by a deletion sweep."""
    job_id, customer_id = running_job_for_subject
    with _get_pg_pool().connection() as conn:
        conn.execute("UPDATE jobs SET status = 'completed' WHERE job_id = %s", (job_id,))
        conn.commit()
        _stop_subject_workloads(conn, [customer_id])
        conn.commit()

    status, phase, _desired, reason_code = _state(job_id)
    assert (status, phase) == ("completed", "succeeded")
    assert reason_code != "privacy_deletion_requested"


def test_it_reports_no_workloads_left_once_they_stop(running_job_for_subject):
    """`remaining` is what holds the deletion open, so it must reach zero."""
    job_id, customer_id = running_job_for_subject
    with _get_pg_pool().connection() as conn:
        _stopped, remaining = _stop_subject_workloads(conn, [customer_id])
        conn.commit()
        assert remaining == 1, "still running, so the deletion correctly stays pending"

        # The agent acknowledges the stop.
        conn.execute("UPDATE jobs SET status = 'stopped' WHERE job_id = %s", (job_id,))
        conn.commit()
        _stopped2, remaining2 = _stop_subject_workloads(conn, [customer_id])
        conn.commit()
    assert remaining2 == 0
