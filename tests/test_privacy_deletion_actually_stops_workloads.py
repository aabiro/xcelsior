"""Privacy withdrawal explicitly sets native stop intent and observed phase.

After projection-trigger contraction the deletion writer must update all three
lifecycle fields itself. Remaining-work counts keep deletion pending until a
terminal phase is recorded; terminal work must never be resurrected.
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
            """INSERT INTO jobs (job_id, status, priority, submitted_at, owner_id, payload, phase, desired_state, effective_priority, queued_at)
               VALUES (%s, 'running', 0, %s, %s, %s, 'running', 'running', 0, NULL)""",
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


def test_the_stop_sets_native_intent_and_phase(running_job_for_subject):
    job_id, customer_id = running_job_for_subject
    assert _state(job_id)[2] == "running"

    with _get_pg_pool().connection() as conn:
        _stop_subject_workloads(conn, [customer_id])
        conn.commit()

    status, phase, desired_state, reason_code = _state(job_id)
    assert desired_state == "stopped", (
        "the writer did not preserve its stop intent — the "
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
        conn.execute(
            """UPDATE jobs SET status='completed', desired_state='running',phase='succeeded' WHERE job_id=%s""",
            (job_id,),
        )
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
        conn.execute(
            "UPDATE jobs SET status='stopped',phase='stopped',desired_state='stopped' WHERE job_id=%s",
            (job_id,),
        )
        conn.commit()
        _stopped2, remaining2 = _stop_subject_workloads(conn, [customer_id])
        conn.commit()
    assert remaining2 == 0
