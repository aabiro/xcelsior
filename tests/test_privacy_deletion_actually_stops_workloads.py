"""Privacy withdrawal stops the subject's compute and then lets the deletion finish.

Placed work stops through its fenced attempt. Unplaced work has no attempt for
anything to stop, so it is terminated directly; marking it ``stopping`` left a
row no controller settles and a deletion that stayed pending forever.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from psycopg.types.json import Jsonb

os.environ.setdefault("XCELSIOR_ENV", "test")

from db import _get_pg_pool
from privacy_sinks import _stop_subject_workloads, delete_authoritative_subject
from tests.fixtures.native_runtime import running_instance
from tests.native_jobs import admit_fixture_job
from tests.test_privacy_sinks import subject  # noqa: F401


def _insert_queued_job(customer_id: str) -> str:
    job_id = f"j-priv-{uuid.uuid4().hex[:8]}"
    with _get_pg_pool().connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, owner_id, payload,
                   phase, desired_state, effective_priority, queued_at)
               VALUES (%s, 'queued', 0, %s, %s, %s, 'pending', 'running', 0,
                   clock_timestamp())""",
            (
                job_id,
                time.time(),
                customer_id,
                Jsonb({"job_id": job_id, "owner": customer_id, "image": "python:3.11"}),
            ),
        )
        admit_fixture_job(conn, job_id)
    return job_id


def _delete_job(job_id: str) -> None:
    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM outbox_events WHERE aggregate_id = %s", (job_id,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


@pytest.fixture
def queued_job_for_subject():
    customer_id = f"cust-priv-{uuid.uuid4().hex[:8]}"
    job_id = _insert_queued_job(customer_id)
    yield job_id, customer_id
    _delete_job(job_id)


def _state(job_id: str) -> tuple:
    with _get_pg_pool().connection() as conn:
        return conn.execute(
            "SELECT status, phase, desired_state, lifecycle_intent FROM jobs WHERE job_id = %s",
            (job_id,),
        ).fetchone()


def _stop(customer_ids: list[str]) -> tuple[int, int]:
    with _get_pg_pool().connection() as conn:
        result = _stop_subject_workloads(conn, customer_ids)
        conn.commit()
    return result


def test_unplaced_work_is_terminated_not_left_stopping(queued_job_for_subject):
    job_id, customer_id = queued_job_for_subject

    assert _stop([customer_id]) == (1, 0), "nothing is running, so nothing holds the deletion"

    assert _state(job_id) == ("terminated", "stopped", "stopped", None)


def test_unplaced_work_an_earlier_pass_marked_stopping_is_settled(queued_job_for_subject):
    """The shape the old writer left on queued work: no attempt, so nothing settled it."""
    job_id, customer_id = queued_job_for_subject
    with _get_pg_pool().connection() as conn:
        conn.execute(
            """UPDATE jobs SET status='stopping', phase='running', desired_state='stopped',
                   generation=generation+1, reason_code='privacy_deletion_requested'
                WHERE job_id=%s""",
            (job_id,),
        )

    assert _stop([customer_id]) == (1, 0)
    assert _state(job_id)[:3] == ("terminated", "stopped", "stopped")


def test_placed_work_stops_through_its_fenced_attempt():
    owner = f"cust-priv-{uuid.uuid4().hex[:8]}"
    with running_instance(owner=owner) as instance:
        job_id = instance["job_id"]

        assert _stop([owner]) == (1, 1), "the attempt is still running until the host reports"
        # A retry while the stop is in flight requests nothing new.
        assert _stop([owner]) == (0, 1)

        assert _state(job_id) == ("stopping", "running", "stopped", "terminate")
        with _get_pg_pool().connection() as conn:
            commands = conn.execute(
                """SELECT attempt_id::text, args->>'intent', args->'preserve'
                     FROM agent_commands WHERE job_id=%s AND command='stop_attempt'""",
                (job_id,),
            ).fetchall()
        assert commands == [(str(instance["authority"]["attempt_id"]), "terminate", False)]


def test_terminal_jobs_are_left_alone(queued_job_for_subject):
    """A finished job is not restarted or re-marked by a deletion sweep."""
    job_id, customer_id = queued_job_for_subject
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='completed', phase='succeeded' WHERE job_id=%s",
            (job_id,),
        )

    assert _stop([customer_id]) == (0, 0)
    assert _state(job_id)[:2] == ("completed", "succeeded")


def test_deletion_of_a_subject_with_queued_work_completes(subject):  # noqa: F811
    job_id = _insert_queued_job(subject["customer_id"])
    try:
        outcome = delete_authoritative_subject(subject["request"], {})

        assert outcome.status == "completed", outcome
        assert _state(job_id)[:2] == ("terminated", "stopped")
    finally:
        _delete_job(job_id)
