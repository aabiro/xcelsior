"""The submit protocol (blueprint 2.2) against real PostgreSQL and a scripted cluster.

A submission becomes one Slurm job. A retry looks the name up and adopts it.
A refusal is terminal. A timeout is not treated as absence until two lookups,
a poll interval apart, have missed it.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
import uuid

import pytest
from psycopg.types.json import Jsonb

from db import pg_connection, pg_transaction
import slurm_submitter as submitter


class FakeSubmit:
    """Answers squeue, sacct, sbatch and scancel. remote=True skips the local sinfo check."""

    remote = True
    ssh_host = "fake"

    def __init__(
        self,
        *,
        squeue=(),
        sbatch_rc=0,
        sbatch_out="4242\n",
        sbatch_err="",
        sacct_rc=0,
        sacct="",
        timeouts=0,
        reveal_after_sbatch=False,
    ):
        self.squeue = list(squeue)
        self.sbatch_rc = sbatch_rc
        self.sbatch_out = sbatch_out
        self.sbatch_err = sbatch_err
        self.sacct_rc = sacct_rc
        self.sacct = sacct
        self.timeouts_left = timeouts
        self.reveal_after_sbatch = reveal_after_sbatch
        self.reveal_line = ""
        self.calls: list[list[str]] = []

    def run(self, argv, *, input_text=None, timeout=30, env=None):
        self.calls.append(list(argv))
        prog = argv[0]
        if prog == "sbatch":
            if self.reveal_after_sbatch and self.reveal_line:
                self.squeue = [self.reveal_line]
            if self.timeouts_left:
                self.timeouts_left -= 1
                raise subprocess.TimeoutExpired(argv, timeout)
            err = self.sbatch_err if self.sbatch_rc else ""
            return subprocess.CompletedProcess(argv, self.sbatch_rc, self.sbatch_out, err)
        if prog == "squeue":
            out = ("\n".join(self.squeue) + "\n") if self.squeue else ""
            return subprocess.CompletedProcess(argv, 0, out, "")
        if prog == "sacct":
            err = "" if self.sacct_rc == 0 else "sacct: error: slurmdbd unreachable"
            return subprocess.CompletedProcess(argv, self.sacct_rc, self.sacct, err)
        if prog == "scancel":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def count(self, prog: str) -> int:
        return sum(1 for call in self.calls if call and call[0] == prog)


@pytest.fixture
def world():
    made: dict[str, list[str]] = {"jobs": [], "clusters": []}

    def add(
        *,
        state="pending",
        sbatch_calls=0,
        checks=0,
        desired="running",
        lifecycle=None,
        submit_started=None,
        last_check=None,
    ):
        marker = uuid.uuid4().hex[:8]
        job_id = f"j{marker}"
        cluster_id = f"c{marker}"
        name = f"{job_id}-a1"
        spec = {"num_gpus": 1, "name": job_id}
        now = time.time()
        with pg_transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, priority, submitted_at, payload, phase, desired_state,
                    effective_priority, queued_at, spec, lifecycle_intent
                )
                VALUES (%s, 'assigned', 10, %s, %s, 'scheduled', %s, 10, to_timestamp(%s), %s, %s)
                """,
                (
                    job_id,
                    now,
                    json.dumps({"job_id": job_id, "num_gpus": 1}),
                    desired,
                    now,
                    Jsonb(spec),
                    lifecycle,
                ),
            )
            attempt = conn.execute(
                """
                INSERT INTO job_attempts
                    (job_id, attempt_number, status, fencing_token, job_generation, created_by)
                VALUES (%s, 1, 'reserved', nextval('placement_fencing_token_seq'), 1, 'test')
                RETURNING attempt_id
                """,
                (job_id,),
            ).fetchone()
            attempt_id = attempt[0]
            conn.execute(
                "UPDATE jobs SET active_attempt_id = %s WHERE job_id = %s",
                (attempt_id, job_id),
            )
            conn.execute(
                """
                INSERT INTO slurm_clusters (
                    cluster_id, profile_key, display_name, enabled, research_only,
                    marketplace, agent_access, poll_interval_seconds
                )
                VALUES (%s, 'generic', 'test', true, false, false, false, 30)
                """,
                (cluster_id,),
            )
            conn.execute(
                """
                INSERT INTO slurm_submissions (
                    submission_name, cluster_id, xcelsior_job_id, attempt_id, state,
                    sbatch_calls, checks, submit_started_at, last_check_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s))
                """,
                (
                    name,
                    cluster_id,
                    job_id,
                    attempt_id,
                    state,
                    sbatch_calls,
                    checks,
                    submit_started,
                    last_check,
                ),
            )
        made["jobs"].append(job_id)
        made["clusters"].append(cluster_id)
        return {"job_id": job_id, "cluster_id": cluster_id, "name": name, "attempt_id": attempt_id}

    yield add
    with pg_transaction() as conn:
        for job_id in made["jobs"]:
            conn.execute("DELETE FROM slurm_job_mappings WHERE xcelsior_job_id = %s", (job_id,))
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id = %s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
        for cluster_id in made["clusters"]:
            conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (cluster_id,))


def _run(fake, now):
    def runner_for(_cluster):
        return contextlib.nullcontext(fake)

    return submitter.submit_due(runner_for=runner_for, now=now)


def _submission(name):
    with pg_connection() as conn:
        row = conn.execute(
            """
            SELECT state, sbatch_calls, checks, error_code, slurm_job_id
              FROM slurm_submissions WHERE submission_name = %s
            """,
            (name,),
        ).fetchone()
    assert row is not None
    return {
        "state": row[0],
        "sbatch_calls": row[1],
        "checks": row[2],
        "error_code": row[3],
        "slurm_job_id": row[4],
    }


def _attempt_and_job(job_id):
    with pg_connection() as conn:
        attempt = conn.execute(
            "SELECT status, failure_code FROM job_attempts WHERE job_id = %s",
            (job_id,),
        ).fetchone()
        job = conn.execute(
            "SELECT status, phase, desired_state FROM jobs WHERE job_id = %s",
            (job_id,),
        ).fetchone()
    return attempt, job


def _squeue(name, sid="4242"):
    return f"{sid}|xcl-{name}|PENDING|None|0:00|10:00|1|tower"


def test_a_name_already_in_squeue_is_adopted_and_not_submitted_again(world):
    row = world()
    fake = FakeSubmit(squeue=[_squeue(row["name"])])
    _run(fake, 1_700_000_000)
    saved = _submission(row["name"])
    assert saved["state"] == "submitted"
    assert saved["slurm_job_id"] == "4242"
    assert saved["sbatch_calls"] == 0
    assert fake.count("sbatch") == 0
    with pg_connection() as conn:
        mapped = conn.execute(
            "SELECT slurm_job_id FROM slurm_job_mappings WHERE submission_name = %s",
            (row["name"],),
        ).fetchone()
    assert mapped == ("4242",)


def test_a_timed_out_sbatch_is_adopted_on_the_next_look_without_a_second_submit(world):
    row = world()
    fake = FakeSubmit(timeouts=1, reveal_after_sbatch=True)
    fake.reveal_line = _squeue(row["name"], "5150")
    start = 1_700_000_000.0
    _run(fake, start)
    after_timeout = _submission(row["name"])
    assert after_timeout["state"] == "unknown"
    assert after_timeout["sbatch_calls"] == 1
    assert after_timeout["checks"] == 0
    assert after_timeout["error_code"] == "timeout"
    assert fake.count("sbatch") == 1

    _run(fake, start + 31)
    adopted = _submission(row["name"])
    assert adopted["state"] == "submitted"
    assert adopted["slurm_job_id"] == "5150"
    assert adopted["sbatch_calls"] == 1
    assert fake.count("sbatch") == 1


def test_two_misses_a_poll_apart_then_one_retry(world):
    start = 1_700_000_000.0
    row = world(state="submitting", sbatch_calls=1, submit_started=start)
    fake = FakeSubmit(sbatch_out="7777\n")
    _run(fake, start)
    assert fake.count("sbatch") == 0
    assert _submission(row["name"])["checks"] == 0

    _run(fake, start + 31)
    assert fake.count("sbatch") == 0
    assert _submission(row["name"])["checks"] == 1
    assert _submission(row["name"])["state"] == "unknown"

    _run(fake, start + 62)
    saved = _submission(row["name"])
    assert fake.count("sbatch") == 1
    assert saved["state"] == "submitted"
    assert saved["slurm_job_id"] == "7777"
    assert saved["sbatch_calls"] == 2


def test_an_invalid_partition_is_rejected_and_never_retried(world):
    row = world()
    fake = FakeSubmit(sbatch_rc=1, sbatch_err="sbatch: error: invalid partition specified: nope\n")
    now = 1_700_000_000.0
    _run(fake, now)
    saved = _submission(row["name"])
    assert saved["state"] == "rejected"
    assert saved["error_code"] == "invalid_partition"
    attempt, job = _attempt_and_job(row["job_id"])
    assert attempt[0] == "failed"
    assert attempt[1] == "invalid_partition"
    assert job == ("failed", "failed", "stopped")

    _run(fake, now + 31)
    assert fake.count("sbatch") == 1
    assert _submission(row["name"])["state"] == "rejected"


def test_a_crash_between_sbatch_and_register_adopts_the_existing_job(world):
    # The claim landed and the process died. squeue already has the name.
    start = 1_700_000_000.0
    row = world(state="submitting", sbatch_calls=1, checks=0, submit_started=start)
    fake = FakeSubmit(squeue=[_squeue(row["name"], "6001")])
    _run(fake, start + 5)
    saved = _submission(row["name"])
    assert saved["state"] == "submitted"
    assert saved["slurm_job_id"] == "6001"
    assert saved["sbatch_calls"] == 1
    assert fake.count("sbatch") == 0


def test_a_stop_requested_before_any_sbatch_submits_nothing(world):
    row = world(desired="stopped")
    fake = FakeSubmit()
    _run(fake, 1_700_000_000)
    saved = _submission(row["name"])
    assert saved["state"] == "rejected"
    assert saved["error_code"] == "cancelled_before_submit"
    assert fake.count("sbatch") == 0
    attempt, job = _attempt_and_job(row["job_id"])
    assert attempt[0] == "cancelled"
    assert job == ("cancelled", "stopped", "stopped")
