"""The submit protocol (blueprint 2.2) against real PostgreSQL and a scripted cluster.

A submission becomes one Slurm job. A retry looks the name up and adopts it.
A refusal is terminal. A timeout is not treated as absence until two lookups,
a poll interval apart, have missed it.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
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
        squeue_rc=0,
        squeue_err="",
    ):
        self.squeue = list(squeue)
        self.sbatch_rc = sbatch_rc
        self.sbatch_out = sbatch_out
        self.sbatch_err = sbatch_err
        self.sacct_rc = sacct_rc
        self.sacct = sacct
        self.timeouts_left = timeouts
        self.reveal_after_sbatch = reveal_after_sbatch
        self.squeue_rc = squeue_rc
        self.squeue_err = squeue_err
        self.reveal_line = ""
        self.calls: list[list[str]] = []
        self.scripts: list[str] = []

    def run(self, argv, *, input_text=None, timeout=30, env=None):
        self.calls.append(list(argv))
        if input_text:
            self.scripts.append(input_text)
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
            if self.squeue_rc:
                err = self.squeue_err or "squeue: error: unable to contact slurm controller\n"
                return subprocess.CompletedProcess(argv, self.squeue_rc, "", err)
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


def _sacct_line(name, sid):
    """One generic-profile sacct -D -X row. Times are epoch seconds."""
    fields = [
        sid,
        f"xcl-{name}",
        "RUNNING",
        "1700000000",
        "1700000001",
        "Unknown",
        "5",
        "0:0",
        "cpu=1,gres/gpu=1",
        "tower",
        "0:0",
        "00:00:00",
        "10",
    ]
    assert len(fields) == 13
    return "|".join(fields)


def _mapping(name):
    with pg_connection() as conn:
        rows = conn.execute(
            """
            SELECT slurm_job_id, desired_state, terminal_at IS NULL
              FROM slurm_job_mappings WHERE submission_name = %s
            """,
            (name,),
        ).fetchall()
    return rows


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


def test_a_dropped_ssh_submit_is_adopted_later_and_not_submitted_twice(world):
    row = world()
    fake = FakeSubmit(
        sbatch_rc=255,
        sbatch_err="ssh: connect to host tower port 22: Connection timed out\n",
        reveal_after_sbatch=True,
    )
    fake.reveal_line = _squeue(row["name"], "5151")
    start = 1_700_000_000.0
    _run(fake, start)
    unknown = _submission(row["name"])
    assert unknown["state"] == "unknown"
    assert unknown["error_code"] == "transport_error"
    assert unknown["sbatch_calls"] == 1
    assert fake.count("sbatch") == 1

    _run(fake, start + 31)
    adopted = _submission(row["name"])
    assert adopted["state"] == "submitted"
    assert adopted["slurm_job_id"] == "5151"
    assert adopted["sbatch_calls"] == 1
    assert fake.count("sbatch") == 1
    assert _mapping(row["name"]) == [("5151", "running", True)]


def test_an_accounting_outage_does_not_count_as_absence(world):
    pending = world()
    start = 1_700_000_000.0
    inflight = world(state="submitting", sbatch_calls=1, submit_started=start)
    fake = FakeSubmit(sacct_rc=1)
    _run(fake, start + 31)
    assert fake.count("sbatch") == 0
    assert _submission(pending["name"])["state"] == "pending"
    assert _submission(pending["name"])["sbatch_calls"] == 0
    held = _submission(inflight["name"])
    assert held["state"] == "submitting"
    assert held["checks"] == 0
    assert held["sbatch_calls"] == 1


def test_a_name_still_in_squeue_is_adopted_when_accounting_is_down(world):
    row = world()
    fake = FakeSubmit(squeue=[_squeue(row["name"], "6161")], sacct_rc=1)
    _run(fake, 1_700_000_000)
    saved = _submission(row["name"])
    assert saved["state"] == "submitted"
    assert saved["slurm_job_id"] == "6161"
    assert fake.count("sbatch") == 0


def test_a_job_that_left_the_controller_is_adopted_from_accounting(world):
    row = world()
    fake = FakeSubmit(sacct=_sacct_line(row["name"], "7171") + "\n")
    _run(fake, 1_700_000_000)
    saved = _submission(row["name"])
    assert saved["state"] == "submitted"
    assert saved["slurm_job_id"] == "7171"
    assert fake.count("sbatch") == 0


def test_two_ids_for_one_name_are_lost_and_neither_is_adopted(world):
    row = world()
    fake = FakeSubmit(squeue=[_squeue(row["name"], "111"), _squeue(row["name"], "222")])
    start = 1_700_000_000.0
    _run(fake, start)
    looking = _submission(row["name"])
    assert looking["state"] == "unknown"
    assert looking["checks"] == 1
    assert looking["error_code"] == "ambiguous_name"
    assert fake.count("sbatch") == 0
    assert fake.count("scancel") == 0

    _run(fake, start + 31)
    lost = _submission(row["name"])
    assert lost["state"] == "lost"
    assert lost["error_code"] == "ambiguous_name"
    assert fake.count("sbatch") == 0
    assert _mapping(row["name"]) == []
    attempt, job = _attempt_and_job(row["job_id"])
    assert attempt == ("lost", "ambiguous_name")
    assert job == ("failed", "failed", "stopped")


def test_an_ambiguous_name_is_not_cancelled_when_the_customer_wants_a_stop(world):
    row = world(desired="stopped")
    fake = FakeSubmit(squeue=[_squeue(row["name"], "111"), _squeue(row["name"], "222")])
    start = 1_700_000_000.0
    _run(fake, start)
    _run(fake, start + 31)
    saved = _submission(row["name"])
    assert saved["state"] == "lost"
    assert saved["error_code"] == "ambiguous_name"
    assert fake.count("sbatch") == 0
    assert fake.count("scancel") == 0
    attempt, _job = _attempt_and_job(row["job_id"])
    assert attempt[0] == "lost"


def test_two_accounting_records_for_one_name_are_lost_as_ambiguous(world):
    row = world()
    fake = FakeSubmit(
        sacct=_sacct_line(row["name"], "111") + "\n" + _sacct_line(row["name"], "222") + "\n"
    )
    start = 1_700_000_000.0
    _run(fake, start)
    _run(fake, start + 31)
    saved = _submission(row["name"])
    assert saved["state"] == "lost"
    assert saved["error_code"] == "ambiguous_name"
    assert fake.count("sbatch") == 0
    assert _mapping(row["name"]) == []


def test_a_stop_after_the_name_is_found_scancels_once(world):
    row = world(desired="stopped")
    fake = FakeSubmit(squeue=[_squeue(row["name"], "8181")])
    _run(fake, 1_700_000_000)
    saved = _submission(row["name"])
    assert saved["state"] == "submitted"
    assert saved["slurm_job_id"] == "8181"
    assert fake.count("sbatch") == 0
    assert fake.count("scancel") == 1
    assert _mapping(row["name"]) == [("8181", "cancelled", True)]
    # The attempt stays reserved until settlement (Phase 2.4). The mapping
    # is what the reconciler watches, and scancel has been asked once.
    attempt, job = _attempt_and_job(row["job_id"])
    assert attempt[0] == "reserved"
    assert job[0] == "assigned"


def test_a_stop_while_submit_is_in_flight_waits_for_two_misses(world):
    start = 1_700_000_000.0
    row = world(state="submitting", sbatch_calls=1, submit_started=start, desired="stopped")
    fake = FakeSubmit()
    _run(fake, start)
    assert _submission(row["name"])["state"] == "submitting"
    assert fake.count("sbatch") == 0

    _run(fake, start + 31)
    looking = _submission(row["name"])
    assert looking["state"] == "unknown"
    assert looking["checks"] == 1
    assert looking["error_code"] != "cancelled"

    _run(fake, start + 62)
    saved = _submission(row["name"])
    assert saved["state"] == "rejected"
    assert saved["error_code"] == "cancelled"
    assert fake.count("sbatch") == 0
    assert fake.count("scancel") == 0
    attempt, job = _attempt_and_job(row["job_id"])
    assert attempt[0] == "cancelled"
    assert job == ("cancelled", "stopped", "stopped")


def test_unparseable_sbatch_output_is_resolved_by_name(world):
    row = world()
    fake = FakeSubmit(sbatch_rc=0, sbatch_out="not a job id\n", reveal_after_sbatch=True)
    fake.reveal_line = _squeue(row["name"], "9191")
    start = 1_700_000_000.0
    _run(fake, start)
    unknown = _submission(row["name"])
    assert unknown["state"] == "unknown"
    assert unknown["error_code"] == "unparseable_sbatch_output"
    assert fake.count("sbatch") == 1

    _run(fake, start + 31)
    adopted = _submission(row["name"])
    assert adopted["state"] == "submitted"
    assert adopted["slurm_job_id"] == "9191"
    assert fake.count("sbatch") == 1


def test_a_refused_host_key_is_rejected_once(world):
    row = world()
    fake = FakeSubmit(
        sbatch_rc=255,
        sbatch_err="Host key verification failed.\n",
        reveal_after_sbatch=True,
    )
    fake.reveal_line = _squeue(row["name"], "3333")
    start = 1_700_000_000.0
    _run(fake, start)
    saved = _submission(row["name"])
    assert saved["state"] == "rejected"
    assert saved["error_code"] == "host_key_mismatch"
    attempt, job = _attempt_and_job(row["job_id"])
    assert attempt == ("failed", "host_key_mismatch")
    assert job[0] == "failed"

    _run(fake, start + 31)
    assert fake.count("sbatch") == 1
    assert _submission(row["name"])["state"] == "rejected"
    assert _mapping(row["name"]) == []


def test_an_unclassified_sbatch_error_is_rejected_without_a_name_lookup(world):
    row = world()
    fake = FakeSubmit(
        sbatch_rc=1,
        sbatch_err="sbatch: error: something novel went wrong\n",
        reveal_after_sbatch=True,
    )
    fake.reveal_line = _squeue(row["name"], "3434")
    start = 1_700_000_000.0
    _run(fake, start)
    saved = _submission(row["name"])
    assert saved["state"] == "rejected"
    assert saved["error_code"] == "sbatch_rejected"

    _run(fake, start + 31)
    assert fake.count("sbatch") == 1
    assert _submission(row["name"])["state"] == "rejected"
    assert _mapping(row["name"]) == []


def test_a_disabled_cluster_is_not_due(world):
    row = world()
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE slurm_clusters SET enabled = false WHERE cluster_id = %s",
            (row["cluster_id"],),
        )
    fake = FakeSubmit()
    outcomes = _run(fake, 1_700_000_000)
    assert not any(item.get("submission_name") == row["name"] for item in outcomes)
    assert _submission(row["name"])["state"] == "pending"
    assert row["name"] not in "\n".join(fake.scripts)
    assert fake.count("sbatch") == 0


def test_a_controller_outage_does_not_submit_or_close_the_row(world):
    row = world()
    fake = FakeSubmit(squeue_rc=1)
    outcomes = _run(fake, 1_700_000_000)
    ours = [item for item in outcomes if item.get("submission_name") == row["name"]]
    assert ours == [
        {
            "submission_name": row["name"],
            "action": "poll_failed",
            "error_code": "controller_unreachable",
        }
    ]
    assert _submission(row["name"])["state"] == "pending"
    assert fake.count("sbatch") == 0


def test_a_different_slurm_id_for_the_same_submission_is_a_conflict(world):
    import slurm_adapter as sa

    row = world()
    sa.register_slurm_job(
        row["job_id"],
        "111",
        cluster_id=row["cluster_id"],
        submission_name=row["name"],
    )
    fake = FakeSubmit(squeue=[_squeue(row["name"], "222")])
    start = 1_700_000_000.0
    _run(fake, start)
    saved = _submission(row["name"])
    assert saved["state"] == "rejected"
    assert saved["error_code"] == "mapping_conflict"
    assert fake.count("sbatch") == 0
    assert _mapping(row["name"]) == [("111", "running", True)]

    _run(fake, start + 31)
    assert fake.count("sbatch") == 0
    assert _submission(row["name"])["state"] == "rejected"


def test_three_submits_then_two_misses_is_lost_without_a_fourth(world):
    start = 1_700_000_000.0
    row = world(state="submitting", sbatch_calls=3, submit_started=start)
    fake = FakeSubmit(sbatch_out="9999\n")
    _run(fake, start)
    assert fake.count("sbatch") == 0
    assert _submission(row["name"])["checks"] == 0

    _run(fake, start + 31)
    assert _submission(row["name"])["checks"] == 1
    assert _submission(row["name"])["state"] == "unknown"
    assert fake.count("sbatch") == 0

    _run(fake, start + 62)
    saved = _submission(row["name"])
    assert saved["state"] == "lost"
    assert saved["error_code"] == "lost"
    assert saved["sbatch_calls"] == 3
    assert fake.count("sbatch") == 0
    attempt, job = _attempt_and_job(row["job_id"])
    assert attempt == ("lost", "lost")
    assert job[0] == "failed"


def test_two_passes_at_once_submit_once(world):
    row = world()
    barrier = threading.Barrier(2)

    class Racing(FakeSubmit):
        def run(self, argv, *, input_text=None, timeout=30, env=None):
            if argv and argv[0] == "squeue":
                barrier.wait(timeout=10)
            with barrier_lock:
                return super().run(argv, input_text=input_text, timeout=timeout, env=env)

    barrier_lock = threading.Lock()
    fake = Racing(sbatch_out="9001\n")
    errors: list[BaseException] = []

    def go():
        try:
            _run(fake, 1_700_000_000)
        except BaseException as exc:  # noqa: BLE001 — the other thread must hear it
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=go) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert errors == []
    assert fake.count("sbatch") == 1
    saved = _submission(row["name"])
    assert saved["state"] == "submitted"
    assert saved["slurm_job_id"] == "9001"
    assert saved["sbatch_calls"] == 1
    assert _mapping(row["name"]) == [("9001", "running", True)]


def test_one_hundred_injected_crashes_leave_one_job_each(world):
    """Local stand-in for the blueprint's 100 crash injections.

    Each pass is a scripted cluster, not xcl-local. A crash before the
    mapping is written, a dropped ssh, and a job that already exists all
    leave one Slurm id and one mapping.
    """
    start = 1_700_000_000.0
    for i in range(100):
        kind = i % 5
        sid = str(10_000 + i)
        if kind in (0, 1, 3):
            row = world()
            if kind == 0:
                fake = FakeSubmit(timeouts=1, reveal_after_sbatch=True, sbatch_out=f"{sid}\n")
            elif kind == 1:
                fake = FakeSubmit(
                    sbatch_rc=255,
                    sbatch_err="ssh: Connection reset by peer\n",
                    reveal_after_sbatch=True,
                    sbatch_out=f"{sid}\n",
                )
            else:
                fake = FakeSubmit(
                    sbatch_out="not a job id\n",
                    reveal_after_sbatch=True,
                )
            fake.reveal_line = _squeue(row["name"], sid)
            _run(fake, start)
            assert _submission(row["name"])["state"] == "unknown"
            assert fake.count("sbatch") == 1
            _run(fake, start + 31)
        elif kind == 2:
            row = world(state="submitting", sbatch_calls=1, submit_started=start)
            fake = FakeSubmit(squeue=[_squeue(row["name"], sid)])
            _run(fake, start + 5)
            assert fake.count("sbatch") == 0
        else:
            row = world()
            fake = FakeSubmit(squeue=[_squeue(row["name"], sid)])
            _run(fake, start)
            assert fake.count("sbatch") == 0
        saved = _submission(row["name"])
        assert saved["state"] == "submitted", kind
        assert saved["slurm_job_id"] == sid
        assert fake.count("sbatch") <= 1
        assert _mapping(row["name"]) == [(sid, "running", True)]
