"""Shadow billing rules and settlement for one Slurm attempt (decision D8).

The rule checks do not touch the database. Settlement checks call
``settle_finished_attempt`` on ledger rows ``write_segments`` stored.
"""

from __future__ import annotations

import inspect
import json
import uuid

from psycopg.types.json import Jsonb

import slurm_adapter as sa
import slurm_billing as sb
from db import pg_connection, pg_transaction


def _run(**overrides):
    record = {
        "state": "COMPLETED",
        "started_at": 1_794_000_005,
        "elapsed_sec": 60,
        "suspended_sec": 0,
        "time_limit_sec": 600,
    }
    record.update(overrides)
    return record


def test_a_job_that_never_started_bills_nothing():
    seconds, rule = sb.segment_charge(_run(started_at=None, state="CANCELLED", elapsed_sec=0))
    assert (seconds, rule) == (0, sb.RULE_NEVER_STARTED)


def test_a_timeout_is_capped_at_the_requested_limit():
    # The lab ran a 60s job for 78s. The extra is the cluster's, not the tenant's.
    seconds, rule = sb.segment_charge(_run(state="TIMEOUT", elapsed_sec=78, time_limit_sec=60))
    assert (seconds, rule) == (60, sb.RULE_TIMEOUT_CAPPED)


def test_suspended_time_bills_only_while_the_job_holds_its_gpus():
    record = _run(state="TIMEOUT", elapsed_sec=78, suspended_sec=10, time_limit_sec=60)
    held, held_rule = sb.segment_charge(record, suspend_holds_gres=True)
    free, free_rule = sb.segment_charge(record, suspend_holds_gres=False)
    assert (held, held_rule) == (70, "timeout_capped_to_limit+suspended_gres_held")
    assert (free, free_rule) == (60, "timeout_capped_to_limit+suspended_gres_released")


def test_a_preemption_bills_to_the_preemption_and_a_requeue_bills_its_own_run():
    preempted, preempted_rule = sb.segment_charge(_run(state="PREEMPTED", elapsed_sec=15))
    requeued, requeued_rule = sb.segment_charge(_run(state="NODE_FAIL", elapsed_sec=9))
    assert (preempted, preempted_rule) == (15, sb.RULE_PREEMPTED)
    assert (requeued, requeued_rule) == (9, sb.RULE_REQUEUED_RUN)


def test_drift_compares_raw_elapsed_gpu_seconds_not_the_suspension_rule():
    # D8 changes what the tenant would pay. It must not show up as drift:
    # both sides use ElapsedRaw × GPUs, and ElapsedRaw already excludes the
    # suspended time.
    records = {
        "7": [
            {
                "name": "xcl-job-a1",
                "ended_at": 1_500.0,
                "submitted_at": 1_000,
                "elapsed_sec": 78,
                "suspended_sec": 10,
                "alloc_tres": "gres/gpu=1",
            }
        ]
    }
    segments = [("7", 1_000, 78, 1)]
    report = sb.compare_day(records, segments, start=1_000.0, end=2_000.0)
    assert report["sacct_gpu_seconds"] == 78
    assert report["ledger_gpu_seconds"] == 78
    assert report["drift_pct"] == 0.0
    assert report["unmatched"]["differing"] == []


def _acct_run(name, sid_submit, elapsed, gpus=1, ended=1_500.0):
    return {
        "name": name,
        "ended_at": ended,
        "submitted_at": sid_submit,
        "elapsed_sec": elapsed,
        "suspended_sec": 0,
        "alloc_tres": f"cpu=2,gres/gpu={gpus},node=1" if gpus else "cpu=2,node=1",
    }


def test_drift_counts_only_what_the_platform_submitted():
    # 2026-11-08 on staging read 100 %: every run that day was the validation
    # kit's own suite, named xcl-… by the adapter but never submitted through
    # the platform, so no ledger row could exist for it.
    records = {
        "7": [_acct_run("xcl-jabc-a1", 1_000, 120)],
        "8": [_acct_run("xcl-scen-requeue", 1_010, 300)],
        "9": [_acct_run("xcl-suite-v3", 1_020, 27)],
    }
    segments = [("7", 1_000, 120, 1)]
    report = sb.compare_day(
        records, segments, start=1_000.0, end=2_000.0, platform_names={"7": {"xcl-jabc-a1"}}
    )
    assert (report["sacct_gpu_seconds"], report["ledger_gpu_seconds"]) == (120, 120)
    assert report["drift_pct"] == 0.0
    assert report["unmatched"]["unattributed"] == ["8@1010", "9@1020"]
    assert report["unmatched"]["unattributed_gpu_seconds"] == 327


def test_a_platform_run_missing_from_the_ledger_is_still_drift():
    records = {"7": [_acct_run("xcl-jabc-a1", 1_000, 120)]}
    report = sb.compare_day(
        records, [], start=1_000.0, end=2_000.0, platform_names={"7": {"xcl-jabc-a1"}}
    )
    assert report["drift_pct"] == 100.0
    assert report["unmatched"]["only_in_sacct"] == ["7@1000"]


def test_a_reused_slurm_id_is_matched_by_name_not_by_id_alone():
    # A rebuilt controller renumbers from 1, so id 7 can be a kit job today and
    # a platform job's id in the mappings table from another day.
    records = {"7": [_acct_run("xcl-suite-v3", 1_000, 60)]}
    report = sb.compare_day(
        records, [], start=1_000.0, end=2_000.0, platform_names={"7": {"xcl-jabc-a1"}}
    )
    assert report["sacct_gpu_seconds"] == 0 and report["drift_pct"] == 0.0
    assert report["unmatched"]["unattributed"] == ["7@1000"]


def test_a_gpu_hour_rate_prices_the_billable_seconds():
    card = {"gpu_hour_micros": 3_600_000}
    assert sb.segment_cost_micros(card, billable_sec=70, gpu_count=1, cpu_count=4) == 70_000
    assert sb.segment_cost_micros({}, billable_sec=70, gpu_count=1, cpu_count=4) is None
    assert sb.segment_cost_micros(card, billable_sec=70, gpu_count=1, cpu_count=0) == 70_000
    # A GPU run on a card that prices only CPU hours is not guessed.
    assert (
        sb.segment_cost_micros({"cpu_hour_micros": 100}, billable_sec=70, gpu_count=1, cpu_count=4)
        is None
    )


def _run_row(**overrides):
    record = {
        "state": "COMPLETED",
        "started_at": 1_794_000_005,
        "ended_at": 1_794_000_065,
        "elapsed_sec": 60,
        "suspended_sec": 0,
        "time_limit_sec": 600,
        "alloc_tres": "cpu=4,gres/gpu=1,node=1",
        "submitted_at": 1_794_000_000,
        "name": "x",
        "raw_state": "COMPLETED",
        "exit_code": "0:0",
        "derived_exit_code": "0:0",
        "node_list": "tower",
    }
    record.update(overrides)
    return record


def _settle(runs, *, holds=True):
    """Write the shipped ledger rows, then settle them. Returns the outcome."""
    marker = uuid.uuid4().hex[:8]
    job_id = f"jsettle{marker}"
    cluster_id = f"csettle{marker}"
    customer = f"settle-{marker}"
    card = {"gpu_hour_micros": 3_600_000}
    with pg_transaction() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, priority, submitted_at, payload, phase, desired_state,
                effective_priority, spec
            )
            VALUES (%s, 'assigned', 10, 1794000000, %s, 'scheduled', 'running', 10, %s)
            """,
            (
                job_id,
                json.dumps({"owner": customer, "num_gpus": 1}),
                Jsonb({"num_gpus": 1, "name": job_id}),
            ),
        )
        attempt_id = conn.execute(
            """
            INSERT INTO job_attempts
                (job_id, attempt_number, status, fencing_token, job_generation, created_by)
            VALUES (%s, 1, 'reserved', nextval('placement_fencing_token_seq'), 1, 'test')
            RETURNING attempt_id
            """,
            (job_id,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE jobs SET active_attempt_id = %s WHERE job_id = %s",
            (attempt_id, job_id),
        )
        conn.execute(
            """
            INSERT INTO slurm_clusters (
                cluster_id, profile_key, display_name, enabled, research_only,
                marketplace, agent_access, poll_interval_seconds, rate_card,
                suspend_holds_gres
            )
            VALUES (%s, 'generic', 'test', false, false, false, false, 60, %s, %s)
            """,
            (cluster_id, Jsonb(card), holds),
        )
        conn.execute(
            """
            INSERT INTO wallets (customer_id, balance_micros, status, created_at, updated_at)
            VALUES (%s, %s, 'active', 1794000000, 1794000000)
            """,
            (customer, 100_000_000),
        )
    sa.register_slurm_job(
        job_id,
        "4242",
        cluster_id=cluster_id,
        attempt_id=str(attempt_id),
        submission_name=job_id,
    )
    try:
        with pg_connection() as conn:
            mapping = conn.execute(
                """
                SELECT mapping_id, cluster_id, slurm_job_id, xcelsior_job_id,
                       xcelsior_attempt_id, tenant_id
                  FROM slurm_job_mappings WHERE xcelsior_job_id = %s
                """,
                (job_id,),
            ).fetchone()
        mapping = {
            "mapping_id": mapping[0],
            "cluster_id": mapping[1],
            "slurm_job_id": mapping[2],
            "xcelsior_job_id": mapping[3],
            "xcelsior_attempt_id": mapping[4],
            "tenant_id": mapping[5],
        }
        with pg_transaction() as conn:
            sb.write_segments(
                conn,
                mapping,
                runs,
                {"rate_card": card, "suspend_holds_gres": holds},
            )
            before = sb.settle_finished_attempt(conn, str(attempt_id))
        with pg_connection() as conn:
            job = conn.execute(
                "SELECT status, phase FROM jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
            attempt = conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id = %s", (attempt_id,)
            ).fetchone()
            segments = conn.execute(
                """
                SELECT run_index, billable_sec, billing_rule, shadow_cost_micros, charged
                  FROM slurm_billing_segments WHERE attempt_id = %s ORDER BY run_index
                """,
                (attempt_id,),
            ).fetchall()
            debit = conn.execute(
                """
                SELECT COALESCE(SUM(-amount_micros), 0)
                  FROM wallet_transactions
                 WHERE customer_id = %s AND tx_type = 'charge'
                """,
                (customer,),
            ).fetchone()[0]
        return {
            "result": before,
            "job": job,
            "attempt": attempt[0],
            "segments": segments,
            "debit_micros": int(debit),
            "attempt_id": str(attempt_id),
        }
    finally:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id = %s", (customer,))
            conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cluster_id,))
            conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (cluster_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
            conn.execute("DELETE FROM wallets WHERE customer_id = %s", (customer,))


def test_a_job_that_never_started_settles_at_zero_and_leaves_reserved():
    outcome = _settle(
        [
            _run_row(
                state="CANCELLED",
                raw_state="CANCELLED",
                started_at=None,
                ended_at=None,
                elapsed_sec=0,
            )
        ]
    )
    assert outcome["result"]["charged_micros"] == 0
    assert outcome["debit_micros"] == 0
    assert outcome["segments"] == [(0, 0, "never_started", 0, True)]
    assert outcome["job"] == ("cancelled", "stopped")
    assert outcome["attempt"] == "cancelled"


def test_a_timeout_with_held_gpus_charges_seventy_seconds():
    outcome = _settle(
        [
            _run_row(
                state="TIMEOUT",
                raw_state="TIMEOUT",
                elapsed_sec=78,
                suspended_sec=10,
                time_limit_sec=60,
                ended_at=1_794_000_083,
            )
        ]
    )
    assert outcome["segments"] == [
        (0, 70, "timeout_capped_to_limit+suspended_gres_held", 70_000, True)
    ]
    assert outcome["result"]["charged_micros"] == 70_000
    assert outcome["debit_micros"] == 70_000
    assert outcome["job"] == ("failed", "failed")
    assert outcome["attempt"] == "failed"


def test_suspended_time_is_free_when_the_cluster_releases_gres():
    outcome = _settle(
        [
            _run_row(
                state="TIMEOUT",
                raw_state="TIMEOUT",
                elapsed_sec=78,
                suspended_sec=10,
                time_limit_sec=60,
            )
        ],
        holds=False,
    )
    assert outcome["segments"][0][1:4] == (
        60,
        "timeout_capped_to_limit+suspended_gres_released",
        60_000,
    )
    assert outcome["debit_micros"] == 60_000
    assert outcome["segments"][0][4] is True


def test_a_requeued_job_charges_each_run_on_its_own_record():
    outcome = _settle(
        [
            _run_row(
                state="NODE_FAIL",
                raw_state="NODE_FAIL",
                elapsed_sec=9,
                ended_at=1_794_000_014,
                exit_code="1:0",
            ),
            _run_row(
                state="COMPLETED",
                raw_state="COMPLETED",
                submitted_at=1_794_000_100,
                started_at=1_794_000_110,
                ended_at=1_794_000_125,
                elapsed_sec=15,
            ),
        ]
    )
    assert [row[1:4] for row in outcome["segments"]] == [
        (9, "requeued_run", 9_000),
        (15, "ran_to_end", 15_000),
    ]
    assert all(row[4] is True for row in outcome["segments"])
    assert outcome["debit_micros"] == 24_000
    assert outcome["job"] == ("completed", "succeeded")
    assert outcome["attempt"] == "succeeded"


def test_settling_twice_does_not_charge_again():
    marker = uuid.uuid4().hex[:8]
    job_id = f"jtwice{marker}"
    cluster_id = f"ctwice{marker}"
    customer = f"twice-{marker}"
    card = {"gpu_hour_micros": 3_600_000}
    with pg_transaction() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, priority, submitted_at, payload, phase, desired_state,
                effective_priority, spec
            )
            VALUES (%s, 'assigned', 10, 1794000000, %s, 'scheduled', 'running', 10, %s)
            """,
            (
                job_id,
                json.dumps({"owner": customer}),
                Jsonb({"num_gpus": 1}),
            ),
        )
        attempt_id = conn.execute(
            """
            INSERT INTO job_attempts
                (job_id, attempt_number, status, fencing_token, job_generation, created_by)
            VALUES (%s, 1, 'reserved', nextval('placement_fencing_token_seq'), 1, 'test')
            RETURNING attempt_id
            """,
            (job_id,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE jobs SET active_attempt_id = %s WHERE job_id = %s", (attempt_id, job_id)
        )
        conn.execute(
            """
            INSERT INTO slurm_clusters (
                cluster_id, profile_key, display_name, enabled, research_only,
                marketplace, agent_access, poll_interval_seconds, rate_card
            )
            VALUES (%s, 'generic', 'test', false, false, false, false, 60, %s)
            """,
            (cluster_id, Jsonb(card)),
        )
        conn.execute(
            """
            INSERT INTO wallets (customer_id, balance_micros, status, created_at, updated_at)
            VALUES (%s, 100000000, 'active', 1794000000, 1794000000)
            """,
            (customer,),
        )
    sa.register_slurm_job(job_id, "5151", cluster_id=cluster_id, attempt_id=str(attempt_id))
    try:
        with pg_connection() as conn:
            mapping_row = conn.execute(
                """
                SELECT mapping_id, cluster_id, slurm_job_id, xcelsior_job_id,
                       xcelsior_attempt_id, tenant_id
                  FROM slurm_job_mappings WHERE submission_name = %s
                """,
                (job_id,),
            ).fetchone()
        mapping = dict(
            zip(
                (
                    "mapping_id",
                    "cluster_id",
                    "slurm_job_id",
                    "xcelsior_job_id",
                    "xcelsior_attempt_id",
                    "tenant_id",
                ),
                mapping_row,
            )
        )
        with pg_transaction() as conn:
            sb.write_segments(
                conn,
                mapping,
                [_run_row(elapsed_sec=10, ended_at=1_794_000_015)],
                {"rate_card": card, "suspend_holds_gres": True},
            )
            first = sb.settle_finished_attempt(conn, str(attempt_id))
            second = sb.settle_finished_attempt(conn, str(attempt_id))
        assert first["charged_micros"] == 10_000
        assert second["already_settled"] is True
        assert second["charged_micros"] == 0
        with pg_connection() as conn:
            debit = conn.execute(
                """
                SELECT COALESCE(SUM(-amount_micros), 0)
                  FROM wallet_transactions
                 WHERE customer_id = %s AND tx_type = 'charge'
                """,
                (customer,),
            ).fetchone()[0]
        assert int(debit) == 10_000
    finally:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id = %s", (customer,))
            conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cluster_id,))
            conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (cluster_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
            conn.execute("DELETE FROM wallets WHERE customer_id = %s", (customer,))


def test_placement_and_the_shadow_write_do_not_call_charging():
    import bg_worker
    import slurm_reconciler
    from control_plane.scheduler import slurm_placement

    for source in (
        inspect.getsource(slurm_placement.place_on_cluster),
        inspect.getsource(slurm_reconciler._write),
        inspect.getsource(bg_worker),
    ):
        assert "settle_finished_attempt" not in source
        assert "mark_charged" not in source
        assert "_charge_on_connection" not in source
    drift = inspect.getsource(slurm_reconciler.report_shadow_drift)
    assert "report_due_clusters" in drift
    assert "settle_finished_attempt" not in drift
    assert "mark_charged" not in drift
    assert "_charge_on_connection" not in drift


def test_the_parser_ends_every_run_but_the_latest_by_its_successor():
    # REQUEUED names the job's fate; a run with a later run under the same id
    # is over. The latest run still follows the finished-state rule.
    fields = "JobIDRaw|JobName|State|Submit|Start|End|ElapsedRaw|ExitCode|AllocTRES|NodeList|DerivedExitCode|Suspended|TimelimitRaw"
    assert fields.split("|") == list(sa._WINDOW_FIELDS)
    lines = "\n".join(
        [
            "7|xcl-j-a1|REQUEUED|1000|1000|1075|75|0:0|gres/gpu=1|tower|0:0|00:00:00|10",
            "7|xcl-j-a1|PENDING|1077|Unknown|Unknown|0|0:0||None|0:0|00:00:00|10",
            "8|xcl-k-a1|REQUEUED|1100|1100|1130|30|0:0|gres/gpu=1|tower|0:0|00:00:00|10",
        ]
    )
    records = sa._window_records(lines, {}, lambda row: True)
    first, pending = records["7"]
    assert first["ended_at"] == 1075.0 and pending["ended_at"] is None
    # A lone REQUEUED run is the latest one: no successor yet, no end.
    assert records["8"][0]["ended_at"] is None
    assert all("_end_if_over" not in run for runs in records.values() for run in runs)
