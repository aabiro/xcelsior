"""Shadow billing rules for one Slurm accounting run (decision D8, blueprint 2.3).

These checks do not touch the database. The ledger row is written when a
mapping settles; that path is covered with the reconciler.
"""

from __future__ import annotations

import slurm_billing as sb


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
