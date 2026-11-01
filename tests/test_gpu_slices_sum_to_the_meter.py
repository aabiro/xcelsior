"""Every GPU slice of an attempt sums to exactly what its meter says it cost.

Provider settlement pays out only when a job's wallet charges equal its terminal
usage meters to the micro (`billing_not_reconciled` otherwise). The meter rounds
the whole attempt once. Periodic billing used to round each five-minute slice
on its own, so any job billed in more than one slice drifted by up to 50 micros
a slice and its provider was never paid.

A slice is now what the attempt owes through its end, rounded once, less what
it already paid. Jittered slice boundaries cannot move the total.
"""

from __future__ import annotations

import uuid

from billing import get_billing_engine
from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance
from tests.test_data_layer_readiness import finish_compute


def _exact_meter_start(job_id: str) -> float:
    """The fixture writes the meter start as a Python float; production uses the attempt's."""
    with _get_pg_pool().connection() as conn:
        started = conn.execute(
            "SELECT EXTRACT(EPOCH FROM a.started_at)::float8 FROM jobs j "
            "JOIN job_attempts a ON a.attempt_id=j.active_attempt_id WHERE j.job_id=%s",
            (job_id,),
        ).fetchone()[0]
        conn.execute("UPDATE usage_meters SET started_at=%s WHERE job_id=%s", (started, job_id))
    return started


def _charged_and_metered(job_id: str) -> tuple[int, int]:
    with _get_pg_pool().connection() as conn:
        charged = conn.execute(
            "SELECT COALESCE(-SUM(amount_micros),0) FROM wallet_transactions "
            "WHERE job_id=%s AND tx_type='charge'",
            (job_id,),
        ).fetchone()[0]
        metered = conn.execute(
            "SELECT total_cost_micros FROM usage_meters WHERE job_id=%s AND completed_at>0",
            (job_id,),
        ).fetchone()[0]
    return int(charged), int(metered)


def test_jittered_periodic_slices_and_the_final_slice_equal_the_meter():
    owner = f"slice-sum-{uuid.uuid4().hex[:10]}"
    engine = get_billing_engine()
    engine.deposit(owner, 100)
    # 3.6/h is 0.001 CAD a second, so each 300.04 s slice is 0.30004 CAD:
    # rounded alone it loses 40 micros, ten times over.
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=3600) as native:
        job_id = native["job_id"]
        started = _exact_meter_start(job_id)
        for k in range(1, 11):
            result = engine.bill_running_period(job_id, period_end=started + k * 300.04)
            assert result["billed"] is True, result
        finish_compute(native)
        assert engine.auto_billing_cycle()["errors"] == 0
        charged, metered = _charged_and_metered(job_id)
        assert metered > 0
        assert charged == metered, (
            f"slices charged {charged} micros but the meter says {metered}; provider "
            "settlement refuses any job whose charges and meter differ"
        )
        with _get_pg_pool().connection() as conn:
            cycles = conn.execute(
                "SELECT count(*),COALESCE(SUM(amount_micros),0) FROM billing_cycles "
                "WHERE job_id=%s AND resource_type='gpu' AND status='charged'",
                (job_id,),
            ).fetchone()
        assert cycles == (11, metered)


def test_a_slice_is_never_negative_after_earlier_rounding():
    """Slices rounded under the old rule may have overpaid; the next is clamped at zero."""
    owner = f"slice-clamp-{uuid.uuid4().hex[:10]}"
    engine = get_billing_engine()
    engine.deposit(owner, 100)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=600) as native:
        job_id = native["job_id"]
        started = _exact_meter_start(job_id)
        assert engine.bill_running_period(job_id, period_end=started + 300)["billed"]
        with _get_pg_pool().connection() as conn:
            # An earlier slice that charged 0.0004 more than it owed.
            conn.execute(
                "UPDATE billing_cycles SET amount_micros=amount_micros+400 "
                "WHERE job_id=%s AND status='charged'",
                (job_id,),
            )
        before = engine.get_wallet(owner)["balance_micros"]
        result = engine.bill_running_period(job_id, period_end=started + 300.2)
        assert result["billed"] is False
        assert engine.get_wallet(owner)["balance_micros"] == before
