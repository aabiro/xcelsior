"""A wallet that cannot cover a GPU slice pays what it has and is suspended.

`_charge_on_connection` refused a charge larger than a positive balance, debited
nothing, and left the wallet active. The failed cycle queued a stop, but start
and restart only required `available > 0`, so the customer could stop and start
again and receive another unpaid slice each time, indefinitely.

Policy now:

* the remaining balance is collected and journaled, the shortfall is recorded
  as an unpaid (failed) cycle, and the wallet is suspended — the hard stop the
  existing stop paths and start gates already act on;
* the shortfall is owed again by the attempt's next charge, so a top-up pays
  exactly what is missing, never the collected part twice;
* start and restart require one billing slice of the workload to be available.

The ledger invariant holds throughout: the balance equals the journal sum.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

import billing as billing_module
from billing import get_billing_engine
from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance
from tests.native_jobs import admit_fixture_job
from tests.test_data_layer_readiness import finish_compute


def _ledger(owner: str) -> tuple[int, int, str]:
    with _get_pg_pool().connection() as conn:
        balance, status = conn.execute(
            "SELECT balance_micros,status FROM wallets WHERE customer_id=%s", (owner,)
        ).fetchone()
        journal = conn.execute(
            "SELECT COALESCE(SUM(amount_micros),0) FROM wallet_transactions WHERE customer_id=%s",
            (owner,),
        ).fetchone()[0]
    return int(balance), int(journal), str(status)


def _exact_meter_start(job_id: str) -> None:
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE usage_meters m SET started_at=EXTRACT(EPOCH FROM a.started_at)::float8 "
            "FROM job_attempts a WHERE m.job_id=%s AND a.attempt_id::text=m.attempt_id",
            (job_id,),
        )


def test_partial_balance_is_collected_the_wallet_suspended_and_the_job_stopped():
    owner = f"partial-{uuid.uuid4().hex[:10]}"
    engine = get_billing_engine()
    engine.deposit(owner, 0.25)
    # 3.6/h for 600 s owes 0.60 against a 0.25 balance.
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=600) as native:
        job_id = native["job_id"]
        _exact_meter_start(job_id)
        result = engine.auto_billing_cycle()
        assert result["errors"] == 0

        balance, journal, status = _ledger(owner)
        assert balance == 0, "the remaining balance must be collected, not left unbilled"
        assert balance == journal, "wallet balance must equal its journal"
        assert status == "suspended", (
            "a wallet that cannot cover running compute must hard-stop, or start/stop "
            "loops get an unpaid slice every time"
        )
        with _get_pg_pool().connection() as conn:
            rows = conn.execute(
                "SELECT status,amount_micros,period_start,period_end FROM billing_cycles "
                "WHERE job_id=%s AND resource_type='gpu' ORDER BY period_start",
                (job_id,),
            ).fetchall()
            charged = [r for r in rows if r[0] == "charged"]
            failed = [r for r in rows if r[0] == "failed"]
            assert [r[1] for r in charged] == [250_000]
            assert len(failed) == 1 and failed[0][1] > 0
            assert charged[0][3] == failed[0][2], "the shortfall begins where payment ended"
            stop = conn.execute(
                "SELECT payload FROM outbox_events WHERE aggregate_id=%s "
                "AND destination_class='billing_stop'",
                (job_id,),
            ).fetchone()
            assert stop is not None, "the existing stop path must fire"
            charge_tx = conn.execute(
                "SELECT amount_micros,balance_after_micros FROM wallet_transactions "
                "WHERE customer_id=%s AND tx_type='charge'",
                (owner,),
            ).fetchall()
        assert charge_tx == [(-250_000, 0)]

        gate = engine.wallet_has_available_funds(owner)
        assert gate["ok"] is False and gate["reason"] == "wallet_suspended"

        # Top up, reactivate, finish: the final slice owes the shortfall plus
        # the rest of the run, and never the 0.25 already paid.
        engine.deposit(owner, 5)
        engine.reactivate_wallet(owner)
        finish_compute(native)
        assert engine.auto_billing_cycle()["errors"] == 0
        with _get_pg_pool().connection() as conn:
            metered = conn.execute(
                "SELECT total_cost_micros FROM usage_meters WHERE job_id=%s", (job_id,)
            ).fetchone()[0]
            paid = conn.execute(
                "SELECT -SUM(amount_micros) FROM wallet_transactions "
                "WHERE job_id=%s AND tx_type='charge'",
                (job_id,),
            ).fetchone()[0]
        assert paid == metered, "after the top-up the attempt is paid exactly once in full"
        balance, journal, status = _ledger(owner)
        assert balance == journal and status == "active"


def test_volume_charge_the_balance_cannot_cover_suspends_without_a_partial_debit(
    monkeypatch,
):
    """Volume cycles cannot record a partial payment, so they debit nothing but still stop."""
    from volumes import VolumeEngine

    owner = f"partial-vol-{uuid.uuid4().hex[:10]}"
    engine = get_billing_engine()
    engine.deposit(owner, 0.000001)
    volumes = VolumeEngine()
    monkeypatch.setattr(volumes, "_provision_volume_storage", lambda *a, **kw: True)
    monkeypatch.setattr(volumes, "_emit_event", lambda *a, **kw: None)
    volume = volumes.create_volume(owner, "partial-vol", 100)
    try:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE volumes SET created_at=EXTRACT(EPOCH FROM clock_timestamp())-86400 "
                "WHERE volume_id=%s",
                (volume["volume_id"],),
            )
        engine.auto_billing_cycle()
        balance, journal, status = _ledger(owner)
        assert balance == 1 and journal == 1, "no partial debit without a record for it"
        assert status == "suspended"
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT status FROM billing_cycles WHERE job_id=%s", (volume["volume_id"],)
            ).fetchall() == [("failed",)]
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (volume["volume_id"],))
            conn.execute("DELETE FROM volumes WHERE volume_id=%s", (volume["volume_id"],))


@pytest.fixture
def stopped_job():
    made: dict[str, list[str]] = {"customers": [], "jobs": [], "hosts": []}

    def make(owner: str) -> str:
        job_id = f"j-slice-{uuid.uuid4().hex[:8]}"
        host_id = f"h-slice-{uuid.uuid4().hex[:8]}"
        now = time.time()
        payload = {"owner": owner, "status": "stopped", "container_name": f"xcl-{job_id}"}
        with _get_pg_pool().connection() as conn:
            conn.execute(
                """INSERT INTO hosts (host_id,status,registered_at,payload,admission_state,
                   administrative_state,availability_state)
                   VALUES (%s,'active',%s,%s,'admitted','admitted','ready')""",
                (host_id, now, json.dumps({"host_id": host_id})),
            )
            conn.execute(
                """INSERT INTO jobs (job_id,status,priority,submitted_at,host_id,payload,phase,
                   desired_state,effective_priority,queued_at)
                   VALUES (%s,'stopped',0,%s,%s,%s,'stopped','stopped',0,NULL)""",
                (job_id, now - 1000, host_id, json.dumps(payload)),
            )
            admit_fixture_job(conn, job_id)
            conn.execute(
                """INSERT INTO job_attempts(attempt_id,job_id,attempt_number,status,host_id,
                          fencing_token,spec_hash,ended_at)
                   SELECT %s,job_id,1,'cancelled',host_id,nextval('placement_fencing_token_seq'),
                          spec_hash,clock_timestamp() FROM jobs WHERE job_id=%s""",
                (str(uuid.uuid4()), job_id),
            )
        made["customers"].append(owner)
        made["jobs"].append(job_id)
        made["hosts"].append(host_id)
        return job_id

    yield make
    with _get_pg_pool().connection() as conn:
        for job_id in made["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        for host_id in made["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
        for owner in made["customers"]:
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id=%s", (owner,))
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (owner,))


def test_start_and_restart_require_one_billing_slice(stopped_job, monkeypatch):
    monkeypatch.setattr("routes.agent.enqueue_agent_command", lambda *a, **k: 1)
    engine = get_billing_engine()
    owner = f"slice-gate-{uuid.uuid4().hex[:10]}"
    # A workload that never ran is priced at the launch estimate (0.20/h).
    engine.deposit(owner, 0.01)
    job_id = stopped_job(owner)
    slice_cad = 0.20 * billing_module.PERIODIC_BILLING_INTERVAL_SEC / 3600
    assert slice_cad > 0.01

    refused = engine.start_instance(job_id)
    assert refused["started"] is False and refused["reason"] == "insufficient_balance"
    assert refused["required_cad"] == pytest.approx(slice_cad, abs=1e-6)
    assert engine.restart_instance(job_id)["restarted"] is False

    engine.deposit(owner, 0.01)
    started = engine.start_instance(job_id)
    assert started["started"] is True, started


def test_the_start_gate_prices_a_slice_at_the_rate_the_workload_last_ran_at():
    owner = f"slice-rate-{uuid.uuid4().hex[:10]}"
    get_billing_engine().deposit(owner, 10)
    with running_instance(owner=owner, cost_per_hour=3.6) as native:
        from psycopg.rows import dict_row

        with _get_pg_pool().connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                "SELECT job_id,spec,spec_hash FROM jobs WHERE job_id=%s", (native["job_id"],)
            ).fetchone()
            assert get_billing_engine().resume_slice_cad(conn, job) == pytest.approx(
                3.6 * billing_module.PERIODIC_BILLING_INTERVAL_SEC / 3600
            )
