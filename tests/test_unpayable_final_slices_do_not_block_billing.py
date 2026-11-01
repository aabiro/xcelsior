"""Final slices that cannot be billed must not stop every other final slice.

`_settle_terminal_gpu_periods` selected pending final slices oldest first with
`LIMIT 200`, and re-selected a slice that had already failed whenever its
wallet's `updated_at` moved — which any successful charge or deposit does. A
customer with many finished but unpayable attempts was retried every tick,
recorded a fresh failed cycle every tick, and once 200 rows were stuck the
batch never reached anyone else's final slice.

Now a failed final slice is retried only when the wallet can cover the amount
that failed, and the pending set is keyset-paged so rows that keep failing for
any reason cannot occupy the whole batch.
"""

from __future__ import annotations

import json
import time
import uuid

from psycopg.types.json import Jsonb

from billing import get_billing_engine
from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance
from tests.native_jobs import admit_fixture_job
from tests.test_data_layer_readiness import finish_compute


def _stuck_attempts(owner: str, count: int, *, ended_ago_sec: float) -> tuple[str, list[str]]:
    """Finished native attempts with a positive meter that billing can never price.

    They carry no captured billing snapshot, so every settlement attempt raises
    and the row stays pending — the shape of any final slice that keeps failing.
    """
    host_id = f"stuck-host-{uuid.uuid4().hex[:8]}"
    jobs: list[str] = []
    now = time.time()
    with _get_pg_pool().connection() as conn:
        conn.execute(
            """INSERT INTO hosts(host_id,status,registered_at,payload,admission_state,
               administrative_state,availability_state) VALUES(%s,'active',%s,%s,
               'admitted','admitted','ready')""",
            (host_id, now, json.dumps({"host_id": host_id})),
        )
        for i in range(count):
            job_id = f"stuck-{uuid.uuid4().hex[:12]}"
            conn.execute(
                """INSERT INTO jobs(job_id,status,priority,submitted_at,host_id,payload,phase,
                   desired_state,effective_priority,owner_id)
                   VALUES(%s,'completed',0,%s,%s,%s,'succeeded','stopped',0,%s)""",
                (job_id, now, host_id, Jsonb({"owner": owner, "name": job_id}), owner),
            )
            admit_fixture_job(conn, job_id)
            ended = now - ended_ago_sec + i * 0.001
            attempt_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO job_attempts(attempt_id,job_id,attempt_number,status,host_id,
                   fencing_token,spec_hash,started_at,ended_at)
                   SELECT %s,job_id,1,'succeeded',host_id,nextval('placement_fencing_token_seq'),
                          spec_hash,to_timestamp(%s),to_timestamp(%s) FROM jobs WHERE job_id=%s""",
                (attempt_id, ended - 600, ended, job_id),
            )
            conn.execute(
                """INSERT INTO usage_meters(meter_id,job_id,host_id,owner,started_at,completed_at,
                   duration_sec,base_rate_per_hour,total_cost_micros,created_at,attempt_id)
                   VALUES(%s,%s,%s,%s,%s,%s,600,1,166700,%s,%s)""",
                (
                    uuid.uuid4().hex[:12],
                    job_id,
                    host_id,
                    owner,
                    ended - 600,
                    ended,
                    now,
                    attempt_id,
                ),
            )
            jobs.append(job_id)
    return host_id, jobs


def _cleanup(host_id: str, jobs: list[str], owners: list[str]) -> None:
    with _get_pg_pool().connection() as conn:
        for job_id in jobs:
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM usage_meters WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
        for owner in owners:
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id=%s", (owner,))
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (owner,))


def test_more_than_a_batch_of_stuck_final_slices_cannot_starve_a_payable_one():
    engine = get_billing_engine()
    stuck_owner = f"stuck-owner-{uuid.uuid4().hex[:8]}"
    engine.deposit(stuck_owner, 100)
    host_id, stuck = _stuck_attempts(stuck_owner, 201, ended_ago_sec=86_400)
    owner = f"payable-{uuid.uuid4().hex[:8]}"
    engine.deposit(owner, 100)
    try:
        with running_instance(owner=owner, cost_per_hour=3.6, age_sec=600) as native:
            finish_compute(native)
            engine.auto_billing_cycle()
            with _get_pg_pool().connection() as conn:
                paid = conn.execute(
                    "SELECT count(*) FROM billing_cycles WHERE job_id=%s AND status='charged'",
                    (native["job_id"],),
                ).fetchone()[0]
            assert paid == 1, (
                "201 older final slices that never settle filled the 200-row batch, so this "
                "payable final slice was never billed"
            )
    finally:
        _cleanup(host_id, stuck, [stuck_owner, owner])


def test_a_failed_final_slice_is_retried_only_once_the_wallet_can_pay_it():
    engine = get_billing_engine()
    owner = f"retry-{uuid.uuid4().hex[:8]}"
    engine.deposit(owner, 100)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=600) as native:
        job_id = native["job_id"]
        finish_compute(native)
        with _get_pg_pool().connection() as conn:
            ended = conn.execute(
                "SELECT EXTRACT(EPOCH FROM ended_at)::float8,EXTRACT(EPOCH FROM started_at)::float8 "
                "FROM job_attempts WHERE job_id=%s",
                (job_id,),
            ).fetchone()
            # The final slice already failed for 0.60, and the wallet now holds
            # 0.10: active, positive, and still unable to pay it.
            conn.execute(
                """INSERT INTO billing_cycles(cycle_id,job_id,customer_id,host_id,resource_type,
                   period_start,period_end,duration_seconds,rate_per_hour,amount_micros,status,
                   created_at) VALUES(%s,%s,%s,%s,'gpu',%s,%s,600,3.6,600000,'failed',%s)""",
                (
                    "BC-failed-" + uuid.uuid4().hex[:6],
                    job_id,
                    owner,
                    native["host_id"],
                    ended[1],
                    ended[0],
                    time.time() - 60,
                ),
            )
            conn.execute("UPDATE wallets SET balance_micros=100000 WHERE customer_id=%s", (owner,))
            conn.execute(
                "INSERT INTO wallet_transactions(tx_id,customer_id,tx_type,amount_micros,"
                "balance_after_micros,description,created_at) VALUES(%s,%s,'charge',%s,100000,"
                "'test drawdown',%s)",
                ("TX-test-" + uuid.uuid4().hex[:8], owner, -99_900_000, time.time()),
            )
        # Any later wallet change (another job's charge, a deposit) used to
        # re-qualify the failed slice.
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE wallets SET updated_at=EXTRACT(EPOCH FROM clock_timestamp())+1 "
                "WHERE customer_id=%s",
                (owner,),
            )
        for _ in range(2):
            engine.auto_billing_cycle()
        with _get_pg_pool().connection() as conn:
            rows = conn.execute(
                "SELECT status,count(*) FROM billing_cycles WHERE job_id=%s GROUP BY status",
                (job_id,),
            ).fetchall()
        assert dict(rows) == {
            "failed": 1
        }, f"an unpayable final slice was retried anyway: {dict(rows)}"
        assert engine.get_wallet(owner)["status"] == "active"

        engine.deposit(owner, 1)
        engine.auto_billing_cycle()
        with _get_pg_pool().connection() as conn:
            rows = conn.execute(
                "SELECT status,count(*) FROM billing_cycles WHERE job_id=%s GROUP BY status",
                (job_id,),
            ).fetchall()
        assert dict(rows).get("charged") == 1, "once the wallet covers it, the slice is billed"
