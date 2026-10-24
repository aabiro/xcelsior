"""Credits require exact historical evidence and preserve source financial rows."""

from concurrent.futures import ThreadPoolExecutor
import json
import time
import uuid

import pytest
from psycopg.errors import InsufficientPrivilege

from billing import get_billing_engine
from db import _get_pg_pool
from control_plane.financial_history import apply_cluster, qualify_cluster
from tests.test_db_role_workflows import restricted_logins  # noqa: F401
from tests.test_db_service_roles import provisioned_roles  # noqa: F401
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401


@pytest.fixture
def duplicate_cluster():
    owner, job = ["historical-" + uuid.uuid4().hex for _ in range(2)]
    billing = get_billing_engine()
    billing.deposit(owner, 1)
    start = time.time() - 10000
    with _get_pg_pool().connection() as conn:
        for index in range(3):
            end = start + 62 + index * 0.02
            conn.execute(
                """INSERT INTO billing_cycles(cycle_id,job_id,customer_id,host_id,resource_type,period_start,period_end,duration_seconds,rate_per_hour,tier_multiplier,amount_micros,status,created_at)
                VALUES(%s,%s,%s,'','volume',%s,%s,%s,0.004795,1,100,'charged',%s)""",
                (job + str(index), job, owner, start, end, end - start, end),
            )
            conn.execute(
                "UPDATE wallets SET balance_micros=balance_micros-100 WHERE customer_id=%s",
                (owner,),
            )
            conn.execute(
                """INSERT INTO wallet_transactions(tx_id,customer_id,tx_type,amount_micros,balance_after_micros,job_id,created_at)
                VALUES(%s,%s,'charge',-100,%s,%s,%s)""",
                (job + str(index), owner, 999900 - index * 100, job, end + 0.2),
            )
    try:
        yield owner, job, start
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM billing_history_dispositions WHERE job_id=%s", (job,))
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (job,))
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id=%s", (owner,))
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (owner,))


def source_rows(conn, job):
    return (
        conn.execute(
            "SELECT to_jsonb(b) FROM billing_cycles b WHERE job_id=%s ORDER BY cycle_id", (job,)
        ).fetchall(),
        conn.execute(
            "SELECT to_jsonb(t) FROM wallet_transactions t WHERE job_id=%s AND tx_type='charge' ORDER BY tx_id",
            (job,),
        ).fetchall(),
    )


def test_credit_is_atomic_idempotent_and_preserves_source(duplicate_cluster):
    owner, job, start = duplicate_cluster
    with _get_pg_pool().connection() as conn:
        before = source_rows(conn, job)
        plan = qualify_cluster(conn, job, start)
        # The actual CLI round-trips the qualified private plan through JSON.
        plan = json.loads(json.dumps(plan))
        assert plan["credit_micros"] == 200
        result = apply_cluster(conn, plan, max_credit_micros=200)
        assert not result["replayed"]
    with _get_pg_pool().connection() as conn:
        assert apply_cluster(conn, plan, max_credit_micros=200)["replayed"]
        assert source_rows(conn, job) == before
        assert conn.execute(
            "SELECT balance_micros,total_refunded_micros FROM wallets WHERE customer_id=%s",
            (owner,),
        ).fetchone() == (999900, 200)
        assert conn.execute(
            "SELECT balance_micros=(SELECT sum(amount_micros) FROM wallet_transactions WHERE customer_id=%s) FROM wallets WHERE customer_id=%s",
            (owner, owner),
        ).fetchone() == (True,)


def test_failed_receipt_rolls_back_credit(duplicate_cluster):
    owner, job, start = duplicate_cluster
    with _get_pg_pool().connection() as conn:
        plan = qualify_cluster(conn, job, start)
        before = conn.execute(
            "SELECT balance_micros FROM wallets WHERE customer_id=%s", (owner,)
        ).fetchone()
        conn.execute("SAVEPOINT before_credit")
        apply_cluster(conn, plan, max_credit_micros=200)
        conn.execute("ROLLBACK TO SAVEPOINT before_credit")
        assert (
            conn.execute(
                "SELECT balance_micros FROM wallets WHERE customer_id=%s", (owner,)
            ).fetchone()
            == before
        )
        assert conn.execute(
            "SELECT count(*) FROM billing_history_dispositions WHERE job_id=%s", (job,)
        ).fetchone() == (0,)


def test_concurrent_disposition_credits_once(duplicate_cluster):
    _, job, start = duplicate_cluster
    with _get_pg_pool().connection() as conn:
        plan = qualify_cluster(conn, job, start)

    def apply(_):
        with _get_pg_pool().connection() as conn:
            return apply_cluster(conn, plan, max_credit_micros=200)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(apply, range(2)))
    assert sum(not row["replayed"] for row in results) == 1


@pytest.mark.parametrize(
    "change",
    [
        "DELETE FROM wallet_transactions WHERE tx_id=%s",
        "UPDATE billing_cycles SET amount_micros=200 WHERE cycle_id=%s",
        "UPDATE billing_cycles SET period_end=period_end+60 WHERE cycle_id=%s",
    ],
)
def test_ambiguous_or_changed_evidence_never_credits(duplicate_cluster, change):
    owner, job, start = duplicate_cluster
    with _get_pg_pool().connection() as conn:
        plan = qualify_cluster(conn, job, start)
        conn.execute(change, (job + "1",))
        with pytest.raises(ValueError):
            apply_cluster(conn, plan, max_credit_micros=200)
        assert conn.execute(
            "SELECT count(*) FROM billing_history_dispositions WHERE job_id=%s", (job,)
        ).fetchone() == (0,)


def test_credit_budget_cannot_be_exceeded(duplicate_cluster):
    _, job, start = duplicate_cluster
    with _get_pg_pool().connection() as conn:
        with pytest.raises(ValueError, match="budget"):
            apply_cluster(conn, qualify_cluster(conn, job, start), max_credit_micros=199)


def test_runtime_cannot_rewrite_disposition_receipts(restricted_logins):
    import psycopg

    for dsn in restricted_logins.values():
        with psycopg.connect(dsn) as conn:
            with pytest.raises(InsufficientPrivilege):
                conn.execute("DELETE FROM billing_history_dispositions")


@pytest.mark.parametrize("retained_receipt", [False, True])
def test_migration_round_trip_and_retained_receipt_refusal(empty_db, retained_receipt):
    import psycopg
    from tests.test_migration_from_production_snapshot import _alembic

    _alembic(empty_db, "upgrade", "137")
    _alembic(empty_db, "upgrade", "138")
    if retained_receipt:
        with psycopg.connect(empty_db) as conn:
            conn.execute(
                "INSERT INTO wallets(customer_id,balance_micros) VALUES('history-migration',100)"
            )
            conn.execute(
                """INSERT INTO wallet_transactions(tx_id,customer_id,tx_type,amount_micros,balance_after_micros,created_at)
                VALUES('history-migration','history-migration','refund',100,100,EXTRACT(EPOCH FROM clock_timestamp()))"""
            )
            conn.execute(
                """INSERT INTO billing_history_dispositions(disposition_key,tenant_id,job_id,policy_version,disposition,evidence,evidence_sha256,credit_micros,credit_tx_id,created_by)
                VALUES('test','history-migration','historic-volume','volume_overlap/v1','credited_duplicate_interval','{}',repeat('a',64),100,'history-migration',current_user)"""
            )
        result = _alembic(empty_db, "downgrade", "137", check=False)
        assert result.returncode != 0
        with psycopg.connect(empty_db) as conn:
            assert conn.execute("SELECT count(*) FROM billing_history_dispositions").fetchone() == (
                1,
            )
            assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("138",)
    else:
        _alembic(empty_db, "downgrade", "137")
        _alembic(empty_db, "upgrade", "138")
        with psycopg.connect(empty_db) as conn:
            assert conn.execute("SELECT count(*) FROM billing_history_dispositions").fetchone() == (
                0,
            )
