"""Wallet ledger in integer minor units (Track B B9.5a, companion §4.4 r6).

The defect these tests pin is measured, not theoretical. Against this
database, 1000 postings of $0.07 accumulate to 69.99999999999966 rather
than 70.00. Per operation that is negligible; what breaks is that
`sum(amount)` over the ledger stops equalling the stored balance — which
is the comparison `DA§8.7`'s finance reconciliation makes, so the drift
turns a reconciliation report permanently noisy.

Migration 068 makes the minor column authoritative and projects the float
from it, with a **bidirectional** trigger so a rolling deploy where one
replica still writes floats does not lose that write.
"""

from __future__ import annotations

import random
import uuid

import pytest

from db import pg_transaction
from money import cad_to_micros, micros_to_cad

WALLET_MINOR_TABLES = {
    "wallets": ("balance_micros", "total_deposited_micros", "total_spent_micros",
                "total_refunded_micros"),
    "wallet_transactions": ("amount_micros", "balance_after_micros"),
    "wallet_holds": ("amount_micros",),
}


@pytest.fixture
def wallet():
    created: list[str] = []

    def _make(balance_micros: int = 0) -> str:
        customer_id = f"wtest-{uuid.uuid4().hex[:10]}"
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO wallets (customer_id, balance_micros) VALUES (%s, %s)",
                (customer_id, balance_micros),
            )
        created.append(customer_id)
        return customer_id

    yield _make

    if created:
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM wallet_transactions WHERE customer_id = ANY(%s)",
                (created,),
            )
            conn.execute(
                "DELETE FROM wallets WHERE customer_id = ANY(%s)", (created,)
            )


def _row(customer_id: str):
    with pg_transaction() as conn:
        return conn.execute(
            "SELECT balance_micros, balance_cad FROM wallets WHERE customer_id = %s",
            (customer_id,),
        ).fetchone()


# ───────────────────── conversion helpers ─────────────────────


@pytest.mark.parametrize(
    "dollars,cents",
    [(0.07, 70_000), (19.99, 19_990_000), (0.0073, 7_300), (0.000001, 1),
     ("12.34", 12_340_000)],
)
def test_cad_to_micros_is_exact(dollars, cents):
    assert cad_to_micros(dollars) == cents


def test_naive_float_conversion_is_wrong():
    """Why the Decimal path exists. Ordinary value, wrong answer."""
    assert int(1.15 * 100) == 114
    assert cad_to_micros(1.15) == 1_150_000


def test_micros_to_cad_round_trips():
    for cents in (1, 7, 1999, 2500, 100_000, 99_999_999):
        assert cad_to_micros(micros_to_cad(cents)) == cents


# ───────────────────── the drift itself ─────────────────────


def test_integer_accumulation_does_not_drift(wallet):
    """The regression, driven against the real column.

    A float balance incremented 1000 times by $0.07 lands on
    69.99999999999966. The integer column must land on exactly 7000.
    """
    customer_id = wallet(0)

    with pg_transaction() as conn:
        for _ in range(1000):
            conn.execute(
                "UPDATE wallets SET balance_micros = balance_micros + 70000 "
                "WHERE customer_id = %s",
                (customer_id,),
            )

    row = _row(customer_id)
    assert row is not None
    assert row[0] == 70_000_000, f"integer balance drifted: {row[0]}"
    assert row[1] == 70.0, f"projected float is wrong: {row[1]!r}"


def test_float_column_would_have_drifted():
    """Documents the baseline this migration removes.

    If this ever stops holding, binary floating point changed and the
    whole item deserves revisiting.
    """
    total = 0.0
    for _ in range(1000):
        total += 0.07
    assert total != 70.0
    assert abs(total - 70.0) < 1e-9  # small, but not zero — that is the point


def test_ledger_sum_equals_stored_balance(wallet):
    """The equality finance reconciliation depends on (`DA§8.7`)."""
    customer_id = wallet(0)
    postings = [random.randint(-5000, 5000) for _ in range(200)]

    with pg_transaction() as conn:
        running = 0
        for i, amount in enumerate(postings):
            running += amount
            conn.execute(
                "UPDATE wallets SET balance_micros = balance_micros + %s "
                "WHERE customer_id = %s",
                (amount, customer_id),
            )
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_micros,
                    balance_after_micros, description, created_at, idempotency_key)
                   VALUES (%s, %s, 'adjust', %s, %s, '', 0, %s)""",
                (f"tx-{customer_id}-{i}", customer_id, amount, running,
                 f"key-{customer_id}-{i}"),
            )

    with pg_transaction() as conn:
        ledger_sum = conn.execute(
            "SELECT sum(amount_micros) FROM wallet_transactions WHERE customer_id = %s",
            (customer_id,),
        ).fetchone()
        stored = conn.execute(
            "SELECT balance_micros FROM wallets WHERE customer_id = %s",
            (customer_id,),
        ).fetchone()

    assert ledger_sum is not None and stored is not None
    assert ledger_sum[0] == stored[0] == sum(postings), (
        "the ledger sum and the stored balance disagree; this is exactly "
        "the discrepancy a finance reconciliation would report"
    )


# ───────────── rolling-deploy safety: both writer generations ─────────────


def test_new_writer_updates_micros_and_float_follows(wallet):
    customer_id = wallet(1_000_000)

    with pg_transaction() as conn:
        conn.execute(
            "UPDATE wallets SET balance_micros = balance_micros + 333 "
            "WHERE customer_id = %s",
            (customer_id,),
        )

    assert _row(customer_id) == (1_000_333, 1.000333)


def test_legacy_writer_updates_float_and_micros_follows(wallet):
    """The property that makes a rolling deploy safe.

    An un-upgraded replica writes only `balance_cad`. A one-way trigger
    would overwrite that float from a stale `balance_micros`, silently
    discarding the write.
    """
    customer_id = wallet(1_000_000)

    with pg_transaction() as conn:
        conn.execute(
            "UPDATE wallets SET balance_cad = balance_cad + 1.11 "
            "WHERE customer_id = %s",
            (customer_id,),
        )

    row = _row(customer_id)
    assert row is not None
    assert row[0] == 2_110_000, (
        f"legacy float write was lost or mis-projected: {row}"
    )
    # The float keeps the legacy writer's own arithmetic result
    # (1.0 + 1.11 == 2.1100000000000003). That is deliberate: the legacy
    # branch derives micros from the float and does NOT reproject the
    # float, because clobbering it would discard the write this branch
    # exists to preserve.
    assert row[1] == pytest.approx(2.11)


def test_insert_from_either_generation_is_consistent():
    """Both insert shapes must leave the row self-consistent."""
    minor_only = f"wtest-{uuid.uuid4().hex[:10]}"
    float_only = f"wtest-{uuid.uuid4().hex[:10]}"
    try:
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO wallets (customer_id, balance_micros) VALUES (%s, %s)",
                (minor_only, 4_242_000),
            )
            conn.execute(
                "INSERT INTO wallets (customer_id, balance_cad) VALUES (%s, %s)",
                (float_only, 12.34),
            )
        assert _row(minor_only) == (4_242_000, 4.242)
        assert _row(float_only) == (12_340_000, 12.34)
    finally:
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM wallets WHERE customer_id = ANY(%s)",
                ([minor_only, float_only],),
            )


# ───────────────────── schema contract ─────────────────────


def test_every_wallet_money_column_has_a_micros_twin():
    with pg_transaction() as conn:
        for table, columns in WALLET_MINOR_TABLES.items():
            present = {
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s",
                    (table,),
                ).fetchall()
            }
            missing = set(columns) - present
            assert not missing, f"{table} is missing minor columns: {missing}"


def test_micros_columns_are_integers():
    with pg_transaction() as conn:
        for table, columns in WALLET_MINOR_TABLES.items():
            for column in columns:
                row = conn.execute(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name = %s",
                    (table, column),
                ).fetchone()
                assert row is not None and row[0] == "bigint", (
                    f"{table}.{column} is {row[0] if row else 'missing'}, not "
                    f"bigint — money must not be a binary float"
                )


def test_accumulating_totals_cannot_go_negative(wallet):
    import psycopg

    customer_id = wallet(0)
    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE wallets SET total_spent_micros = -1 WHERE customer_id = %s",
                (customer_id,),
            )


def test_balance_may_be_negative(wallet):
    """Overdraft/grace is legitimate; only the totals are monotonic."""
    customer_id = wallet(0)
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE wallets SET balance_micros = -5000000 WHERE customer_id = %s",
            (customer_id,),
        )
    assert _row(customer_id) == (-5_000_000, -5.0)
