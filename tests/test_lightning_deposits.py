"""Lightning deposit integrity (Track B B9.3b).

The Lightning module was migrated from SQLite to PostgreSQL by migration
`060_shared_state_to_pg` but its query layer was never converted or
exercised — `dict_row` was imported and unused, four functions called
`dict(row)` on `tuple_row` connections, and three carried SQLite `?`
placeholders. The result was that **no paid Lightning deposit was ever
credited**: `process_ln_deposits` raised `ValueError` on the first pending
row, outside any try block, killing the sweep for every customer.

Naively fixing only the placeholders would have been worse than the bug:
`mark_credited` failing after a successful credit leaves the deposit
`paid`, and the watcher re-runs every 5 seconds — an unbounded re-credit
loop, because neither credit callback passed an idempotency key.

Every test here drives the real module against a real database.

Companion §10.1 (Lightning requirements), §2.7, §4.4; Track B §B9.3.
"""

from __future__ import annotations

import time
import uuid

import pytest

import lightning as ln
from db import pg_transaction

pytestmark = pytest.mark.usefixtures("_ln_enabled")


@pytest.fixture
def _ln_enabled(monkeypatch):
    """The module gates every entry point on this flag."""
    monkeypatch.setattr(ln, "LN_ENABLED", True)


def _insert_deposit(
    deposit_id: str, *, status: str, customer_id: str, amount_cad: float = 25.0
) -> None:
    with pg_transaction() as conn:
        conn.execute(
            """INSERT INTO ln_deposits
                (deposit_id,customer_id,label,bolt11,payment_hash,
                 amount_msat,amount_sats,amount_micros,amount_cad_minor,
                 btc_cad_rate_exact,status,created_at_ts,expires_at_ts)
            VALUES(%s,%s,%s,%s,%s,1000000,1000,%s,%s,100000,%s,
                clock_timestamp(),clock_timestamp()+interval '1 day')""",
            (
                deposit_id,
                customer_id,
                f"xcelsior-{deposit_id}",
                f"lnbc-{deposit_id}",
                f"hash-{deposit_id}",
                round(amount_cad * 1_000_000),
                round(amount_cad * 100),
                status,
            ),
        )


def _status_of(deposit_id: str) -> str | None:
    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT status FROM ln_deposits WHERE deposit_id = %s", (deposit_id,)
        ).fetchone()
        return None if row is None else row[0]


@pytest.fixture
def deposits():
    """Track seeded deposit ids and clean up only those rows."""
    created: list[str] = []

    def _make(
        status: str, *, customer_id: str | None = None, amount_cad: float = 25.0
    ) -> tuple[str, str]:
        deposit_id = f"lntest-{uuid.uuid4().hex[:12]}"
        cust = customer_id or f"cust-{uuid.uuid4().hex[:8]}"
        _insert_deposit(deposit_id, status=status, customer_id=cust, amount_cad=amount_cad)
        created.append(deposit_id)
        return deposit_id, cust

    yield _make

    if created:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM ln_deposits WHERE deposit_id = ANY(%s)", (created,))


# ───────────────────── query-layer regressions ─────────────────────


def test_pending_query_returns_mappings_not_tuples(deposits):
    """`dict(row)` on a tuple_row connection raised ValueError.

    This is the failure that killed the whole sweep, because
    `get_pending_deposits()` is called outside any try block.
    """
    deposit_id, customer_id = deposits("pending")

    rows = ln.get_pending_deposits()

    match = [r for r in rows if r["deposit_id"] == deposit_id]
    assert match, "seeded pending deposit not returned"
    assert match[0]["customer_id"] == customer_id
    assert match[0]["status"] == "pending"


def test_paid_uncredited_query_returns_mappings(deposits):
    deposit_id, _ = deposits("paid")
    rows = ln.get_paid_uncredited()
    assert any(r["deposit_id"] == deposit_id for r in rows)


def test_customer_deposit_history_uses_postgres_placeholders(deposits):
    """`get_customer_deposits` carried SQLite `?` placeholders.

    psycopg3 raises ProgrammingError("the query has 0 placeholders but 2
    parameters were passed") — so a customer's deposit history was a 500.
    """
    customer_id = f"cust-{uuid.uuid4().hex[:8]}"
    first, _ = deposits("credited", customer_id=customer_id)
    second, _ = deposits("pending", customer_id=customer_id)

    rows = ln.get_customer_deposits(customer_id, limit=10)

    returned = {r["deposit_id"] for r in rows}
    assert {first, second} <= returned
    assert all(r["customer_id"] == customer_id for r in rows)


def test_customer_deposit_history_respects_limit(deposits):
    customer_id = f"cust-{uuid.uuid4().hex[:8]}"
    for _ in range(3):
        deposits("credited", customer_id=customer_id)
    assert len(ln.get_customer_deposits(customer_id, limit=2)) == 2


# ───────────────────── mark_credited semantics ─────────────────────


def test_mark_credited_transitions_paid_to_credited(deposits):
    """`mark_credited` carried `?` placeholders and always raised."""
    deposit_id, _ = deposits("paid")

    assert ln.mark_credited(deposit_id) is True
    assert _status_of(deposit_id) == "credited"


def test_mark_credited_is_compare_and_swap(deposits):
    """A second mark must not re-transition an already-credited deposit."""
    deposit_id, _ = deposits("paid")

    assert ln.mark_credited(deposit_id) is True
    assert ln.mark_credited(deposit_id) is False, (
        "mark_credited must compare-and-swap on status='paid'; a duplicate "
        "call reporting success would mask a double-credit"
    )
    assert _status_of(deposit_id) == "credited"


def test_mark_credited_refuses_a_pending_deposit(deposits):
    """Only a *paid* deposit may be credited."""
    deposit_id, _ = deposits("pending")

    assert ln.mark_credited(deposit_id) is False
    assert _status_of(deposit_id) == "pending"


# ───────────────────── sweep behaviour ─────────────────────


def test_sweep_credits_a_paid_deposit_exactly_once(deposits):
    deposit_id, customer_id = deposits("paid", amount_cad=42.5)
    credits: list[tuple] = []

    ln.process_ln_deposits(credit_callback=lambda c, a, d: credits.append((c, a, d)))

    assert credits == [
        (customer_id, 42.5, deposit_id)
    ], f"expected exactly one credit for the paid deposit, got {credits}"
    assert _status_of(deposit_id) == "credited"


def test_sweep_does_not_recredit_on_the_next_pass(deposits):
    """The regression that would have printed money.

    Before the fix, `mark_credited` raised, the exception was swallowed,
    the deposit stayed `paid`, and the 5-second watcher credited it again
    forever.
    """
    deposit_id, customer_id = deposits("paid")
    credits: list[tuple] = []

    def _cb(c, a, d):
        credits.append((c, a, d))

    for _ in range(3):
        ln.process_ln_deposits(credit_callback=_cb)

    assert len(credits) == 1, (
        f"deposit {deposit_id} was credited {len(credits)} times across three "
        f"sweeps; it must be credited once and then leave the "
        f"paid-uncredited set"
    )
    assert _status_of(deposit_id) == "credited"


def test_one_failing_deposit_does_not_stop_the_others(deposits):
    """Per-deposit isolation — the sweep serves every customer.

    The original code raised out of `get_pending_deposits()` before the
    loop even started, so one bad row starved the entire fleet.
    """
    bad_id, _ = deposits("paid", customer_id="cust-explodes")
    good_id, good_cust = deposits("paid", customer_id="cust-fine")
    credited: list[str] = []

    def _cb(customer_id, amount_cad, deposit_id):
        if deposit_id == bad_id:
            raise RuntimeError("simulated wallet outage")
        credited.append(deposit_id)

    ln.process_ln_deposits(credit_callback=_cb)

    assert good_id in credited, "a healthy deposit was starved by a failing one"
    assert _status_of(good_id) == "credited"
    # The failed one stays claimable so the next sweep retries it.
    assert _status_of(bad_id) == "paid"


def test_failed_credit_is_retried_and_settles(deposits):
    """A transient wallet failure must not strand the deposit."""
    deposit_id, _ = deposits("paid")
    attempts: list[str] = []
    fail_first = {"n": 0}

    def _cb(customer_id, amount_cad, dep_id):
        attempts.append(dep_id)
        fail_first["n"] += 1
        if fail_first["n"] == 1:
            raise RuntimeError("transient")

    ln.process_ln_deposits(credit_callback=_cb)
    assert _status_of(deposit_id) == "paid", "a failed credit must not mark credited"

    ln.process_ln_deposits(credit_callback=_cb)
    assert _status_of(deposit_id) == "credited"
    assert len(attempts) == 2


def test_sweep_is_a_noop_when_disabled(monkeypatch, deposits):
    deposit_id, _ = deposits("paid")
    monkeypatch.setattr(ln, "LN_ENABLED", False)
    credits: list[tuple] = []

    ln.process_ln_deposits(credit_callback=lambda *a: credits.append(a))

    assert credits == []
    assert _status_of(deposit_id) == "paid"


# ───────────────── idempotency contract at the call sites ─────────────────


def test_credit_idempotency_key_is_deposit_scoped():
    key = ln.credit_idempotency_key("ln-abc123")
    assert key == "ln-deposit:ln-abc123"
    assert ln.credit_idempotency_key("ln-other") != key


def test_every_credit_callback_passes_an_idempotency_key():
    """Structural gate — the ledger dedupe is not optional.

    `process_ln_deposits` retries a deposit whose `mark_credited` did not
    land. That retry is only safe because the wallet credit deduplicates
    on the deposit id. A new call site that forgets the key silently
    reintroduces the re-credit loop, so fail the build instead.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    call_sites = []

    for path in (root / "workers" / "bg_worker.py", root / "api" / "api.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if "ln_credit_callback" not in node.name:
                continue
            deposits_called = [
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "deposit"
            ]
            assert deposits_called, (
                f"{path.name}:{node.name} no longer calls .deposit(); update "
                f"this gate to follow the credit path"
            )
            for call in deposits_called:
                kwargs = {kw.arg for kw in call.keywords}
                assert "idempotency_key" in kwargs, (
                    f"{path.name}:{node.name} calls BillingEngine.deposit "
                    f"without idempotency_key. process_ln_deposits retries "
                    f"failed credits, so this re-credits the wallet on every "
                    f"sweep. Pass "
                    f"idempotency_key=lightning.credit_idempotency_key(deposit_id)."
                )
            call_sites.append(f"{path.name}:{node.name}")

    assert len(call_sites) == 2, (
        f"expected the two known Lightning credit callbacks, found "
        f"{call_sites}. A new one must also pass an idempotency key."
    )


# ───────────────── payment-endpoint integrity (companion §2.7) ─────────────────


def test_clnrest_tls_context_verifies_certificates():
    """A database transaction cannot compensate for an unverified endpoint."""
    import ssl

    assert (
        ln._ssl_ctx.verify_mode == ssl.CERT_REQUIRED
    ), "the clnrest SSL context must verify certificates"
    assert ln._ssl_ctx.check_hostname is True, "the clnrest SSL context must verify the hostname"


def test_no_sqlite_database_path_remains():
    """Companion §10.1/§2.7: no file-based authority for deposit state."""
    assert not hasattr(ln, "LN_DB_PATH"), (
        "LN_DB_PATH is dead SQLite configuration; deposit state is owned by "
        "PostgreSQL (migration 060)"
    )


# ───────────── typed money and time (Track B B9.3a, migration 066) ─────────────


@pytest.mark.parametrize(
    "dollars,cents",
    [
        (0.07, 7),  # 0.07 is 0.070000000000000007 in binary float
        (19.99, 1999),
        (25.0, 2500),
        (1000.00, 100_000),
        (0.01, 1),
        (1.005, 101),  # ROUND_HALF_UP, the currency convention
        ("12.34", 1234),  # a decimal string never touches float at all
    ],
)
def test_cad_to_minor_is_exact(dollars, cents):
    """Companion §4.4 rule 6 — money is integer minor units, not a float.

    `int(0.07 * 100)` is 7 only by luck of rounding; `int(1.15 * 100)` is
    114. Going through Decimal(str(x)) reads the decimal value the caller
    meant rather than the float's binary expansion.
    """
    assert ln.cad_to_minor(dollars) == cents


def test_minor_to_cad_round_trips():
    for cents in (1, 7, 1999, 2500, 100_000, 99_999_999):
        assert ln.cad_to_minor(ln.minor_to_cad(cents)) == cents


def test_naive_float_conversion_would_be_wrong():
    """Documents why the Decimal path exists rather than `int(x * 100)`."""
    assert int(1.15 * 100) == 114, "float math no longer misbehaves; revisit"
    assert ln.cad_to_minor(1.15) == 115


def test_new_deposits_store_exact_minor_units(deposits):
    """A row written today carries integer cents and a NUMERIC rate."""
    deposit_id, _ = deposits("pending", amount_cad=19.99)

    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT amount_micros::double precision / 1000000.0 AS amount_cad, "
            "amount_cad_minor, btc_cad_rate_exact "
            "FROM ln_deposits WHERE deposit_id = %s",
            (deposit_id,),
        ).fetchone()

    assert row is not None
    amount_cad, amount_minor, rate_exact = row
    assert (
        amount_minor == 1999
    ), f"expected native 1999 cents, got {amount_minor} for CAD {amount_cad}"
    assert rate_exact is not None
    from decimal import Decimal as _D

    assert isinstance(
        rate_exact, _D
    ), f"btc_cad_rate_exact must come back as Decimal, got {type(rate_exact)}"


def test_writer_that_omits_typed_columns_is_refused_after_contract():
    """No projection silently supplies authority to an obsolete writer."""
    from psycopg.errors import NotNullViolation

    marker = "lntest-legacy-" + uuid.uuid4().hex
    with pytest.raises(NotNullViolation):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO ln_deposits(deposit_id,customer_id,label,bolt11,payment_hash,amount_msat,amount_sats,amount_micros,status) "
                "VALUES(%s,'customer',%s,'bolt',%s,1000000,1000,3330000,'pending')",
                (marker, marker, marker),
            )


def test_zero_epoch_sentinels_project_to_null(deposits):
    """`paid_at = 0` means "never paid", not 1970-01-01.

    The legacy schema defaults paid_at/credited_at to 0. Projecting that
    to an epoch timestamp would make every unpaid deposit look 55 years
    overdue to any query ordering on time.
    """
    deposit_id, _ = deposits("pending")

    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT paid_at_ts,credited_at_ts FROM ln_deposits WHERE deposit_id=%s",
            (deposit_id,),
        ).fetchone()
    assert row == (None, None)
    projection = next(r for r in ln.get_pending_deposits() if r["deposit_id"] == deposit_id)
    assert projection["paid_at"] == projection["credited_at"] == 0


def test_credit_and_mark_stamp_typed_timestamps(deposits):
    """The lifecycle writes typed time and derives the API epoch at read time."""
    deposit_id, _ = deposits("paid")

    assert ln.mark_credited(deposit_id) is True

    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT credited_at_ts FROM ln_deposits WHERE deposit_id = %s",
            (deposit_id,),
        ).fetchone()

    assert row is not None
    assert row[0] is not None
    projection = next(
        r
        for r in ln.get_customer_deposits(ln.check_deposit(deposit_id)["customer_id"])
        if r["deposit_id"] == deposit_id
    )
    assert projection["credited_at"] == row[0].timestamp()


def test_sweep_credits_from_exact_cents_not_the_float(deposits):
    """The wallet is credited from integer cents, never the stored float.

    A stored `amount_cad` of 25.0 can be 24.999999999999996; posting that
    to the ledger writes the float error into a customer's balance.
    """
    deposit_id, customer_id = deposits("paid", amount_cad=19.99)

    # The contract now rejects monetary history edits instead of carrying an
    # independently mutable legacy projection alongside exact cents.
    import psycopg

    with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE ln_deposits SET amount_micros=19980000 WHERE deposit_id=%s",
                (deposit_id,),
            )

    credits: list[tuple] = []
    ln.process_ln_deposits(credit_callback=lambda c, a, d: credits.append((c, a, d)))

    assert len(credits) == 1
    _, amount, _ = credits[0]
    assert amount == 19.99, (
        f"wallet credited {amount!r}; it must come from amount_cad_minor "
        f"(1999 cents), not the legacy float column"
    )


def test_amount_check_constraint_rejects_non_positive():
    """A zero or negative deposit is a data bug, not a valid row."""
    import psycopg

    deposit_id = f"lntest-bad-{uuid.uuid4().hex[:8]}"
    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_transaction() as conn:
            conn.execute(
                """INSERT INTO ln_deposits
                    (deposit_id,customer_id,label,bolt11,payment_hash,
                     amount_msat,amount_sats,amount_micros,amount_cad_minor,
                     btc_cad_rate_exact,status,created_at_ts,expires_at_ts)
                VALUES(%s,'c',%s,'b',%s,0,0,0,0,100000,'pending',
                    clock_timestamp(),clock_timestamp()+interval '1 hour')""",
                (deposit_id, deposit_id, deposit_id),
            )
