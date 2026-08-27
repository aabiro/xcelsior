"""A grace period must not silence the warning that the balance is low.

`wallets.grace_until` carried two incompatible meanings on one column.

`serverless/service.py` refuses work at zero balance *unless* the customer is
inside a grace window. `BillingEngine.maybe_warn_low_balance` wrote `now` into the
same column as a rate-limit watermark, with a comment saying so — *"Reuse
grace_until as last-warn watermark when still positive balance."*

Nothing ever set it to a **future** time. Every write was `0` or `now`, so the
serverless grace clause could not be false for the reason it was written: it
read as "unless they are in a grace period", and no grace period existed.

That alone is only misleading. The trap is what happens when someone fixes it.

Grant a real window — `grace_until = now + 3 days` — and the rate-limiter reads
that future value as *"last warned at now + 3 days"*. `now - last` is
**negative**, which is less than six hours, so the warning is suppressed. For
the whole grace period, and past it until the value falls six hours behind. The
customer stops being told their balance is low exactly while a window they did
not ask for runs out.

Neither function is wrong alone. The defect lived in the column they shared, and
nobody reading either one would find it.

Migration 116 gives the watermark `low_balance_warned_at`. These tests are the
reason to keep it separate: the first two would both have failed before it.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from billing import get_billing_engine  # noqa: E402
from db import _get_pg_pool  # noqa: E402


@pytest.fixture
def wallet_id():
    cid = f"grace-probe-{uuid.uuid4().hex[:10]}"
    yield cid
    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM wallets WHERE customer_id = %s", (cid,))


def _set(cid: str, **cols) -> None:
    sets = ", ".join(f"{k} = %s" for k in cols)
    with _get_pg_pool().connection() as conn:
        conn.execute(
            f"UPDATE wallets SET {sets} WHERE customer_id = %s",
            (*cols.values(), cid),
        )


def _read(cid: str, column: str):
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            f"SELECT {column} FROM wallets WHERE customer_id = %s", (cid,)
        ).fetchone()
    return row[0] if row else None


def test_a_future_grace_window_does_not_suppress_the_low_balance_warning(wallet_id):
    """The trap, asserted directly. This failed before migration 116."""
    engine = get_billing_engine()
    engine.get_wallet(wallet_id)  # creates the row

    # A real grace window, which is what `serverless/service.py` has always been
    # asking for and nothing ever granted.
    _set(wallet_id, grace_until=time.time() + 3 * 86_400)

    result = engine.maybe_warn_low_balance(wallet_id, balance_cad=0.50)
    assert result.get("reason") != "rate_limited", (
        "a future grace_until silenced the low-balance warning — the rate "
        "limiter read it as 'last warned in three days' time', so `now - last` "
        "was negative and compared as 'warned recently'. The customer stops "
        "hearing that their balance is low while the grace window runs out."
    )
    assert result.get("warned") is True


def test_the_watermark_no_longer_writes_to_grace_until(wallet_id):
    """Warning someone must not silently extend or destroy their grace window."""
    engine = get_billing_engine()
    engine.get_wallet(wallet_id)
    deadline = time.time() + 3 * 86_400
    _set(wallet_id, grace_until=deadline)

    engine.maybe_warn_low_balance(wallet_id, balance_cad=0.50)

    after = float(_read(wallet_id, "grace_until") or 0)
    assert abs(after - deadline) < 1.0, (
        f"grace_until moved from {deadline} to {after} because a warning was "
        "emitted. A notification must not rewrite a billing deadline."
    )
    assert _read(wallet_id, "low_balance_warned_at") is not None, (
        "the warning was not recorded anywhere, so the rate limiter has nothing "
        "to work from and every check will send another notification"
    )


def test_the_rate_limit_still_works(wallet_id):
    """The behaviour being preserved — one warning per six hours, not per call."""
    engine = get_billing_engine()
    engine.get_wallet(wallet_id)

    first = engine.maybe_warn_low_balance(wallet_id, balance_cad=0.50)
    assert first.get("warned") is True

    second = engine.maybe_warn_low_balance(wallet_id, balance_cad=0.50)
    assert second.get("warned") is False
    assert second.get("reason") == "rate_limited"


def test_a_warning_older_than_six_hours_fires_again(wallet_id):
    engine = get_billing_engine()
    engine.get_wallet(wallet_id)
    engine.maybe_warn_low_balance(wallet_id, balance_cad=0.50)

    _set(wallet_id, low_balance_warned_at=time.time() - 7 * 3600)
    again = engine.maybe_warn_low_balance(wallet_id, balance_cad=0.50)
    assert again.get("warned") is True, "a seven-hour-old warning should not rate-limit"


def test_a_balance_above_the_threshold_is_not_warned_about(wallet_id):
    engine = get_billing_engine()
    engine.get_wallet(wallet_id)
    result = engine.maybe_warn_low_balance(wallet_id, balance_cad=10_000.0)
    assert result.get("warned") is False


def test_the_schema_refuses_a_nonsense_watermark(wallet_id):
    """`0` meant "none" on `grace_until`; here NULL does, and 0 is refused."""
    import psycopg

    get_billing_engine().get_wallet(wallet_id)
    with pytest.raises(psycopg.errors.CheckViolation):
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE wallets SET low_balance_warned_at = 0 WHERE customer_id = %s",
                (wallet_id,),
            )
