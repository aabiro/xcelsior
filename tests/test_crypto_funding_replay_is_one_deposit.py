"""Gate P1 clause 2, the crypto rails: replaying a deposit yields one deposit.

Both of them. The clause says *"the crypto **rails**"*, plural, and there are
two — on-chain BTC and Lightning. Fixing only the first would have left the
clause half met while reading as done.

The clause names three rails — *"manual top-up, auto-top-up, and the crypto
rails"*. Two were asserted by `test_funding_replay_is_one_charge.py`. The third
was not, and it turned out not to be unasserted so much as **unimplemented**:
`create_deposit` took no key and deduplicated nothing, so every call ran
`get_new_address`, inserted a row and locked a fresh BTC/CAD rate.

A retried request therefore produced a *second Bitcoin address* for one intended
deposit. That is a worse shape than a duplicated card charge, not a milder one:
a duplicate charge is visible and refundable, whereas two live addresses for one
intent are not obviously duplicates to whoever is looking at them. Whichever
gets paid, the other stays open and the wallet has burned an address it now has
to watch.

## What each test here is for

The first is the clause. The rest are the ways a replay guard is usually wrong:

- returning a *new* rate or address on the replay, so the caller's QR code
  changes underneath them
- deduplicating across customers, which turns the guard into a way to read
  another tenant's row
- accepting the same key for a different amount, which funds the wrong number
- losing the race between the read and the insert, which produces exactly the
  second address the mechanism exists to prevent

Every one of them passes a naive `SELECT ... IF NOT FOUND: INSERT`.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

bitcoin = pytest.importorskip("bitcoin")

try:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as _c:
        _has = _c.execute("SELECT to_regclass('crypto_deposits')").fetchone()[0] is not None
        _col = _c.execute(
            "SELECT 1 FROM information_schema.columns "
            " WHERE table_name = 'crypto_deposits' AND column_name = 'idempotency_key'"
        ).fetchone()
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no database: {_e}")
else:
    if not _has or not _col:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 109")


@pytest.fixture
def customer():
    return f"cust-{uuid.uuid4().hex[:12]}"


@pytest.fixture(autouse=True)
def _fixed_rate(monkeypatch):
    """A deterministic rate, so a changed address is not mistaken for a changed rate.

    Left un-fixed, a replay returning a *different* rate would be invisible
    whenever the live rate happened not to move between the two calls — the
    test would pass for a reason unrelated to the guard.
    """
    rates = iter([50_000.0, 90_000.0, 123_456.0, 7.0] * 40)
    monkeypatch.setattr(bitcoin, "get_btc_cad_rate", lambda: next(rates))
    addresses = (f"bc1qtest{n:034d}" for n in range(1, 10_000))
    monkeypatch.setattr(bitcoin, "get_new_address", lambda _label: next(addresses))


def _cleanup(customer_id: str) -> None:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as c:
        c.execute("DELETE FROM crypto_deposits WHERE customer_id = %s", (customer_id,))
        c.commit()


# ── The clause ────────────────────────────────────────────────────────


def test_replaying_a_deposit_with_the_same_key_creates_exactly_one(customer):
    try:
        first = bitcoin.create_deposit(customer, 50.0, "idem-abc")
        second = bitcoin.create_deposit(customer, 50.0, "idem-abc")

        assert first["deposit_id"] == second["deposit_id"]

        from db import _get_pg_pool

        with _get_pg_pool().connection() as c:
            rows = c.execute(
                "SELECT count(*) FROM crypto_deposits WHERE customer_id = %s",
                (customer,),
            ).fetchone()[0]
        assert rows == 1, f"{rows} deposits exist for one replayed request"
    finally:
        _cleanup(customer)


def test_the_replay_returns_the_original_address_and_rate(customer):
    """Not merely "one row" — the *same* answer.

    A guard that dedupes the row but recomputes the response hands the caller a
    different address and a freshly-fetched rate for the deposit they already
    showed the user. The fixed-rate fixture moves the rate between calls, so a
    recomputed response is visible rather than coincidentally equal.
    """
    try:
        first = bitcoin.create_deposit(customer, 25.0, "idem-same")
        second = bitcoin.create_deposit(customer, 25.0, "idem-same")

        assert first["btc_address"] == second["btc_address"]
        assert first["btc_cad_rate"] == second["btc_cad_rate"]
        assert first["amount_btc"] == second["amount_btc"]
        assert first["expires_at"] == second["expires_at"]
        assert first["qr_data"] == second["qr_data"]
    finally:
        _cleanup(customer)


def test_without_a_key_every_call_is_a_new_deposit(customer):
    """The negative control, and the documented behaviour.

    No key means no claim of replay-safety. If this stopped being true the test
    above would pass for the wrong reason — because *nothing* creates a second
    deposit — and the guard would be indistinguishable from a broken endpoint.
    """
    try:
        first = bitcoin.create_deposit(customer, 10.0)
        second = bitcoin.create_deposit(customer, 10.0)
        assert first["deposit_id"] != second["deposit_id"]
        assert first["btc_address"] != second["btc_address"]
    finally:
        _cleanup(customer)


# ── The ways a replay guard is usually wrong ──────────────────────────


def test_the_same_key_for_a_different_amount_is_refused(customer):
    """A caller contradicting itself, not a duplicate and not a fresh request.

    Returning the original would fund the wrong amount; creating a new deposit
    would break the key's only guarantee. Refusing is the one answer that is
    not silently wrong.
    """
    try:
        bitcoin.create_deposit(customer, 40.0, "idem-conflict")
        with pytest.raises(bitcoin.DepositKeyConflict):
            bitcoin.create_deposit(customer, 41.0, "idem-conflict")
    finally:
        _cleanup(customer)


def test_two_customers_may_use_the_same_key(customer):
    """Keys are caller-chosen strings, meaningful only inside one account.

    A global unique index would make one tenant's key collide with another's —
    turning a replay guard into a way to probe whether a key exists, and
    handing a caller someone else's deposit row on a collision.
    """
    other = f"cust-{uuid.uuid4().hex[:12]}"
    try:
        mine = bitcoin.create_deposit(customer, 15.0, "shared-key")
        theirs = bitcoin.create_deposit(other, 15.0, "shared-key")
        assert mine["deposit_id"] != theirs["deposit_id"]
        assert mine["btc_address"] != theirs["btc_address"]
    finally:
        _cleanup(customer)
        _cleanup(other)


def test_a_concurrent_replay_still_yields_one_deposit(customer):
    """The race a read-then-insert loses.

    Two replays can both miss the SELECT and both proceed to INSERT. The
    guarantee is held by the unique index — `ON CONFLICT DO NOTHING` makes the
    loser a no-op, and it then reads the winner's row — rather than by the
    timing of the check. Without that, this is exactly how the second address
    gets created.
    """
    import threading

    results: list[dict] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def go():
        try:
            barrier.wait(timeout=10)
            results.append(bitcoin.create_deposit(customer, 60.0, "idem-race"))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    try:
        threads = [threading.Thread(target=go) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"concurrent replays raised: {errors}"
        assert len(results) == 4

        ids = {r["deposit_id"] for r in results}
        assert len(ids) == 1, f"{len(ids)} distinct deposits from one key: {ids}"

        from db import _get_pg_pool

        with _get_pg_pool().connection() as c:
            rows = c.execute(
                "SELECT count(*) FROM crypto_deposits WHERE customer_id = %s",
                (customer,),
            ).fetchone()[0]
        assert rows == 1, f"{rows} rows written by 4 concurrent replays"
    finally:
        _cleanup(customer)


# ── Lightning: the same clause, the sharper failure ───────────────────
#
# On-chain, two addresses at least belong to one wallet and both credit if
# paid. A second bolt11 is a distinct payment request: a wallet that pays the
# first settles nothing against the second, and whoever holds the second is
# waiting on a payment that already happened.

lightning = pytest.importorskip("lightning")


@pytest.fixture
def ln_customer():
    return f"lncust-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def _ln_stubs(monkeypatch):
    """A real row, a fake node. The invoice is the part that must not repeat."""
    monkeypatch.setattr(lightning, "LN_ENABLED", True, raising=False)
    monkeypatch.setattr(lightning, "get_btc_cad_rate", lambda: 50_000.0)
    seq = iter(range(1, 10_000))

    def _invoice(_msat, label, _desc):
        n = next(seq)
        return {
            "bolt11": f"lnbc{n}test{label[-6:]}",
            "payment_hash": f"{n:064x}",
            "expires_at": 4_000_000_000.0 + n,
        }

    monkeypatch.setattr(lightning, "create_invoice", _invoice)


def _ln_cleanup(customer_id: str) -> None:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as c:
        c.execute("DELETE FROM ln_deposits WHERE customer_id = %s", (customer_id,))
        c.commit()


def test_replaying_a_lightning_deposit_returns_the_same_invoice(ln_customer, _ln_stubs):
    try:
        first = lightning.create_deposit(ln_customer, 30.0, "ln-idem-1")
        second = lightning.create_deposit(ln_customer, 30.0, "ln-idem-1")

        assert first["deposit_id"] == second["deposit_id"]
        assert first["bolt11"] == second["bolt11"], (
            "the replay minted a second invoice; a wallet paying the first "
            "settles nothing against it"
        )
        assert first["payment_hash"] == second["payment_hash"]

        from db import _get_pg_pool

        with _get_pg_pool().connection() as c:
            rows = c.execute(
                "SELECT count(*) FROM ln_deposits WHERE customer_id = %s",
                (ln_customer,),
            ).fetchone()[0]
        assert rows == 1, f"{rows} lightning deposits for one replayed request"
    finally:
        _ln_cleanup(ln_customer)


def test_a_lightning_deposit_without_a_key_is_a_new_invoice(ln_customer, _ln_stubs):
    """Negative control: no key means no claim of replay-safety."""
    try:
        first = lightning.create_deposit(ln_customer, 12.0)
        second = lightning.create_deposit(ln_customer, 12.0)
        assert first["deposit_id"] != second["deposit_id"]
        assert first["bolt11"] != second["bolt11"]
    finally:
        _ln_cleanup(ln_customer)


def test_a_lightning_key_reused_for_a_different_amount_is_refused(ln_customer, _ln_stubs):
    try:
        lightning.create_deposit(ln_customer, 20.0, "ln-conflict")
        with pytest.raises(lightning.DepositKeyConflict):
            lightning.create_deposit(ln_customer, 21.0, "ln-conflict")
    finally:
        _ln_cleanup(ln_customer)


# ── Auto-top-up: the third rail the clause names ──────────────────────


def test_the_auto_topup_key_is_stable_inside_one_sweep_interval():
    """Two sweeps in one interval must derive the *same* key.

    Auto-top-up reaches the already-asserted `charge_saved_card`, so what is
    unproven is not the charge — it is the **key derivation**, which is the
    only thing that decides whether two sweeps are one charge or two. The
    balance does not move until Stripe confirms, so a wallet stays eligible for
    the whole interval between the charge and the confirmation; if the key
    varied inside that window the sweep would charge again.

    Derived from the source rather than by running a sweep, because running one
    needs Stripe, a saved payment method and a funded wallet — none of which
    make the key derivation any more or less true.
    """
    import inspect
    import re

    from billing import BillingEngine

    source = inspect.getsource(BillingEngine.check_low_balance_and_topup)
    match = re.search(r'f"autotopup:\{([^}]+)\}:\{([^}]+)\}:"', source)
    assert match, (
        "the auto-topup idempotency key is no longer derived as "
        "autotopup:{customer}:{amount}; two sweeps in one interval may now "
        "produce two charges"
    )
    assert "now // _TOPUP_INFLIGHT_SECONDS" in source, (
        "the auto-topup key is no longer bucketed by the sweep interval. An "
        "unbucketed timestamp makes every sweep a new key, which is exactly a "
        "double charge; a constant key would make a genuinely later top-up "
        "impossible."
    )


def test_the_auto_topup_bucket_changes_between_intervals_and_not_within_one():
    """The arithmetic itself, not a claim about it.

    A bucket that never changes blocks legitimate later top-ups; one that
    changes every second is no bucket at all. Both read as "there is a key".
    """
    from billing import _TOPUP_INFLIGHT_SECONDS

    assert _TOPUP_INFLIGHT_SECONDS > 0

    def bucket(t):
        return int(t // _TOPUP_INFLIGHT_SECONDS)

    # Aligned to a boundary on purpose. The first draft of this test used an
    # arbitrary timestamp and failed, which was the test being wrong and the
    # scheme being worth stating precisely: the bucket is a **fixed grid**, not
    # a window measured from the first charge.
    base = 1_000_000_000 // _TOPUP_INFLIGHT_SECONDS * _TOPUP_INFLIGHT_SECONDS
    assert bucket(base) == bucket(base + _TOPUP_INFLIGHT_SECONDS - 1), (
        "the key changes within a single bucket"
    )
    assert bucket(base) != bucket(base + _TOPUP_INFLIGHT_SECONDS), (
        "the key never changes between buckets, so a genuinely later top-up "
        "would be suppressed forever"
    )

    # And the consequence, asserted rather than assumed: two sweeps one second
    # apart get *different* keys when they straddle a boundary. That is why the
    # code calls in-flight suppression the primary guard and this key the last
    # line of defence — the key alone does not close the window, and a comment
    # claiming otherwise would be wrong.
    edge = base + _TOPUP_INFLIGHT_SECONDS
    assert bucket(edge - 1) != bucket(edge), (
        "a boundary straddle no longer produces distinct keys; if that is now "
        "intended, the in-flight suppression comment needs revisiting"
    )
