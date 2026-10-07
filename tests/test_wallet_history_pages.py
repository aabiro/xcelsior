"""Wallet history comes a page at a time, and paging never repeats or skips a row.

The billing page rendered every transaction the API returned as one list. The
route now pages with a keyset cursor (`before` = the last row's
`(created_at, tx_id)`), not an offset: a ledger grows at the head while someone
reads it, and with offsets each new charge shifts the rows under the reader so
page two repeats page one's last row.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from fastapi.testclient import TestClient  # noqa: E402

import routes.billing as billing_routes  # noqa: E402
from api import app  # noqa: E402

client = TestClient(app)


def _insert(customer_id: str, tx_id: str, created_at: float, micros: int = 1_000_000) -> None:
    from db import pg_connection

    with pg_connection() as conn:
        conn.execute(
            """INSERT INTO wallet_transactions
               (tx_id, customer_id, tx_type, description, created_at, amount_micros, balance_after_micros)
               VALUES (%s, %s, 'deposit', %s, %s, %s, 0)""",
            (tx_id, customer_id, f"tx {tx_id}", created_at, micros),
        )


@pytest.fixture
def ledger(monkeypatch):
    """Seven transactions; three share one instant, so the tiebreak is exercised."""
    from db import pg_connection

    monkeypatch.setattr(billing_routes, "_require_customer_access", lambda *a, **k: None)
    billing_routes.get_billing_engine().get_wallet(cid := f"cust-pages-{uuid.uuid4().hex[:8]}")
    base = time.time() - 10_000
    times = [base + 1, base + 2, base + 3, base + 3, base + 3, base + 4, base + 5]
    for i, at in enumerate(times):
        _insert(cid, f"{cid}-tx{i}", at)
    # Newest first, ties broken by tx_id descending: the order the API promises.
    expected = [f"{cid}-tx{i}" for i, _ in sorted(enumerate(times), key=lambda p: (p[1], f"{cid}-tx{p[0]}"), reverse=True)]
    yield cid, expected
    with pg_connection() as conn:
        conn.execute("DELETE FROM wallet_transactions WHERE customer_id = %s", (cid,))
        conn.execute("DELETE FROM wallets WHERE customer_id = %s", (cid,))


def _page(cid: str, limit: int, before: str | None = None) -> dict:
    params: dict[str, str | int] = {"limit": limit}
    if before:
        params["before"] = before
    r = client.get(f"/api/billing/wallet/{cid}/history", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_paging_visits_every_row_once_newest_first(ledger):
    cid, expected = ledger
    seen, cursor, pages = [], None, 0
    while True:
        body = _page(cid, 3, cursor)
        assert body["total"] == 7
        seen += [t["tx_id"] for t in body["transactions"]]
        pages += 1
        cursor = body["next_cursor"]
        if cursor is None:
            break
        assert pages < 10, "the cursor never ran out"
    assert seen == expected, "pages repeated, skipped or reordered a row"
    assert pages == 3


def test_a_charge_landing_mid_read_does_not_shift_the_next_page(ledger):
    cid, expected = ledger
    first = _page(cid, 3)
    _insert(cid, f"{cid}-new", time.time())  # newest of all, arrives while reading
    second = _page(cid, 3, first["next_cursor"])
    assert [t["tx_id"] for t in second["transactions"]] == expected[3:6], (
        "with offsets the new head row pushes page one's last row onto page two"
    )


def test_the_last_page_says_so(ledger):
    cid, _ = ledger
    assert _page(cid, 7)["next_cursor"] is None
    assert _page(cid, 6)["next_cursor"] is not None


def test_limit_only_callers_get_what_they_always_did(ledger):
    """The MCP tool and older clients send only `limit`."""
    cid, expected = ledger
    body = _page(cid, 4)
    assert [t["tx_id"] for t in body["transactions"]] == expected[:4]
    assert body["transactions"][0]["amount_cad"] == 1.0


@pytest.mark.parametrize("bad", ["not-base64!", "WzEsMiwzXQ", "e30"])
def test_a_malformed_cursor_is_refused_not_a_500(ledger, bad):
    cid, _ = ledger
    r = client.get(f"/api/billing/wallet/{cid}/history", params={"before": bad})
    assert r.status_code == 400, r.text
