"""Wallet history must state money in CAD, as every other API money field does.

`get_wallet_history` returns the stored row, and since money moved to integer
micros that row has `amount_micros` / `balance_after_micros` and nothing in
CAD. The route passed it through. Measured on production on 2026-10-06 with a
signed-in Playwright run: the billing page crashed on `amount_cad.toFixed` for
an account with one transaction, and analytics summed undefined into `$NaN`.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

from fastapi.testclient import TestClient  # noqa: E402

import routes.billing as billing_routes  # noqa: E402

ROW = {
    "tx_id": "tx-1",
    "customer_id": "cust-1",
    "tx_type": "refund",
    "amount_micros": 5_250_000,
    "balance_after_micros": 20_010_000,
    "description": "Refund: host failure",
    "job_id": "job-1",
    "created_at": 1791283275.3,
}


def test_cad_is_derived_from_micros_exactly() -> None:
    out = billing_routes._wallet_transaction_for_api(ROW)
    assert out["amount_cad"] == 5.25
    assert out["balance_after_cad"] == 20.01


def test_the_micros_fields_stay_for_existing_clients() -> None:
    """MCP agents already receive these; removing them would break them."""
    out = billing_routes._wallet_transaction_for_api(ROW)
    assert out["amount_micros"] == 5_250_000 and out["balance_after_micros"] == 20_010_000
    assert out["tx_type"] == "refund" and out["created_at"] == ROW["created_at"]


def test_a_debit_stays_negative() -> None:
    out = billing_routes._wallet_transaction_for_api({**ROW, "amount_micros": -1_500_000})
    assert out["amount_cad"] == -1.5


def test_the_endpoint_applies_it(monkeypatch) -> None:  # noqa: ANN001
    """Wired into the route, not merely defined."""

    class _Engine:
        def get_wallet_history(self, customer_id, limit, *, before=None):  # noqa: ANN001
            return [dict(ROW)]

        def count_wallet_history(self, customer_id):  # noqa: ANN001
            return 1

    monkeypatch.setattr(billing_routes, "get_billing_engine", lambda: _Engine())
    monkeypatch.setattr(billing_routes, "_require_customer_access", lambda *a, **k: None)
    from api import app

    body = TestClient(app).get("/api/billing/wallet/cust-1/history").json()
    tx = body["transactions"][0]
    assert tx["amount_cad"] == 5.25, tx
    assert tx["balance_after_cad"] == 20.01
