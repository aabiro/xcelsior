"""Cancelling a reserved commitment must succeed and record the cancellation.

`MarketplaceEngine.cancel_reservation` read `res["monthly_rate_cad"]` off a row
that came straight from `SELECT * FROM reservations`, with no projection step
like the one `get_wallet` and `bitcoin.get_deposit` use. Migration `097` left
only `monthly_rate_micros`, so the key raised `KeyError` before the termination
fee or the UPDATE was reached, and
`DELETE /api/v2/marketplace/reservations/{id}` has no handler around the call —
so every customer who tried to cancel got a 500 and the reservation stayed
active.

Behind it sat a second fault that the first one hid: the UPDATE also set
`updated_at`, which `reservations` does not have either. Fixing only the read
would have moved the failure from `KeyError` to `UndefinedColumn` at the same
call.

The existing coverage was `assert hasattr(MarketplaceEngine, "cancel_reservation")`
in `test_host_verification_features.py` — a mention, which is exactly what a
broken method still satisfies. This exercises the method and the route.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_PERSISTENT_AUTH", "true")

from api import app
from db import _get_pg_pool
from marketplace import get_marketplace_engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


@pytest.fixture
def reservation():
    me = get_marketplace_engine()
    customer_id = f"cust-resv-{uuid.uuid4().hex[:8]}"
    res = me.create_reservation(
        customer_id=customer_id, gpu_model="RTX 4090", gpu_count=1, period_months=6
    )
    yield res, customer_id
    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM reservations WHERE reservation_id = %s", (res["reservation_id"],))
        conn.commit()


def test_cancelling_a_reservation_returns_a_fee_not_an_error(reservation):
    res, customer_id = reservation
    me = get_marketplace_engine()

    out = me.cancel_reservation(reservation_id=res["reservation_id"], customer_id=customer_id)

    assert "error" not in out, out
    # Six months ahead at the reserved rate, half of it: a real number, and one
    # that could only be computed from the row's stored rate.
    assert out["early_termination_fee_cad"] > 0
    assert out["monthly_rate_cad"] == pytest.approx(res["monthly_rate_cad"], rel=0.01)


def test_a_cancelled_reservation_is_actually_cancelled(reservation):
    res, customer_id = reservation
    me = get_marketplace_engine()

    me.cancel_reservation(reservation_id=res["reservation_id"], customer_id=customer_id)

    rows = me.get_customer_reservations(customer_id)
    assert [r["status"] for r in rows] == ["cancelled"]

    # And it cannot be cancelled twice.
    again = me.cancel_reservation(reservation_id=res["reservation_id"], customer_id=customer_id)
    assert "error" in again
