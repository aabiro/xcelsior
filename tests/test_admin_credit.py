"""Admin credits: a platform admin can fund a wallet without paying, and it is
never mistaken for money received."""

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app
from billing import get_billing_engine

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def customer():
    email = f"admincredit-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Credit Target"},
    ).json()
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"}).json()
    return {
        "email": email,
        "customer_id": reg["user"]["customer_id"],
        "headers": {"Authorization": f"Bearer {login['access_token']}"},
    }


def _credit(cid: str, headers: dict, **body):
    payload = {"amount_cad": 25.0, "reason": "Testing the launch flow", "idempotency_key": uuid.uuid4().hex, **body}
    return client.post(f"/api/billing/wallet/{cid}/admin-credit", json=payload, headers=headers)


def _wallet(cid: str) -> dict:
    return get_billing_engine().get_wallet(cid)


def test_admin_credit_funds_the_wallet_as_a_credit_not_a_deposit(customer):
    cid = customer["customer_id"]
    before = _wallet(cid)
    r = _credit(cid, _admin_headers())
    assert r.status_code == 200, r.text
    after = _wallet(cid)
    assert after["balance_cad"] == pytest.approx(before["balance_cad"] + 25.0)
    # Revenue and the FINTRAC aggregate count deposits as money received.
    assert after["total_deposited_cad"] == pytest.approx(before["total_deposited_cad"])

    be = get_billing_engine()
    with be._conn() as conn:
        row = conn.execute(
            "SELECT tx_type, description FROM wallet_transactions WHERE tx_id = %s",
            (r.json()["tx_id"],),
        ).fetchone()
    assert row["tx_type"] == "credit"
    assert row["description"] == "Admin credit: Testing the launch flow"


def test_a_customer_cannot_credit_even_their_own_wallet(customer):
    r = _credit(customer["customer_id"], customer["headers"])
    assert r.status_code == 403


def test_a_retried_grant_lands_once(customer):
    cid = customer["customer_id"]
    before = _wallet(cid)["balance_cad"]
    key = uuid.uuid4().hex
    first = _credit(cid, _admin_headers(), idempotency_key=key)
    second = _credit(cid, _admin_headers(), idempotency_key=key)
    assert first.status_code == second.status_code == 200
    assert second.json().get("dedup") is True
    assert _wallet(cid)["balance_cad"] == pytest.approx(before + 25.0)


def test_an_unknown_customer_is_not_given_a_wallet():
    r = _credit(f"cust-nobody-{uuid.uuid4().hex[:8]}", _admin_headers())
    assert r.status_code == 404


def test_a_grant_must_say_why(customer):
    r = _credit(customer["customer_id"], _admin_headers(), reason="")
    assert r.status_code == 422


def test_admin_user_list_carries_the_wallet_key(customer):
    r = client.get("/api/admin/users", headers=_admin_headers())
    assert r.status_code == 200
    row = next(u for u in r.json()["users"] if u["email"] == customer["email"])
    assert row["customer_id"] == customer["customer_id"]


@pytest.mark.parametrize("body", [
    {"amount_cad": "NaN"}, {"amount_cad": "Infinity"}, {"amount_cad": 0},
    {"amount_cad": -1}, {"amount_cad": 0.001}, {"amount_cad": 10000.01},
    {"reason": "   "}, {"idempotency_key": ""}, {"idempotency_key": None},
])
def test_invalid_grants_do_not_change_the_wallet(customer, body):
    cid = customer["customer_id"]
    before = _wallet(cid)["balance_cad"]
    assert _credit(cid, _admin_headers(), **body).status_code == 422
    assert _wallet(cid)["balance_cad"] == before


def test_admin_credit_is_available_outside_relaxed_environments(customer, monkeypatch):
    import env_config
    monkeypatch.setattr(env_config, "is_relaxed_env", lambda: False)
    assert _credit(customer["customer_id"], _admin_headers()).status_code == 200
    assert _credit(customer["customer_id"], customer["headers"]).status_code == 403


def test_request_id_cannot_be_reused_for_a_different_amount(customer):
    cid = customer["customer_id"]
    before = _wallet(cid)["balance_cad"]
    key = uuid.uuid4().hex
    assert _credit(cid, _admin_headers(), idempotency_key=key).status_code == 200
    assert _credit(cid, _admin_headers(), idempotency_key=key, amount_cad=50).status_code == 409
    assert _wallet(cid)["balance_cad"] == pytest.approx(before + 25)


def test_simultaneous_retries_return_the_same_grant(customer):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    be = get_billing_engine()
    cid = customer["customer_id"]
    before = _wallet(cid)["balance_cad"]
    key = f"admin-credit:{cid}:{uuid.uuid4().hex}"
    start = Barrier(2)

    def grant():
        start.wait(timeout=10)
        return be.grant_credit(cid, 25, "Admin credit: concurrent retry", key)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(grant) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert results[0]["tx_id"] == results[1]["tx_id"]
    assert sum(bool(result.get("dedup")) for result in results) == 1
    assert _wallet(cid)["balance_cad"] == pytest.approx(before + 25)
