"""Billing route security sweep — customer scope on mutating endpoints."""

import os
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def two_users():
    users = []
    for label in ("a", "b"):
        email = f"bsec-{label}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
        reg = client.post(
            "/api/auth/register",
            json={"email": email, "password": "StrongPass123!", "name": f"Sec {label}"},
        ).json()
        login = client.post(
            "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
        ).json()
        users.append(
            {
                "email": email,
                "customer_id": reg["user"]["customer_id"],
                "headers": {"Authorization": f"Bearer {login['access_token']}"},
            }
        )
    return users[0], users[1]


_CUSTOMER_MUTATIONS = [
    (
        "POST",
        "/api/billing/payment-intent",
        lambda u: {"customer_id": u["customer_id"], "amount_cad": 10},
    ),
    (
        "POST",
        "/api/billing/paypal/create-order",
        lambda u: {"customer_id": u["customer_id"], "amount_cad": 10},
    ),
    (
        "POST",
        "/api/billing/paypal/capture-order",
        lambda u: {"customer_id": u["customer_id"], "order_id": "ORDER_PROBE"},
    ),
    (
        "POST",
        f"/api/billing/wallet/{{cid}}/deposit",
        lambda u: {"amount_cad": 10.0, "description": "probe"},
    ),
    (
        "POST",
        "/api/billing/crypto/deposit",
        lambda u: {"customer_id": u["customer_id"], "amount_cad": 10},
    ),
    (
        "POST",
        "/api/billing/lightning/deposit",
        lambda u: {"customer_id": u["customer_id"], "amount_cad": 10},
    ),
    (
        "POST",
        "/api/billing/paypal/marketplace/create-order",
        lambda u: {
            "customer_id": u["customer_id"],
            "provider_id": "prov-probe",
            "job_id": "job-probe",
            "amount_cad": 10,
        },
    ),
    (
        "POST",
        "/api/billing/paypal/marketplace/capture-order",
        lambda u: {
            "customer_id": u["customer_id"],
            "provider_id": "prov-probe",
            "order_id": "ORDER_PROBE",
        },
    ),
    (
        "POST",
        "/api/pricing/reserve",
        lambda u: {
            "customer_id": u["customer_id"],
            "gpu_model": "RTX 4090",
            "commitment_type": "1_month",
            "quantity": 1,
            "province": "ON",
        },
    ),
]


@pytest.mark.parametrize("method,path_tpl,body_fn", _CUSTOMER_MUTATIONS)
def test_billing_mutation_requires_auth(method, path_tpl, body_fn, two_users):
    user_a, _ = two_users
    path = path_tpl.format(cid=user_a["customer_id"]) if "{cid}" in path_tpl else path_tpl
    fresh = TestClient(app)
    r = fresh.request(method, path, json=body_fn(user_a))
    assert r.status_code in (401, 503), f"{method} {path} -> {r.status_code}: {r.text[:120]}"


@pytest.mark.parametrize("method,path_tpl,body_fn", _CUSTOMER_MUTATIONS)
def test_billing_mutation_forbidden_cross_account(method, path_tpl, body_fn, two_users):
    user_a, user_b = two_users
    path = path_tpl.format(cid=user_b["customer_id"]) if "{cid}" in path_tpl else path_tpl
    body = body_fn(user_b)
    r = client.request(method, path, json=body, headers=user_a["headers"])
    assert r.status_code in (403, 503), f"{method} {path} -> {r.status_code}: {r.text[:120]}"