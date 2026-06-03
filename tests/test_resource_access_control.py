"""Cross-account access-control regression tests for resource-scoped routes."""

import os
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _fund_wallet(customer_id: str, headers: dict) -> None:
    r = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "Test credits"},
        headers=headers,
    )
    assert r.status_code == 200, r.text[:200]


@pytest.fixture(scope="module")
def two_users():
    users = []
    for label in ("a", "b"):
        email = f"idor-{label}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
        reg = client.post(
            "/api/auth/register",
            json={"email": email, "password": "StrongPass123!", "name": f"Idor {label}"},
        ).json()
        login = client.post(
            "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
        ).json()
        headers = {"Authorization": f"Bearer {login['access_token']}"}
        customer_id = reg["user"]["customer_id"]
        _fund_wallet(customer_id, headers)
        users.append(
            {
                "email": email,
                "customer_id": customer_id,
                "user_id": reg["user"].get("user_id", email),
                "headers": headers,
            }
        )
    return users[0], users[1]


def test_instance_get_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job = client.post(
        "/instance",
        json={"name": "owned-by-a", "vram_needed_gb": 1},
        headers=user_a["headers"],
    ).json()["instance"]
    job_id = job["job_id"]
    r = client.get(f"/instance/{job_id}", headers=user_b["headers"])
    assert r.status_code == 403


def test_instance_cancel_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job = client.post(
        "/instance",
        json={"name": "cancel-a", "vram_needed_gb": 1},
        headers=user_a["headers"],
    ).json()["instance"]
    r = client.post(
        f"/instances/{job['job_id']}/cancel",
        headers=user_b["headers"],
    )
    assert r.status_code == 403


def test_billing_refund_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job = client.post(
        "/instance",
        json={"name": "refund-a", "vram_needed_gb": 1},
        headers=user_a["headers"],
    ).json()["instance"]
    r = client.post(
        "/api/billing/refund",
        json={"job_id": job["job_id"], "exit_code": 1, "failure_reason": "hardware"},
        headers=user_b["headers"],
    )
    assert r.status_code == 403


def test_provider_earnings_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    provider_id = f"prov-{uuid.uuid4().hex[:8]}"
    client.post(
        "/api/providers/register",
        json={
            "provider_id": provider_id,
            "email": user_a["email"],
            "provider_type": "individual",
            "legal_name": "Test Provider",
            "province": "ON",
        },
        headers=user_a["headers"],
    )
    r = client.get(
        f"/api/providers/{provider_id}/earnings",
        headers=user_b["headers"],
    )
    assert r.status_code == 403