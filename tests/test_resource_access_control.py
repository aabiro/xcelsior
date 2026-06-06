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


def test_billing_dump_requires_admin_not_regular_user(two_users):
    user_a, _user_b = two_users
    r = client.get("/billing", headers=user_a["headers"])
    assert r.status_code == 403


def test_events_all_requires_admin(two_users):
    _, user_b = two_users
    r = client.get("/api/events", headers=user_b["headers"])
    assert r.status_code == 403


def test_provider_list_only_own_account(two_users):
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
    r = client.get("/api/providers", headers=user_b["headers"])
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_provider_register_email_mismatch_forbidden(two_users):
    user_a, user_b = two_users
    r = client.post(
        "/api/providers/register",
        json={
            "provider_id": f"prov-{uuid.uuid4().hex[:8]}",
            "email": user_a["email"],
            "provider_type": "individual",
            "legal_name": "Evil",
            "province": "ON",
        },
        headers=user_b["headers"],
    )
    assert r.status_code == 403


def test_provider_earnings_forbidden_cross_account(two_users):
    import time

    from db import UserStore
    from stripe_connect import get_stripe_manager

    user_a, user_b = two_users
    provider_id = f"prov-{uuid.uuid4().hex[:8]}"
    reg = client.post(
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
    if reg.status_code != 200:
        # CI has no .env.test Stripe key — seed provider row so access control is testable.
        now = time.time()
        mgr = get_stripe_manager()
        with mgr._conn() as conn:
            conn.execute(
                """INSERT INTO provider_accounts
                   (provider_id, provider_type, stripe_account_id, status,
                    corporation_name, business_number, gst_hst_number,
                    email, legal_name, country, province, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'CA', %s, %s)
                   ON CONFLICT (provider_id) DO NOTHING""",
                (
                    provider_id,
                    "individual",
                    "",
                    "onboarding",
                    "",
                    "",
                    "",
                    user_a["email"],
                    "Test Provider",
                    "ON",
                    now,
                ),
            )
            conn.commit()
        UserStore.update_user(user_a["email"], {"provider_id": provider_id, "role": "provider"})
    r = client.get(
        f"/api/providers/{provider_id}/earnings",
        headers=user_b["headers"],
    )
    assert r.status_code == 403


def test_events_entity_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job = client.post(
        "/instance",
        json={"name": "events-a", "vram_needed_gb": 1},
        headers=user_a["headers"],
    ).json()["instance"]
    r = client.get(
        f"/api/events/job/{job['job_id']}",
        headers=user_b["headers"],
    )
    assert r.status_code == 403


def test_api_inference_get_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    submitted = client.post(
        "/api/inference",
        json={
            "model": "distilbert-base-uncased-finetuned-sst-2-english",
            "inputs": ["hello"],
        },
        headers=user_a["headers"],
    )
    assert submitted.status_code == 200, submitted.text[:200]
    job_id = submitted.json()["job_id"]
    r = client.get(f"/api/inference/{job_id}", headers=user_b["headers"])
    assert r.status_code == 403


def test_v1_inference_poll_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    submitted = client.post(
        "/api/inference",
        json={
            "model": "distilbert-base-uncased-finetuned-sst-2-english",
            "inputs": ["hello"],
        },
        headers=user_a["headers"],
    )
    assert submitted.status_code == 200, submitted.text[:200]
    job_id = submitted.json()["job_id"]
    r = client.get(f"/v1/inference/{job_id}", headers=user_b["headers"])
    assert r.status_code == 403


def test_residency_trace_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    created = client.post(
        "/instance",
        json={"name": "residency-a", "vram_needed_gb": 1},
        headers=user_a["headers"],
    )
    if created.status_code == 200:
        job_id = created.json()["instance"]["job_id"]
    else:
        listed = client.get("/instances", headers=user_a["headers"])
        assert listed.status_code == 200, listed.text[:200]
        instances = listed.json().get("instances") or []
        assert instances, "need an owned instance for residency IDOR test"
        job_id = instances[0]["job_id"]
    r = client.get(
        f"/api/jurisdiction/residency-trace/{job_id}",
        headers=user_b["headers"],
    )
    assert r.status_code == 403
