"""Instance & scheduler authorization regression tests."""

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"

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
        email = f"inst-auth-{label}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
        reg = client.post(
            "/api/auth/register",
            json={"email": email, "password": "StrongPass123!", "name": f"Inst {label}"},
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
                "headers": headers,
            }
        )
    return users[0], users[1]


def _create_job(headers: dict, name: str) -> str:
    r = client.post(
        "/instance",
        json={"name": name, "vram_needed_gb": 1},
        headers=headers,
    )
    assert r.status_code == 200, r.text[:200]
    return r.json()["instance"]["job_id"]


@pytest.fixture
def auth_required(monkeypatch):
    import routes._deps as deps

    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)


def test_instances_list_isolated_per_user(two_users):
    user_a, user_b = two_users
    job_a = _create_job(user_a["headers"], "list-a")
    _create_job(user_b["headers"], "list-b")

    r_a = client.get("/instances", headers=user_a["headers"])
    assert r_a.status_code == 200
    ids_a = {j["job_id"] for j in r_a.json()["instances"]}
    assert job_a in ids_a

    r_b = client.get("/instances", headers=user_b["headers"])
    assert r_b.status_code == 200
    ids_b = {j["job_id"] for j in r_b.json()["instances"]}
    assert job_a not in ids_b


def test_instances_list_requires_auth_when_enforced(auth_required):
    fresh = TestClient(app)
    r = fresh.get("/instances")
    assert r.status_code == 401


def test_instance_log_stream_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job_id = _create_job(user_a["headers"], "logs-a")
    r = client.get(
        f"/instances/{job_id}/logs/stream",
        headers=user_b["headers"],
    )
    assert r.status_code == 403


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/queue/process"),
        ("POST", "/failover"),
        ("POST", "/api/v2/scheduler/process-binpack"),
        ("POST", "/queue/process/ca"),
        ("POST", "/api/queue/process-sovereign"),
    ],
)
def test_scheduler_triggers_forbidden_for_regular_user(two_users, method, path):
    _, user_b = two_users
    body = (
        {"canada_only": True}
        if "sovereign" in path
        else None
    )
    r = client.request(method, path, json=body, headers=user_b["headers"])
    assert r.status_code == 403, f"{method} {path} -> {r.status_code}"


def test_instance_rename_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job_id = _create_job(user_a["headers"], "rename-a")
    r = client.patch(
        f"/instance/{job_id}/name",
        json={"name": "stolen"},
        headers=user_b["headers"],
    )
    assert r.status_code == 403


def test_instance_stop_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job_id = _create_job(user_a["headers"], "stop-a")
    r = client.post(f"/instances/{job_id}/stop", headers=user_b["headers"])
    assert r.status_code == 403