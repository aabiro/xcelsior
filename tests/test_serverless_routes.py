"""Smoke coverage for routes/serverless.py management + queue API."""

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from api import app
from db import UserStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


def _register_and_fund() -> dict:
    email = f"slvr-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Slvr Cov"},
    )
    assert reg.status_code == 200, reg.text[:300]
    reg_body = reg.json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text[:200]
    body = login.json()
    user = reg_body.get("user") or body.get("user") or {}
    customer_id = user["customer_id"]
    assert UserStore.get_user(email) is not None
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    deposit = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "serverless route test credits"},
        headers=headers,
    )
    assert deposit.status_code == 200, deposit.text[:200]
    return headers


@pytest.fixture(scope="module")
def user_headers():
    return _register_and_fund()


@pytest.fixture(scope="module")
def endpoint_id(user_headers):
    r = client.post(
        "/api/v2/serverless/endpoints",
        headers=user_headers,
        json={
            "name": "test-llama",
            "mode": "preset",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "gpu_type": "",
            "min_workers": 0,
            "max_workers": 1,
        },
    )
    assert r.status_code == 200, r.text[:300]
    return r.json()["endpoint"]["endpoint_id"]


def test_serverless_list_without_session_token():
    """In test env anonymous access is allowed when AUTH_REQUIRED=false."""
    r = client.get("/api/v2/serverless/endpoints")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_serverless_list(user_headers):
    r = client.get("/api/v2/serverless/endpoints", headers=user_headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_serverless_get_endpoint(user_headers, endpoint_id):
    r = client.get(
        f"/api/v2/serverless/endpoints/{endpoint_id}",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json()["endpoint"]["endpoint_id"] == endpoint_id


def test_serverless_endpoint_health(user_headers, endpoint_id):
    r = client.get(
        f"/api/v2/serverless/endpoints/{endpoint_id}/health",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "health" in r.json()


def test_serverless_endpoint_usage(user_headers, endpoint_id):
    r = client.get(
        f"/api/v2/serverless/endpoints/{endpoint_id}/usage",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "usage" in r.json()


def test_serverless_get_not_found(user_headers):
    r = client.get(
        "/api/v2/serverless/endpoints/sep-nonexistent",
        headers=user_headers,
    )
    assert r.status_code == 404


def test_serverless_run_job(user_headers, endpoint_id):
    r = client.post(
        f"/v1/serverless/{endpoint_id}/run",
        headers=user_headers,
        json={"input": {"prompt": "hello"}},
    )
    assert r.status_code == 200
    assert r.json().get("status") == "IN_QUEUE"
    assert r.json().get("id")


def test_serverless_openai_models(user_headers, endpoint_id):
    r = client.get(
        f"/v1/serverless/{endpoint_id}/openai/v1/models",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("object") == "list"


def test_serverless_openai_chat_requires_worker(user_headers, endpoint_id):
    r = client.post(
        f"/v1/serverless/{endpoint_id}/openai/v1/chat/completions",
        headers=user_headers,
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code in (503, 502, 504)
    assert r.status_code != 500


def test_serverless_delete_endpoint(user_headers, endpoint_id):
    r = client.delete(
        f"/api/v2/serverless/endpoints/{endpoint_id}",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True