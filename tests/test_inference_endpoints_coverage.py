"""Smoke coverage for routes/inference.py v2 endpoints (UNTESTED_ENDPOINTS.md)."""

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def user_headers():
    email = f"infcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Inf Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.fixture(scope="module")
def endpoint_id(user_headers):
    r = client.post(
        "/api/v2/inference/endpoints",
        headers=user_headers,
        json={
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "gpu_type": "",
            "min_workers": 0,
            "max_workers": 1,
        },
    )
    assert r.status_code == 200, r.text[:300]
    return r.json()["endpoint"]["endpoint_id"]


def test_inference_list_requires_auth():
    r = client.get("/api/v2/inference/endpoints")
    assert r.status_code == 401


def test_inference_list(user_headers):
    r = client.get("/api/v2/inference/endpoints", headers=user_headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_inference_get_endpoint(user_headers, endpoint_id):
    r = client.get(
        f"/api/v2/inference/endpoints/{endpoint_id}",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json()["endpoint"]["endpoint_id"] == endpoint_id


def test_inference_endpoint_health(user_headers, endpoint_id):
    r = client.get(
        f"/api/v2/inference/endpoints/{endpoint_id}/health",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "health" in r.json()


def test_inference_endpoint_usage(user_headers, endpoint_id):
    r = client.get(
        f"/api/v2/inference/endpoints/{endpoint_id}/usage",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "usage" in r.json()


def test_inference_get_not_found(user_headers):
    r = client.get(
        "/api/v2/inference/endpoints/ep-nonexistent",
        headers=user_headers,
    )
    assert r.status_code == 404


def test_inference_delete_endpoint(user_headers, endpoint_id):
    r = client.delete(
        f"/api/v2/inference/endpoints/{endpoint_id}",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_openai_chat_completions(user_headers):
    r = client.post(
        "/v1/chat/completions",
        headers=user_headers,
        json={
            "model": "distilbert-base-uncased-finetuned-sst-2-english",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code in (200, 503)
    assert r.status_code != 500
    if r.status_code == 200:
        assert r.json().get("object") == "chat.completion"