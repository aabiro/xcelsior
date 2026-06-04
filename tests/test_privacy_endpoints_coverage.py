"""Smoke coverage for routes/privacy.py (UNTESTED_ENDPOINTS.md)."""

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
    email = f"privcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Privacy Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200
    return email, {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_privacy_retention_policies():
    r = client.get("/api/privacy/retention-policies")
    assert r.status_code == 200
    assert "policies" in r.json()


def test_privacy_retention_summary():
    r = client.get("/api/privacy/retention-summary")
    assert r.status_code == 200


def test_privacy_purge_expired():
    r = client.post("/api/privacy/purge-expired")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "purged" in r.json()


def test_privacy_config_save_and_get():
    org_id = f"org-{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/privacy/config",
        json={
            "org_id": org_id,
            "privacy_level": "strict",
            "privacy_officer_name": "Officer",
            "privacy_officer_email": "privacy@xcelsior.ca",
        },
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("org_id") == org_id

    r2 = client.get(f"/api/privacy/config/{org_id}")
    assert r2.status_code == 200
    assert r2.json().get("privacy_level") == "strict"


def test_privacy_consent_v1_crud():
    entity_id = f"ent-{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/privacy/consent",
        json={
            "entity_id": entity_id,
            "consent_type": "data_collection",
            "details": {"source": "coverage"},
        },
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True

    r2 = client.get(f"/api/privacy/consent/{entity_id}")
    assert r2.status_code == 200
    assert "consents" in r2.json()

    r3 = client.delete(f"/api/privacy/consent/{entity_id}/data_collection")
    assert r3.status_code == 200
    assert r3.json().get("ok") is True


def test_privacy_v2_consent_list_record_withdraw(user_headers):
    _, headers = user_headers
    purpose = f"marketing-{uuid.uuid4().hex[:6]}"
    r = client.post(
        "/api/v2/privacy/consent",
        json={"purpose": purpose, "consent_type": "express"},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True

    r2 = client.get("/api/v2/privacy/consents", headers=headers)
    assert r2.status_code == 200
    assert r2.json().get("ok") is True
    assert isinstance(r2.json().get("consents"), list)

    r3 = client.delete(f"/api/v2/privacy/consent/{purpose}", headers=headers)
    assert r3.status_code == 200
    assert r3.json().get("ok") is True


def test_privacy_v2_erase(user_headers, monkeypatch):
    _, headers = user_headers

    def _mock_erasure(user_id: str) -> dict:
        return {"user_id": user_id, "actions": ["encryption_key_destroyed"]}

    monkeypatch.setattr("routes.privacy.execute_right_to_erasure", _mock_erasure)
    r = client.post("/api/v2/privacy/erase", headers=headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "erasure" in r.json()


def test_privacy_v2_consent_requires_auth():
    r = client.post(
        "/api/v2/privacy/consent",
        json={"purpose": "newsletter", "consent_type": "express"},
    )
    assert r.status_code == 401