"""Smoke coverage for routes/admin.py (UNTESTED_ENDPOINTS.md)."""

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def target_user_email():
    email = f"admincov-target-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Admin Target"},
    )
    return email


def test_admin_users():
    r = client.get("/api/admin/users", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_admin_overview():
    r = client.get("/api/admin/overview", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_admin_teams():
    r = client.get("/api/admin/teams", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_admin_revenue():
    r = client.get("/api/admin/revenue", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_admin_infrastructure():
    r = client.get("/api/admin/infrastructure", headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    volumes = body.get("volumes")
    assert isinstance(volumes, dict)
    assert "nfs" in volumes or "error" in volumes


def test_admin_activity():
    r = client.get("/api/admin/activity", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_admin_verification_queue():
    r = client.get("/api/admin/verification-queue", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_admin_ai_stats():
    r = client.get("/api/admin/ai-stats", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_admin_ai_conversations():
    r = client.get("/api/admin/ai-conversations", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_admin_set_user_role(target_user_email):
    r = client.post(
        f"/api/admin/users/{target_user_email}/role",
        params={"role": "submitter"},
        headers=_admin_headers(),
    )
    assert r.status_code == 200
    assert r.json().get("role") == "submitter"


def test_admin_toggle_admin(target_user_email):
    r = client.post(
        f"/api/admin/users/{target_user_email}/toggle-admin",
        headers=_admin_headers(),
    )
    assert r.status_code == 200
    assert "is_admin" in r.json()


def test_admin_remove_team_member_not_found():
    r = client.delete(
        "/api/admin/teams/nonexistent-team-id/members/nobody@xcelsior.ca",
        headers=_admin_headers(),
    )
    assert r.status_code == 404


def test_admin_agent_rollout_validation():
    r = client.post(
        "/api/admin/agent/rollout",
        headers=_admin_headers(),
        json={},
    )
    assert r.status_code == 400