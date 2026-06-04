"""Smoke coverage for routes/notifications.py (UNTESTED_ENDPOINTS.md)."""

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app
from db import NotificationStore

client = TestClient(app)


@pytest.fixture(scope="module")
def user_ctx():
    email = f"notifcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Notif Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    notif_id = NotificationStore.create(
        email,
        "system",
        "Coverage notification",
        body="Smoke test",
    )
    return {"email": email, "headers": headers, "notification_id": notif_id}


def test_notifications_unread_count_requires_auth():
    r = client.get("/api/notifications/unread-count")
    assert r.status_code == 401


def test_notifications_unread_count(user_ctx):
    r = client.get("/api/notifications/unread-count", headers=user_ctx["headers"])
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("unread_count", 0) >= 1


def test_notifications_mark_read(user_ctx):
    nid = user_ctx["notification_id"]
    r = client.post(
        f"/api/notifications/{nid}/read",
        headers=user_ctx["headers"],
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_notifications_read_all(user_ctx):
    NotificationStore.create(
        user_ctx["email"],
        "system",
        "Second notification",
        body="Still unread",
    )
    r = client.post("/api/notifications/read-all", headers=user_ctx["headers"])
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("marked", 0) >= 0


def test_notifications_delete(user_ctx):
    nid = NotificationStore.create(
        user_ctx["email"],
        "system",
        "To delete",
        body="",
    )
    r = client.delete(
        f"/api/notifications/{nid}",
        headers=user_ctx["headers"],
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_notifications_delete_missing(user_ctx):
    r = client.delete(
        "/api/notifications/notif-does-not-exist",
        headers=user_ctx["headers"],
    )
    assert r.status_code == 404


def test_push_subscription_status(user_ctx):
    r = client.get(
        "/api/notifications/push/subscription",
        headers=user_ctx["headers"],
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "configured" in r.json()


def test_push_subscription_upsert(user_ctx):
    r = client.post(
        "/api/notifications/push/subscription",
        headers=user_ctx["headers"],
        json={
            "endpoint": "https://push.example/sub/1",
            "keys": {"p256dh": "a" * 20, "auth": "b" * 16},
        },
    )
    assert r.status_code in (200, 503)
    assert r.status_code != 500


def test_push_subscription_delete(user_ctx):
    r = client.request(
        "DELETE",
        "/api/notifications/push/subscription",
        headers=user_ctx["headers"],
        json={"endpoint": "https://push.example/sub/unused"},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True