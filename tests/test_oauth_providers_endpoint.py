"""The login and register pages show only sign-in providers configured here."""

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

from fastapi.testclient import TestClient

import routes.auth as auth
from api import app

client = TestClient(app)


def test_lists_only_providers_with_a_client_id(monkeypatch):
    monkeypatch.setitem(auth._OAUTH_PROVIDERS["google"], "client_id", "google-id")
    monkeypatch.setitem(auth._OAUTH_PROVIDERS["facebook"], "client_id", "")
    r = client.get("/api/auth/oauth/providers")
    assert r.status_code == 200
    providers = r.json()["providers"]
    assert "google" in providers
    assert "facebook" not in providers


def test_is_public():
    assert client.get("/api/auth/oauth/providers").status_code == 200
