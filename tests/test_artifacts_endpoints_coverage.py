"""Smoke coverage for routes/artifacts.py — POST /api/artifacts/download."""

import os
import uuid
from unittest.mock import patch

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def user_headers():
    email = f"artcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Artifacts Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    return headers


def test_artifacts_download(user_headers):
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    with patch("routes.instances._check_job_access"):
        r = client.post(
            "/api/artifacts/download",
            json={
                "job_id": job_id,
                "filename": "coverage-output.bin",
                "artifact_type": "job_output",
            },
            headers=user_headers,
        )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "url" in r.json()


def test_artifacts_download_requires_auth():
    r = client.post(
        "/api/artifacts/download",
        json={"job_id": "x", "filename": "out.bin", "artifact_type": "job_output"},
    )
    assert r.status_code == 401