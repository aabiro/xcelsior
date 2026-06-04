"""Smoke coverage for routes/instances.py (UNTESTED_ENDPOINTS.md)."""

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


def _fund_wallet(customer_id: str, headers: dict) -> None:
    r = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "Instance cov"},
        headers=headers,
    )
    assert r.status_code == 200, r.text[:200]


@pytest.fixture(scope="module")
def instance_ctx():
    email = f"instcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Inst Cov"},
    ).json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    customer_id = reg["user"]["customer_id"]
    _fund_wallet(customer_id, headers)
    job = client.post(
        "/instance",
        json={"name": f"cov-{uuid.uuid4().hex[:6]}", "vram_needed_gb": 1},
        headers=headers,
    ).json()["instance"]
    job_id = job["job_id"]
    from scheduler import _set_job_fields

    _set_job_fields(job_id, host_id="cov-host-1", status="queued")
    return {"email": email, "customer_id": customer_id, "headers": headers, "job_id": job_id}


def test_image_templates():
    r = client.get("/api/images/templates")
    assert r.status_code == 200
    assert "templates" in r.json()
    assert len(r.json()["templates"]) >= 1


def test_instance_log_stream(instance_ctx, monkeypatch):
    import routes.instances as inst_mod

    async def _short_log_stream(request, job_id):
        yield 'event: connected\ndata: {"status":"streaming"}\n\n'

    monkeypatch.setattr(inst_mod, "_job_log_generator", _short_log_stream)
    job_id = instance_ctx["job_id"]
    r = client.get(
        f"/instances/{job_id}/logs/stream",
        headers=instance_ctx["headers"],
    )
    assert r.status_code == 200
    assert "text/event-stream" in (r.headers.get("content-type") or "")
    assert "connected" in r.text


def test_instance_log_stream_not_found(instance_ctx):
    r = client.get(
        "/instances/nonexistent-job-id/logs/stream",
        headers=instance_ctx["headers"],
    )
    assert r.status_code == 404


def test_instance_rename(instance_ctx):
    job_id = instance_ctx["job_id"]
    r = client.patch(
        f"/instance/{job_id}/name",
        headers=instance_ctx["headers"],
        json={"name": "renamed-coverage"},
    )
    assert r.status_code == 200
    assert r.json().get("name") == "renamed-coverage"


def test_instance_lock_unlock(instance_ctx):
    job_id = instance_ctx["job_id"]
    headers = instance_ctx["headers"]
    r_lock = client.post(f"/instances/{job_id}/lock", headers=headers)
    assert r_lock.status_code == 200
    assert r_lock.json().get("locked") is True

    r_unlock = client.post(f"/instances/{job_id}/unlock", headers=headers)
    assert r_unlock.status_code == 200
    assert r_unlock.json().get("locked") is False


def test_instance_reset_not_running(instance_ctx):
    job_id = instance_ctx["job_id"]
    r = client.post(f"/instances/{job_id}/reset", headers=instance_ctx["headers"])
    assert r.status_code in (400, 423)
    assert r.status_code != 500


def test_admin_reinject_shell(instance_ctx):
    job_id = instance_ctx["job_id"]
    r = client.post(
        f"/admin/instances/{job_id}/reinject-shell",
        headers=_admin_headers(),
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("command_id") is not None


def test_admin_reinject_shell_missing_job():
    r = client.post(
        "/admin/instances/nonexistent-job-id/reinject-shell",
        headers=_admin_headers(),
    )
    assert r.status_code == 404