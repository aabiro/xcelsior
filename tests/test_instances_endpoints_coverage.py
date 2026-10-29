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
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    customer_id = reg["user"]["customer_id"]
    _fund_wallet(customer_id, headers)
    job = client.post(
        "/instance",
        json={"name": f"cov-{uuid.uuid4().hex[:6]}", "vram_needed_gb": 1},
        headers=headers,
    ).json()["instance"]
    job_id = job["job_id"]
    # host_id is an authority column a payload patch may not set.
    from psycopg.types.json import Jsonb
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET host_id=%s, status='queued', payload=payload||%s WHERE job_id=%s",
            ("cov-host-1", Jsonb({"host_id": "cov-host-1", "status": "queued"}), job_id),
        )
    return {"email": email, "customer_id": customer_id, "headers": headers, "job_id": job_id}


def test_image_templates():
    r = client.get("/api/images/templates")
    assert r.status_code == 200
    assert "templates" in r.json()
    assert len(r.json()["templates"]) >= 1


def test_exclusive_gpu_launch_is_preserved_for_worker(instance_ctx):
    response = client.post(
        "/instance",
        json={"name": "exclusive-contract", "vram_needed_gb": 1, "exclusive_gpu": True},
        headers=instance_ctx["headers"],
    )
    assert response.status_code == 200, response.text[:200]
    job = response.json()["instance"]
    from scheduler import get_job

    assert job["exclusive_gpu"] is True
    assert get_job(job["job_id"])["exclusive_gpu"] is True


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


def test_admin_reinject_shell():
    """The directive targets the running native attempt's container."""
    from db import _get_pg_pool
    from tests.fixtures.native_runtime import running_instance

    with running_instance(owner=f"reinject-{uuid.uuid4().hex[:8]}") as instance:
        r = client.post(
            f"/admin/instances/{instance['job_id']}/reinject-shell",
            headers=_admin_headers(),
        )
        assert r.status_code == 200, r.text
        assert r.json().get("ok") is True
        with _get_pg_pool().connection() as conn:
            row = conn.execute(
                "SELECT attempt_id::text, args->>'container_name' FROM agent_commands WHERE id=%s",
                (r.json()["command_id"],),
            ).fetchone()
    attempt_id = instance["authority"]["attempt_id"]
    assert row == (str(attempt_id), f"xcl-{instance['job_id']}-{str(attempt_id)[:8]}")


def test_admin_reinject_shell_needs_a_live_attempt(instance_ctx):
    """A queued row with a host name but no placement has no container to touch."""
    r = client.post(
        f"/admin/instances/{instance_ctx['job_id']}/reinject-shell",
        headers=_admin_headers(),
    )
    assert r.status_code == 409
    assert "fenced_residual_requires_live_authority" in r.text


def test_admin_reinject_shell_missing_job():
    r = client.post(
        "/admin/instances/nonexistent-job-id/reinject-shell",
        headers=_admin_headers(),
    )
    assert r.status_code == 404
