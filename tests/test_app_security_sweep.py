"""App-wide authorization sweep — infrastructure guards and cross-account IDOR."""

import os
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

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
        email = f"appsec-{label}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
        reg = client.post(
            "/api/auth/register",
            json={"email": email, "password": "StrongPass123!", "name": f"AppSec {label}"},
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


_INFRA_ADMIN_ONLY = [
    ("POST", "/token/generate", None),
    ("GET", "/api/nfs/config", None),
    ("GET", "/builds", None),
    ("POST", "/build/probe-model/dockerfile", None),
]

_INFRA_ADMIN_ONLY_AUTH_ENFORCED = _INFRA_ADMIN_ONLY + [
    ("POST", "/build", {"model": "probe-model", "push": False}),
]


@pytest.mark.parametrize("method,path,body", _INFRA_ADMIN_ONLY)
def test_infrastructure_endpoints_require_admin(method, path, body, two_users):
    _, user_b = two_users
    r = client.request(method, path, json=body, headers=user_b["headers"])
    assert r.status_code == 403, f"{method} {path} -> {r.status_code}: {r.text[:120]}"


@pytest.fixture
def auth_required(monkeypatch):
    import routes._deps as deps

    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)


@pytest.mark.parametrize("method,path,body", _INFRA_ADMIN_ONLY_AUTH_ENFORCED)
def test_infrastructure_endpoints_reject_anonymous(method, path, body, auth_required):
    fresh = TestClient(app)
    r = fresh.request(method, path, json=body)
    assert r.status_code in (401, 403), f"{method} {path} -> {r.status_code}"


_SLA_PROTECTED = [
    ("GET", "/api/sla/hosts-summary"),
    ("GET", "/api/sla/host-probe"),
    ("GET", "/api/sla/violations/host-probe"),
    ("GET", "/api/sla/downtimes"),
]


@pytest.mark.parametrize("method,path", _SLA_PROTECTED)
def test_sla_data_requires_auth(method, path):
    fresh = TestClient(app)
    r = fresh.request(method, path)
    assert r.status_code == 401, f"{method} {path} -> {r.status_code}"


def test_sla_targets_public_reference():
    r = client.get("/api/sla/targets")
    assert r.status_code == 200
    assert r.json().get("ok") is True


_HOST_READS = [
    ("GET", "/hosts/ca", None),
    ("POST", "/api/jurisdiction/hosts", {"canada_only": True}),
]


@pytest.mark.parametrize("method,path,body", _HOST_READS)
def test_host_inventory_reads_require_auth(method, path, body, auth_required):
    fresh = TestClient(app)
    r = fresh.request(method, path, json=body)
    assert r.status_code == 401, f"{method} {path} -> {r.status_code}"


def test_host_inventory_reads_allowed_for_authenticated_user(two_users):
    _, user_b = two_users
    r = client.get("/hosts/ca", headers=user_b["headers"])
    assert r.status_code == 200
    assert "hosts" in r.json()


def test_volume_get_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    created = client.post(
        "/api/v2/volumes",
        json={"name": "sec-vol", "size_gb": 1},
        headers=user_a["headers"],
    )
    assert created.status_code == 200, created.text[:200]
    volume_id = created.json()["volume"]["volume_id"]
    r = client.get(f"/api/v2/volumes/{volume_id}", headers=user_b["headers"])
    assert r.status_code in (403, 404)


def test_artifacts_list_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job_id = _create_job(user_a["headers"], "art-list-a")
    r = client.get(f"/api/artifacts/{job_id}", headers=user_b["headers"])
    assert r.status_code == 403


def test_artifacts_download_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job_id = _create_job(user_a["headers"], "art-dl-a")
    r = client.post(
        "/api/artifacts/download",
        json={"job_id": job_id, "filename": "output.txt"},
        headers=user_b["headers"],
    )
    assert r.status_code == 403


def test_artifacts_upload_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job_id = _create_job(user_a["headers"], "art-up-a")
    r = client.post(
        "/api/artifacts/upload",
        json={"job_id": job_id, "filename": "output.txt"},
        headers=user_b["headers"],
    )
    assert r.status_code == 403


def test_team_get_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    team = client.post(
        "/api/teams",
        json={"name": "Team A", "plan": "free"},
        headers=user_a["headers"],
    )
    assert team.status_code == 200, team.text[:200]
    team_id = team.json()["team_id"]
    r = client.get(f"/api/teams/{team_id}", headers=user_b["headers"])
    assert r.status_code == 403


def test_ssh_key_delete_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    key = client.post(
        "/api/ssh/keys",
        json={
            "name": "probe",
            "public_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl test@xcelsior",
        },
        headers=user_a["headers"],
    )
    assert key.status_code == 200, key.text[:200]
    key_id = key.json()["id"]
    r = client.delete(f"/api/ssh/keys/{key_id}", headers=user_b["headers"])
    assert r.status_code == 404


def test_terminal_ticket_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job_id = _create_job(user_a["headers"], "term-a")
    r = client.post(
        "/api/terminal/ticket",
        json={"instance_id": job_id},
        headers=user_b["headers"],
    )
    assert r.status_code == 403


def test_stream_ticket_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    job_id = _create_job(user_a["headers"], "stream-a")
    r = client.post(f"/api/instances/{job_id}/stream-ticket", headers=user_b["headers"])
    assert r.status_code == 403