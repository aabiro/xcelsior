"""P1 team tenancy — role guards and consistent job access."""

import os
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

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


def _register_and_login(label: str) -> dict:
    email = f"team-p1-{label}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": f"Team P1 {label}"},
    )
    assert reg.status_code == 200, reg.text[:200]
    reg_body = reg.json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text[:200]
    body = login.json()
    user = reg_body.get("user") or body.get("user") or {}
    assert UserStore.get_user(email) is not None
    return {
        "email": email,
        "customer_id": user["customer_id"],
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
    }


def _fund_wallet(customer_id: str, headers: dict) -> None:
    r = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "Team P1 credits"},
        headers=headers,
    )
    assert r.status_code == 200, r.text[:200]


def _create_job(headers: dict, name: str) -> str:
    r = client.post(
        "/instance",
        json={"name": name, "vram_needed_gb": 1},
        headers=headers,
    )
    assert r.status_code == 200, r.text[:200]
    return r.json()["instance"]["job_id"]


@pytest.fixture(scope="module")
def team_roles():
    admin = _register_and_login("admin")
    member = _register_and_login("member")
    viewer = _register_and_login("viewer")
    outsider = _register_and_login("outsider")

    team_resp = client.post(
        "/api/teams",
        json={"name": "P1 Role Team", "plan": "free"},
        headers=admin["headers"],
    )
    assert team_resp.status_code == 200, team_resp.text[:200]
    team_id = team_resp.json()["team_id"]
    billing_customer_id = UserStore.get_team(team_id)["billing_customer_id"]
    _fund_wallet(billing_customer_id, admin["headers"])

    for email, role in ((member["email"], "member"), (viewer["email"], "viewer")):
        add = client.post(
            f"/api/teams/{team_id}/members",
            json={"email": email, "role": role},
            headers=admin["headers"],
        )
        assert add.status_code == 200, add.text[:200]
        assert not add.json().get("invited"), add.text[:200]

    job_id = _create_job(member["headers"], "team-role-job")

    return {
        "admin": admin,
        "member": member,
        "viewer": viewer,
        "outsider": outsider,
        "team_id": team_id,
        "billing_customer_id": billing_customer_id,
        "job_id": job_id,
    }


def test_auth_me_includes_team_context(team_roles):
    for key, role in (
        ("admin", "admin"),
        ("member", "member"),
        ("viewer", "viewer"),
    ):
        r = client.get("/api/auth/me", headers=team_roles[key]["headers"])
        assert r.status_code == 200, r.text[:200]
        user = r.json()["user"]
        assert user["team_id"] == team_roles["team_id"]
        assert user["team_role"] == role
        assert user["billing_customer_id"] == team_roles["billing_customer_id"]
        assert user["team_can_write_instances"] == (role != "viewer")
        assert user["team_can_manage_billing"] == (role == "admin")


def test_member_can_read_team_wallet(team_roles):
    r = client.get(
        f"/api/billing/wallet/{team_roles['billing_customer_id']}",
        headers=team_roles["member"]["headers"],
    )
    assert r.status_code == 200, r.text[:200]


def test_member_cannot_deposit_team_wallet(team_roles):
    r = client.post(
        f"/api/billing/wallet/{team_roles['billing_customer_id']}/deposit",
        json={"amount_cad": 5.0},
        headers=team_roles["member"]["headers"],
    )
    assert r.status_code == 403, r.text[:200]


def test_viewer_can_read_team_instance(team_roles):
    r = client.get(
        f"/instance/{team_roles['job_id']}",
        headers=team_roles["viewer"]["headers"],
    )
    assert r.status_code == 200, r.text[:200]


def test_viewer_cannot_rename_team_instance(team_roles):
    r = client.patch(
        f"/instance/{team_roles['job_id']}/name",
        json={"name": "blocked"},
        headers=team_roles["viewer"]["headers"],
    )
    assert r.status_code == 403, r.text[:200]
    assert "viewer" in r.text.lower()


def test_viewer_cannot_stop_team_instance(team_roles):
    r = client.post(
        f"/instances/{team_roles['job_id']}/stop",
        headers=team_roles["viewer"]["headers"],
    )
    assert r.status_code == 403, r.text[:200]
    assert "viewer" in r.text.lower()


def test_member_can_rename_team_instance(team_roles):
    r = client.patch(
        f"/instance/{team_roles['job_id']}/name",
        json={"name": "member-renamed"},
        headers=team_roles["member"]["headers"],
    )
    assert r.status_code == 200, r.text[:200]


def test_member_restart_passes_team_access(team_roles):
    """Queued jobs return 400 on restart; outsiders would get 403 first."""
    r = client.post(
        f"/instances/{team_roles['job_id']}/restart",
        headers=team_roles["member"]["headers"],
    )
    assert r.status_code == 400, r.text[:200]
    assert "must be running or stopped" in r.text.lower()


def test_outsider_cannot_restart_team_instance(team_roles):
    r = client.post(
        f"/instances/{team_roles['job_id']}/restart",
        headers=team_roles["outsider"]["headers"],
    )
    assert r.status_code == 403, r.text[:200]


def test_outsider_cannot_read_team_instance(team_roles):
    r = client.get(
        f"/instance/{team_roles['job_id']}",
        headers=team_roles["outsider"]["headers"],
    )
    assert r.status_code == 403, r.text[:200]


def test_team_members_share_concurrency_pool(team_roles, monkeypatch):
    import routes.instances as inst

    # Fixture already created one team job; cap=2 leaves room for one more.
    monkeypatch.setattr(inst, "_TEAM_PLAN_CONCURRENCY_CAPS", {"free": 2, "pro": 2, "enterprise": 2})

    first = client.post(
        "/instance",
        json={"name": "pool-a", "vram_needed_gb": 1},
        headers=team_roles["member"]["headers"],
    )
    assert first.status_code == 200, first.text[:200]

    # Admin shares the same pool — second slot is now full.
    second = client.post(
        "/instance",
        json={"name": "pool-b", "vram_needed_gb": 1},
        headers=team_roles["admin"]["headers"],
    )
    assert second.status_code == 429, second.text[:200]
    assert "concurrent instance limit" in second.text.lower()


def test_login_includes_team_context(team_roles):
    login = client.post(
        "/api/auth/login",
        json={"email": team_roles["member"]["email"], "password": "StrongPass123!"},
    )
    assert login.status_code == 200, login.text[:200]
    user = login.json()["user"]
    assert user["team_id"] == team_roles["team_id"]
    assert user["team_role"] == "member"
    assert user["billing_customer_id"] == team_roles["billing_customer_id"]
    assert user["team_can_write_instances"] is True
    assert user["team_can_manage_billing"] is False


def test_single_team_user_can_switch_to_personal_workspace():
    solo = _register_and_login("solo-one-team")
    team_resp = client.post(
        "/api/teams",
        json={"name": "Solo Team", "plan": "free"},
        headers=solo["headers"],
    )
    assert team_resp.status_code == 200, team_resp.text[:200]
    team_id = team_resp.json()["team_id"]
    billing_customer_id = UserStore.get_team(team_id)["billing_customer_id"]

    on_team = client.get("/api/auth/me", headers=solo["headers"])
    assert on_team.status_code == 200, on_team.text[:200]
    team_user = on_team.json()["user"]
    assert team_user["team_id"] == team_id
    assert team_user["billing_customer_id"] == billing_customer_id

    personal_switch = client.patch(
        "/api/teams/active",
        json={"team_id": None},
        headers=solo["headers"],
    )
    assert personal_switch.status_code == 200, personal_switch.text[:200]
    assert personal_switch.json().get("active_team_id") is None

    personal = client.get("/api/auth/me", headers=solo["headers"])
    assert personal.status_code == 200, personal.text[:200]
    personal_user = personal.json()["user"]
    assert not personal_user.get("team_id")
    assert personal_user["billing_customer_id"] == solo["customer_id"]


def test_member_volume_scoped_to_team_wallet(team_roles):
    name = f"team-vol-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v2/volumes",
        json={"name": name, "size_gb": 1},
        headers=team_roles["member"]["headers"],
    )
    assert created.status_code == 200, created.text[:200]
    volume_id = created.json()["volume"]["volume_id"]

    fetched = client.get(
        f"/api/v2/volumes/{volume_id}",
        headers=team_roles["member"]["headers"],
    )
    assert fetched.status_code == 200, fetched.text[:200]
    assert fetched.json()["volume"]["owner_id"] == team_roles["billing_customer_id"]

    listed = client.get("/api/v2/volumes", headers=team_roles["viewer"]["headers"])
    assert listed.status_code == 200, listed.text[:200]
    ids = [v["volume_id"] for v in listed.json().get("volumes") or []]
    assert volume_id in ids

    client.delete(f"/api/v2/volumes/{volume_id}", headers=team_roles["admin"]["headers"])


def test_member_can_launch_instance_with_team_volume(team_roles):
    name = f"launch-vol-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v2/volumes",
        json={"name": name, "size_gb": 1},
        headers=team_roles["member"]["headers"],
    )
    assert created.status_code == 200, created.text[:200]
    volume_id = created.json()["volume"]["volume_id"]

    launched = client.post(
        "/instance",
        json={
            "name": "team-vol-launch",
            "vram_needed_gb": 1,
            "volume_ids": [volume_id],
        },
        headers=team_roles["member"]["headers"],
    )
    assert launched.status_code == 200, launched.text[:300]

    client.delete(f"/api/v2/volumes/{volume_id}", headers=team_roles["admin"]["headers"])


def test_outsider_cannot_launch_with_team_volume(team_roles):
    name = f"outsider-vol-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v2/volumes",
        json={"name": name, "size_gb": 1},
        headers=team_roles["member"]["headers"],
    )
    assert created.status_code == 200, created.text[:200]
    volume_id = created.json()["volume"]["volume_id"]

    outsider = team_roles["outsider"]
    _fund_wallet(outsider["customer_id"], outsider["headers"])

    blocked = client.post(
        "/instance",
        json={
            "name": "blocked-vol-launch",
            "vram_needed_gb": 1,
            "volume_ids": [volume_id],
        },
        headers=outsider["headers"],
    )
    assert blocked.status_code == 404, blocked.text[:200]

    client.delete(f"/api/v2/volumes/{volume_id}", headers=team_roles["admin"]["headers"])


def test_viewer_cannot_create_team_volume(team_roles):
    blocked = client.post(
        "/api/v2/volumes",
        json={"name": f"viewer-vol-{uuid.uuid4().hex[:8]}", "size_gb": 1},
        headers=team_roles["viewer"]["headers"],
    )
    assert blocked.status_code == 403, blocked.text[:200]
    assert "viewer" in blocked.text.lower()


def test_viewer_cannot_attach_or_delete_team_volume(team_roles):
    created = client.post(
        "/api/v2/volumes",
        json={"name": f"viewer-guard-{uuid.uuid4().hex[:8]}", "size_gb": 1},
        headers=team_roles["member"]["headers"],
    )
    assert created.status_code == 200, created.text[:200]
    volume_id = created.json()["volume"]["volume_id"]
    job_id = team_roles["job_id"]

    attach_blocked = client.post(
        f"/api/v2/volumes/{volume_id}/attach",
        json={"instance_id": job_id, "mount_path": "/workspace"},
        headers=team_roles["viewer"]["headers"],
    )
    assert attach_blocked.status_code == 403, attach_blocked.text[:200]

    delete_blocked = client.delete(
        f"/api/v2/volumes/{volume_id}",
        headers=team_roles["viewer"]["headers"],
    )
    assert delete_blocked.status_code == 403, delete_blocked.text[:200]

    client.delete(f"/api/v2/volumes/{volume_id}", headers=team_roles["admin"]["headers"])


def test_personal_user_concurrency_isolated_from_team(team_roles, monkeypatch):
    import routes.instances as inst

    monkeypatch.setattr(inst, "_TEAM_PLAN_CONCURRENCY_CAPS", {"free": 1, "pro": 1, "enterprise": 1})

    # Team already has one queued job from fixture setup.
    blocked = client.post(
        "/instance",
        json={"name": "team-blocked", "vram_needed_gb": 1},
        headers=team_roles["member"]["headers"],
    )
    assert blocked.status_code == 429, blocked.text[:200]

    # Outsider is not on the team — separate personal pool and wallet.
    outsider = team_roles["outsider"]
    _fund_wallet(outsider["customer_id"], outsider["headers"])
    personal = client.post(
        "/instance",
        json={"name": "personal-ok", "vram_needed_gb": 1},
        headers=outsider["headers"],
    )
    assert personal.status_code == 200, personal.text[:200]