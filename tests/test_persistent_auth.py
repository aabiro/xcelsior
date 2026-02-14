"""Tests for Phase 10 — Persistent Auth, Teams & Provider UX (v2.6.0).

Covers:
- UserStore CRUD (users, sessions, API keys, teams)
- Team management (create, add/remove members, capacity limits, delete)
- Auth endpoints with persistent storage integration
- Team API endpoints
- SLA enforcement endpoint
- Notification preferences
"""

import json
import os
import tempfile
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_DB_BACKEND", "sqlite")
os.environ.setdefault("XCELSIOR_PERSISTENT_AUTH", "true")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

import db as db_mod
from db import UserStore, auth_connection


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_auth_db(tmp_path, monkeypatch):
    """Redirect auth DB to a temp directory for test isolation."""
    auth_file = str(tmp_path / "test_auth.db")
    monkeypatch.setenv("XCELSIOR_AUTH_DB_PATH", auth_file)
    monkeypatch.setattr(db_mod, "AUTH_DB_FILE", auth_file)
    # Also isolate the main DB
    db_file = str(tmp_path / "test_xcelsior.db")
    monkeypatch.setenv("XCELSIOR_DB_PATH", db_file)
    monkeypatch.setattr(db_mod, "DEFAULT_DB_FILE", db_file)
    yield auth_file


@pytest.fixture
def sample_user():
    return {
        "email": "test@xcelsior.ca",
        "user_id": f"user-{uuid.uuid4().hex[:8]}",
        "name": "Test User",
        "password_hash": "hashed_pw",
        "salt": "salt123",
        "role": "submitter",
        "customer_id": "cust-test01",
        "country": "CA",
        "province": "ON",
        "created_at": time.time(),
    }


@pytest.fixture
def sample_user_2():
    return {
        "email": "provider@xcelsior.ca",
        "user_id": f"user-{uuid.uuid4().hex[:8]}",
        "name": "Provider User",
        "password_hash": "hashed_pw2",
        "salt": "salt456",
        "role": "provider",
        "customer_id": "cust-prov01",
        "country": "CA",
        "province": "BC",
        "created_at": time.time(),
    }


# ── UserStore: Users ──────────────────────────────────────────────────


class TestUserStoreCRUD:
    """Verify UserStore user CRUD operations."""

    def test_create_and_get_user(self, sample_user):
        UserStore.create_user(sample_user)
        user = UserStore.get_user(sample_user["email"])
        assert user is not None
        assert user["email"] == sample_user["email"]
        assert user["name"] == "Test User"
        assert user["role"] == "submitter"
        assert user["country"] == "CA"

    def test_get_user_by_id(self, sample_user):
        UserStore.create_user(sample_user)
        user = UserStore.get_user_by_id(sample_user["user_id"])
        assert user is not None
        assert user["email"] == sample_user["email"]

    def test_user_exists(self, sample_user):
        assert not UserStore.user_exists(sample_user["email"])
        UserStore.create_user(sample_user)
        assert UserStore.user_exists(sample_user["email"])

    def test_update_user(self, sample_user):
        UserStore.create_user(sample_user)
        UserStore.update_user(sample_user["email"], {"name": "Updated Name", "role": "admin"})
        user = UserStore.get_user(sample_user["email"])
        assert user["name"] == "Updated Name"
        assert user["role"] == "admin"

    def test_update_user_ignores_disallowed_fields(self, sample_user):
        UserStore.create_user(sample_user)
        UserStore.update_user(sample_user["email"], {"email": "evil@hacker.com", "name": "Good Name"})
        user = UserStore.get_user(sample_user["email"])
        assert user["email"] == sample_user["email"]  # unchanged
        assert user["name"] == "Good Name"

    def test_delete_user(self, sample_user):
        UserStore.create_user(sample_user)
        UserStore.delete_user(sample_user["email"])
        assert UserStore.get_user(sample_user["email"]) is None

    def test_delete_user_cascades_sessions_keys(self, sample_user):
        UserStore.create_user(sample_user)
        sess = {
            "token": "tok-del-test",
            "email": sample_user["email"],
            "user_id": sample_user["user_id"],
            "role": "submitter",
            "expires_at": time.time() + 3600,
        }
        UserStore.create_session(sess)
        key_data = {
            "key": "xc_del_test_key_1234567890abcdef",
            "name": "del-key",
            "email": sample_user["email"],
            "user_id": sample_user["user_id"],
        }
        UserStore.create_api_key(key_data)
        UserStore.delete_user(sample_user["email"])
        assert UserStore.get_session("tok-del-test") is None
        assert UserStore.list_api_keys(sample_user["email"]) == []

    def test_list_users(self, sample_user, sample_user_2):
        UserStore.create_user(sample_user)
        UserStore.create_user(sample_user_2)
        users = UserStore.list_users()
        assert len(users) == 2

    def test_get_nonexistent_user_returns_none(self):
        assert UserStore.get_user("nobody@xcelsior.ca") is None
        assert UserStore.get_user_by_id("user-nope") is None


# ── UserStore: Sessions ───────────────────────────────────────────────


class TestUserStoreSessions:
    """Verify session lifecycle."""

    def test_create_and_get_session(self, sample_user):
        UserStore.create_user(sample_user)
        sess = {
            "token": f"tok-{uuid.uuid4().hex[:12]}",
            "email": sample_user["email"],
            "user_id": sample_user["user_id"],
            "role": "submitter",
            "name": "Test User",
            "expires_at": time.time() + 3600,
        }
        UserStore.create_session(sess)
        result = UserStore.get_session(sess["token"])
        assert result is not None
        assert result["email"] == sample_user["email"]

    def test_expired_session_not_returned(self, sample_user):
        UserStore.create_user(sample_user)
        sess = {
            "token": "tok-expired",
            "email": sample_user["email"],
            "user_id": sample_user["user_id"],
            "role": "submitter",
            "expires_at": time.time() - 100,
        }
        UserStore.create_session(sess)
        assert UserStore.get_session("tok-expired") is None

    def test_delete_session(self, sample_user):
        UserStore.create_user(sample_user)
        sess = {
            "token": "tok-delete-me",
            "email": sample_user["email"],
            "user_id": sample_user["user_id"],
            "role": "submitter",
            "expires_at": time.time() + 3600,
        }
        UserStore.create_session(sess)
        UserStore.delete_session("tok-delete-me")
        assert UserStore.get_session("tok-delete-me") is None

    def test_delete_user_sessions(self, sample_user):
        UserStore.create_user(sample_user)
        for i in range(3):
            UserStore.create_session({
                "token": f"tok-multi-{i}",
                "email": sample_user["email"],
                "user_id": sample_user["user_id"],
                "role": "submitter",
                "expires_at": time.time() + 3600,
            })
        UserStore.delete_user_sessions(sample_user["email"])
        for i in range(3):
            assert UserStore.get_session(f"tok-multi-{i}") is None

    def test_cleanup_expired_sessions(self, sample_user):
        UserStore.create_user(sample_user)
        # Create 2 expired + 1 valid
        for i in range(2):
            UserStore.create_session({
                "token": f"tok-exp-{i}",
                "email": sample_user["email"],
                "user_id": sample_user["user_id"],
                "role": "submitter",
                "expires_at": time.time() - 100,
            })
        UserStore.create_session({
            "token": "tok-valid",
            "email": sample_user["email"],
            "user_id": sample_user["user_id"],
            "role": "submitter",
            "expires_at": time.time() + 3600,
        })
        cleaned = UserStore.cleanup_expired_sessions()
        assert cleaned == 2
        assert UserStore.get_session("tok-valid") is not None


# ── UserStore: API Keys ───────────────────────────────────────────────


class TestUserStoreAPIKeys:
    """Verify API key storage."""

    def test_create_and_get_key(self, sample_user):
        UserStore.create_user(sample_user)
        key_val = "xc_test_" + uuid.uuid4().hex
        UserStore.create_api_key({
            "key": key_val,
            "name": "test-key",
            "email": sample_user["email"],
            "user_id": sample_user["user_id"],
        })
        result = UserStore.get_api_key(key_val)
        assert result is not None
        assert result["name"] == "test-key"
        assert result["email"] == sample_user["email"]

    def test_list_keys(self, sample_user):
        UserStore.create_user(sample_user)
        for i in range(3):
            UserStore.create_api_key({
                "key": f"xc_list_{i}_" + uuid.uuid4().hex,
                "name": f"key-{i}",
                "email": sample_user["email"],
                "user_id": sample_user["user_id"],
            })
        keys = UserStore.list_api_keys(sample_user["email"])
        assert len(keys) == 3

    def test_delete_key_by_preview(self, sample_user):
        UserStore.create_user(sample_user)
        key_val = "xc_preview_test_abcdefghij"
        UserStore.create_api_key({
            "key": key_val,
            "name": "preview-key",
            "email": sample_user["email"],
            "user_id": sample_user["user_id"],
        })
        preview = key_val[:12] + "..." + key_val[-4:]
        deleted = UserStore.delete_api_key_by_preview(sample_user["email"], preview)
        assert deleted is True
        assert UserStore.get_api_key(key_val) is None

    def test_get_nonexistent_key_returns_none(self):
        assert UserStore.get_api_key("xc_no_such_key") is None


# ── UserStore: Teams ──────────────────────────────────────────────────


class TestUserStoreTeams:
    """Verify team/org management operations."""

    def test_create_team(self, sample_user):
        UserStore.create_user(sample_user)
        team_id = f"team-{uuid.uuid4().hex[:8]}"
        UserStore.create_team({
            "team_id": team_id,
            "name": "Test Team",
            "owner_email": sample_user["email"],
            "plan": "free",
            "max_members": 5,
        })
        team = UserStore.get_team(team_id)
        assert team is not None
        assert team["name"] == "Test Team"
        assert team["plan"] == "free"

    def test_owner_auto_added_as_admin(self, sample_user):
        UserStore.create_user(sample_user)
        team_id = f"team-{uuid.uuid4().hex[:8]}"
        UserStore.create_team({
            "team_id": team_id,
            "name": "Auto Admin Team",
            "owner_email": sample_user["email"],
        })
        members = UserStore.list_team_members(team_id)
        assert len(members) == 1
        assert members[0]["email"] == sample_user["email"]
        assert members[0]["role"] == "admin"

    def test_add_team_member(self, sample_user, sample_user_2):
        UserStore.create_user(sample_user)
        UserStore.create_user(sample_user_2)
        team_id = f"team-{uuid.uuid4().hex[:8]}"
        UserStore.create_team({
            "team_id": team_id,
            "name": "Addable Team",
            "owner_email": sample_user["email"],
            "max_members": 5,
        })
        result = UserStore.add_team_member(team_id, sample_user_2["email"], "member")
        assert result is True
        members = UserStore.list_team_members(team_id)
        assert len(members) == 2
        emails = [m["email"] for m in members]
        assert sample_user_2["email"] in emails

    def test_capacity_limit_enforced(self, sample_user, sample_user_2):
        UserStore.create_user(sample_user)
        UserStore.create_user(sample_user_2)
        team_id = f"team-{uuid.uuid4().hex[:8]}"
        UserStore.create_team({
            "team_id": team_id,
            "name": "Tiny Team",
            "owner_email": sample_user["email"],
            "max_members": 1,
        })
        result = UserStore.add_team_member(team_id, sample_user_2["email"])
        assert result is False  # capacity = 1, owner already takes the slot
        members = UserStore.list_team_members(team_id)
        assert len(members) == 1

    def test_remove_team_member(self, sample_user, sample_user_2):
        UserStore.create_user(sample_user)
        UserStore.create_user(sample_user_2)
        team_id = f"team-{uuid.uuid4().hex[:8]}"
        UserStore.create_team({
            "team_id": team_id,
            "name": "Remove Test",
            "owner_email": sample_user["email"],
            "max_members": 5,
        })
        UserStore.add_team_member(team_id, sample_user_2["email"])
        UserStore.remove_team_member(team_id, sample_user_2["email"])
        members = UserStore.list_team_members(team_id)
        assert len(members) == 1
        # Verify user's team_id is cleared
        user2 = UserStore.get_user(sample_user_2["email"])
        assert user2["team_id"] is None

    def test_delete_team(self, sample_user, sample_user_2):
        UserStore.create_user(sample_user)
        UserStore.create_user(sample_user_2)
        team_id = f"team-{uuid.uuid4().hex[:8]}"
        UserStore.create_team({
            "team_id": team_id,
            "name": "Deletable Team",
            "owner_email": sample_user["email"],
            "max_members": 5,
        })
        UserStore.add_team_member(team_id, sample_user_2["email"])
        UserStore.delete_team(team_id)
        assert UserStore.get_team(team_id) is None
        assert UserStore.list_team_members(team_id) == []
        # Users' team_id should be cleared
        u1 = UserStore.get_user(sample_user["email"])
        assert u1["team_id"] is None

    def test_get_user_teams(self, sample_user):
        UserStore.create_user(sample_user)
        for i in range(3):
            UserStore.create_team({
                "team_id": f"team-multi-{i}",
                "name": f"Team {i}",
                "owner_email": sample_user["email"],
            })
        teams = UserStore.get_user_teams(sample_user["email"])
        assert len(teams) == 3

    def test_get_nonexistent_team_returns_none(self):
        assert UserStore.get_team("team-nope") is None


# ── API Integration: Team Endpoints ───────────────────────────────────


class TestTeamEndpoints:
    """Verify team management API endpoints via FastAPI TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)

    @pytest.fixture
    def auth_token(self, client):
        """Register + login, return Bearer token."""
        client.post("/api/auth/register", json={
            "email": "team-leader@xcelsior.ca",
            "password": "StrongPass123!",
            "name": "Team Leader",
        })
        r = client.post("/api/auth/login", json={
            "email": "team-leader@xcelsior.ca",
            "password": "StrongPass123!",
        })
        return r.json()["access_token"]

    def test_create_team_endpoint(self, client, auth_token):
        r = client.post("/api/teams", json={"name": "API Team", "plan": "free"},
                        headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "API Team"
        assert "team_id" in d

    def test_get_my_teams(self, client, auth_token):
        client.post("/api/teams", json={"name": "My Team"},
                     headers={"Authorization": f"Bearer {auth_token}"})
        r = client.get("/api/teams/me",
                       headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200
        assert len(r.json()["teams"]) >= 1

    def test_get_team_detail(self, client, auth_token):
        cr = client.post("/api/teams", json={"name": "Detail Team"},
                         headers={"Authorization": f"Bearer {auth_token}"})
        team_id = cr.json()["team_id"]
        r = client.get(f"/api/teams/{team_id}",
                       headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200
        d = r.json()
        assert d["team"]["name"] == "Detail Team"
        assert "members" in d

    def test_add_member_endpoint(self, client, auth_token):
        cr = client.post("/api/teams", json={"name": "Add Member Team"},
                         headers={"Authorization": f"Bearer {auth_token}"})
        team_id = cr.json()["team_id"]
        # Register another user
        client.post("/api/auth/register", json={
            "email": "member@xcelsior.ca",
            "password": "MemberPass123!",
            "name": "Team Member",
        })
        r = client.post(f"/api/teams/{team_id}/members",
                        json={"email": "member@xcelsior.ca", "role": "member"},
                        headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200

    def test_remove_member_endpoint(self, client, auth_token):
        cr = client.post("/api/teams", json={"name": "Remove Member Team"},
                         headers={"Authorization": f"Bearer {auth_token}"})
        team_id = cr.json()["team_id"]
        client.post("/api/auth/register", json={
            "email": "removable@xcelsior.ca",
            "password": "RemovePass123!",
            "name": "Removable",
        })
        client.post(f"/api/teams/{team_id}/members",
                     json={"email": "removable@xcelsior.ca", "role": "member"},
                     headers={"Authorization": f"Bearer {auth_token}"})
        r = client.delete(f"/api/teams/{team_id}/members/removable@xcelsior.ca",
                          headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200

    def test_delete_team_endpoint(self, client, auth_token):
        cr = client.post("/api/teams", json={"name": "Deletable Team"},
                         headers={"Authorization": f"Bearer {auth_token}"})
        team_id = cr.json()["team_id"]
        r = client.delete(f"/api/teams/{team_id}",
                          headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200

    def test_create_team_unauthenticated(self, client):
        r = client.post("/api/teams", json={"name": "No Auth"})
        assert r.status_code == 401

    def test_nonmember_cannot_view_team(self, client, auth_token):
        cr = client.post("/api/teams", json={"name": "Private Team"},
                         headers={"Authorization": f"Bearer {auth_token}"})
        team_id = cr.json()["team_id"]
        # Register another user who is NOT a member
        client.post("/api/auth/register", json={
            "email": "outsider@xcelsior.ca",
            "password": "OutsiderPass123!",
            "name": "Outsider",
        })
        lr = client.post("/api/auth/login", json={
            "email": "outsider@xcelsior.ca",
            "password": "OutsiderPass123!",
        })
        outsider_token = lr.json()["access_token"]
        r = client.get(f"/api/teams/{team_id}",
                       headers={"Authorization": f"Bearer {outsider_token}"})
        assert r.status_code == 403


# ── API Integration: Persistent Auth ──────────────────────────────────


class TestPersistentAuthEndpoints:
    """Verify auth endpoints work with persistent storage."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)

    def test_register_creates_persistent_user(self, client):
        r = client.post("/api/auth/register", json={
            "email": "persist@xcelsior.ca",
            "password": "PersistPass123!",
            "name": "Persist User",
        })
        assert r.status_code == 200
        # Verify user exists in persistent store
        user = UserStore.get_user("persist@xcelsior.ca")
        assert user is not None
        assert user["name"] == "Persist User"

    def test_login_returns_session(self, client):
        client.post("/api/auth/register", json={
            "email": "sesstest@xcelsior.ca",
            "password": "SessTest123!",
            "name": "Sess Test",
        })
        r = client.post("/api/auth/login", json={
            "email": "sesstest@xcelsior.ca",
            "password": "SessTest123!",
        })
        assert r.status_code == 200
        d = r.json()
        assert "access_token" in d
        # Session should be in persistent store
        sess = UserStore.get_session(d["access_token"])
        assert sess is not None

    def test_me_endpoint_uses_persistent_auth(self, client):
        client.post("/api/auth/register", json={
            "email": "metest@xcelsior.ca",
            "password": "MeTest123!",
            "name": "Me Test",
        })
        lr = client.post("/api/auth/login", json={
            "email": "metest@xcelsior.ca",
            "password": "MeTest123!",
        })
        token = lr.json()["access_token"]
        r = client.get("/api/auth/me",
                       headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["user"]["email"] == "metest@xcelsior.ca"

    def test_api_key_persistent_storage(self, client):
        client.post("/api/auth/register", json={
            "email": "apikey@xcelsior.ca",
            "password": "ApiKey123!",
            "name": "API Key User",
        })
        lr = client.post("/api/auth/login", json={
            "email": "apikey@xcelsior.ca",
            "password": "ApiKey123!",
        })
        token = lr.json()["access_token"]
        r = client.post("/api/keys/generate?name=persist-key",
                        headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        key_val = r.json()["key"]
        # Key should be in persistent store
        stored = UserStore.get_api_key(key_val)
        assert stored is not None
        assert stored["name"] == "persist-key"


# ── SLA Enforcement ───────────────────────────────────────────────────


class TestSLAEnforcement:
    """Verify SLA enforcement admin endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)

    def test_sla_enforce_endpoint_exists(self, client):
        import datetime
        month = datetime.datetime.now().strftime("%Y-%m")
        r = client.post("/api/sla/enforce", json={
            "host_id": "gpu-test-01",
            "month": month,
            "tier": "community",
            "monthly_spend_cad": 100.0,
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True


# ── Dashboard Version ────────────────────────────────────────────────


class TestDashboardVersion:
    """Verify dashboard was bumped to v2.8.0."""

    def test_version_string(self):
        with open("templates/dashboard.html") as f:
            content = f.read()
        assert "v2.8.0" in content
        assert "v2.5.0" not in content
