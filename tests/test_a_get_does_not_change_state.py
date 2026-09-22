"""A GET must be safe. `GET /api/teams/invite/{token}` was not.

If the invited address already had an account, fetching that URL added them to
the team and deleted the single-use token. Nobody had to be signed in, and the
address joined was the invite's rather than the caller's.

The practical failure is not exotic. Anything that fetches a URL triggered it:
a browser prefetch, a crawler, a Slack or iMessage link preview, an Outlook
Safe Links or Proofpoint scanner. An invitation emailed to someone behind a
corporate mail scanner was accepted and consumed before they opened it, and the
page they eventually reached said "Invitation Not Found" — so invitations
failed most reliably for the people most likely to have a scanner in front of
them.

It also made the frontend's consent flow decorative. `accept-invite/page.tsx`
fetches this on mount only to render the invitation, warns "this invitation is
for X, you're signed in as Y", and accepts via `POST .../accept` when the user
presses a button. That POST requires an interactive session and checks the
invite belongs to the caller. The join had already happened on page load.
"""

from __future__ import annotations

import ast
import os
import secrets
import time
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

import api as _api_mod
import routes._deps as _deps_mod
import routes.auth as _auth_mod

from api import app
from db import UserStore

client = TestClient(app)

#: Writes that must never happen during a GET.
MUTATORS = {"add_team_member", "delete_team_invite", "create_team_invite", "remove_team_member"}


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    monkeypatch.setattr(_api_mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(_deps_mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(_auth_mod, "_USE_PERSISTENT_AUTH", True)
    _deps_mod._AUTH_RATE_BUCKETS.clear()
    monkeypatch.setattr(_deps_mod, "_AUTH_RATE_LIMIT_REQUESTS", 5000)


def _register(email: str) -> dict:
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Invitee"},
    )
    assert r.status_code == 200, r.text[:300]
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def invite():
    """A team with a live invitation to an account that already exists."""
    leader = f"lead-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    leader_headers = _register(leader)
    team = client.post(
        "/api/teams", json={"name": f"Team {uuid.uuid4().hex[:6]}"}, headers=leader_headers
    )
    assert team.status_code == 200, team.text[:300]
    team_id = team.json()["team_id"]

    invitee = f"invitee-{uuid.uuid4().hex[:8]}@xcelsior.ca".lower()
    _register(invitee)

    token = secrets.token_urlsafe(32)
    UserStore.create_team_invite(
        {
            "token": token,
            "team_id": team_id,
            "email": invitee,
            "role": "member",
            "invited_by": leader,
            "created_at": time.time(),
            "expires_at": time.time() + 86400,
        }
    )
    return {"token": token, "team_id": team_id, "invitee": invitee, "leader": leader_headers}


def _members(team_id: str, headers: dict) -> set[str]:
    r = client.get(f"/api/teams/{team_id}", headers=headers)
    assert r.status_code == 200, r.text[:300]
    # `members` is a top-level key, not nested under `team`.
    return {str(m.get("email", "")).lower() for m in (r.json().get("members") or [])}


def test_previewing_an_invite_does_not_join_anyone(invite) -> None:
    """The bug, stated directly."""
    before = _members(invite["team_id"], invite["leader"])
    assert invite["invitee"] not in before

    r = client.get(f"/api/teams/invite/{invite['token']}")
    assert r.status_code == 200, r.text[:300]

    after = _members(invite["team_id"], invite["leader"])
    assert invite["invitee"] not in after, (
        "a GET added the invited user to the team — a link preview or mail "
        "scanner now joins people to teams"
    )


def test_previewing_does_not_consume_the_single_use_token(invite) -> None:
    """Otherwise the first fetch wins and the human gets "not found"."""
    first = client.get(f"/api/teams/invite/{invite['token']}")
    assert first.status_code == 200
    second = client.get(f"/api/teams/invite/{invite['token']}")
    assert second.status_code == 200, (
        "the first GET consumed the invitation; a scanner that follows links in "
        "email would burn it before the recipient clicks"
    )


def test_the_preview_still_describes_the_invitation(invite) -> None:
    """Read-only is not the same as useless — the page needs these fields."""
    body = client.get(f"/api/teams/invite/{invite['token']}").json()
    assert body["pending"] is True
    assert body["email"] == invite["invitee"]
    assert body["role"] == "member"
    assert body["team_name"]
    assert body["account_exists"] is True


def test_the_post_is_what_actually_joins(invite) -> None:
    """The half that was always correct: interactive session, identity checked."""
    invitee_headers = _register_existing(invite["invitee"])
    r = client.post(f"/api/teams/invite/{invite['token']}/accept", headers=invitee_headers)
    assert r.status_code == 200, r.text[:300]
    assert invite["invitee"] in _members(invite["team_id"], invite["leader"])


def _register_existing(email: str) -> dict:
    r = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    assert r.status_code == 200, r.text[:300]
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_no_get_handler_in_teams_calls_a_membership_mutator() -> None:
    """Structural, because the behavioural tests above only cover one route.

    A GET that writes is not a typo — it is a shape that looks reasonable while
    being written, which is why this checks every GET in the module rather than
    the one that had the bug.
    """
    import routes.teams as teams_mod

    tree = ast.parse(open(teams_mod.__file__, encoding="utf-8").read())
    offenders: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_get = any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "get"
            for d in node.decorator_list
        )
        if not is_get:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                name = getattr(sub.func, "attr", None) or getattr(sub.func, "id", None)
                if name in MUTATORS:
                    # Deleting an already-expired invite changes no outcome.
                    if name == "delete_team_invite":
                        continue
                    offenders.append(f"{node.name} (line {node.lineno}) calls {name}")

    assert not offenders, (
        "these GET handlers change membership state, so any prefetch, crawler or "
        "link scanner triggers the write:\n  " + "\n  ".join(offenders)
    )
