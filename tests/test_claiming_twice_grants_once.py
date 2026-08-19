"""`claim_reputation_milestones` publishes `idempotency: "keyed"`. Prove it.

The surface tells clients *"safe to repeat — already-claimed milestones are
skipped rather than granted twice"*, and `tool-surface.json` carries the
machine-readable form. A client that believes it and retries a call that appeared
to time out is doing exactly what the contract invites.

That promise was wrong across seven tools once already. The Unreleased changelog
leads with it: `idempotency` defaulted to `"keyed"` for everything not read-only,
so 25 tools advertised that repeating was free when only four sent a key, and
`run_training_job` would have launched **a second billed instance**. The default
was removed and every write now has to declare. Declaring is not proving, and
this one is declared on the strength of a claim in a docstring.

Points are not money, so a double grant costs nobody a charge. It inflates a
score — and score sets tier, tier sets the commission the platform takes and the
premium the provider may charge. A reputation that can be inflated by retrying is
a pricing bug wearing a gamification costume.

## It is defended twice, which matters if you try to verify this

Removing the route's `if m["id"] in claimed: continue` leaves the test green.
So does removing the engine's `if milestone_id in self.claimed_milestones(...)`
guard in `grant_milestone`. Only removing **both** turns it red — at which point
a second claim re-grants all six milestones, 275 points.

That is defence in depth working, not a redundant line to tidy away. It is
recorded because verifying this file with a single injection produces a
convincing false conclusion that the test is inert, which is how a real guard
gets deleted for being untested.

## What is asserted

A second claim with no new activity grants nothing and leaves the score
unmoved. Not that the engine's `grant_milestone` is idempotent in isolation —
that is `tests/test_reputation.py`'s business — but that the **route** is, which
is what the tool calls and what the contract describes.
"""

from __future__ import annotations

import json
import pathlib
import uuid

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _register_and_login(email: str) -> dict[str, str]:
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Claimant"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.fixture
def claimant():
    """Function-scoped. Each test needs an account that has claimed nothing
    yet — a shared one carries the previous test's grants and turns "granted
    nothing the second time" into "had nothing to grant at all"."""
    return _register_and_login(f"claim-{uuid.uuid4().hex[:10]}@xcelsior.ca")


@pytest.fixture
def all_milestones_earned(monkeypatch):
    """Report every milestone as met, so the route has something to grant.

    The signals are the *input* to the claim, not the claim itself: the route
    still decides what to grant, records it, and skips what it already holds —
    which is the logic under test. Supplying them is what makes the property
    testable at all, because in this environment a fresh account earns nothing
    (registration does not persist a name here, so not even `complete_profile`
    is reachable) and every assertion below would pass against a route that
    grants nobody anything. That is precisely the guard-that-cannot-fail shape
    this file is meant to avoid, and the first version of it had exactly that
    bug — caught by its own calibration.
    """
    import routes.reputation as rep

    monkeypatch.setattr(
        rep, "_journey_signals", lambda user: {m["id"]: int(m["target"]) for m in rep._MILESTONES}
    )
    return rep


def test_the_first_claim_grants_and_the_second_does_not(claimant, all_milestones_earned):
    """The property, and its calibration, in one assertion pair."""
    first = client.post("/api/reputation/me/claim", headers=claimant)
    assert first.status_code == 200, first.text
    granted = first.json()["newly_granted"]
    assert granted, (
        "the first claim granted nothing, so the repeat assertion below proves "
        "nothing — the route never had anything to skip"
    )

    second = client.post("/api/reputation/me/claim", headers=claimant)
    assert second.status_code == 200, second.text
    assert second.json()["newly_granted"] == [], (
        f"claiming twice granted {second.json()['newly_granted']} again. The "
        "tool publishes idempotency 'keyed' and its description says repeats "
        "are skipped, so a client retrying a timed-out call inflates its own "
        "reputation — and tier sets both the platform commission and the "
        "pricing premium"
    )


def test_the_score_does_not_move_on_the_second_claim(claimant, all_milestones_earned):
    """`newly_granted == []` could be true while points were added anyway."""
    client.post("/api/reputation/me/claim", headers=claimant)
    before = client.get("/api/reputation/me", headers=claimant).json().get("score")
    client.post("/api/reputation/me/claim", headers=claimant)
    after = client.get("/api/reputation/me", headers=claimant).json().get("score")
    assert before == after, f"score moved on a repeat claim: {before} -> {after}"


def test_nothing_is_granted_when_nothing_is_earned():
    """The other direction: a claim cannot invent progress.

    A **fresh** account, deliberately — not the module-scoped `claimant`, which
    by this point has already claimed everything the patched fixture reported as
    earned. Against that account this would pass whether or not the earned-check
    still existed, which is a false green of the exact kind this file is about.

    If this ever starts granting, the check has stopped being enforced and the
    tool is a free-points button.
    """
    headers = _register_and_login(f"unearned-{uuid.uuid4().hex[:10]}@xcelsior.ca")
    response = client.post("/api/reputation/me/claim", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["newly_granted"] == []


def test_the_surface_still_publishes_the_promise_this_proves():
    """If the contract is changed to `none`, this file's subject is gone and it
    should be deleted deliberately rather than left asserting a property nobody
    relies on."""
    surface = json.loads((ROOT / "mcp" / "tool-surface.json").read_text(encoding="utf-8"))
    tool = next(t for t in surface["tools"] if t["name"] == "claim_reputation_milestones")
    assert tool["idempotency"] == "keyed", (
        f"the tool now publishes idempotency {tool['idempotency']!r}; this file "
        "exists to back the 'keyed' claim"
    )
