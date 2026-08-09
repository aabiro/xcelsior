"""Gate P0 clause 1, against a live server with a real token.

The clause: *"Every access and billing endpoint named above refuses a token
missing its new scope, asserted with a real token against a live server."*

Everything else asserting these refusals runs against a `TestClient`. §1 of the
plan says why that is not enough — *"a mock is what passed while production did
not"* — and this file is the only thing that drives the deployed code with a
credential the server actually minted.

## What makes the credential the right one

`_require_scope` is a **no-op for interactive sessions**, by design: a browser
session carries OIDC identity scopes that say nothing about API authority. So a
refusal test driven by a logged-in user's token would pass while proving nothing.
This mints a `client_credentials` token holding exactly `instances:read` — the
credential class Quick Connect issues, and the one the scopes are for.

## Why the positive control is not optional

A server that answers 403 to everything passes a refusal-only test perfectly.
`sibling` is a path the same token *may* reach, and it must return a non-403 for
any refusal below to mean anything. `tests/live/test_scope_refusals_live.py`
learned this the hard way on 2026-08-04, when Cloudflare's browser-integrity
check returned 403 before the origin was ever reached and the suite reported a
fix that had not happened.

## Running it

    XCELSIOR_LIVE_BASE_URL=http://127.0.0.1:9600 \
    XCELSIOR_LIVE_USER_TOKEN=<a non-admin session token> \
    pytest tests/live/test_named_scopes_refuse_live.py

The session token is used **only** to register the narrow OAuth client; every
assertion runs against the machine token that client mints.
"""

from __future__ import annotations

import os
import uuid

import pytest

httpx = pytest.importorskip("httpx", reason="live gates install httpx explicitly")

BASE = (os.environ.get("XCELSIOR_LIVE_BASE_URL") or os.environ.get("XCELSIOR_STAGING_URL") or "").rstrip("/")
USER_TOKEN = os.environ.get("XCELSIOR_LIVE_USER_TOKEN", "")

#: The scope the throwaway client is given — deliberately unrelated to billing
#: and ssh, so every refusal below is about the *missing* scope rather than
#: about holding none at all.
HELD_SCOPE = "instances:read"

#: A path `HELD_SCOPE` genuinely reaches. Its job is to fail loudly if the
#: server has started refusing everything.
SIBLING = "/instances"

#: The endpoints Gate P0 names by hand.
REFUSED = [
    ("POST", "/api/billing/setup-intent"),
    ("POST", "/api/billing/portal-session"),
    ("GET", "/api/ssh/keys"),
    ("POST", "/api/ssh/keys"),
]

_NEEDS_LIVE = (
    "live gate: set XCELSIOR_LIVE_BASE_URL and XCELSIOR_LIVE_USER_TOKEN "
    "(a NON-admin session token)"
)

HEADERS = {"User-Agent": "xcelsior-live-gates/1.0 (+named scope refusal)", "Accept": "application/json"}

#: Markers identifying a 403 as an edge's rather than the application's.
_EDGE_MARKERS = ("error code:", "cloudflare", "attention required", "<!doctype html")


def _from_the_origin(response) -> None:
    """A 403 only means "scope refused" if the application produced it."""
    body = (response.text or "").lower()
    for marker in _EDGE_MARKERS:
        assert marker not in body, (
            f"the 403 came from an edge, not the origin (matched {marker!r}). "
            "This gate would otherwise report enforcement that was never reached."
        )


@pytest.fixture(scope="module")
def narrow_token() -> str:
    """A client_credentials token holding only `HELD_SCOPE`."""
    if not BASE or not USER_TOKEN:
        pytest.skip(_NEEDS_LIVE)

    with httpx.Client(base_url=BASE, timeout=30, headers=HEADERS) as client:
        created = client.post(
            "/api/oauth/clients",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={
                "client_name": f"live-gate-{uuid.uuid4().hex[:8]}",
                "client_type": "confidential",
                "redirect_uris": [],
                "grant_types": ["client_credentials"],
                "scopes": [HELD_SCOPE],
            },
        )
        assert created.status_code == 200, (
            f"could not register the throwaway client: {created.status_code} "
            f"{created.text[:200]}"
        )
        oauth_client = created.json()["client"]

        issued = client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": oauth_client["client_id"],
                "client_secret": oauth_client["client_secret"],
                "scope": HELD_SCOPE,
            },
        )
        assert issued.status_code == 200, f"token issuance failed: {issued.text[:200]}"
        token = issued.json()["access_token"]
        assert token, "the server issued an empty access token"
        return token


def test_the_token_can_reach_what_its_scope_allows(narrow_token):
    """The positive control, first, because everything else depends on it.

    If this is not a 200 the server is refusing broadly and every refusal below
    is meaningless.
    """
    with httpx.Client(base_url=BASE, timeout=30, headers=HEADERS) as client:
        response = client.get(SIBLING, headers={"Authorization": f"Bearer {narrow_token}"})
    assert response.status_code == 200, (
        f"a token holding {HELD_SCOPE} could not reach {SIBLING} "
        f"({response.status_code}). Without this, the refusals prove nothing."
    )


@pytest.mark.parametrize("method,path", REFUSED, ids=[f"{m} {p}" for m, p in REFUSED])
def test_the_named_endpoints_refuse_a_token_missing_their_scope(narrow_token, method, path):
    """Gate P0 clause 1, endpoint by endpoint."""
    with httpx.Client(base_url=BASE, timeout=30, headers=HEADERS) as client:
        response = client.request(
            method, path,
            headers={"Authorization": f"Bearer {narrow_token}", "Content-Type": "application/json"},
            json={},
        )

    assert response.status_code != 404, (
        f"{method} {path} does not exist on this deployment — the gate is "
        "asserting against a route that is not there, which passes for the "
        "wrong reason"
    )
    assert response.status_code == 403, (
        f"{method} {path} answered {response.status_code} to a credential "
        f"holding only {HELD_SCOPE}. Gate P0 requires a refusal."
    )
    _from_the_origin(response)


def test_a_refusal_says_it_is_about_scope(narrow_token):
    """The message matters: an agent has to be able to tell *why* it was refused.

    A bare 403 is indistinguishable from an ownership failure, and an agent that
    cannot tell them apart will retry the one that will never succeed.
    """
    with httpx.Client(base_url=BASE, timeout=30, headers=HEADERS) as client:
        response = client.post(
            "/api/billing/setup-intent",
            headers={"Authorization": f"Bearer {narrow_token}", "Content-Type": "application/json"},
            json={},
        )
    assert "scope" in response.text.lower(), (
        f"the refusal does not mention scope: {response.text[:200]}"
    )
