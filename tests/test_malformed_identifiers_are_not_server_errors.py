"""A bad id in a URL is the caller's mistake, not a server fault.

Most identifier columns in this schema are `uuid`. Most routes accept their
path parameter as a plain `str` and hand it to Postgres unexamined, so Postgres
is the first thing that ever looks at the value — and when it cannot parse one
it raises `InvalidTextRepresentation` from inside the handler. The caller got a
500.

Three routes were confirmed to do this before the fix
(`/api/admin/reconciler/findings/{id}/dismiss` and `/enforce`, and
`/api/v2/privacy/erase/{id}`), all reachable by anyone who mistypes a URL. A
5xx says the server is broken: it pages whoever is on call and spends the error
budget, for a typo. There are enough `uuid` columns behind enough `str` path
parameters that fixing the three and stopping would leave the class open, so
there is a floor in `api.py` as well as the three specific repairs.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

MALFORMED = ("does-not-exist", "123", "' OR 1=1 --", "null")


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_headers():
    email = f"malformed-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Malformed Probe"},
    )
    token = r.json().get("access_token")
    assert token, f"could not register a probe user: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def a_finding():
    """One real, unresolved finding, cleaned up afterwards."""
    from db import _get_pg_pool

    finding_id = str(uuid.uuid4())
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO reconciliation_findings
                (finding_id, resource_type, resource_id, finding_type, severity, summary)
            VALUES (%s, 'host', 'probe-host', 'orphaned_allocation', 'warning', 'probe finding')
            """,
            (finding_id,),
        )
    yield finding_id
    with pool.connection() as conn:
        conn.execute("DELETE FROM reconciliation_findings WHERE finding_id = %s", (finding_id,))


# ── the three routes that were confirmed to 500 ───────────────────────────


@pytest.mark.parametrize("bad", MALFORMED)
@pytest.mark.parametrize("action", ("dismiss", "enforce"))
def test_reconciler_routes_answer_404_for_an_unparseable_finding_id(action, bad) -> None:
    r = client.post(
        f"/api/admin/reconciler/findings/{bad}/{action}", headers=_admin_headers()
    )
    assert r.status_code < 500, (
        f"/{action} returned {r.status_code} for finding_id={bad!r} — a malformed "
        "id must not read as a server fault"
    )
    assert r.status_code == 404, (
        f"/{action} returned {r.status_code} for finding_id={bad!r}; the "
        "valid-but-absent case already answers 404 and there is nothing useful "
        "to tell the caller that distinguishes the two"
    )


@pytest.mark.parametrize("bad", MALFORMED)
def test_privacy_erasure_status_answers_404_for_an_unparseable_request_id(
    bad, user_headers
) -> None:
    """And 404 specifically, because this route's 404s are load-bearing.

    `get_deletion_status` returns the same "not found" for a request that does
    not exist and one that is not the caller's, on purpose — the comment in
    `privacy_deletion.py` says so. A 500 for a malformed id was a third,
    distinguishable answer that undid that.
    """
    r = client.get(f"/api/v2/privacy/erase/{bad}", headers=user_headers)
    assert r.status_code == 404, (
        f"returned {r.status_code} for request_id={bad!r}; missing, forbidden and "
        "malformed must all look alike from outside"
    )


# ── the floor under every other route ─────────────────────────────────────


def test_a_data_error_anywhere_becomes_a_400_not_a_500() -> None:
    """The handler in `api.py`, exercised through a route that never learned.

    Registered against `psycopg.DataError` rather than the one subclass seen so
    far: a numeric column given a word, or a timestamp given a sentence, is the
    same mistake by the same caller.
    """
    from fastapi import APIRouter
    import psycopg

    probe = APIRouter()

    @probe.get("/__test__/data-error")
    def _raise_data_error():
        raise psycopg.errors.InvalidTextRepresentation(
            'invalid input syntax for type uuid: "nope"'
        )

    app.include_router(probe)
    try:
        r = client.get("/__test__/data-error")
        assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text[:200]}"
        assert r.json()["error"]["code"] == "invalid_identifier"
    finally:
        app.router.routes = [
            route
            for route in app.router.routes
            if getattr(route, "path", None) != "/__test__/data-error"
        ]


def test_the_400_body_does_not_echo_the_database_error() -> None:
    """The psycopg message quotes the offending value and names the column.

    Someone who pastes a token into the wrong URL should not get it reflected
    back in the response body, where it lands in their browser history, any
    intermediary's access log, and a screenshot in a support ticket.
    """
    from fastapi import APIRouter
    import psycopg

    # Not shaped like a real credential — see the note in
    # tests/test_control_plane_v1_instance_routes.py. A secret scanner cannot
    # tell a fake key from a real one, and this file's point is that the
    # *value* must not be echoed, whatever it looks like.
    secret = "ECHO-CANARY-must-not-be-reflected"
    probe = APIRouter()

    @probe.get("/__test__/data-error-secret")
    def _raise_with_secret():
        raise psycopg.errors.InvalidTextRepresentation(
            f'invalid input syntax for type uuid: "{secret}"'
        )

    app.include_router(probe)
    try:
        r = client.get("/__test__/data-error-secret")
        assert secret not in r.text, f"the response echoed the offending value: {r.text[:300]}"
    finally:
        app.router.routes = [
            route
            for route in app.router.routes
            if getattr(route, "path", None) != "/__test__/data-error-secret"
        ]


# ── dismiss told the truth about what it did ──────────────────────────────


def test_dismissing_an_unknown_finding_is_not_reported_as_success() -> None:
    """The UPDATE matched nothing and the route said `{"ok": true}` anyway.

    An operator working through a queue of findings by id would have been told
    each mistyped one was handled.
    """
    r = client.post(
        f"/api/admin/reconciler/findings/{uuid.uuid4()}/dismiss", headers=_admin_headers()
    )
    assert r.status_code == 404, (
        f"dismissing a finding that does not exist returned {r.status_code}: {r.text[:200]}"
    )


def test_dismissing_a_real_finding_resolves_it_once(a_finding) -> None:
    """And the second attempt says so rather than repeating the success."""
    first = client.post(
        f"/api/admin/reconciler/findings/{a_finding}/dismiss", headers=_admin_headers()
    )
    assert first.status_code == 200, first.text[:200]
    body = first.json()
    assert body["ok"] is True
    assert body["resolved_at"], "a dismissal that resolved nothing reported no timestamp"

    second = client.post(
        f"/api/admin/reconciler/findings/{a_finding}/dismiss", headers=_admin_headers()
    )
    assert second.status_code == 400, (
        f"re-dismissing an already-resolved finding returned {second.status_code}; "
        "'you already did this' and 'done' are different answers"
    )
