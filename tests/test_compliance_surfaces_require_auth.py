"""The legal-request ledger is not writable by an anonymous caller.

All three transparency routes carried this shape:

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "transparency:write")
    ...write...

The scope check runs *only for callers who already authenticated*. An anonymous
caller — `user is None` — skips it and falls straight through to the write.
There is no `_require_auth` in front of it, so nothing else refuses them either.
The check that looks like the guard is the branch that unauthenticated callers
never enter.

What that reaches:

* `POST /api/transparency/legal-request` inserts a row into `legal_requests` and
  appends a `transparency.legal_request` event to the hash-chained audit store,
  attributed to `actor="admin"`.
* `POST /api/transparency/legal-request/{id}/respond` marks an existing request
  `responded`, and sets `complied` / `challenged` — the fields a transparency
  report is computed from.
* `GET /api/transparency/report` discloses every legal request and data
  disclosure in the window.

So an unauthenticated caller could fabricate subpoena records, mark a real one
as complied-with, or read the lot. This is a compliance artifact — CLOUD Act
diligence — and its integrity is the whole point of it existing.

The fix is `_require_auth` *before* the scope check, unconditionally. The scope
check then narrows further, but is no longer the only thing standing there.

**Anonymous is asserted separately from under-scoped**, because they fail for
different reasons and a single test that accepted 401-or-403 could not tell a
working guard from a broken one — the `assert r.status_code in (401, 403, 200)`
defect in a new costume.

**Auth has to be forced on, and that is why this defect survived.**
`tests/conftest.py`'s autouse `_pin_test_auth_env` sets
`routes._deps.AUTH_REQUIRED = False`, and `_require_auth` then hands an
anonymous caller `{"is_admin": True}` rather than raising. So under the default
suite configuration *no test can observe an endpoint refusing an anonymous
caller* — every route looks authenticated because every caller is silently
promoted. A conditional-scope guard is invisible in that environment.

`auth_enforced` below declares `_pin_test_auth_env` as an explicit dependency so
it is ordered *after* it. Fixture ordering is not a security boundary, so each
test also re-asserts the flag in its own body: an autouse fixture that reverted
the pin would otherwise leave these tests passing against auth-off, which is
exactly how a previous run was reported as "verified under enforced auth" when
it was not.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)

#: (method, path, json-body) for every route that reads or writes the ledger.
#: Listed individually so removing one is visible in review rather than showing
#: up as a count change.
TRANSPARENCY_ROUTES = [
    (
        "POST",
        "/api/transparency/legal-request",
        {
            "request_type": "subpoena",
            "requesting_country": "CA",
            "authority": "anonymous-probe",
            "scope": "probe",
            "notes": "written by tests/test_compliance_surfaces_require_auth.py",
        },
    ),
    ("POST", "/api/transparency/legal-request/probe-id/respond", None),
    ("GET", "/api/transparency/report", None),
]

#: `routes/privacy.py` carried the same shape at eight sites. Seven are listed
#: here; the eighth — `GET /api/privacy/retention-policies` — is deliberately
#: public and appears in `PUBLIC_BY_DESIGN` below.
#:
#: The line between them: this list is per-entity data or a mutation. That one
#: renders the `RETENTION_POLICIES` constant, which is the platform's published
#: policy and discloses nothing about anybody.
PRIVACY_ROUTES = [
    ("GET", "/api/privacy/retention-summary", None),
    ("POST", "/api/privacy/purge-expired", None),
    ("POST", "/api/privacy/config", {"org_id": "probe", "level": "strict"}),
    ("GET", "/api/privacy/config/probe-org", None),
    (
        "POST",
        "/api/privacy/consent",
        {"entity_id": "probe", "consent_type": "telemetry"},
    ),
    ("DELETE", "/api/privacy/consent/probe/telemetry", None),
    ("GET", "/api/privacy/consent/probe", None),
]

#: Routes where `if user: _require_scope(...)` is correct rather than a defect:
#: an anonymous caller gets the public view, an authenticated one is additionally
#: held to their scope. Named explicitly so the distinction is a decision on the
#: record, not an omission — and so the guard below cannot quietly grow.
PUBLIC_BY_DESIGN = [
    ("GET", "/api/privacy/retention-policies", None),  # a static constant
    ("GET", "/marketplace", None),  # public catalogue
]


def _call(method: str, path: str, body: dict | None, headers: dict | None = None):
    kwargs: dict = {"headers": headers or {}}
    if body is not None:
        kwargs["json"] = body
    return client.request(method, path, **kwargs)


@pytest.fixture
def auth_enforced(_pin_test_auth_env, monkeypatch):
    """Turn authentication back on for this test.

    `_pin_test_auth_env` is named as a parameter deliberately: it is autouse and
    sets `AUTH_REQUIRED = False`, and declaring it as a dependency is what
    guarantees this fixture runs *after* it rather than being silently undone.
    """
    import routes._deps as deps

    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)
    return deps


def _assert_enforced(deps) -> None:
    """Re-checked in the test body; fixture ordering is not a security boundary."""
    assert deps.AUTH_REQUIRED is True, (
        "AUTH_REQUIRED was reverted to False after auth_enforced ran — this "
        "test would have passed against an unauthenticated configuration"
    )


@pytest.mark.parametrize("method,path,body", TRANSPARENCY_ROUTES + PRIVACY_ROUTES)
def test_an_anonymous_caller_is_refused(method, path, body, auth_enforced):
    """No credential at all must be refused with 401, before any write."""
    _assert_enforced(auth_enforced)
    r = _call(method, path, body)
    assert r.status_code == 401, (
        f"{method} {path} returned {r.status_code} to a caller with no "
        f"credential. The legal-request ledger is a compliance artifact; an "
        f"anonymous caller must not read or write it. Body: {r.text[:200]}"
    )


def test_the_anonymous_write_does_not_reach_the_ledger(auth_enforced):
    """The consequence, asserted at the data rather than at the status code.

    A route could refuse *after* writing — the status code alone would not
    distinguish that from refusing before. This counts the rows.
    """
    _assert_enforced(auth_enforced)
    from routes.transparency import _transparency_db

    marker = f"anon-probe-{uuid.uuid4().hex[:10]}"

    with _transparency_db() as conn:
        before = conn.execute(
            "SELECT count(*) AS c FROM legal_requests WHERE authority = %s", (marker,)
        ).fetchone()["c"]

    _call(
        "POST",
        "/api/transparency/legal-request",
        {
            "request_type": "subpoena",
            "requesting_country": "CA",
            "authority": marker,
            "scope": "probe",
            "notes": "must not be recorded",
        },
    )

    with _transparency_db() as conn:
        after = conn.execute(
            "SELECT count(*) AS c FROM legal_requests WHERE authority = %s", (marker,)
        ).fetchone()["c"]

    assert after == before == 0, (
        f"an unauthenticated POST inserted {after - before} row(s) into "
        "legal_requests — the refusal, if any, happened after the write"
    )


@pytest.mark.parametrize("method,path,body", PUBLIC_BY_DESIGN)
def test_a_deliberately_public_route_stays_reachable(method, path, body, auth_enforced):
    """The other half of the rule, and the one that keeps it honest.

    Without this, "add `_require_auth` everywhere" would satisfy every assertion
    above while breaking the public catalogue and the published retention
    policy. The guard has to distinguish the defect from the design, so both
    directions are asserted.
    """
    _assert_enforced(auth_enforced)
    r = _call(method, path, body)
    assert r.status_code != 401, (
        f"{method} {path} is public by design and now refuses anonymous "
        f"callers: {r.text[:200]}"
    )


def test_an_authenticated_caller_is_not_refused_as_anonymous(auth_enforced):
    """The calibration control: a real credential must not read as 401.

    Without it, a route that refused *everything* would satisfy every assertion
    above and look like a working guard.
    """
    _assert_enforced(auth_enforced)
    email = f"transp-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Transp"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    r = client.get("/api/transparency/report", headers=headers)
    assert r.status_code != 401, (
        "an authenticated user was refused as if unauthenticated; the two "
        "failures must stay distinguishable"
    )
