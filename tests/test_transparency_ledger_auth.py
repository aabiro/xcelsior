"""The legal-request ledger must not accept anonymous writes.

`POST /api/transparency/legal-request` and `/api/transparency/legal-response`
record subpoenas, warrants and MLAT requests, and whether the platform complied
or challenged them. That ledger is the evidence behind the published
transparency report.

Both handlers authorize like this:

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "transparency:write")

`_get_current_user` returns `None` for an unauthenticated caller, so the `if`
is skipped and the insert proceeds. The scope check only ever runs for someone
who already authenticated — exactly backwards. An anonymous request writes to
the ledger; a legitimate one is the only kind that can be refused.

The damage is not data theft, it is falsification: anyone reachable by the API
can insert warrants that were never served, or mark a real request "complied"
or "challenged". A transparency report is worth precisely what its underlying
records are worth.

`if user:` is a plausible-looking way to write "only check scopes when we have
someone to check", which is why this needs a test rather than a code comment.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


@pytest.fixture
def auth_enforced():
    """Turn the dev escape hatch off for the duration of a test.

    `routes/_deps.AUTH_REQUIRED` is False whenever `XCELSIOR_ENV` is dev or
    test, and `_require_auth` then hands an unauthenticated caller a synthetic
    *admin* principal. Under that default every assertion below passes for the
    wrong reason — the endpoint looks open because the whole suite runs with
    authentication disabled, which says nothing about production.

    So these tests set the condition they claim to test. Without this fixture
    they would have gone green the moment the handler changed, while proving
    nothing.
    """
    import routes._deps as deps

    original = deps.AUTH_REQUIRED
    deps.AUTH_REQUIRED = True
    try:
        yield
    finally:
        deps.AUTH_REQUIRED = original

_LEGAL_REQUEST = {
    "request_type": "subpoena",
    "requesting_country": "CA",
    "authority": "Test Authority",
    "scope": "test",
    "notes": "auth guard test",
}


@pytest.mark.enforced_auth
def test_anonymous_cannot_record_a_legal_request(auth_enforced):
    """No credential at all must not be the way past the scope check."""
    r = client.post("/api/transparency/legal-request", json=_LEGAL_REQUEST)
    assert r.status_code in (401, 403), (
        "an unauthenticated caller wrote to the legal-request ledger: "
        f"{r.status_code} {r.text[:200]}"
    )


@pytest.mark.enforced_auth
def test_anonymous_cannot_record_a_legal_response(auth_enforced):
    """The response side decides whether a request reads as complied or challenged."""
    r = client.post(
        "/api/transparency/legal-request/does-not-exist/respond",
        params={"complied": True, "challenged": False, "notes": "auth guard test"},
    )
    assert r.status_code in (401, 403), (
        "an unauthenticated caller wrote to the legal-response ledger: "
        f"{r.status_code} {r.text[:200]}"
    )


@pytest.mark.enforced_auth
def test_a_bogus_bearer_token_is_refused(auth_enforced):
    """Presenting a credential must not be worse than presenting none."""
    r = client.post(
        "/api/transparency/legal-request",
        json=_LEGAL_REQUEST,
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert r.status_code in (401, 403), r.text[:200]


def test_the_handlers_do_not_make_the_scope_check_conditional():
    """Pin the shape, not just the status code.

    A future edit could restore `if user:` while some *other* guard happens to
    return 401 for the cases above, and the behavioural tests would still pass
    while the ledger was open again to any authenticated principal without the
    scope.
    """
    import inspect

    import routes.transparency as mod

    for name in ("api_record_legal_request", "api_respond_legal_request"):
        source = inspect.getsource(getattr(mod, name))
        # Comments describe the defect on purpose; only executable lines count.
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        assert "if user:" not in code, (
            f"{name} guards its scope check behind `if user:`, so an "
            "unauthenticated caller skips it entirely"
        )
        assert "_require_scope" in code, f"{name} no longer checks a scope at all"
