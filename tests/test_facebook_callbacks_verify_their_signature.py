"""Facebook's callbacks must actually reject a forged `signed_request`.

Both callbacks carried their own copy of the verification, and in both copies
the `HTTPException` raised on a signature mismatch was thrown *inside* a `try`
whose `except Exception` sat two lines below it. `HTTPException` is an
`Exception`, so the refusal was swallowed by the handler's own error path,
logged as a parse failure, and the request went on to return
`{"status": "success"}`.

The verification was therefore inert in production, with the app secret
correctly configured. What makes it worth a file of its own is that someone had
already found and repaired the *other* defect in that block — an environment
check that only refused on a literal `"production"`, with a comment explaining
the staging case it missed — and that repair could never have taken effect. Two
bugs in the same ten lines, one hiding the fix for the other, and every
observable signal said the endpoint worked.

Only a request with a deliberately wrong signature can tell the working state
from the broken one. That is this file.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

import routes.auth as auth_mod
from api import app

client = TestClient(app)

SECRET = "facebook-app-secret"
PATHS = (
    "/api/auth/oauth/facebook/deauthorize",
    "/api/auth/oauth/facebook/delete-data",
)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _payload(user_id: str = "fb-user-123") -> str:
    return _b64(json.dumps({"user_id": user_id, "algorithm": "HMAC-SHA256"}).encode())


def _signed(secret: str, payload: str) -> str:
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return f"{_b64(sig)}.{payload}"


def _forged(payload: str) -> str:
    return f"{_b64(b'not-the-real-signature')}.{payload}"


@pytest.fixture
def production(monkeypatch):
    """A configured, non-relaxed deployment — where verification must bite."""
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.setitem(auth_mod._OAUTH_PROVIDERS["facebook"], "client_secret", SECRET)


@pytest.mark.parametrize("path", PATHS)
def test_a_forged_signature_is_refused(production, path) -> None:
    r = client.post(path, data={"signed_request": _forged(_payload())})
    assert r.status_code == 400, (
        f"{path} accepted a forged signed_request with {r.status_code}: {r.text[:200]}. "
        "Anyone who knows the URL can name any Facebook user in a deletion callback."
    )


@pytest.mark.parametrize("path", PATHS)
def test_a_valid_signature_is_accepted(production, path) -> None:
    """The other half: a refusal that refuses everything is not a fix."""
    r = client.post(path, data={"signed_request": _signed(SECRET, _payload())})
    assert r.status_code == 200, f"{path} rejected a correctly signed request: {r.text[:200]}"


@pytest.mark.parametrize("path", PATHS)
def test_a_missing_signed_request_is_refused(production, path) -> None:
    """No payload is not an anonymous success.

    The old handlers fell through to `fb_user_id = "unknown"` and returned
    `{"status": "success"}` — for the deletion callback, a confirmation code
    for a deletion nobody asked for and nothing performed.
    """
    r = client.post(path, data={})
    assert r.status_code == 400, f"{path} returned {r.status_code} for a request with no payload"


@pytest.mark.parametrize("path", PATHS)
def test_an_unconfigured_secret_fails_closed(monkeypatch, path) -> None:
    """With no secret there is nothing to verify against, so nothing may pass.

    Failing open here is the more tempting mistake: the callback still "works"
    in every test that does not forge a signature, and the deployment that
    forgot the secret is exactly the one that will not notice.
    """
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.setitem(auth_mod._OAUTH_PROVIDERS["facebook"], "client_secret", "")
    r = client.post(path, data={"signed_request": _signed(SECRET, _payload())})
    assert r.status_code == 400, (
        f"{path} accepted a signed_request with no configured secret ({r.status_code})"
    )


@pytest.mark.parametrize("path", PATHS)
def test_development_stays_permissive(monkeypatch, path) -> None:
    """Local development has no app secret and must still be able to call these."""
    monkeypatch.setenv("XCELSIOR_ENV", "dev")
    monkeypatch.setitem(auth_mod._OAUTH_PROVIDERS["facebook"], "client_secret", "")
    r = client.post(path, data={"signed_request": _forged(_payload())})
    assert r.status_code == 200, f"{path} broke local development: {r.text[:200]}"


def test_the_refusal_is_not_reachable_through_a_bare_except() -> None:
    """The shape of the original bug, asserted structurally.

    A `try` that wraps the signature check and catches bare `Exception` will
    swallow the `HTTPException` again, and every behavioural test above would
    still pass if the refusal were later moved back inside one. The verifier is
    small enough to require that it contains no such handler at all.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(auth_mod._facebook_signed_request))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught = node.type
        names = []
        if isinstance(caught, ast.Name):
            names = [caught.id]
        elif isinstance(caught, ast.Tuple):
            names = [e.id for e in caught.elts if isinstance(e, ast.Name)]
        elif caught is None:
            names = ["<bare except>"]
        assert "Exception" not in names and "<bare except>" not in names, (
            "_facebook_signed_request catches Exception, which is how the "
            "HTTPException refusal became unreachable the first time; catch the "
            "specific decode errors instead"
        )
