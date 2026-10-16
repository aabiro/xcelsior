"""`/oauth/authorize` refuses a bad request before sending anyone to log in.

It used to redirect every signed-out request to `/login` first, so an unknown
or unresolvable client id answered 302 and the person only learned the
connection was broken after signing in. GX0's "an unresolvable
metadata-document client id is rejected" check failed on exactly that.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_API_TOKEN", "testtoken")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)

PARAMS = {
    "response_type": "code",
    "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
    "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
    "code_challenge_method": "S256",
}


def _authorize(**overrides):
    client.cookies.clear()
    return client.get(
        "/oauth/authorize",
        params={**PARAMS, **overrides},
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )


def test_unknown_client_is_refused_without_a_login_detour():
    response = _authorize(client_id="no-such-client")
    assert response.status_code == 400, response.headers.get("location")


def test_unresolvable_metadata_document_client_is_refused_signed_out():
    response = _authorize(client_id="https://gx0-nonexistent.invalid/client.json")
    assert response.status_code >= 400, response.headers.get("location")


def test_unregistered_redirect_is_refused_signed_out():
    response = _authorize(client_id="xcelsior-connector", redirect_uri="https://evil.example/cb")
    assert response.status_code == 400


def test_a_valid_request_still_goes_to_login_when_signed_out():
    response = _authorize(client_id="xcelsior-connector")
    assert response.status_code == 302
    assert response.headers["location"].startswith("/login?redirect=")
