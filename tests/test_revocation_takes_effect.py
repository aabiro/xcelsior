"""Revocation is asserted by its effect, not by its response.

Both revocation endpoints returned `{"ok": true}` and left the presented bearer
working. Measured against production on 2026-08-04:

    POST   /api/auth/logout               → {"ok": true}
    GET    /api/auth/me                   → 200   (immediately, and 30s later)
    DELETE /api/auth/sessions/l7Q9Yf4r    → {"ok": true, "message": "Session revoked"}
    GET    /api/auth/me                   → 200

The cause: an opaque access token resolves *only* from the auth cache, and both
paths deleted the session row — which nothing on the request path consults. The
token stayed valid for the remainder of its TTL, which for a browser session is
30 days.

It survived because both endpoints had their **success** asserted and neither had
its **effect** asserted. That is the whole defect class in one sentence, and it is
why every test here revokes and then re-resolves the same token.
"""

from __future__ import annotations

import time

import pytest

import oauth_service


@pytest.fixture(autouse=True)
def clean_cache():
    oauth_service.reset_auth_cache_for_tests()
    yield
    oauth_service.reset_auth_cache_for_tests()


def _mint(session_token: str = "sess-abc") -> str:
    """One cached access token, shaped as `issue_user_tokens` writes it."""
    token = f"xoa_{session_token}_bearer"
    oauth_service._cache_set_json(
        "access_token",
        token,
        {
            "auth_type": "oauth_access_token",
            "email": "probe@example.com",
            "session_token": session_token,
            "expires_at": time.time() + 3600,
        },
        3600,
    )
    return token


def test_a_freshly_minted_token_resolves():
    """The control. Without it, every assertion below passes on a broken mint."""
    token = _mint()
    assert oauth_service.resolve_opaque_access_token(token) is not None


def test_revoking_the_token_stops_it_resolving():
    token = _mint()
    assert oauth_service.resolve_opaque_access_token(token) is not None
    oauth_service.revoke_access_token(token)
    assert oauth_service.resolve_opaque_access_token(token) is None, (
        "the token still resolves after revoke_access_token — this is the "
        "production defect, where logout returned ok and the bearer kept working"
    )


def test_revoking_the_session_stops_every_token_minted_from_it():
    """Revocation by session, which is all the prefix endpoint can identify."""
    first = _mint("sess-xyz")
    second = "xoa_sess-xyz_second"
    oauth_service._cache_set_json(
        "access_token",
        second,
        {"auth_type": "oauth_access_token", "email": "probe@example.com",
         "session_token": "sess-xyz", "expires_at": time.time() + 3600},
        3600,
    )
    assert oauth_service.resolve_opaque_access_token(first) is not None
    assert oauth_service.resolve_opaque_access_token(second) is not None

    oauth_service.revoke_session("sess-xyz")

    assert oauth_service.resolve_opaque_access_token(first) is None
    assert oauth_service.resolve_opaque_access_token(second) is None, (
        "a sibling bearer from the same session survived revocation — revoking a "
        "session must reach every credential minted from it, or 'revoked' names "
        "only the one token the caller happened to hold"
    )


def test_revoking_one_session_leaves_another_alone():
    """Revocation is narrow. A blunt revocation is its own outage."""
    keep = _mint("sess-keep")
    drop = _mint("sess-drop")
    oauth_service.revoke_session("sess-drop")
    assert oauth_service.resolve_opaque_access_token(drop) is None
    assert oauth_service.resolve_opaque_access_token(keep) is not None, (
        "revoking one session killed another — the marker is not session-scoped"
    )


def test_expiry_still_works_independently_of_revocation():
    """The path that already worked keeps working."""
    token = "xoa_expired"
    oauth_service._cache_set_json(
        "access_token",
        token,
        {"auth_type": "oauth_access_token", "session_token": "s", "expires_at": time.time() - 1},
        3600,
    )
    assert oauth_service.resolve_opaque_access_token(token) is None


def test_the_revocation_marker_outlives_the_credential_it_revokes():
    """A marker that expires first is a revocation with a gap in it.

    Asserted on the constant rather than by waiting: the marker TTL must exceed
    the longest access-token lifetime, or a revoked session becomes valid again
    when the marker lapses.
    """
    longest = max(oauth_service.BROWSER_SESSION_TTL_SEC, oauth_service.ACCESS_TOKEN_TTL_SEC)
    import inspect

    src = inspect.getsource(oauth_service.revoke_session)
    assert "max(BROWSER_SESSION_TTL_SEC, ACCESS_TOKEN_TTL_SEC)" in src, (
        "revoke_session no longer derives its TTL from the longest credential "
        f"lifetime (currently {longest}s); a shorter marker would let a revoked "
        "session resume"
    )
