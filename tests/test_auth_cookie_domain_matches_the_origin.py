"""An auth cookie scoped to a domain you are not on is silently discarded.

The domain was the literal `".xcelsior.ca"` whenever `XCELSIOR_BASE_URL` began
with `https` — a *scheme* test standing in for an *identity* test. Any other
HTTPS origin (staging, a preview deploy, a tunnel) was handed a cookie for a
domain it is not on, and browsers drop those without complaint: login returns
200 and sets a cookie, and the next request has no session.

Found while trying to reach the Quick Connect page, which needs an interactive
session. Against a local API the response carried

    Set-Cookie: xcelsior_session=…; Domain=.xcelsior.ca

which no client talking to 127.0.0.1 will store. There is no error to see.

Production is the case that must not move: an `xcelsior.ca` origin still gets
the shared parent domain, which is what lets subdomains see one session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routes._deps import _auth_cookie_domain  # noqa: E402


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("XCELSIOR_COOKIE_DOMAIN", raising=False)


@pytest.mark.parametrize(
    "base",
    ["https://xcelsior.ca", "https://www.xcelsior.ca", "https://dashboard.xcelsior.ca"],
)
def test_production_origins_keep_the_shared_parent_domain(base: str) -> None:
    """The behaviour that must not change: one session across subdomains."""
    assert _auth_cookie_domain(base) == ".xcelsior.ca"


@pytest.mark.parametrize(
    "base",
    [
        "https://staging.example.com",
        "https://xcelsior-preview.vercel.app",
        "https://abc123.ngrok.io",
        "http://127.0.0.1:9500",
        "http://localhost:3000",
    ],
)
def test_other_origins_get_a_host_only_cookie(base: str) -> None:
    """None means no Domain attribute — valid on any host, including an IP."""
    assert _auth_cookie_domain(base) is None, (
        f"{base} would be sent a cookie for another domain, which the client "
        "discards silently: login succeeds and the session never exists"
    )


def test_an_https_origin_is_not_assumed_to_be_xcelsior() -> None:
    """The exact substitution that caused this: scheme used as identity."""
    assert _auth_cookie_domain("https://someone-elses-host.test") != ".xcelsior.ca"


def test_a_lookalike_domain_is_not_matched() -> None:
    """`endswith` on a bare suffix would match an attacker-registered name."""
    assert _auth_cookie_domain("https://notxcelsior.ca") is None
    assert _auth_cookie_domain("https://xcelsior.ca.evil.test") is None


def test_an_explicit_override_wins(monkeypatch) -> None:
    """A deployment that genuinely needs cross-subdomain on its own name."""
    monkeypatch.setenv("XCELSIOR_COOKIE_DOMAIN", ".example.com")
    assert _auth_cookie_domain("https://app.example.com") == ".example.com"
    assert _auth_cookie_domain("https://xcelsior.ca") == ".example.com"


def test_the_clearer_uses_the_same_domain_as_the_setter() -> None:
    """A delete_cookie whose Domain differs clears nothing — logout would lie."""
    import inspect

    from routes import _deps

    setter = inspect.getsource(_deps._set_auth_cookie)
    clearer = inspect.getsource(_deps._clear_auth_cookie)
    for src, name in ((setter, "_set_auth_cookie"), (clearer, "_clear_auth_cookie")):
        assert "_auth_cookie_domain(" in src, (
            f"{name} does not derive its domain from _auth_cookie_domain; if the "
            "two disagree, logout appears to work and leaves the session live"
        )
