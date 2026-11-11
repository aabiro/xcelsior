"""The "Authentication complete" page holds long enough to read, then hands back.

At 1.2 s the page flashed past on the way back to the app and read as a jolt.
The user asked for at least two seconds more (2026-11-10). The handoff bar
fills over the same interval, so the wait looks intentional, not stalled.

A loopback return also says what the browser's "can't be reached" means: the
app that opened the port stopped waiting (Claude Desktop gives a starting
MCP server a minute), which is what the user hit with no explanation.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import routes.auth as auth  # noqa: E402


def _page(*, approved=True, loopback=True, name="Claude"):
    callback = (
        "http://127.0.0.1:42418/oauth/callback?code=c&state=s"
        if loopback
        else "https://claude.ai/api/mcp/auth_callback?code=c&state=s"
    )
    return auth._oauth_decision_page(
        approved=approved, client={"client_name": name}, callback_url=callback, is_loopback=loopback
    )


def test_the_page_waits_at_least_three_seconds_before_returning():
    assert auth._OAUTH_RETURN_DELAY_MS >= 3000
    page = _page()
    delays = re.findall(r"setTimeout\(function\(\)\{window\.location\.replace\(.*?\);\},(\d+)\)", page)
    assert delays == [str(auth._OAUTH_RETURN_DELAY_MS)]


def test_the_handoff_bar_fills_over_the_same_wait():
    page = _page()
    assert f"animation: handoff-fill {auth._OAUTH_RETURN_DELAY_MS}ms linear forwards" in page
    assert "__HANDOFF_MS__" not in page
    # Reduced motion keeps the bar, without the movement.
    assert re.search(r"prefers-reduced-motion: reduce\)[^@]*\.handoff::after \{ animation: none; \}", page)


def test_only_an_approved_loopback_return_explains_a_refused_connection():
    hint = "stopped waiting for this sign-in"
    assert hint in _page(approved=True, loopback=True)
    assert hint not in _page(approved=True, loopback=False)
    assert hint not in _page(approved=False, loopback=True)


def test_the_client_name_in_the_hint_is_escaped():
    page = _page(name='<img src=x onerror="alert(1)">')
    assert "<img src=x" not in page and "&lt;img src=x" in page


def test_safe_to_close_notice_links_the_close_word_and_preserves_callback_handoff():
    for page in (_page(approved=True), _page(approved=False), auth._oauth_error_page("Expired", "Try again.", 400)):
        assert "It’s safe to " in page
        assert '<a class="close-link" href="#close" role="button" data-close-window>close</a> this window.' in page
        assert "window.close()" in page
        assert "onclick=" not in page

    decision_page = _page(approved=True)
    # Clicking close before the 3.2 s timer fires still hands the callback URL over first.
    assert "if(!handedOff){handedOff=true;window.location.replace(callbackUrl);" in decision_page
