"""`xcelsior login` could not log anyone in, and its test said it could.

`cmd_login` was written against the legacy device endpoints that once lived at
`/_internal/legacy-auth/*`, which took no client identity at all. When the OAuth
service took over `/api/auth/*`, every device call started going through
`authenticate_client`, which looks the client up before anything else:

    POST /api/auth/device        (no body)
    → 401 {"error": "invalid_client", "error_description": "Unknown OAuth client"}

So the first step of the first command a new user runs failed, every time, on
every deployment. What it printed was worse than the failure:

    Error: Could not reach Xcelsior API at https://xcelsior.ca: 401 ...

because step 1's `except Exception` reported an HTTP status as unreachability.
That sends whoever hits it to DNS, the VPN and the firewall — none of which are
wrong — while the API is up and answering precisely.

The existing `test_cmd_login_success` passed throughout. It patched
`requests.post` with two canned successes and never looked at what the CLI
sent, so it asserted the plumbing between the two responses and nothing about
the protocol. A test that cannot observe the request cannot fail for a wrong
request.

These tests forward the CLI's real calls into the real app. The server decides
whether the CLI is speaking its protocol, so the assertion cannot drift away
from the endpoint the way a hand-written mock did.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse

import pytest
import requests
from fastapi.testclient import TestClient

import cli
from api import app

client = TestClient(app)

APPROVER = {
    "user_id": "cli-login-test-user",
    "email": "cli-login-test@xcelsior.ca",
    "role": "submitter",
    "name": "CLI Login Test",
}


class _ForwardToApp:
    """Stands in for `requests.post`, forwarding to the app under test.

    Records every call so a test can assert on what the CLI actually sent, and
    approves the device code the first time the token endpoint is polled —
    standing in for the human who opens the verification page.
    """

    def __init__(self, *, approve: bool = True) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.user_code: str | None = None
        self._approve = approve
        self._approved = False

    def __call__(self, url, json=None, timeout=None, **kwargs):  # noqa: A002
        path = urlparse(url).path
        self.calls.append((path, dict(json or {})))

        if self._approve and self._approved is False and "token" in path:
            self._approve_now()
            self._approved = True

        response = client.post(path, json=json or {})
        if response.status_code == 200 and "device" in path:
            self.user_code = response.json().get("user_code")
        return response

    def _approve_now(self) -> None:
        from oauth_service import approve_device_code

        assert self.user_code, "the device call never returned a user_code to approve"
        approve_device_code(user=APPROVER, user_code=self.user_code)


def _run_login(post) -> None:
    with patch("requests.post", post), patch("webbrowser.open"), patch("cli.time.sleep"):
        with patch("cli.get_api_url", return_value="http://testserver"):
            with patch("cli._save_token") as save:
                cli.cmd_login(SimpleNamespace(api_url=None))
    _run_login.saved = save  # type: ignore[attr-defined]


def test_the_device_request_is_one_the_server_accepts() -> None:
    """The regression, stated as narrowly as it can be.

    Whatever else `cmd_login` does, its first request has to be one the server
    will answer. Before the fix this was a 401 `invalid_client`, because the
    CLI sent no `client_id` and `authenticate_client` rejects an unknown client
    before it looks at anything else.
    """
    post = _ForwardToApp(approve=False)

    with patch("requests.post", post), patch("webbrowser.open"), patch("cli.time.sleep"):
        with patch("cli.get_api_url", return_value="http://testserver"):
            with patch("cli._save_token"):
                # Never approved, so the poll loop runs to its deadline and
                # exits non-zero. Step 1 is what this test is about.
                with pytest.raises(SystemExit):
                    cli.cmd_login(SimpleNamespace(api_url=None))

    assert post.calls, "cmd_login made no HTTP call at all"
    path, body = post.calls[0]
    replay = client.post(path, json=body)
    assert replay.status_code == 200, (
        f"the server refused the very request `xcelsior login` opens with: "
        f"POST {path} {body} → {replay.status_code} {replay.text[:200]}. "
        "Every login fails here, before the user is shown a code."
    )


def test_login_completes_against_the_real_endpoints() -> None:
    """The whole flow, with the server — not a mock — judging each request."""
    post = _ForwardToApp()

    with patch("requests.post", post), patch("webbrowser.open"), patch("cli.time.sleep"):
        with patch("cli.get_api_url", return_value="http://testserver"):
            with patch("cli._save_token") as save:
                cli.cmd_login(SimpleNamespace(api_url=None))

    save.assert_called_once()
    token = save.call_args[0][0]
    assert token.get("access_token"), f"login saved no access token: {token}"
    assert token.get("token_type", "").lower() == "bearer"


def test_every_device_call_identifies_the_client() -> None:
    """Both calls need it, and the poll needs a grant_type as well.

    Fixing only the first call moves the 401 from step 1 to the first poll,
    where it surfaces as "Unexpected response: 401" after the user has already
    entered their code — a worse place to fail than the start.
    """
    post = _ForwardToApp()

    with patch("requests.post", post), patch("webbrowser.open"), patch("cli.time.sleep"):
        with patch("cli.get_api_url", return_value="http://testserver"):
            with patch("cli._save_token"):
                cli.cmd_login(SimpleNamespace(api_url=None))

    assert len(post.calls) >= 2, f"expected an authorize and at least one poll, got {post.calls}"
    for path, body in post.calls:
        assert body.get("client_id"), f"POST {path} carried no client_id: {body}"
    poll_path, poll_body = post.calls[-1]
    assert poll_body.get("grant_type") == "urn:ietf:params:oauth:grant-type:device_code", (
        f"the token poll must name the device_code grant; POST {poll_path} sent "
        f"{poll_body.get('grant_type')!r}, which the server answers with "
        "400 unsupported_grant_type"
    )


def test_the_client_id_the_cli_sends_is_a_real_client() -> None:
    """A constant that names a client nobody seeded is the same 401 in disguise."""
    from oauth_service import get_client

    record = get_client(cli.OAUTH_CLIENT_ID)
    assert record, (
        f"cli.OAUTH_CLIENT_ID={cli.OAUTH_CLIENT_ID!r} is not a registered OAuth "
        "client, so every login answers 401 invalid_client"
    )
    assert "urn:ietf:params:oauth:grant-type:device_code" in list(
        record.get("grant_types") or []
    ), f"{cli.OAUTH_CLIENT_ID!r} exists but is not allowed the device_code grant"
    assert record.get("client_type") == "public", (
        f"{cli.OAUTH_CLIENT_ID!r} is confidential, so it requires a client_secret "
        "the CLI cannot hold on a user's laptop"
    )


def test_an_http_refusal_is_not_reported_as_unreachability(capsys) -> None:
    """The message that hid this for as long as it was broken.

    A 401 from a healthy API and a DNS failure are different problems with
    different fixes, and only one of them is "could not reach".
    """

    def refuse(url, json=None, timeout=None, **kwargs):  # noqa: A002
        response = requests.Response()
        response.status_code = 401
        response._content = b'{"error":"invalid_client"}'
        response.url = url
        return response

    with patch("requests.post", refuse), patch("cli.get_api_url", return_value="http://testserver"):
        with pytest.raises(SystemExit):
            cli.cmd_login(SimpleNamespace(api_url=None))

    err = capsys.readouterr().err
    assert "Could not reach" not in err, (
        f"a 401 from a responding API was reported as unreachability: {err}"
    )
    assert "401" in err, f"the refusal does not say what the API answered: {err}"
    assert "invalid_client" in err, (
        f"the body says exactly what is wrong and the message drops it: {err}"
    )


def test_an_unreachable_api_still_reports_as_unreachable(capsys) -> None:
    """The other half: don't over-correct into calling an outage a refusal."""

    def unreachable(url, json=None, timeout=None, **kwargs):  # noqa: A002
        raise requests.exceptions.ConnectionError("Name or service not known")

    with (
        patch("requests.post", unreachable),
        patch("cli.get_api_url", return_value="http://testserver"),
    ):
        with pytest.raises(SystemExit):
            cli.cmd_login(SimpleNamespace(api_url=None))

    err = capsys.readouterr().err
    assert "Could not reach" in err, f"a transport failure must read as one: {err}"


def test_the_endpoints_login_uses_are_reachable_without_a_token() -> None:
    """The blind spot in every other test in this file.

    `AUTH_REQUIRED` is off under pytest, so `TokenAuthMiddleware` never runs and
    the tests above would stay green even if the login endpoints were dropped
    from the unauthenticated allowlist — which is a 401 on `xcelsior login` for
    exactly the reason this whole file exists, arriving from the other
    direction. `login` has no token yet by definition, so both calls must be
    reachable without one.
    """
    from routes._deps import PUBLIC_PATHS, PUBLIC_PATH_PREFIXES

    for path in ("/oauth/device/authorize", "/oauth/token"):
        assert path in PUBLIC_PATHS or path.startswith(PUBLIC_PATH_PREFIXES), (
            f"{path} needs a bearer token, but `xcelsior login` is how a user "
            "gets one. TokenAuthMiddleware answers 401 before the handler runs, "
            "so every login fails with no server-side log of why."
        )


def test_the_internal_legacy_auth_surface_is_not_publicly_reachable() -> None:
    """What makes the superseded device flow harmless is the allowlist, only.

    `/_internal/legacy-auth/*` is still mounted. It keeps its codes in
    `routes.health._device_codes`, a per-process dict — so with more than one
    worker a code issued by one worker cannot be verified by another — and its
    verify handler falls back to minting a session for a caller with no user at
    all. Nothing calls it: the CLI moved to `/oauth/*`, and `oauth_service`'s
    cache-backed implementation is the real one.

    None of that matters while the path is absent from the unauthenticated
    allowlist, which is the whole of its protection. Adding `/_internal/` there
    — or serving the API on a port that does not go through nginx's prefix
    allowlist — turns it into an endpoint that hands out 30-day bearer tokens.
    That is a one-line mistake, so it gets a test rather than a comment.
    """
    from api import app
    from routes._deps import PUBLIC_PATHS, PUBLIC_PATH_PREFIXES

    internal = sorted(
        {
            path
            for route in app.routes
            if (path := getattr(route, "path", "")).startswith("/_internal/")
        }
    )
    assert internal, (
        "no /_internal/ routes are mounted any more — if they were deleted, "
        "delete this test with them"
    )
    for path in internal:
        assert path not in PUBLIC_PATHS and not path.startswith(PUBLIC_PATH_PREFIXES), (
            f"{path} is reachable without a token. If that is deliberate, the "
            "anonymous session fallback in `api_auth_verify_device` has to go "
            "first — as written it issues a real bearer token to a caller who "
            "never authenticated."
        )
