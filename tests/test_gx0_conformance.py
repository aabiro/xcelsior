"""Exercise the standalone runner without network access or a database."""

import json
import signal
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import httpx
import pytest

from scripts import gx0_conformance as gx0


ISSUER = "https://issuer.example"
RESOURCE = "https://mcp.example/mcp"
METADATA = {
    "authorization_endpoint": ISSUER + "/oauth/authorize",
    "token_endpoint": ISSUER + "/oauth/token",
}


# The checker imports only httpx. Override the application fixtures so its
# regression tests can also run without loading API routes or reaching Postgres.
@pytest.fixture
def _reset_shared_runtime_limits():
    pass


@pytest.fixture
def _reset_public_status_cache():
    pass


@pytest.fixture
def _pin_test_auth_env():
    pass


class OAuthServer:
    """Tokens are unique per client/grant, and revocation is observable."""

    def __init__(self):
        self.codes = {}
        self.tokens = {}
        self.revoked = set()
        self.consumed = set()
        self.calls = []
        self.logouts = []
        self.refresh_probes = []
        self.tool_error = None
        self.broken_logout = None

    def issue(self, client):
        key = str(len(self.tokens))
        bundle = {
            "access_token": "test-access-" + key,
            "refresh_token": "test-refresh-" + key,
            "expires_in": 3600,
            "resource": RESOURCE,
        }
        self.tokens[bundle["access_token"]] = (client, bundle["refresh_token"])
        return bundle

    def __call__(self, request):
        path = request.url.path
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        if path == "/api/auth/login":
            bundle = self.issue("web")
            return httpx.Response(
                200,
                json={"access_token": bundle["access_token"]},
                headers={
                    "set-cookie": "xcelsior_refresh="
                    + bundle["refresh_token"]
                    + "; Path=/; HttpOnly",
                },
            )
        if path == "/gx0/client-v1.json":
            document = json.loads(Path("api/scripts/fixtures/gx0-client-v1.json").read_text())
            return httpx.Response(200, json=document)
        if path == "/oauth/authorize":
            params = request.url.params
            if params["resource"] != RESOURCE:
                return httpx.Response(400)
            code = "code-" + str(len(self.codes))
            self.codes[code] = params["client_id"]
            return httpx.Response(
                302,
                headers={
                    "location": params["redirect_uri"]
                    + "?"
                    + urlencode({"code": code, "state": params["state"]})
                },
            )
        if path == "/oauth/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            if form["grant_type"] == "authorization_code":
                assert form["client_id"] == self.codes[form["code"]]
                return httpx.Response(200, json=self.issue(form["client_id"]))
            token = form["refresh_token"]
            if token in self.revoked or token in self.consumed:
                self.refresh_probes.append(token)
                return httpx.Response(401, json={"error": "invalid_grant"})
            self.consumed.add(token)
            return httpx.Response(200, json=self.issue("refreshed"))
        if path == "/api/auth/logout":
            cookie = request.headers["cookie"]
            assert cookie.startswith("xcelsior_refresh=") and ";" not in cookie
            token = cookie.split("=", 1)[1]
            assert self.tokens[bearer][1] not in self.revoked, "login must be revoked last"
            self.logouts.append(token)
            if token != self.broken_logout:
                self.revoked.add(token)
            return httpx.Response(200, json={"ok": True})
        assert "cookie" not in request.headers, (
            "probe must authenticate with the tested bearer only"
        )
        client, token = self.tokens.get(bearer, ("unknown", "unknown"))
        if token in self.revoked:
            return httpx.Response(401)
        if path == "/api/auth/me":
            return httpx.Response(200, json={"user": {}})
        assert path == "/mcp"
        body = json.loads(request.content)
        method = body["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            data = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {}}
        elif method == "tools/list":
            if "mcp-session-id" not in request.headers:
                return httpx.Response(400)
            assert request.headers["mcp-session-id"] == "test-session"
            data = {
                "tools": [
                    {
                        "name": "list_instances",
                        "annotations": {"readOnlyHint": True},
                        "outputSchema": {"type": "object"},
                    }
                ]
            }
        else:
            assert method == "tools/call"
            assert body["params"] == {"name": "list_instances", "arguments": {"limit": 1}}
            self.calls.append((client, bearer))
            data = self.tool_error or {"structuredContent": {"instances": []}}
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": data},
            headers={"mcp-session-id": "test-session"},
        )


def test_each_client_and_refreshed_token_execute_a_tool_and_all_sessions_are_revoked(capsys):
    server = OAuthServer()
    results = gx0.Results()
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        cleanup = gx0.SessionCleanup(results, http, ISSUER, METADATA["token_endpoint"], RESOURCE)
        try:
            gx0.run_authenticated_checks(
                results,
                http,
                ISSUER,
                "reviewer@example.test",
                "not-real",
                METADATA,
                RESOURCE,
                "xcelsior-connector",
                gx0.DEFAULT_CIMD_CLIENT_ID,
                {"client_id": "new-dcr-client"},
                cleanup,
            )
        finally:
            cleanup.cleanup()
        assert len(http.cookies) == 0
    assert [client for client, _ in server.calls] == [
        "new-dcr-client",
        gx0.DEFAULT_CIMD_CLIENT_ID,
        "xcelsior-connector",
        "refreshed",
    ]
    assert len({token for _, token in server.calls}) == 4
    assert (
        len(server.tokens)
        == len(server.logouts)
        == len(server.revoked)
        == len(server.refresh_probes)
        == 7
    )
    assert server.logouts[-1] == "test-refresh-0"
    assert results.failures == results.blocks == 0
    output = capsys.readouterr().out
    for secret in list(server.tokens) + [t for _, t in server.tokens.values()]:
        assert secret not in output


@pytest.mark.parametrize(
    "tool_error",
    [
        {"isError": True, "structuredContent": {"instances": []}},
        {"structuredContent": {"error": "insufficient_scope", "instances": []}},
        {"structuredContent": {"ok": False, "instances": []}},
        {"content": [{"type": "text", "text": '{"error":"upstream unavailable"}'}]},
        {"structuredContent": {}},
    ],
)
def test_http_200_tool_errors_are_failures(tool_error):
    server = OAuthServer()
    server.tool_error = tool_error
    bundle = server.issue("DCR")
    results = gx0.Results()
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        gx0.check_authenticated_tools_list(results, http, RESOURCE, bundle["access_token"], "DCR")
    assert results.failures == 1
    assert results.rows[-1][0] == gx0.FAIL
    assert "tools/call" in results.rows[-1][1]


@pytest.mark.parametrize(
    "exception", [RuntimeError("secret must not appear"), KeyboardInterrupt(), "SIGTERM"]
)
def test_main_revokes_on_failure_or_interrupt(monkeypatch, exception, capsys):
    server = OAuthServer()
    http_class = httpx.Client
    monkeypatch.setattr(
        gx0.httpx, "Client", lambda **kw: http_class(transport=httpx.MockTransport(server), **kw)
    )
    monkeypatch.setattr(gx0, "check_external_vantage", lambda *a: None)
    monkeypatch.setattr(gx0, "check_tls", lambda *a: None)
    monkeypatch.setattr(gx0, "check_challenge", lambda *a: "metadata")
    monkeypatch.setattr(
        gx0,
        "check_protected_resource",
        lambda *a: {"resource": RESOURCE, "authorization_servers": [ISSUER]},
    )
    monkeypatch.setattr(gx0, "check_authorization_server", lambda *a: METADATA)
    monkeypatch.setattr(gx0, "check_cimd_rejection", lambda *a: None)
    monkeypatch.setattr(gx0, "check_dcr", lambda *a: {"client_id": "new-dcr-client"})
    monkeypatch.setenv("GX0_EMAIL", "reviewer@example.test")
    monkeypatch.setenv("GX0_PASSWORD", "not-real")
    monkeypatch.setattr("sys.argv", ["gx0", "--base", RESOURCE])

    def fail(*args):
        cleanup = args[-1]
        assert gx0.sign_in(cleanup.http, ISSUER, "reviewer", "not-real", cleanup)[0] == "pass"
        cleanup.track(server.issue("DCR"), "DCR", "DCR client")
        cleanup.track(server.issue("refreshed"), "DCR", "refreshed token")
        if exception == "SIGTERM":
            signal.raise_signal(signal.SIGTERM)
        raise exception

    monkeypatch.setattr(gx0, "run_authenticated_checks", fail)
    previous_handler = signal.getsignal(signal.SIGTERM)
    assert gx0.main() == 1
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert len(server.revoked) == len(server.tokens) == 3
    assert "secret must not appear" not in capsys.readouterr().out


def test_logout_200_noop_fails_and_does_not_skip_other_sessions():
    server = OAuthServer()
    results = gx0.Results()
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        cleanup = gx0.SessionCleanup(results, http, ISSUER, METADATA["token_endpoint"], RESOURCE)
        gx0.sign_in(http, ISSUER, "reviewer", "not-real", cleanup)
        bundle = server.issue("DCR")
        server.broken_logout = bundle["refresh_token"]
        cleanup.track(bundle, "DCR", "DCR client")
        cleanup.cleanup()
    assert results.failures >= 2
    assert "test-refresh-0" in server.revoked
    assert "test-refresh-2" in server.revoked, "revoke credential minted by failed revocation probe"


@pytest.mark.parametrize(
    "body",
    [
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603}},
        {"jsonrpc": "2.0", "id": 7, "result": {}},
        {"jsonrpc": "2.0", "id": 1, "result": [], "error": {}},
    ],
)
def test_rpc_error_or_wrong_id_never_passes(body):
    assert gx0.rpc_result(httpx.Response(200, json=body), 1) is None


def test_sse_parses_multiple_events_and_ignores_notifications():
    response = httpx.Response(
        200,
        text=': heartbeat\n\nevent: message\ndata: {"jsonrpc":"2.0","method":"notification"}\n\nevent: message\ndata: {"jsonrpc":"2.0","id":1,\ndata: "result":{"tools":[]}}\n\n',
    )
    assert gx0.rpc_result(response, 1) == {"tools": []}
