"""Only the ASGI server's trusted-proxy boundary may choose the client IP."""

from collections import defaultdict, deque

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.websockets import WebSocket
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from routes import _deps as deps
from routes.billing import _client_ip as billing_client_ip


def scope(kind="http", peer="203.0.113.25", **headers):
    return {
        "type": kind, "http_version": "1.1", "scheme": "https" if kind == "http" else "wss",
        "method": "GET", "path": "/", "query_string": b"",
        "client": (peer, 54321) if peer else None, "server": ("localhost", 9500),
        "headers": [(key.replace("_", "-").encode(), value.encode()) for key, value in headers.items()],
    }


async def unused(*args):
    raise AssertionError("IP extraction must not read or send a request body")


def addresses(asgi_scope):
    if asgi_scope["type"] == "websocket":
        return [deps._get_ws_client_ip(WebSocket(asgi_scope, unused, unused))]
    request = Request(asgi_scope)
    return [deps._get_real_client_ip(request), billing_client_ip(request)]


@pytest.mark.parametrize("kind", ["http", "websocket"])
@pytest.mark.parametrize("header", ["cf_connecting_ip", "x_real_ip", "x_forwarded_for"])
def test_direct_clients_cannot_override_their_address(kind, header):
    assert set(addresses(scope(kind, **{header: "198.51.100.99"}))) == {"203.0.113.25"}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["http", "websocket"])
@pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
async def test_trusted_nginx_chain_uses_the_last_untrusted_hop(kind, peer):
    observed = []

    async def app(asgi_scope, receive, send):
        observed.extend(addresses(asgi_scope))

    # Nginx appends the actual remote address to any caller-supplied XFF.
    # The unrelated CF header must not override that verified result.
    middleware = ProxyHeadersMiddleware(app, trusted_hosts=["127.0.0.1", "::1"])
    await middleware(scope(kind, peer, x_forwarded_for="198.51.100.99, 203.0.113.25",
                           x_real_ip="203.0.113.25", cf_connecting_ip="198.51.100.99"), unused, unused)
    assert set(observed) == {"203.0.113.25"}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["http", "websocket"])
async def test_an_untrusted_peer_cannot_present_a_proxy_chain(kind):
    observed = []

    async def app(asgi_scope, receive, send):
        observed.extend(addresses(asgi_scope))

    await ProxyHeadersMiddleware(app)(scope(kind, x_forwarded_for="198.51.100.99"), unused, unused)
    assert set(observed) == {"203.0.113.25"}


def test_missing_peer_is_unknown_even_with_forwarded_headers():
    request = Request(scope(peer=None, x_real_ip="198.51.100.99"))
    assert deps._get_real_client_ip(request) == "unknown"
    assert billing_client_ip(request) == ""


def test_rotating_headers_does_not_grant_new_login_budgets(monkeypatch, _pin_test_auth_env):
    monkeypatch.setattr(deps, "_AUTH_RATE_LIMIT_REQUESTS", 2)
    monkeypatch.setattr(deps, "_AUTH_RATE_BUCKETS", defaultdict(deque))
    monkeypatch.setattr(deps, "_shared_state_update", lambda *args: (False, None))
    for suffix in (1, 2):
        deps._check_auth_rate_limit(Request(scope(cf_connecting_ip=f"198.51.100.{suffix}")))
    with pytest.raises(HTTPException) as error:
        deps._check_auth_rate_limit(Request(scope(cf_connecting_ip="198.51.100.3")))
    assert error.value.status_code == 429


def test_a_stolen_ticket_cannot_be_redeemed_by_spoofing_its_ip(monkeypatch):
    monkeypatch.setattr(deps, "_USE_SHARED_RUNTIME_LIMITS", False)
    monkeypatch.setattr(deps, "_WS_TICKETS", {})
    ticket = deps._issue_ws_ticket(
        {"user_id": "owner", "email": "owner@example.test"}, purpose="terminal",
        target="instance", client_ip="203.0.113.25",
    )["ticket"]
    attacker = WebSocket(scope("websocket", peer="198.51.100.99",
                               cf_connecting_ip="203.0.113.25"), unused, unused)
    assert deps._consume_ws_ticket(ticket, attacker, purpose="terminal", target="instance") is None
