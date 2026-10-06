"""The client address the app sees must be the visitor, not Cloudflare.

fa37079 made `_get_real_client_ip` read only `request.client.host`, as uvicorn
resolves it from X-Forwarded-For — right for spoofing, but with only loopback
trusted, the walk stopped at the Cloudflare edge nginx appends. Measured on
production on 2026-10-06: a request from 207.219.90.237 was logged as
172.69.130.143. Every per-IP control was shared by all visitors on an edge:
the login brute-force limit, the flood limit, the WebSocket ticket pin, the
demo gate.

These run uvicorn's real ProxyHeadersMiddleware with the trusted list from
gunicorn.conf.py, on the header chain production actually produces:
Cloudflare sets X-Forwarded-For to the visitor; nginx appends its peer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

ROOT = Path(__file__).resolve().parents[1]


def _trusted() -> list[str]:
    import os

    # Loaded as one worker: the test environment runs in-memory auth, which
    # the config rightly refuses to combine with several workers.
    saved = os.environ.get("GUNICORN_WORKERS")
    os.environ["GUNICORN_WORKERS"] = "1"
    try:
        ns: dict = {}
        exec((ROOT / "gunicorn.conf.py").read_text(), ns)
    finally:
        if saved is None:
            os.environ.pop("GUNICORN_WORKERS", None)
        else:
            os.environ["GUNICORN_WORKERS"] = saved
    # gunicorn's own default when the config does not set it: loopback only.
    return [h.strip() for h in ns.get("forwarded_allow_ips", "127.0.0.1").split(",")]


def _resolve(peer: str, xff: str) -> str:
    seen: dict = {}

    async def app(scope, receive, send):  # noqa: ANN001
        seen["client"] = scope["client"][0]

    mw = ProxyHeadersMiddleware(app, trusted_hosts=_trusted())
    scope = {
        "type": "http", "scheme": "http", "client": (peer, 0),
        "headers": [(b"x-forwarded-for", xff.encode())],
    }
    asyncio.run(mw(scope, None, None))
    return seen["client"]


def test_a_visitor_behind_cloudflare_is_the_client() -> None:
    """The production chain: CF wrote the visitor, nginx appended the CF edge."""
    assert _resolve("127.0.0.1", "207.219.90.237, 172.69.130.143") == "207.219.90.237"


def test_an_ipv6_edge_is_walked_past_too() -> None:
    assert _resolve("127.0.0.1", "203.0.113.9, 2606:4700::6810:84e5") == "203.0.113.9"


def test_a_forger_bypassing_cloudflare_gets_their_own_address() -> None:
    """They can write anything on the left; nginx appends who they really are."""
    assert _resolve("127.0.0.1", "207.219.90.237, 198.51.100.7") == "198.51.100.7"


def test_a_forged_cloudflare_hop_does_not_skip_the_real_peer() -> None:
    """Writing a Cloudflare address yourself does not make you Cloudflare."""
    assert _resolve("127.0.0.1", "1.2.3.4, 172.69.1.1, 198.51.100.7") == "198.51.100.7"


def test_a_direct_untrusted_peer_cannot_be_overridden() -> None:
    """Not behind nginx at all: the header is ignored outright."""
    assert _resolve("198.51.100.7", "207.219.90.237") == "198.51.100.7"


@pytest.mark.parametrize("edge", ["172.69.130.143", "104.16.0.1", "162.158.1.1"])
def test_the_observed_and_common_edges_are_trusted(edge: str) -> None:
    import ipaddress

    nets = [ipaddress.ip_network(t) for t in _trusted() if "/" in t]
    assert any(ipaddress.ip_address(edge) in n for n in nets), edge
