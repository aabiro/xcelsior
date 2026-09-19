"""The 443 ingress is split across three files that must agree.

`agent.xcelsior.ca` needs the worker's client certificate to survive to Envoy,
so nginx must not terminate its TLS. Six other vhosts on the same port must keep
terminating exactly as before. The arrangement:

    stream (443, ssl_preread)  ──agent.xcelsior.ca──►  Envoy 9443   (Phase B)
                               └──everything else───►  nginx 8444

Three ways this breaks silently, each leaving every component individually
healthy while all traffic fails:

* **Two things bind 443.** The stream router and a leftover `listen 443 ssl`
  cannot coexist; nginx refuses to start, and on a reload that means the whole
  box loses TLS, not just one vhost.
* **PROXY protocol mismatch.** The router sends the PROXY header to *every*
  backend. A backend not expecting it reads that line as the first bytes of a
  ClientHello and fails as malformed TLS.
* **Real client IP lost.** After passthrough the peer is 127.0.0.1, so without
  `real_ip_header proxy_protocol` six services silently start logging,
  rate-limiting and geolocating against localhost — the one thing the
  restructure was required not to change.

And 8444 rather than 8443 because Headscale already holds `127.0.0.1:8443` on
that host.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
NGINX = ROOT / "nginx"
ROUTER = NGINX / "stream-tls-router.conf"
ENVOY = ROOT / "infra" / "envoy" / "agent-gateway.yaml"

HEADSCALE_PORT = 8443
BACKEND_PORT = 8444

#: vhosts that moved behind the router.
MOVED = [
    "xcelsior.conf",
    "mcp-xcelsior.conf",
    "docs-xcelsior.conf",
    "downloads-xcelsior.conf",
    "headscale.conf",
    "unmatched-hosts.conf",
    "agent-xcelsior.conf",
]


def _conf(name: str) -> str:
    return (NGINX / name).read_text(encoding="utf-8")


def test_only_the_stream_router_binds_443() -> None:
    offenders = [
        p.name
        for p in NGINX.glob("*.conf")
        if p.name != ROUTER.name and re.search(r"^\s*listen\s+(\[::\]:)?443\b", p.read_text(), re.M)
    ]
    assert not offenders, (
        f"{offenders} still bind 443 alongside the stream router; nginx will "
        "refuse to start and the box loses TLS for every vhost, not just these"
    )
    assert re.search(r"^\s*listen\s+443\b", ROUTER.read_text(), re.M), (
        "the stream router does not bind 443, so nothing does"
    )


def test_the_router_does_not_send_traffic_to_headscales_port() -> None:
    # Comments stripped first: this file *explains* why it avoids 8443, and
    # matching that sentence would fail the check on its own rationale.
    body = "\n".join(
        ln.split("#", 1)[0] for ln in ROUTER.read_text().splitlines()
    )
    ports = {int(m) for m in re.findall(r"127\.0\.0\.1:(\d+)", body)}
    assert HEADSCALE_PORT not in ports, (
        f"the router proxies to {HEADSCALE_PORT}, which Headscale already holds "
        "on the control-plane host"
    )


@pytest.mark.parametrize("name", MOVED)
def test_each_moved_vhost_expects_proxy_protocol(name: str) -> None:
    # Only TLS listeners moved. Plain :80 redirect blocks and internal
    # loopback listeners never fronted TLS and the router does not touch them.
    listens = [
        ln
        for ln in re.findall(r"^\s*listen\s+[^\n;]*;", _conf(name), re.M)
        if re.search(r"\bssl\b", ln)
    ]
    assert listens, f"{name} has no TLS listener"
    for ln in listens:
        assert f"{BACKEND_PORT}" in ln, f"{name}: {ln.strip()!r} is not on {BACKEND_PORT}"
        assert "proxy_protocol" in ln, (
            f"{name}: {ln.strip()!r} does not expect PROXY protocol, but the "
            "router in front of it always sends it — every request fails as "
            "malformed TLS"
        )


@pytest.mark.parametrize("name", MOVED)
def test_each_moved_vhost_restores_the_real_client_ip(name: str) -> None:
    body = _conf(name)
    assert "set_real_ip_from 127.0.0.1;" in body, (
        f"{name} does not trust the router's PROXY header, so real_ip_header is "
        "ignored and this service sees every client as 127.0.0.1"
    )
    assert re.search(r"real_ip_header\s+proxy_protocol;", body), (
        f"{name} does not read the client address from the PROXY header; logs "
        "and rate limits would silently see localhost"
    )


def test_envoy_consumes_the_proxy_header_the_router_sends() -> None:
    listener = yaml.safe_load(ENVOY.read_text())["static_resources"]["listeners"][0]
    names = [f.get("name", "") for f in listener.get("listener_filters", [])]
    assert any("proxy_protocol" in n for n in names), (
        "the envoy listener has no proxy_protocol listener filter, but the "
        f"stream router sends PROXY to every backend: envoy would read that "
        f"line as a ClientHello. filters={names}"
    )
