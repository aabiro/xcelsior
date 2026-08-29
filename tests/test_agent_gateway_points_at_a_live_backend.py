"""The agent gateway must not proxy at a port nothing is listening on.

Two independent config faults with the same signature — every worker request
fails while Envoy, mTLS, SPIRE and the API are each individually healthy, so
the investigation goes looking for an identity problem that does not exist.

1. **Upstream pinned to the down slot.** Production runs blue/green on 9500 and
   9501, and only one is live at a time (9501 as of 2026-08-29; 9500 has
   nothing bound). `nginx/agent-xcelsior.conf` was pinned to 9500 alone and
   502'd every request; `infra/envoy/agent-gateway.yaml` was written with the
   same mistake and would have reproduced it at cutover.

2. **Listener colliding with Headscale.** The Envoy listener binds `0.0.0.0`
   under `network_mode: host`, and Headscale already holds `127.0.0.1:8443` on
   that same host. Binding 8443 there is a fight over the tailnet control plane.

Both are checked against the config, because there is no environment where
this is exercised before it matters.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
ENVOY = ROOT / "infra" / "envoy" / "agent-gateway.yaml"
NGINX = ROOT / "nginx" / "agent-xcelsior.conf"

#: Headscale's port on the control-plane host. Not negotiable from here.
HEADSCALE_PORT = 8443


def _envoy() -> dict:
    return yaml.safe_load(ENVOY.read_text(encoding="utf-8"))


def _api_upstream_ports() -> list[int]:
    cluster = next(
        c for c in _envoy()["static_resources"]["clusters"] if c["name"] == "xcelsior_api"
    )
    return [
        ep["endpoint"]["address"]["socket_address"]["port_value"]
        for ep in cluster["load_assignment"]["endpoints"][0]["lb_endpoints"]
    ]


def test_envoy_upstream_includes_both_blue_green_slots() -> None:
    ports = _api_upstream_ports()
    assert {9500, 9501} <= set(ports), (
        f"envoy proxies the API at {ports}; it must list both blue/green slots, "
        "or it will point at whichever one is currently down"
    )


def test_envoy_listener_does_not_collide_with_headscale() -> None:
    port = _envoy()["static_resources"]["listeners"][0]["address"]["socket_address"][
        "port_value"
    ]
    assert port != HEADSCALE_PORT, (
        f"the envoy listener binds {port}, which Headscale already holds on the "
        "control-plane host; under network_mode:host that is a direct collision"
    )


def test_nginx_gateway_upstream_also_lists_both_slots() -> None:
    """The interim gateway and its replacement must agree about the backend."""
    ports = {int(m) for m in re.findall(r"server\s+127\.0\.0\.1:(\d+)", NGINX.read_text())}
    assert {9500, 9501} <= ports, (
        f"nginx agent gateway upstream is {sorted(ports)}; it must mirror the "
        "blue/green pair like the public site does"
    )
