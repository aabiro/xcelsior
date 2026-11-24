"""The URLs an MCP client and our own runbook probe must answer what is true.

Two endpoints gave a misleading answer, both found on 2026-11-24 by probing
production from outside:

1. `https://xcelsior.ca/.well-known/openid-configuration` answered 401.
   Under the 2025-11-25 MCP authorization spec this is the second URL a client
   tries during authorization-server discovery. We do not serve OIDC discovery,
   so the honest answer is 404. The 401 came from `TokenAuthMiddleware`, which
   runs before routing and answers 401 to every path it does not know.

2. `https://mcp.xcelsior.ca/mcp/health` and `/readyz` answered 404. The
   `mcp.xcelsior.ca` vhost only had `location = /mcp`, an exact match. But
   `runbooks/mcp-edge.md` checks `tool_profile` (operator tools leaking onto the
   public connector) and `resource_audience` on that host. Those checks could
   only be run against the `xcelsior.ca/mcp` compatibility route, which is not
   the edge they describe.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "10000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "10000")

from routes._deps import PUBLIC_PATHS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MCP_VHOST = ROOT / "api" / "nginx" / "mcp-xcelsior.conf"
RUNBOOK = ROOT / "api" / "runbooks" / "mcp-edge.md"

OIDC_DISCOVERY = "/.well-known/openid-configuration"


def test_oidc_discovery_probe_reaches_the_router():
    """Public means the router answers it, and the router has no such route: 404."""
    assert OIDC_DISCOVERY in PUBLIC_PATHS


def test_oidc_discovery_is_not_served_by_any_route():
    """If someone later serves it, it must be a real document, not a stub.
    Then this test should change on purpose."""
    from api import app

    paths = {getattr(r, "path", None) for r in app.routes}
    assert OIDC_DISCOVERY not in paths


def _location_block(conf: str, path: str) -> str:
    m = re.search(r"location\s*=\s*" + re.escape(path) + r"\s*\{(.*?)\n    \}", conf, re.S)
    assert m, f"mcp-xcelsior.conf has no `location = {path}` block"
    return m.group(1)


def test_runbook_probes_on_the_mcp_host_are_routed():
    conf = MCP_VHOST.read_text()
    runbook = RUNBOOK.read_text()
    probed = set(re.findall(r"https://mcp\.xcelsior\.ca(/[A-Za-z0-9_./-]*)", runbook))
    assert {"/mcp/health", "/readyz"} <= probed, "runbook no longer probes these; revisit this test"
    for path in ("/mcp/health", "/readyz"):
        block = _location_block(conf, path)
        assert "proxy_pass http://xcelsior_mcp_canonical;" in block
        # /readyz makes outbound fetches per call; an unthrottled public probe
        # would be an amplifier.
        assert "limit_req zone=" in block
