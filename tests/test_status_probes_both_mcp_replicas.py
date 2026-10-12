"""The status page has to find the live MCP replica, whichever colour it is.

deploy.sh promotes `mcp` (:8770) and `mcp-blue` (:8771) in turn and stops the
replica it replaced. `/api/status` probed :8770 alone, so after every other
deploy it reported "MCP connector: down — Connection refused" while
mcp.xcelsior.ca served normally from :8771. The CLI wizard showed that to every
user as "Some services are degraded" (found 2026-10-12, live colour blue).
"""

from __future__ import annotations

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

import routes.health as health  # noqa: E402


def _serve(status: int, body: dict) -> tuple[HTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:  # noqa: ANN002
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/readyz"


def _closed_url() -> str:
    """A loopback port with nothing listening, like a stopped standby replica."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/readyz"


@pytest.fixture
def replicas(monkeypatch):  # noqa: ANN001
    servers: list[HTTPServer] = []

    def configure(*urls: str) -> None:
        monkeypatch.setenv("XCELSIOR_MCP_HEALTH_URL", ",".join(urls))

    def serve(status: int, body: dict) -> str:
        server, url = _serve(status, body)
        servers.append(server)
        return url

    yield configure, serve
    for server in servers:
        server.shutdown()


def test_the_live_replica_on_the_second_port_reads_as_up(replicas) -> None:  # noqa: ANN001
    configure, serve = replicas
    configure(_closed_url(), serve(200, {"ok": True}))
    assert health._probe_mcp_connector() == ("MCP connector", "operational", "ready", False)


def test_both_replicas_stopped_reads_as_down(replicas) -> None:  # noqa: ANN001
    configure, _ = replicas
    configure(_closed_url(), _closed_url())
    name, state, detail, required = health._probe_mcp_connector()
    assert (name, state, required) == ("MCP connector", "down", False)
    assert detail.startswith("unreachable: ")


def test_a_replica_that_answers_not_ready_names_the_failing_check(replicas) -> None:  # noqa: ANN001
    configure, serve = replicas
    configure(_closed_url(), serve(503, {"checks": {"redis": False, "api": True}}))
    assert health._probe_mcp_connector() == ("MCP connector", "down", "redis", False)


def test_the_default_probes_both_colours() -> None:
    assert "127.0.0.1:8770/readyz" in health._MCP_HEALTH_URLS_DEFAULT
    assert "127.0.0.1:8771/readyz" in health._MCP_HEALTH_URLS_DEFAULT


def test_the_example_env_matches_the_default() -> None:
    # An environment copied from .env.example must not pin one colour again.
    line = next(
        line
        for line in (Path(__file__).resolve().parents[1] / ".env.example").read_text().splitlines()
        if line.startswith("XCELSIOR_MCP_HEALTH_URL=")
    )
    assert line.split("=", 1)[1] == health._MCP_HEALTH_URLS_DEFAULT
