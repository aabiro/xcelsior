"""otel.xcelsior.ca is a public door into the private collector; it must stay narrow.

Claude Cowork exports its events from a VM on a user's machine or on
Anthropic's servers, so the endpoint has to be public. What keeps that safe is
all in `api/nginx/otel-xcelsior.conf`, so these tests run that file in a real
nginx in front of a stand-in collector and check, from the outside:

* no token installed on the host → every request refused (503), so the vhost
  can be deployed before the operator writes the secret snippet;
* a wrong or missing bearer token → 401, and the collector sees nothing;
* the right token → the OTLP body reaches the collector at `/v1/logs`, with
  the shared token stripped;
* anything but `POST /v1/logs` → refused, so traces and metrics paths on the
  same receiver stay private.
"""

from __future__ import annotations

import http.server
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api" / "scripts"))

import render_nginx_vhosts as rnv  # noqa: E402

TOKEN = "a" * 64


class _Collector(http.server.BaseHTTPRequestHandler):
    seen: list[tuple[str, str, str | None, bytes]] = []

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).seen.append((self.command, self.path, self.headers.get("Authorization"), body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    do_GET = do_POST  # noqa: N815

    def log_message(self, *_):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ingest(tmp_path):
    nginx, openssl = shutil.which("nginx"), shutil.which("openssl")
    if not nginx or not openssl:
        pytest.skip("needs the nginx and openssl binaries")
    (tmp_path / "logs").mkdir()
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            "/CN=otel.xcelsior.ca",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    _Collector.seen = []
    collector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=collector.serve_forever, daemon=True).start()
    tls = _free_port()

    conf = rnv.render(
        (ROOT / "api" / "nginx" / "otel-xcelsior.conf").read_text(),
        topology="direct",
        api_live="green",
        mcp_live="green",
    )
    conf = conf.replace("127.0.0.1:4318", f"127.0.0.1:{collector.server_address[1]}")
    conf = re.sub(r"/etc/letsencrypt/live/[^;]*/fullchain\.pem", str(cert), conf)
    conf = re.sub(r"/etc/letsencrypt/live/[^;]*/privkey\.pem", str(key), conf)
    conf = conf.replace(
        "/etc/nginx/snippets/xcelsior-abuse-deny*.conf", f"{tmp_path}/abuse-deny*.conf"
    )
    conf = conf.replace("/etc/nginx/snippets/otel-ingest-token.conf*", f"{tmp_path}/token.conf*")
    conf = re.sub(r"listen \[::\]:\d+\b[^;]*;", "", conf)
    conf = re.sub(r"listen 80\b[^;]*;", lambda _m: f"listen 127.0.0.1:{_free_port()};", conf)
    conf = re.sub(r"listen 443\b", f"listen 127.0.0.1:{tls}", conf)
    (tmp_path / "site.conf").write_text(conf)
    (tmp_path / "nginx.conf").write_text(
        f"pid {tmp_path}/nginx.pid;\nerror_log {tmp_path}/logs/error.log;\nevents {{}}\n"
        f"http {{\n  access_log {tmp_path}/logs/access.log;\n"
        f"  client_body_temp_path {tmp_path}/body;\n  proxy_temp_path {tmp_path}/proxy;\n"
        f"  include {tmp_path}/site.conf;\n}}\n"
    )

    def start(token: str | None):
        if token is not None:
            (tmp_path / "token.conf").write_text(f'set $xcelsior_otel_ingest_token "{token}";\n')
        r = subprocess.run(
            [nginx, "-p", str(tmp_path), "-c", str(tmp_path / "nginx.conf")],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", tls), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.05)

    def request(method: str, path: str, auth: str | None = None, body: bytes = b"{}") -> int:
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        head = f"{method} {path} HTTP/1.1\r\nHost: otel.xcelsior.ca\r\nConnection: close\r\n"
        head += "Content-Type: application/x-protobuf\r\n"
        if auth is not None:
            head += f"Authorization: {auth}\r\n"
        head += f"Content-Length: {len(body)}\r\n\r\n"
        with ctx.wrap_socket(
            socket.create_connection(("127.0.0.1", tls), timeout=5),
            server_hostname="otel.xcelsior.ca",
        ) as s:
            s.sendall(head.encode() + body)
            data = b""
            while chunk := s.recv(65536):
                data += chunk
        return int(data.split(b"\r\n", 1)[0].split()[1])

    yield start, request
    subprocess.run(
        [nginx, "-p", str(tmp_path), "-c", str(tmp_path / "nginx.conf"), "-s", "stop"],
        capture_output=True,
    )
    collector.shutdown()


def test_without_a_token_on_the_host_everything_is_refused(ingest):
    start, request = ingest
    start(None)
    assert request("POST", "/v1/logs", f"Bearer {TOKEN}") == 503
    assert request("POST", "/v1/logs", "Bearer ") == 503
    assert _Collector.seen == []


def test_a_wrong_or_missing_token_never_reaches_the_collector(ingest):
    start, request = ingest
    start(TOKEN)
    assert request("POST", "/v1/logs") == 401
    assert request("POST", "/v1/logs", "Bearer " + "b" * 64) == 401
    assert request("POST", "/v1/logs", TOKEN) == 401
    assert request("POST", "/v1/logs", "Bearer ") == 401
    assert _Collector.seen == []


def test_the_right_token_delivers_the_logs_without_the_secret(ingest):
    start, request = ingest
    start(TOKEN)
    assert request("POST", "/v1/logs", f"Bearer {TOKEN}", body=b"\x0a\x00otlp") == 200
    assert [(m, p, a, b) for m, p, a, b in _Collector.seen] == [
        ("POST", "/v1/logs", None, b"\x0a\x00otlp")
    ]


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/v1/logs"),
        ("POST", "/v1/traces"),
        ("POST", "/v1/metrics"),
        ("POST", "/"),
        ("POST", "/v1/logs/x"),
    ],
)
def test_only_post_v1_logs_is_served(ingest, method, path):
    start, request = ingest
    start(TOKEN)
    assert request(method, path, f"Bearer {TOKEN}") in (403, 404)
    assert _Collector.seen == []
