"""Real local receivers; DNS policy overrides never escape these tests."""

import datetime
import json
import socket
import ssl
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import public_http as ph


@contextmanager
def receiver(monkeypatch, *, status=200, body=b"ok", slow=False, tls=None):
    received = []
    stop = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.do_POST()

        def do_POST(self):
            received.append(
                (
                    self.path,
                    dict(self.headers),
                    self.rfile.read(int(self.headers.get("Content-Length", 0))),
                )
            )
            try:
                if slow:
                    # A byte every 20ms defeats an ordinary per-read timeout.
                    for char in b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok":
                        self.wfile.write(bytes([char]))
                        self.wfile.flush()
                        if stop.wait(0.02):
                            break
                    return
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/json")
                if 300 <= status < 400:
                    self.send_header("Location", "/forbidden-redirect")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if tls:
        server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    real_dns = socket.getaddrinfo
    calls = []

    def dns(host, port, **kwargs):
        calls.append(host)
        return real_dns("127.0.0.1", port, **kwargs)

    monkeypatch.setattr(ph.socket, "getaddrinfo", dns)
    monkeypatch.setattr(ph, "_public_address", lambda addr: addr == "127.0.0.1")
    try:
        yield f"{'https' if tls else 'http'}://callback.test:{server.server_port}/hook", received, calls
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "::",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "64:ff9b::a00:1",
        "2002:0a00:0001::",
    ],
)
def test_private_and_translation_addresses_are_blocked(ip):
    assert not ph._public_address(ip)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/hook",
        "http://user:secret@example.com/",
        "http://example.com:0/",
        "http://example.com:65536/",
        "http://example.com:bad/",
        "http://example.com:/",
        "http://[broken/",
        "http://example.com/\r\nX:secret",
        "http://example.com/#fragment",
        "http://[fe80::1%25eth0]/",
        "http://a\\b/",
    ],
)
def test_invalid_urls_never_resolve(monkeypatch, url):
    monkeypatch.setattr(ph, "_resolve", lambda *args: pytest.fail("DNS on invalid URL"))
    with pytest.raises(ph.PublicHTTPError):
        ph.request(url)


def test_mixed_or_empty_dns_answers_fail_before_connect(monkeypatch):
    monkeypatch.setattr(ph._PinnedConnection, "connect", lambda *_: pytest.fail("connected"))
    for answers in (
        [],
        [
            (socket.AF_INET, 1, 6, "", ("93.184.216.34", 80)),
            (socket.AF_INET6, 1, 6, "", ("::ffff:127.0.0.1", 80, 0, 0)),
        ],
    ):
        monkeypatch.setattr(ph, "_resolve", lambda *args: answers)
        with pytest.raises(ph.PublicHTTPError):
            ph.request("http://callback.test/")


def test_dns_is_pinned_and_proxy_environment_is_ignored(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    with receiver(monkeypatch) as (url, received, calls):
        initial_dns = ph.socket.getaddrinfo

        def rebind(*args, **kwargs):
            if calls:
                pytest.fail("a second resolution could rebind to an internal address")
            return initial_dns(*args, **kwargs)

        monkeypatch.setattr(ph.socket, "getaddrinfo", rebind)
        assert ph.request(url, method="POST", body=b"signed payload").status == 200
        assert len(calls) == 1
        assert received[0][1]["Host"].startswith("callback.test:")
        assert received[0][2] == b"signed payload"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_are_never_followed(monkeypatch, status):
    with receiver(monkeypatch, status=status) as (url, received, calls):
        assert ph.request(url, method="POST", body=b"secret").status == status
        assert len(received) == len(calls) == 1


def test_body_limit_is_enforced(monkeypatch):
    with receiver(monkeypatch, body=b"x" * 1000) as (url, _, _):
        with pytest.raises(ph.PublicHTTPError, match="too large"):
            ph.request(url, max_response_bytes=32)
        assert ph.request(url, max_response_bytes=0).body == b""


def test_absolute_deadline_stops_slow_headers(monkeypatch):
    with receiver(monkeypatch, slow=True) as (url, _, _):
        started = time.monotonic()
        with pytest.raises(ph.PublicHTTPError):
            ph.request(url, timeout=0.12)
        assert time.monotonic() - started < 0.6


def test_dns_timeout_and_saturation_are_bounded(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(ph, "_resolver_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(ph.socket, "getaddrinfo", lambda *a, **k: release.wait(2))
    try:
        with pytest.raises(ph.PublicHTTPError, match="timed out"):
            ph.request("http://callback.test/", timeout=0.02)
        with pytest.raises(ph.PublicHTTPError, match="resolver is busy"):
            ph.request("http://callback.test/", timeout=0.02)
    finally:
        release.set()


def test_https_keeps_sni_and_verifies_hostname(monkeypatch, tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "callback.test")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("callback.test")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    names = []
    context.set_servername_callback(lambda sock, name, ctx: names.append(name))
    real_context = ssl.create_default_context
    monkeypatch.setattr(
        ph.ssl, "create_default_context", lambda: real_context(cafile=str(cert_path))
    )
    with receiver(monkeypatch, tls=context) as (url, received, _):
        assert ph.request(url).status == 200
        assert names == ["callback.test"]
        with pytest.raises(ph.PublicHTTPError):
            ph.request(url.replace("callback.test", "wrong.test"))
        assert len(received) == 1


def test_webhook_signature_and_retry_resolves_again(monkeypatch):
    from serverless import webhooks as wh

    monkeypatch.setenv("XCELSIOR_SERVERLESS_WEBHOOK_HMAC_SECRET", "isolated-test-key")
    with receiver(monkeypatch) as (url, received, calls):
        assert wh._post_webhook(url, {"id": "synthetic"}) == (True, "")
        path, headers, body = received[0]
        assert json.loads(body) == {"id": "synthetic"}
        assert wh.verify_signature(
            body,
            int(headers["X-Xcelsior-Timestamp"]),
            headers["X-Xcelsior-Signature"].split("=", 1)[1],
        )
        monkeypatch.setattr(
            ph, "_resolve", lambda *a: [(socket.AF_INET, 1, 6, "", ("10.0.0.1", 80))]
        )
        ok, error = wh._post_webhook(url, {"id": "synthetic"})
        assert not ok and "blocked" in error
        assert len(received) == 1


def test_production_has_no_known_signing_secret(monkeypatch):
    import scheduler
    from serverless import webhooks as wh

    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.delenv("XCELSIOR_SERVERLESS_WEBHOOK_HMAC_SECRET", raising=False)
    monkeypatch.setattr(scheduler, "API_TOKEN", "")
    with pytest.raises(ph.PublicHTTPError, match="not configured"):
        wh.sign_payload(b"{}", 1)


def test_cimd_uses_bounded_transport_and_redacts_failures(monkeypatch):
    import oauth_registration as reg
    from oauth_service import OAuthGrantError

    def fetch(url, **kwargs):
        assert kwargs["max_response_bytes"] == reg.CIMD_MAX_DOCUMENT_BYTES
        return ph.Response(200, {"content-type": "application/json"}, b'{"client_name":"test"}')

    monkeypatch.setattr(reg, "public_request", fetch)
    assert reg._fetch_cimd_document("https://callback.test/client") == {"client_name": "test"}

    def fail(*a, **k):
        raise ph.PublicHTTPError("secret-location")

    monkeypatch.setattr(reg, "public_request", fail)
    with pytest.raises(OAuthGrantError) as exc:
        reg._fetch_cimd_document("https://callback.test/client?token=secret")
    assert "secret" not in str(exc.value)
