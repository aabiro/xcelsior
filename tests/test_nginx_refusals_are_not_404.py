"""nginx must never tell a visitor "404 Not Found" when it means "busy" or "down".

Found on production on 2026-10-06, from a signed-in Playwright pass that saw
routes which certainly exist (`/api/auth/sessions`, `/api/v2/serverless/enabled`,
the dashboard's own RSC payloads, `/sw.js`) answer 404 at random.

Two faults compounded:

* `limit_conn xcelsior_conn 20`. Over HTTP/2 every stream counts, and one
  signed-in dashboard load measured 40-44 concurrent requests: API calls, an
  RSC prefetch per sidebar link, the service worker, the open event stream.
  A single visitor loading a single page was refused on every load.
* Refusals are 503 by default, and `error_page 502 503 504 /50x.html` named a
  file production does not have. nginx then served its own 404 page.

So a busy or briefly-down origin read, to the browser and to this app's own
code, as "that does not exist": serverless features hid themselves, the
service worker failed to register, and pages rendered half-empty.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import socket
import ssl
import subprocess
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / "nginx" / "xcelsior.conf"

#: One signed-in dashboard load, measured against production on 2026-10-06.
MEASURED_CONCURRENT_REQUESTS_PER_PAGE = 44


def _conf() -> str:
    return SITE.read_text(encoding="utf-8")


def _directives() -> str:
    """The config without comments, which explain these faults by name."""
    return "\n".join(line.split("#", 1)[0] for line in _conf().splitlines())


def test_one_page_load_fits_under_the_per_visitor_connection_cap():
    caps = [int(n) for n in re.findall(r"^\s*limit_conn\s+\S+\s+(\d+);", _directives(), re.M)]
    assert caps, "no limit_conn in the site config; the guard has nothing to read"
    # Two tabs open at once is ordinary use.
    assert min(caps) >= 2 * MEASURED_CONCURRENT_REQUESTS_PER_PAGE, (
        f"limit_conn {min(caps)} refuses a visitor with two dashboard tabs open "
        f"({MEASURED_CONCURRENT_REQUESTS_PER_PAGE} concurrent requests each)"
    )


def test_refusals_say_slow_down():
    conf = _directives()
    assert re.search(r"^\s*limit_conn_status\s+429;", conf, re.M), "limit_conn refusals are not 429"
    assert re.search(r"^\s*limit_req_status\s+429;", conf, re.M), "limit_req refusals are not 429"


def test_no_error_page_depends_on_a_file_outside_the_repo():
    """The page that went missing lived in /usr/share/nginx/html on the host."""
    assert "/usr/share/nginx/html" not in _directives()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_nginx(tmp_path):
    nginx, openssl = shutil.which("nginx"), shutil.which("openssl")
    if not nginx or not openssl:
        pytest.skip("needs the nginx and openssl binaries")
    (tmp_path / "logs").mkdir()
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=xcelsior.ca",
         "-days", "1", "-keyout", str(key), "-out", str(cert)],
        check=True, capture_output=True,
    )
    dead = _free_port()  # nothing listens here: every upstream is "down"
    tls_port = _free_port()
    conf = _conf()
    conf = re.sub(r"/etc/letsencrypt/live/[^;]*/(fullchain|chain)\.pem", str(cert), conf)
    conf = re.sub(r"/etc/letsencrypt/live/[^;]*/privkey\.pem", str(key), conf)
    conf = conf.replace("/var/log/nginx/", f"{tmp_path}/logs/")
    conf = re.sub(r"server 127\.0\.0\.1:\d+", f"server 127.0.0.1:{dead}", conf)
    conf = re.sub(r"listen \[::\]:\d+\b[^;]*;", "", conf)
    conf = re.sub(r"listen 80\b[^;]*;", lambda _m: f"listen 127.0.0.1:{_free_port()};", conf)
    conf = re.sub(r"listen (443|8444)\b", f"listen 127.0.0.1:{tls_port}", conf)
    conf = re.sub(r"listen 127\.0\.0\.1:9555;", lambda _m: f"listen 127.0.0.1:{_free_port()};", conf)
    tls_listen = re.search(rf"listen 127\.0\.0\.1:{tls_port}[^;]*;", conf)
    assert tls_listen, "the TLS server block was not found to rebind"
    (tmp_path / "site.conf").write_text(conf)
    (tmp_path / "nginx.conf").write_text(
        f"pid {tmp_path}/nginx.pid;\nerror_log {tmp_path}/logs/error.log;\nevents {{}}\n"
        f"http {{\n  access_log {tmp_path}/logs/access.log;\n  include {tmp_path}/site.conf;\n}}\n"
    )
    started = subprocess.run(
        [nginx, "-p", str(tmp_path), "-c", str(tmp_path / "nginx.conf")], capture_output=True, text=True
    )
    if started.returncode != 0:
        pytest.fail(f"nginx would not start on the repo config:\n{started.stderr}")
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", tls_port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        yield tls_port, "proxy_protocol" in tls_listen.group(0)
    finally:
        subprocess.run([nginx, "-p", str(tmp_path), "-c", str(tmp_path / "nginx.conf"), "-s", "stop"], capture_output=True)


def _get(port: int, proxy_protocol: bool, path: str) -> tuple[int, dict[str, str], str]:
    raw = socket.create_connection(("127.0.0.1", port), timeout=5)
    if proxy_protocol:
        raw.sendall(f"PROXY TCP4 127.0.0.1 127.0.0.1 40000 {port}\r\n".encode())
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with ctx.wrap_socket(raw, server_hostname="xcelsior.ca") as tls:
        tls.sendall(f"GET {path} HTTP/1.1\r\nHost: xcelsior.ca\r\nConnection: close\r\n\r\n".encode())
        data = b""
        while chunk := tls.recv(65536):
            data += chunk
    head, _, body = data.decode("utf-8", "replace").partition("\r\n\r\n")
    lines = head.split("\r\n")
    status = int(lines[0].split()[1])
    headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:])}
    return status, headers, body


@pytest.mark.parametrize("path", ["/api/v2/serverless/enabled", "/dashboard", "/sw.js"])
def test_an_unreachable_origin_is_reported_as_unavailable(running_nginx, path):
    port, proxy_protocol = running_nginx
    status, headers, body = _get(port, proxy_protocol, path)
    assert status != 404, f"{path} with every upstream down answered 404 Not Found"
    # error_page without `=code` keeps the upstream's own status: 502 when it
    # cannot be reached, 503 when it is overloaded. Either is the truth.
    assert status in (502, 503, 504), f"{path} answered {status}"
    assert "retry-after" in headers, "a 503 without Retry-After gives a client nothing to act on"
    assert "briefly unavailable" in body
