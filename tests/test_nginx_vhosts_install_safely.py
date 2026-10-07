"""The deploy installs the repo's nginx vhosts on the topology the host has.

Production does not run the SNI-passthrough router the repo's vhosts are
written for, so `deploy.sh` held every nginx install, and production's vhosts
drifted from the repo with each release. Found on 2026-10-06: a connection cap
of 20 refusing ordinary page loads, an error page pointing at a missing file
(both reached visitors as random 404s), and a headscale vhost still naming the
certificate the repo had moved off in 1624d67.

Now `scripts/render_nginx_vhosts.py` renders the vhosts for direct 443 or for
the router, with the upstreams on the live blue-green colours, and
`scripts/install_nginx_vhosts.sh` installs them on the host, restoring the
previous files if `nginx -t` fails or the site stops answering through nginx.
"""

from __future__ import annotations

import http.server
import os
import pathlib
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import render_nginx_vhosts as rnv  # noqa: E402

NGINX = ROOT / "nginx"
INSTALLER = ROOT / "scripts" / "install_nginx_vhosts.sh"


def _source(name: str) -> str:
    return (NGINX / f"{name}.conf").read_text()


def _directives(text: str) -> list[str]:
    return [line.split("#", 1)[0].rstrip() for line in text.splitlines() if line.split("#", 1)[0].strip()]


# ── Rendering ─────────────────────────────────────────────────────────────


def test_the_repo_is_written_for_the_router():
    """If this stops being true the direct rendering is untested, not unneeded."""
    assert any("proxy_protocol" in line for line in _directives(_source("xcelsior")) if "listen" in line)


@pytest.mark.parametrize("name", rnv.VHOSTS)
def test_direct_rendering_serves_tls_on_443_without_proxy_protocol(name):
    out = _directives(rnv.render(_source(name), topology="direct", api_live="green", mcp_live="green"))
    listens = [line for line in out if line.strip().startswith("listen")]
    assert not [line for line in listens if "proxy_protocol" in line]
    assert not [line for line in listens if "8444" in line]
    assert not [line for line in out if re.search(r"real_ip_header\s+proxy_protocol", line)]
    if any("ssl" in line for line in listens):
        assert any(re.search(r"\b443\b", line) for line in listens), f"{name} lost its TLS listener"


@pytest.mark.parametrize("name", rnv.VHOSTS)
def test_router_rendering_keeps_the_layout(name):
    src = _source(name)
    out = rnv.render(src, topology="router", api_live="green", mcp_live="green")
    assert [line for line in _directives(out) if "listen" in line] == [
        line for line in _directives(src) if "listen" in line
    ]


def _upstream(text: str, name: str) -> dict[str, bool]:
    body = re.search(rf"upstream\s+{name}\s*\{{(.*?)\}}", text, re.S).group(1)
    return {m.group(1): bool(m.group(2)) for m in re.finditer(r"server\s+(127\.0\.0\.1:\d+)(\s+backup)?;", body)}


@pytest.mark.parametrize("api", ["green", "blue"])
@pytest.mark.parametrize("mcp", ["green", "blue"])
def test_the_live_colours_are_the_primaries(api, mcp):
    out = rnv.render(_source("xcelsior"), topology="direct", api_live=api, mcp_live=mcp)
    live_api = "127.0.0.1:9500" if api == "green" else "127.0.0.1:9501"
    live_mcp = "127.0.0.1:8770" if mcp == "green" else "127.0.0.1:8771"
    assert [s for s, backup in _upstream(out, "xcelsior_api").items() if not backup] == [live_api]
    assert [s for s, backup in _upstream(out, "xcelsior_mcp").items() if not backup] == [live_mcp]


def test_an_upstream_missing_a_colour_is_refused():
    broken = _source("xcelsior").replace("server 127.0.0.1:9501 backup;", "")
    with pytest.raises(ValueError):
        rnv.render(broken, topology="direct", api_live="blue", mcp_live="green")


# ── The rendered config, in a real nginx ──────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Healthz(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"live")

    def log_message(self, *_):
        pass


def test_a_direct_rendering_routes_to_the_live_colour_in_real_nginx(tmp_path):
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
    live = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Healthz)
    threading.Thread(target=live.serve_forever, daemon=True).start()
    dead, tls = _free_port(), _free_port()

    conf = rnv.render(_source("xcelsior"), topology="direct", api_live="blue", mcp_live="blue")
    conf = conf.replace("127.0.0.1:9501", f"127.0.0.1:{live.server_address[1]}")  # blue is live
    conf = re.sub(r"127\.0\.0\.1:(9500|8770|8771|3000)\b", f"127.0.0.1:{dead}", conf)
    conf = re.sub(r"/etc/letsencrypt/live/[^;]*/(fullchain|chain)\.pem", str(cert), conf)
    conf = re.sub(r"/etc/letsencrypt/live/[^;]*/privkey\.pem", str(key), conf)
    conf = conf.replace("/var/log/nginx/", f"{tmp_path}/logs/")
    conf = re.sub(r"listen \[::\]:\d+\b[^;]*;", "", conf)
    conf = re.sub(r"listen 80\b[^;]*;", lambda _m: f"listen 127.0.0.1:{_free_port()};", conf)
    conf = re.sub(r"listen 443\b", f"listen 127.0.0.1:{tls}", conf)
    conf = re.sub(r"listen 127\.0\.0\.1:9555;", lambda _m: f"listen 127.0.0.1:{_free_port()};", conf)
    (tmp_path / "site.conf").write_text(conf)
    (tmp_path / "nginx.conf").write_text(
        f"pid {tmp_path}/nginx.pid;\nerror_log {tmp_path}/logs/error.log;\nevents {{}}\n"
        f"http {{\n  access_log {tmp_path}/logs/access.log;\n  include {tmp_path}/site.conf;\n}}\n"
    )
    started = subprocess.run([nginx, "-p", str(tmp_path), "-c", str(tmp_path / "nginx.conf")], capture_output=True, text=True)
    assert started.returncode == 0, started.stderr
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", tls), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        # No PROXY header: in the direct layout the client speaks TLS straight away.
        with ctx.wrap_socket(socket.create_connection(("127.0.0.1", tls), timeout=5), server_hostname="xcelsior.ca") as s:
            s.sendall(b"GET /healthz HTTP/1.1\r\nHost: xcelsior.ca\r\nConnection: close\r\n\r\n")
            data = b""
            while chunk := s.recv(65536):
                data += chunk
        status = int(data.split(b"\r\n", 1)[0].split()[1])
        assert status == 200 and b"\r\nlive\r\n" in data, data[:300]
    finally:
        subprocess.run([nginx, "-p", str(tmp_path), "-c", str(tmp_path / "nginx.conf"), "-s", "stop"], capture_output=True)
        live.shutdown()


# ── The installer, run for real against a scratch /etc/nginx ──────────────

STUBS = {
    "sudo": '#!/bin/sh\nexec "$@"\n',
    # `nginx -t` fails if any installed vhost is marked broken, as a bad config would.
    "nginx": '#!/bin/sh\nif [ "$1" = "-t" ]; then ! grep -qs BROKEN "$NGINX_DIR"/sites-available/*; exit $?; fi\nexit 0\n',
    "systemctl": '#!/bin/sh\necho "systemctl $*" >> "$STUB_LOG"\nexit 0\n',
    "curl": '#!/bin/sh\nprintf "%s" "$HEALTH_CODE"\n',
    "sleep": "#!/bin/sh\nexit 0\n",
}


@pytest.fixture
def host(tmp_path):
    bin_dir, etc, src, backups = tmp_path / "bin", tmp_path / "etc", tmp_path / "src", tmp_path / "backups"
    for d in (bin_dir, etc / "sites-available", etc / "sites-enabled", src, backups):
        d.mkdir(parents=True)
    for name, body in STUBS.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    for name in rnv.VHOSTS:
        (etc / "sites-available" / name).write_text(f"old {name}\n")
        (src / f"{name}.conf").write_text(f"new {name}\n")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "NGINX_DIR": str(etc),
        "SRC_DIR": str(src),
        "BACKUP_ROOT": str(backups),
        "STUB_LOG": str(tmp_path / "stub.log"),
        "HEALTH_CODE": "200",
    }
    return etc, src, env


def _install(env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(INSTALLER)], env=env, capture_output=True, text=True, timeout=30)


def _serving(etc) -> dict[str, str]:
    return {n: (etc / "sites-available" / n).read_text().strip() for n in rnv.VHOSTS}


def test_a_healthy_install_replaces_the_vhosts(host):
    etc, _, env = host
    r = _install(env)
    assert r.returncode == 0, r.stderr
    assert _serving(etc) == {n: f"new {n}" for n in rnv.VHOSTS}
    assert all((etc / "sites-enabled" / n).is_symlink() for n in rnv.VHOSTS)


def test_a_config_nginx_rejects_is_rolled_back(host):
    etc, src, env = host
    (src / "xcelsior.conf").write_text("BROKEN\n")
    r = _install(env)
    assert r.returncode != 0
    assert _serving(etc) == {n: f"old {n}" for n in rnv.VHOSTS}, "the previous vhosts must be serving again"


def test_a_site_that_stops_answering_is_rolled_back(host):
    """`nginx -t` passed on 2026-10-05 too; only a request through nginx catches that."""
    etc, _, env = host
    r = _install({**env, "HEALTH_CODE": "400"})
    assert r.returncode != 0
    assert "answered 400" in r.stderr
    assert _serving(etc) == {n: f"old {n}" for n in rnv.VHOSTS}
