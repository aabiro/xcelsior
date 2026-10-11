"""Tests for the /static/* endpoint (P1.1).

Verifies:
- GET /static/worker_agent.py returns 200 + non-empty body
- X-Xcelsior-Agent-SHA256 header matches sha256(body)
- Unknown files under /static/ return 404 (no directory traversal)
- Path traversal attempts are rejected
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_worker_agent_served_with_sha256_header():
    resp = client.get("/static/worker_agent.py")
    assert resp.status_code == 200, resp.text
    body = resp.content
    assert len(body) > 0
    assert b"VERSION" in body  # sanity: looks like the real file

    advertised = resp.headers.get("X-Xcelsior-Agent-SHA256")
    assert advertised, "missing X-Xcelsior-Agent-SHA256 header"
    assert advertised == hashlib.sha256(body).hexdigest()

    # Must match the file actually on disk
    on_disk = (REPO_ROOT / "worker_agent.py").read_bytes()
    assert body == on_disk


def test_worker_agent_content_type_is_python():
    resp = client.get("/static/worker_agent.py")
    assert "text/x-python" in resp.headers.get("content-type", "")


@pytest.mark.parametrize(
    "path",
    [
        "/static/api.py",
        "/static/scheduler.py",
        "/static/does_not_exist.py",
        "/static/README.md",
    ],
)
def test_unknown_files_return_404(path: str):
    resp = client.get(path)
    assert resp.status_code == 404


def test_directory_traversal_rejected():
    # FastAPI normalizes paths, but double-check common traversal shapes.
    for path in (
        "/static/..%2Fapi.py",
        "/static/%2e%2e%2fapi.py",
        "/static/subdir/worker_agent.py",
    ):
        resp = client.get(path)
        assert resp.status_code == 404, f"{path} should 404, got {resp.status_code}"


def test_head_request_supported():
    # Installers + nginx upstream probes may send HEAD.
    resp = client.head("/static/worker_agent.py")
    assert resp.status_code == 200
    assert resp.headers.get("X-Xcelsior-Agent-SHA256")
    # HEAD must carry no body (Starlette enforces this).
    assert resp.content == b""


def test_response_has_nosniff_header():
    resp = client.get("/static/worker_agent.py")
    assert resp.headers.get("x-content-type-options") == "nosniff"


def test_route_does_not_set_duplicate_cache_control():
    # nginx owns Cache-Control (set with `always` directive). The route must
    # not set its own, otherwise clients see two conflicting values.
    resp = client.get("/static/worker_agent.py")
    assert "cache-control" not in {k.lower() for k in resp.headers.keys()}


def test_static_path_exempt_from_auth_middleware():
    # The global auth middleware must allow /static/* unauthenticated. We
    # detect regressions by asserting no WWW-Authenticate / 401 on a fresh
    # client with no credentials.
    resp = client.get("/static/worker_agent.py")
    assert resp.status_code != 401


@pytest.mark.parametrize(
    "name",
    [
        "worker_image_cache.py",
        "worker_nfs.py",
        "worker_nvme_cache.py",
        "worker_luks_volumes.py",
    ],
)
def test_worker_split_modules_served_and_signed(name: str):
    # These modules were extracted out of worker_agent.py and are imported by
    # it at load time, so installers/self-update must fetch + verify them
    # exactly like worker_agent.py itself (B6 supply-chain protection).
    resp = client.get(f"/static/{name}")
    assert resp.status_code == 200, resp.text
    body = resp.content
    assert len(body) > 0

    advertised_sha = resp.headers.get("X-Xcelsior-Agent-SHA256")
    assert advertised_sha == hashlib.sha256(body).hexdigest()

    on_disk = (REPO_ROOT / name).read_bytes()
    assert body == on_disk

    assert resp.headers.get(
        "X-Xcelsior-Agent-Signature"
    ), f"{name} must carry a signature header (it's in _SIGNABLE)"


# ── Installer delivery (2026-10-11) ───────────────────────────────────
# `curl -fsSL https://xcelsior.ca/install.sh | bash` returned 404, and the
# installer's downloads other than worker_agent.py 404'd at nginx, while an
# installed agent could not start without env_config.py. These keep the
# installer, the API whitelist, the signer and nginx in step.

import ast  # noqa: E402
import re  # noqa: E402

from routes import static as static_routes  # noqa: E402

INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _installer_downloads() -> set[str]:
    return set(re.findall(r'download_static_file "([^"]+)"', INSTALL_SH.read_text()))


def _installer_signable() -> set[str]:
    m = re.search(r'^SIGNABLE_FILES="([^"]+)"', INSTALL_SH.read_text(), re.M)
    assert m, "SIGNABLE_FILES not found in install.sh"
    return set(m.group(1).split())


def test_installer_served_from_repo():
    resp = client.get("/install.sh")
    assert resp.status_code == 200, resp.text
    assert resp.content == INSTALL_SH.read_bytes()
    assert resp.headers["X-Xcelsior-Agent-SHA256"] == hashlib.sha256(resp.content).hexdigest()
    assert "shellscript" in resp.headers["content-type"]


def test_every_installer_download_is_served():
    for name in sorted(_installer_downloads()):
        resp = client.get(f"/static/{name}")
        assert (
            resp.status_code == 200
        ), f"install.sh downloads {name}, API answers {resp.status_code}"


def test_signable_lists_agree():
    signer = ast.literal_eval(
        re.search(
            r"SIGNABLE = (\([^)]*\))", (REPO_ROOT / "scripts" / "sign-static-agent.py").read_text()
        ).group(1)
    )
    assert _installer_signable() == set(static_routes._SIGNABLE) == set(signer)
    assert {n for n in _installer_downloads() if n.endswith(".py")} == _installer_signable()


def test_nginx_forwards_exactly_the_installer_files():
    conf = (REPO_ROOT / "nginx" / "xcelsior.conf").read_text()
    assert "location = /install.sh {" in conf
    m = re.search(r"location ~ (\^/static/\([^)]*\)\$) \{", conf)
    assert m, "no /static/ regex location in nginx/xcelsior.conf"
    pattern = re.compile(m.group(1))
    for name in static_routes._ALLOWED:
        assert pattern.match(f"/static/{name}"), f"nginx does not forward /static/{name}"
    for name in ("api.py", "db.py", "worker_agent.py.sig", "worker_agent.pyc"):
        assert not pattern.match(f"/static/{name}")


def _module_level_local_imports(module: str, local: set[str]) -> set[str]:
    tree = ast.parse((REPO_ROOT / f"{module}.py").read_text())
    found = set()
    for node in tree.body:  # only unconditional imports run at load time
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found & local


def test_agent_load_time_imports_are_all_shipped():
    local = {p.stem for p in REPO_ROOT.glob("*.py")}
    shipped = {Path(n).stem for n in _installer_downloads() if n.endswith(".py")}
    seen, todo = set(), ["worker_agent"]
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        for dep in _module_level_local_imports(module, local):
            assert (
                dep in shipped
            ), f"{module}.py imports {dep} at load time, but install.sh does not ship it"
            todo.append(dep)
