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
