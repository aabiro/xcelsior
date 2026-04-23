"""P2.3 — Jupyter / VSCode auto-launch tests.

These tests verify the worker-side `_run_auto_launch` dispatcher:

* A job with no ``auto_launch`` field is a no-op (no docker exec).
* ``auto_launch=["jupyter"]`` issues exactly one `docker exec` whose
  shell payload installs + starts JupyterLab on :8888 with the
  per-instance token.
* ``auto_launch=["vscode"]`` issues exactly one `docker exec` whose
  shell payload starts code-server on :8443 with PASSWORD.
* Non-interactive jobs are never auto-launched.
* Unknown services are rejected (warning log, no exec).

The `subprocess.run` boundary is monkeypatched so the tests never
actually fork docker.
"""
from __future__ import annotations

import subprocess
from typing import Any, List

import pytest


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> List[List[str]]:
    import worker_agent as wa

    calls: List[List[str]] = []

    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd: Any, *a: Any, **kw: Any) -> _FakeCompleted:  # type: ignore[override]
        calls.append(list(cmd))
        return _FakeCompleted()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    # Silence the log-shipping helper so tests don't hit the network.
    monkeypatch.setattr(wa, "_push_log_lines", lambda *a, **kw: None)

    # Don't call the report endpoint during tests.
    monkeypatch.setenv("XCELSIOR_API_URL", "")
    return calls


def test_no_auto_launch_is_noop(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True},
    )
    assert captured_calls == []


def test_auto_launch_requires_interactive(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": False, "auto_launch": ["jupyter"]},
    )
    assert captured_calls == []


def test_jupyter_launch(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": ["jupyter"]},
    )
    assert len(captured_calls) == 1
    cmd = captured_calls[0]
    # docker exec -d <container> bash -lc "<shell>"
    assert cmd[:3] == ["docker", "exec", "-d"]
    assert cmd[3] == "container1"
    assert cmd[4:6] == ["bash", "-lc"]
    shell = cmd[6]
    assert "jupyter lab" in shell
    assert "--ip=0.0.0.0" in shell
    assert "--port=8888" in shell
    assert "--ServerApp.token=" in shell
    assert "--allow-root" in shell


def test_vscode_launch(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": ["vscode"]},
    )
    assert len(captured_calls) == 1
    shell = captured_calls[0][6]
    assert "code-server" in shell
    assert "0.0.0.0:8443" in shell
    assert "PASSWORD=" in shell


def test_both_services_launch(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": ["jupyter", "vscode"]},
    )
    assert len(captured_calls) == 2
    shells = [c[6] for c in captured_calls]
    assert any("jupyter lab" in s for s in shells)
    assert any("code-server" in s for s in shells)


def test_unknown_service_skipped(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": ["ssh-tunnel"]},
    )
    assert captured_calls == []


def test_token_is_deterministic(monkeypatch: pytest.MonkeyPatch):
    import worker_agent as wa

    monkeypatch.setenv("HOST_SECRET", "s3cret")
    t1 = wa._auto_launch_token("jobA")
    t2 = wa._auto_launch_token("jobA")
    t3 = wa._auto_launch_token("jobB")

    assert t1 == t2
    assert t1 != t3
    assert len(t1) == 32


def test_csv_auto_launch_string_accepted(captured_calls):
    # scheduler sometimes passes a csv string instead of a list
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": "jupyter,vscode"},
    )
    assert len(captured_calls) == 2


# ═══════════════════════════════════════════════════════════════════════
# P2.3 Phase 2 — API-level integration tests
# ═══════════════════════════════════════════════════════════════════════
# These hit the real FastAPI app (against the test postgres) and exercise:
#   * JobIn model_validator that auto-injects 8888/8443 into exposed_ports
#   * POST /instances/{id}/auto-launch/report (token persistence)
#   * GET  /instances/{id}/auto-launch (owner auth + URL synthesis)

import json as _ijson
import os as _ios
import time as _itime
import uuid as _iuuid
from typing import Iterator as _IIterator

from fastapi.testclient import TestClient as _ITestClient

_ios.environ.setdefault("XCELSIOR_ENV", "test")
_ios.environ.setdefault("XCELSIOR_ALLOW_UNAUTH_AGENT", "1")
_ios.environ.setdefault("XCELSIOR_PUBLIC_HOST", "xcelsior.test")

import api as _iapi  # noqa: E402
import routes.instances as _irinst  # noqa: E402

_TOKEN_HEX = "a" * 32


# ── JobIn model-validator unit tests ────────────────────────────────────


def test_jobin_auto_launch_jupyter_adds_8888():
    j = _irinst.JobIn(name="t", auto_launch=["jupyter"])
    assert j.exposed_ports == [8888]


def test_jobin_auto_launch_vscode_adds_8443():
    j = _irinst.JobIn(name="t", auto_launch=["vscode"])
    assert j.exposed_ports == [8443]


def test_jobin_auto_launch_both_adds_both_ports():
    j = _irinst.JobIn(name="t", auto_launch=["jupyter", "vscode"])
    assert set(j.exposed_ports or []) == {8888, 8443}


def test_jobin_auto_launch_preserves_user_ports_no_dup():
    j = _irinst.JobIn(
        name="t", auto_launch=["jupyter"], exposed_ports=[7000, 8888]
    )
    assert j.exposed_ports == [7000, 8888]


def test_jobin_auto_launch_merges_user_ports():
    j = _irinst.JobIn(name="t", auto_launch=["jupyter"], exposed_ports=[7000])
    assert j.exposed_ports == [7000, 8888]


def test_jobin_auto_launch_rejects_unknown_service():
    with pytest.raises(Exception):
        _irinst.JobIn(name="t", auto_launch=["ssh"])


def test_jobin_auto_launch_dedups_case_insensitive():
    j = _irinst.JobIn(name="t", auto_launch=["Jupyter", "JUPYTER"])
    assert j.auto_launch == ["jupyter"]
    assert j.exposed_ports == [8888]


def test_jobin_auto_launch_none_leaves_ports_untouched():
    j = _irinst.JobIn(name="t", auto_launch=None, exposed_ports=[7000])
    assert j.exposed_ports == [7000]


# ── Endpoint integration tests ──────────────────────────────────────────


@pytest.fixture(scope="module")
def api_client() -> "_IIterator[_ITestClient]":
    original = _irinst._require_auth

    def _fake_auth(request):
        return {
            "email": "alice@test",
            "user_id": "alice",
            "customer_id": "",
            "role": "user",
            "is_admin": False,
        }

    _irinst._require_auth = _fake_auth  # type: ignore[assignment]
    try:
        with _ITestClient(_iapi.app) as c:
            yield c
    finally:
        _irinst._require_auth = original  # type: ignore[assignment]


@pytest.fixture
def api_seeded_job():
    from db import _get_pg_pool

    pool = _get_pg_pool()
    job_id = f"abcd{_iuuid.uuid4().hex[:20]}"
    host_id = f"host-al-{_iuuid.uuid4().hex[:8]}"
    now = _itime.time()
    host_payload = {
        "host_id": host_id,
        "status": "active",
        "owner": "alice",
        "registered_at": now,
        "ip": "100.64.1.42",
    }
    job_payload = {
        "job_id": job_id,
        "status": "running",
        "owner": "alice",
        "host": host_id,
        "host_id": host_id,
        "image": "python:3.11",
        "name": "al-test",
        "submitted_at": now,
        "auto_launch": ["jupyter", "vscode"],
        "exposed_ports": [8888, 8443],
    }
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload) "
            "VALUES (%s,%s,%s,%s::jsonb) "
            "ON CONFLICT (host_id) DO UPDATE SET payload=EXCLUDED.payload",
            (host_id, "active", now, _ijson.dumps(host_payload)),
        )
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, host_id, payload) "
            "VALUES (%s,%s,0,%s,%s,%s::jsonb)",
            (job_id, "running", now, host_id, _ijson.dumps(job_payload)),
        )
        conn.commit()

    yield {"job_id": job_id, "host_id": host_id, "slug": job_id[:12]}

    with pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
        conn.commit()


def _api_set_http_ports(job_id: str, ports: dict) -> None:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload = jsonb_set(payload, '{http_ports}', %s::jsonb, true) "
            "WHERE job_id=%s",
            (_ijson.dumps(ports), job_id),
        )
        conn.commit()


def test_report_persists_full_token(api_client, api_seeded_job):
    r = api_client.post(
        f"/instances/{api_seeded_job['job_id']}/auto-launch/report",
        json={
            "host_id": api_seeded_job["host_id"],
            "ports": {"jupyter": 8888, "vscode": 8443},
            "token_sha": "abcdef01",
            "token": _TOKEN_HEX,
        },
    )
    assert r.status_code == 200
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT payload->'auto_launch_ports' FROM jobs WHERE job_id=%s",
            (api_seeded_job["job_id"],),
        )
        row = cur.fetchone()
    alp = row[0] if not isinstance(row, dict) else list(row.values())[0]
    if isinstance(alp, str):
        alp = _ijson.loads(alp)
    assert alp["token"] == _TOKEN_HEX
    assert alp["ports"] == {"jupyter": 8888, "vscode": 8443}


def test_report_rejects_wrong_host(api_client, api_seeded_job):
    r = api_client.post(
        f"/instances/{api_seeded_job['job_id']}/auto-launch/report",
        json={
            "host_id": "other-host",
            "ports": {"jupyter": 8888},
            "token": _TOKEN_HEX,
        },
    )
    assert r.status_code == 404


def test_report_rejects_bad_token_format(api_client, api_seeded_job):
    r = api_client.post(
        f"/instances/{api_seeded_job['job_id']}/auto-launch/report",
        json={
            "host_id": api_seeded_job["host_id"],
            "ports": {"jupyter": 8888},
            "token": "not-hex!",
        },
    )
    assert r.status_code >= 400
    assert r.status_code != 200


def test_get_returns_urls_and_token(api_client, api_seeded_job):
    api_client.post(
        f"/instances/{api_seeded_job['job_id']}/auto-launch/report",
        json={
            "host_id": api_seeded_job["host_id"],
            "ports": {"jupyter": 8888, "vscode": 8443},
            "token_sha": "abcdef01",
            "token": _TOKEN_HEX,
        },
    )
    _api_set_http_ports(api_seeded_job["job_id"], {"8888": 55123, "8443": 55456})

    r = api_client.get(f"/instances/{api_seeded_job['job_id']}/auto-launch")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    svcs = body["services"]
    assert "jupyter" in svcs and "vscode" in svcs

    base = _ios.environ["XCELSIOR_PUBLIC_HOST"]
    j = svcs["jupyter"]
    assert j["container_port"] == 8888
    assert j["host_port"] == 55123
    assert j["token"] == _TOKEN_HEX
    assert j["url"] == f"https://{api_seeded_job['slug']}-8888.{base}/?token={_TOKEN_HEX}"

    v = svcs["vscode"]
    assert v["container_port"] == 8443
    assert v["host_port"] == 55456
    assert v["password"] == _TOKEN_HEX
    assert v["url"] == f"https://{api_seeded_job['slug']}-8443.{base}/"


def test_get_ready_false_before_report(api_client, api_seeded_job):
    r = api_client.get(f"/instances/{api_seeded_job['job_id']}/auto-launch")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is False
    assert body["services"] == {}


def test_get_ready_false_when_http_ports_missing(api_client, api_seeded_job):
    api_client.post(
        f"/instances/{api_seeded_job['job_id']}/auto-launch/report",
        json={
            "host_id": api_seeded_job["host_id"],
            "ports": {"jupyter": 8888},
            "token": _TOKEN_HEX,
        },
    )
    r = api_client.get(f"/instances/{api_seeded_job['job_id']}/auto-launch")
    assert r.status_code == 200
    assert r.json()["ready"] is False


def test_get_404_when_auto_launch_not_configured(api_client):
    from db import _get_pg_pool

    pool = _get_pg_pool()
    job_id = f"noal{_iuuid.uuid4().hex[:20]}"
    host_id = f"host-na-{_iuuid.uuid4().hex[:8]}"
    now = _itime.time()
    host_payload = {
        "host_id": host_id,
        "owner": "alice",
        "ip": "100.64.1.42",
        "registered_at": now,
    }
    job_payload = {
        "job_id": job_id,
        "status": "running",
        "owner": "alice",
        "host": host_id,
        "host_id": host_id,
        "name": "no-auto",
        "submitted_at": now,
        "exposed_ports": [7000],
    }
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload) "
            "VALUES (%s,%s,%s,%s::jsonb) "
            "ON CONFLICT (host_id) DO UPDATE SET payload=EXCLUDED.payload",
            (host_id, "active", now, _ijson.dumps(host_payload)),
        )
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, host_id, payload) "
            "VALUES (%s,%s,0,%s,%s,%s::jsonb)",
            (job_id, "running", now, host_id, _ijson.dumps(job_payload)),
        )
        conn.commit()
    try:
        r = api_client.get(f"/instances/{job_id}/auto-launch")
        assert r.status_code == 404
    finally:
        with pool.connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
            conn.commit()


def test_get_403_when_not_owner(api_client, api_seeded_job):
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload = jsonb_set(payload, '{owner}', '\"mallory\"'::jsonb) "
            "WHERE job_id=%s",
            (api_seeded_job["job_id"],),
        )
        conn.commit()

    r = api_client.get(f"/instances/{api_seeded_job['job_id']}/auto-launch")
    assert r.status_code == 403


def test_get_404_when_job_missing(api_client):
    r = api_client.get("/instances/does-not-exist-xyz/auto-launch")
    assert r.status_code == 404
