"""Durable output must remain visible when an API process has stale memory."""

import time
import uuid
from contextlib import contextmanager

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from db import _get_pg_pool
from routes import instances


@pytest.fixture
def authorized(monkeypatch):
    monkeypatch.setattr(instances, "_require_auth", lambda request: {"user_id": "test"})
    monkeypatch.setattr(instances, "_require_scope", lambda *args: None)
    monkeypatch.setattr(instances, "_check_job_access", lambda *args: None)
    return Request({"type": "http", "headers": []})


def test_json_and_download_include_another_processes_persisted_lines(authorized):
    job_id = "log-proof-" + uuid.uuid4().hex
    instances._job_log_buffers[job_id] = [{"message": "stale process buffer"}]
    try:
        with _get_pg_pool().connection() as c:
            c.execute(
                "INSERT INTO job_logs(job_id,ts,level,line) VALUES(%s,%s,'info','durable CUDA output')",
                (job_id, time.time()),
            )
        result = instances.api_instance_logs(job_id, authorized, limit=100)
        assert [row["message"] for row in result["logs"]] == ["durable CUDA output"]
        response = instances.api_instance_logs_download(job_id, authorized)
        assert b"durable CUDA output" in response.body
        assert b"stale process buffer" not in response.body
    finally:
        instances._job_log_buffers.pop(job_id, None)


def test_durable_read_failure_does_not_return_stale_memory(monkeypatch, authorized):
    import db

    class UnavailablePool:
        @contextmanager
        def connection(self):
            raise RuntimeError("database unavailable")
            yield

    monkeypatch.setattr(db, "_get_pg_pool", lambda: UnavailablePool())
    instances._job_log_buffers["outage"] = [{"message": "stale"}]
    try:
        for call in (
            lambda: instances.api_instance_logs("outage", authorized, limit=10),
            lambda: instances.api_instance_logs_download("outage", authorized),
        ):
            with pytest.raises(HTTPException) as exc:
                call()
            assert exc.value.status_code == 503
    finally:
        instances._job_log_buffers.pop("outage", None)


def test_download_refuses_missing_read_scope_before_loading_any_logs(monkeypatch):
    principal = {
        "user_id": "owner",
        "auth_type": "client_credentials_jwt",
        "grant_type": "client_credentials",
        "scopes": ["billing:read"],
    }
    monkeypatch.setattr(instances, "_require_auth", lambda request: principal)
    loaded = []
    monkeypatch.setattr(instances, "_load_pg_logs", lambda *a, **k: loaded.append(True))
    with pytest.raises(HTTPException) as exc:
        instances.api_instance_logs_download("owned-job", Request({"type": "http", "headers": []}))
    assert exc.value.status_code == 403
    assert not loaded
