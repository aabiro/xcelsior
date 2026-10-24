"""Authenticated provenance survives threadpool dispatch and ends with its request."""

import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from db import DatabaseOps, _get_pg_pool
from control_plane.outbox import append_event
from control_plane.request_context import EventRequestContextMiddleware, event_context


def test_events_capture_verified_principal_without_accepting_header_identity(monkeypatch):
    from routes import _deps
    import structured_logging

    monkeypatch.setattr(
        _deps,
        "_get_current_user",
        lambda _: {"user_id": "verified-user", "client_id": "verified-client"},
    )
    monkeypatch.setattr(
        structured_logging, "_get_active_trace_context", lambda: ("a" * 32, "b" * 16)
    )
    app = FastAPI()
    app.add_middleware(EventRequestContextMiddleware)

    @app.get("/{mode}")
    def issue(mode: str, request: Request):
        if mode == "authenticated":
            _deps._require_auth(request)
        job = uuid.uuid4().hex
        with _get_pg_pool().connection() as conn:
            DatabaseOps.upsert_job(
                conn, {"job_id": job, "owner": "resource-owner"}, backend="postgres"
            )
            eid = append_event(
                conn,
                aggregate_type="job",
                aggregate_id=job,
                event_type="job.v1.submitted",
                headers={"actor_id": "forged-argument"} if mode == "authenticated" else {},
            )
            row = conn.execute(
                "SELECT headers,trace_id,tenant_id FROM outbox_events WHERE event_id=%s", (eid,)
            ).fetchone()
            conn.rollback()
        return {"metadata": row[0], "trace_id": row[1], "tenant_id": row[2]}

    with TestClient(app) as client:
        first = client.get(
            "/authenticated",
            headers={"x-actor-id": "forged-header", "x-client-id": "forged-header"},
        ).json()
        second = client.get("/unattributed", headers={"x-actor-id": "forged-header"}).json()
    assert first["metadata"]["actor_id"] == "verified-user"
    assert first["metadata"]["client_id"] == "verified-client"
    assert first["metadata"]["request_id"] != second["metadata"]["request_id"]
    assert "actor_id" not in second["metadata"] and "client_id" not in second["metadata"]
    assert first["trace_id"] == second["trace_id"] == "a" * 32
    assert first["tenant_id"] == second["tenant_id"] == "resource-owner"
    assert "actor_id" not in event_context() and "request_id" not in event_context()


def test_verified_context_crosses_sync_dependency_to_async_endpoint(monkeypatch):
    from routes import _deps

    monkeypatch.setattr(
        _deps,
        "_get_current_user",
        lambda _: {"user_id": "sync-verified-user", "client_id": "sync-verified-client"},
    )
    app = FastAPI()
    app.add_middleware(EventRequestContextMiddleware)

    @app.get("/")
    async def inspect(user=Depends(_deps._require_auth)):
        return event_context()

    with TestClient(app) as client:
        metadata = client.get("/").json()
    assert metadata["actor_id"] == "sync-verified-user"
    assert metadata["client_id"] == "sync-verified-client"
    assert "actor_id" not in event_context() and "request_id" not in event_context()
