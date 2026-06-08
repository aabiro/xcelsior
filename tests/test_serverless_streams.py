"""Phase 9 — SSE streaming hardening."""

import asyncio
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from api import app
from db import UserStore
from serverless.repo import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_IN_PROGRESS,
    EndpointCreate,
    ServerlessRepo,
)
from serverless.service import ServerlessService
from serverless.streams import (
    SSE_RESPONSE_HEADERS,
    async_live_job_stream,
    format_done_chunk,
    format_error_chunk,
    format_ping_comment,
    iter_job_stream,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)
    api_mod._RATE_BUCKETS.clear()


def _register_and_fund() -> dict:
    email = f"sls-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Streams"},
    )
    assert reg.status_code == 200
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    body = login.json()
    user = reg.json().get("user") or body.get("user") or {}
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    deposit = client.post(
        f"/api/billing/wallet/{user['customer_id']}/deposit",
        json={"amount_cad": 50.0},
        headers=headers,
    )
    assert deposit.status_code == 200
    assert UserStore.get_user(email) is not None
    return headers


@pytest.fixture(scope="module")
def user_headers():
    return _register_and_fund()


class TestStreamHelpers:
    def test_ping_comment_format(self):
        assert format_ping_comment() == ": ping\n\n"

    def test_done_and_error_chunks(self):
        assert "[DONE]" in format_done_chunk()
        assert "job_failed" in format_error_chunk("job_failed", "boom")

    def test_replay_after_seq(self):
        repo = ServerlessRepo()
        owner = f"cust-{uuid.uuid4().hex[:8]}"
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner, name="s", mode="custom", image_ref="img")
        )
        job = repo.enqueue_job(ep["endpoint_id"], owner, {})
        repo.append_stream_event(job["job_id"], "token", {"t": "a"})
        repo.append_stream_event(job["job_id"], "token", {"t": "b"})
        repo.complete_job(job["job_id"], output={"ok": True})

        frames = list(iter_job_stream(repo, job["job_id"], after_seq=1))
        body = "".join(frames)
        assert '"t": "b"' in body or '"t":"b"' in body.replace(" ", "")
        assert '"t": "a"' not in body.replace(" ", "")
        assert "[DONE]" in body
        repo.soft_delete_endpoint(ep["endpoint_id"], owner)

    def test_failed_job_emits_error_chunk(self):
        repo = ServerlessRepo()
        owner = f"cust-{uuid.uuid4().hex[:8]}"
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner, name="s", mode="custom", image_ref="img")
        )
        job = repo.enqueue_job(ep["endpoint_id"], owner, {})
        repo.complete_job(job["job_id"], error="boom")
        frames = list(iter_job_stream(repo, job["job_id"]))
        assert "job_failed" in "".join(frames)
        repo.soft_delete_endpoint(ep["endpoint_id"], owner)


class TestRunsyncOutput:
    def test_empty_output_returns_dict(self):
        assert ServerlessService.normalize_runsync_output({"output": None}) == {}
        assert ServerlessService.normalize_runsync_output({"output": {"x": 1}}) == {"x": 1}
        assert ServerlessService.normalize_runsync_output({"output": "text"}) == {
            "output": "text"
        }


class TestCancelInflight:
    def test_cancel_releases_worker_concurrency(self):
        repo = ServerlessRepo()
        svc = ServerlessService(repo)
        owner = f"cust-{uuid.uuid4().hex[:8]}"
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner, name="s", mode="custom", image_ref="img")
        )
        ep_id = ep["endpoint_id"]
        worker = repo.create_worker(ep_id, scheduler_job_id="sched-1")
        worker_id = str(worker["worker_id"])
        job = repo.enqueue_job(ep_id, owner, {"x": 1})
        claimed = repo.claim_next_job(ep_id, worker_id)
        assert claimed is not None
        repo.increment_worker_concurrency(worker_id)

        cancelled = svc.cancel_inflight_job(
            str(job["job_id"]),
            ep_id,
            reason="test cancel",
        )
        assert cancelled is not None
        assert cancelled["status"] == JOB_STATUS_CANCELLED
        w = repo.get_worker(worker_id)
        assert int(w["current_concurrency"]) == 0
        repo.soft_delete_endpoint(ep_id, owner)


class TestStreamRoute:
    def test_stream_has_anti_buffering_headers(self, user_headers):
        repo = ServerlessRepo()
        owner_row = None
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": "stream-ep",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
            },
        )
        assert created.status_code == 200
        ep_id = created.json()["endpoint"]["endpoint_id"]
        owner_row = repo.get_endpoint(ep_id)
        job = repo.enqueue_job(ep_id, str(owner_row["owner_id"]), {"n": 1})
        repo.append_stream_event(job["job_id"], "token", {"t": "hi"})
        repo.complete_job(job["job_id"], output={"done": True})

        with client.stream(
            "GET",
            f"/v1/serverless/{ep_id}/stream/{job['job_id']}?after_seq=0",
            headers=user_headers,
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in (resp.headers.get("content-type") or "")
            assert resp.headers.get("X-Accel-Buffering") == "no"
            body = "".join(resp.iter_text())
        assert "[DONE]" in body
        assert "hi" in body

    def test_disconnect_cancels_inflight_job(self, user_headers, monkeypatch):
        repo = ServerlessRepo()
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": "dc-ep",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
            },
        )
        ep_id = created.json()["endpoint"]["endpoint_id"]
        owner_row = repo.get_endpoint(ep_id)
        worker = repo.create_worker(ep_id, scheduler_job_id="sched-dc")
        worker_id = str(worker["worker_id"])
        job = repo.enqueue_job(ep_id, str(owner_row["owner_id"]), {"x": 1})
        repo.claim_next_job(ep_id, worker_id)
        repo.increment_worker_concurrency(worker_id)

        import routes.serverless as sl_routes

        async def _fake_stream(repo, job_id, *, after_seq=0, request=None, on_disconnect=None, **kwargs):
            if on_disconnect:
                on_disconnect()
            yield format_error_chunk("job_cancelled", "client disconnected")

        monkeypatch.setattr(sl_routes, "async_live_job_stream", _fake_stream)

        r = client.get(
            f"/v1/serverless/{ep_id}/stream/{job['job_id']}",
            headers=user_headers,
        )
        assert r.status_code == 200
        refreshed = repo.get_job(job["job_id"], endpoint_id=ep_id)
        assert refreshed is not None
        assert refreshed["status"] == JOB_STATUS_CANCELLED


class TestAsyncLiveStream:
    def test_terminal_done_emitted(self):
        repo = ServerlessRepo()
        owner = f"cust-{uuid.uuid4().hex[:8]}"
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner, name="s", mode="custom", image_ref="img")
        )
        job = repo.enqueue_job(ep["endpoint_id"], owner, {})
        repo.complete_job(job["job_id"], output={})

        async def _collect():
            frames = []
            async for f in async_live_job_stream(repo, job["job_id"]):
                frames.append(f)
            return frames

        frames = asyncio.run(_collect())
        assert any("[DONE]" in f for f in frames)
        repo.soft_delete_endpoint(ep["endpoint_id"], owner)