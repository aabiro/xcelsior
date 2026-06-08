"""Phase 12 — idempotency dedupe, webhook HMAC delivery, and retry backoff."""

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

os.environ["XCELSIOR_ENV"] = "test"
os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ["XCELSIOR_SERVERLESS_WEBHOOK_HMAC_SECRET"] = "test-idem-secret"
os.environ["XCELSIOR_SERVERLESS_WEBHOOK_MAX_ATTEMPTS"] = "3"
os.environ["XCELSIOR_SERVERLESS_WEBHOOK_BACKOFF_BASE_SEC"] = "0.01"

from serverless.repo import EndpointCreate, JOB_STATUS_COMPLETED, ServerlessRepo
from serverless.webhooks import (
    deliver_job_webhook,
    retry_pending_webhooks,
    verify_signature,
)


@pytest.fixture
def owner_id():
    return f"cust-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def endpoint(repo: ServerlessRepo, owner_id: str):
    ep = repo.create_endpoint(
        EndpointCreate(
            owner_id=owner_id,
            name="idem-ep",
            mode="custom",
            image_ref="xcelsior/serverless-base:cuda12.4-py3.12",
        )
    )
    yield ep
    repo.soft_delete_endpoint(ep["endpoint_id"], owner_id)


@pytest.fixture
def repo():
    return ServerlessRepo()


class TestIdempotencyRace:
    def test_concurrent_enqueue_same_key_returns_one_job(
        self, repo: ServerlessRepo, endpoint: dict, owner_id: str
    ):
        ep_id = endpoint["endpoint_id"]
        key = f"race-{uuid.uuid4().hex[:8]}"
        results: list[str] = []
        barrier = threading.Barrier(4)

        def _enqueue():
            barrier.wait()
            job = repo.enqueue_job(ep_id, owner_id, {"n": 1}, idempotency_key=key)
            results.append(str(job["job_id"]))

        threads = [threading.Thread(target=_enqueue) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 4
        assert len(set(results)) == 1
        assert repo.queue_depth(ep_id) == 1

    def test_different_keys_create_distinct_jobs(
        self, repo: ServerlessRepo, endpoint: dict, owner_id: str
    ):
        ep_id = endpoint["endpoint_id"]
        j1 = repo.enqueue_job(ep_id, owner_id, {"a": 1}, idempotency_key="key-a")
        j2 = repo.enqueue_job(ep_id, owner_id, {"b": 2}, idempotency_key="key-b")
        assert j1["job_id"] != j2["job_id"]
        assert repo.queue_depth(ep_id) == 2


class TestWebhookRetry:
    def _local_hook(self) -> tuple[HTTPServer, str, list[dict]]:
        received: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                sig = self.headers.get("X-Xcelsior-Signature", "")
                ts = int(self.headers.get("X-Xcelsior-Timestamp", "0"))
                assert sig.startswith("sha256=")
                assert verify_signature(body, ts, sig.split("=", 1)[1])
                received.append(json.loads(body.decode("utf-8")))
                self.send_response(500)
                self.end_headers()

            def log_message(self, *_args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{port}/hook", received

    def test_failed_delivery_schedules_retry(self, repo: ServerlessRepo, endpoint: dict, owner_id: str, monkeypatch):
        server, hook_url, _received = self._local_hook()

        def _allow_local(url: str):
            return url if url.startswith("http://127.0.0.1:") else None

        import serverless.webhooks as wh

        monkeypatch.setattr(wh, "validate_webhook_url", _allow_local)

        try:
            job = repo.enqueue_job(
                endpoint["endpoint_id"],
                owner_id,
                {"x": 1},
                webhook_url=hook_url,
            )
            done = repo.complete_job(job["job_id"], output={"ok": True})
            assert done is not None
            assert deliver_job_webhook(repo, done) is False

            refreshed = repo.get_job(job["job_id"])
            assert refreshed is not None
            assert refreshed["webhook_status"] == "pending"
            assert int(refreshed["webhook_attempts"]) == 1
            assert float(refreshed["webhook_next_retry_at"]) > 0
        finally:
            server.shutdown()

    def test_retry_pending_delivers_after_backoff(
        self, repo: ServerlessRepo, endpoint: dict, owner_id: str, monkeypatch
    ):
        server, hook_url, received = self._local_hook()
        attempts = {"n": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                attempts["n"] += 1
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                sig = self.headers.get("X-Xcelsior-Signature", "")
                ts = int(self.headers.get("X-Xcelsior-Timestamp", "0"))
                assert verify_signature(body, ts, sig.split("=", 1)[1])
                received.append(json.loads(body.decode("utf-8")))
                code = 500 if attempts["n"] == 1 else 200
                self.send_response(code)
                self.end_headers()

            def log_message(self, *_args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        hook_url = f"http://127.0.0.1:{port}/hook"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def _allow_local(url: str):
            return url if url.startswith("http://127.0.0.1:") else None

        import serverless.webhooks as wh

        monkeypatch.setattr(wh, "validate_webhook_url", _allow_local)

        try:
            job = repo.enqueue_job(
                endpoint["endpoint_id"],
                owner_id,
                {"x": 1},
                webhook_url=hook_url,
            )
            done = repo.complete_job(job["job_id"], output={"ok": True})
            deliver_job_webhook(repo, done)

            repo.update_job_webhook_status(
                job["job_id"],
                status="pending",
                next_retry_at=time.time() - 1,
            )
            due = repo.get_job(job["job_id"])
            count = retry_pending_webhooks(repo)
            assert count >= 1

            final = repo.get_job(job["job_id"])
            assert final is not None
            assert final["webhook_status"] == "delivered"
            assert len(received) >= 2
        finally:
            server.shutdown()

    def test_max_attempts_marks_failed(self, repo: ServerlessRepo, endpoint: dict, owner_id: str, monkeypatch):
        server, hook_url, _received = self._local_hook()

        def _allow_local(url: str):
            return url if url.startswith("http://127.0.0.1:") else None

        import serverless.webhooks as wh

        monkeypatch.setattr(wh, "validate_webhook_url", _allow_local)
        monkeypatch.setattr(wh, "WEBHOOK_MAX_ATTEMPTS", 2)

        try:
            job = repo.enqueue_job(
                endpoint["endpoint_id"],
                owner_id,
                {"x": 1},
                webhook_url=hook_url,
            )
            done = repo.complete_job(job["job_id"], output={"ok": True})
            assert done is not None
            assert done["status"] == JOB_STATUS_COMPLETED

            deliver_job_webhook(repo, done)
            second = repo.get_job(job["job_id"])
            deliver_job_webhook(repo, second)

            final = repo.get_job(job["job_id"])
            assert final is not None
            assert final["webhook_status"] == "failed"
            assert int(final["webhook_attempts"]) == 2
        finally:
            server.shutdown()