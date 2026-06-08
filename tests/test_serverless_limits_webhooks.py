"""Phase 8 — rate limits, queue backpressure, idempotency, webhooks."""

import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")
os.environ["XCELSIOR_SERVERLESS_WEBHOOK_HMAC_SECRET"] = "test-webhook-secret"
os.environ["XCELSIOR_SERVERLESS_WEBHOOK_MAX_ATTEMPTS"] = "1"

from api import app
from db import UserStore
from serverless.keys import create_endpoint_key
from serverless.limits import RateLimitExceeded, check_key_rate_limit
from serverless.repo import EndpointCreate, ServerlessRepo
from serverless.webhooks import (
    build_webhook_payload,
    deliver_job_webhook,
    sign_payload,
    validate_webhook_url,
    verify_signature,
)

client = TestClient(app)


def _error_code(response) -> str | None:
    body = response.json()
    if isinstance(body.get("error"), dict):
        return body["error"].get("code")
    detail = body.get("detail")
    if isinstance(detail, dict):
        err = detail.get("error")
        if isinstance(err, dict):
            return err.get("code")
    return None


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
    email = f"sl8-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Phase8"},
    )
    assert reg.status_code == 200, reg.text[:300]
    reg_body = reg.json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text[:200]
    body = login.json()
    user = reg_body.get("user") or body.get("user") or {}
    customer_id = user["customer_id"]
    assert UserStore.get_user(email) is not None
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    deposit = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "phase8 credits"},
        headers=headers,
    )
    assert deposit.status_code == 200, deposit.text[:200]
    return headers


@pytest.fixture(scope="module")
def user_headers():
    return _register_and_fund()


@pytest.fixture(scope="module")
def endpoint_id(user_headers):
    r = client.post(
        "/api/v2/serverless/endpoints",
        headers=user_headers,
        json={
            "name": "phase8-ep",
            "mode": "custom",
            "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
            "min_workers": 0,
            "max_workers": 1,
        },
    )
    assert r.status_code == 200, r.text[:300]
    return r.json()["endpoint"]["endpoint_id"]


@pytest.fixture(scope="module")
def api_key_headers(user_headers, endpoint_id):
    r = client.post(
        f"/api/v2/serverless/endpoints/{endpoint_id}/keys",
        headers=user_headers,
        json={"name": "phase8-key", "rate_limit_rpm": 2},
    )
    assert r.status_code == 200, r.text[:300]
    raw = r.json()["key"]["api_key"]
    return {"Authorization": f"Bearer {raw}"}


class TestWebhookHelpers:
    def test_ssrf_blocks_localhost(self):
        assert validate_webhook_url("http://localhost/hook") is None
        assert validate_webhook_url("https://127.0.0.1/hook") is None

    def test_hmac_roundtrip(self):
        body = b'{"id":"sjob-abc","status":"COMPLETED"}'
        ts = 1710000000
        sig = sign_payload(body, ts)
        assert verify_signature(body, ts, sig)
        assert not verify_signature(body, ts, "bad")

    def test_build_payload_shape(self):
        job = {
            "job_id": "sjob-1",
            "endpoint_id": "sep-1",
            "status": "COMPLETED",
            "output": {"ok": True},
        }
        payload = build_webhook_payload(job)
        assert payload["id"] == "sjob-1"
        assert payload["status"] == "COMPLETED"


class TestRateLimitUnit:
    def test_per_key_rpm_enforced(self):
        key_id = f"skey-{uuid.uuid4().hex[:8]}"
        check_key_rate_limit(key_id, 2)
        check_key_rate_limit(key_id, 2)
        with pytest.raises(RateLimitExceeded):
            check_key_rate_limit(key_id, 2)


class TestServerlessPhase8Routes:
    def test_idempotency_dedupes_run(self, api_key_headers, endpoint_id):
        idem = f"idem-{uuid.uuid4().hex[:8]}"
        headers = {**api_key_headers, "Idempotency-Key": idem}
        r1 = client.post(
            f"/v1/serverless/{endpoint_id}/run",
            headers=headers,
            json={"input": {"n": 1}},
        )
        r2 = client.post(
            f"/v1/serverless/{endpoint_id}/run",
            headers=headers,
            json={"input": {"n": 2}},
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["id"] == r2.json()["id"]

    def test_rate_limit_returns_429_with_headers(self, user_headers):
        import serverless.limits as limits

        limits._RATE_BUCKETS.clear()
        ep = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": f"rpm-{uuid.uuid4().hex[:6]}",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
            },
        )
        assert ep.status_code == 200
        endpoint_id = ep.json()["endpoint"]["endpoint_id"]
        created = client.post(
            f"/api/v2/serverless/endpoints/{endpoint_id}/keys",
            headers=user_headers,
            json={"name": "rpm-test", "rate_limit_rpm": 2},
        )
        assert created.status_code == 200
        key_headers = {"Authorization": f"Bearer {created.json()['key']['api_key']}"}
        for _ in range(2):
            r = client.post(
                f"/v1/serverless/{endpoint_id}/run",
                headers=key_headers,
                json={"input": {"x": 1}},
            )
            assert r.status_code == 200, r.text[:200]
        r = client.post(
            f"/v1/serverless/{endpoint_id}/run",
            headers=key_headers,
            json={"input": {"x": 2}},
        )
        assert r.status_code == 429, r.text[:300]
        assert _error_code(r) == "rate_limited", r.text[:300]
        assert r.headers.get("X-RateLimit-Limit") == "2"
        assert r.headers.get("X-RateLimit-Remaining") == "0"

    def test_revoked_key_blocked(self, user_headers, endpoint_id):
        created = client.post(
            f"/api/v2/serverless/endpoints/{endpoint_id}/keys",
            headers=user_headers,
            json={"name": "revoke-me", "rate_limit_rpm": 60},
        )
        assert created.status_code == 200
        key_id = created.json()["key"]["key_id"]
        raw = created.json()["key"]["api_key"]
        revoked = client.delete(
            f"/api/v2/serverless/endpoints/{endpoint_id}/keys/{key_id}",
            headers=user_headers,
        )
        assert revoked.status_code == 200
        r = client.post(
            f"/v1/serverless/{endpoint_id}/run",
            headers={"Authorization": f"Bearer {raw}"},
            json={"input": {}},
        )
        assert r.status_code == 401


class TestWebhookDelivery:
    def test_deliver_marks_delivered(self, monkeypatch):
        repo = ServerlessRepo()
        owner = f"cust-{uuid.uuid4().hex[:12]}"
        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner,
                name="wh-ep",
                mode="custom",
                image_ref="xcelsior/serverless-base:cuda12.4-py3.12",
            )
        )
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
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        hook_url = f"http://127.0.0.1:{port}/hook"

        def _allow_local(url: str):
            return url if url.startswith("http://127.0.0.1:") else validate_webhook_url(url)

        try:
            job = repo.enqueue_job(
                ep["endpoint_id"],
                owner,
                {"x": 1},
                webhook_url=hook_url,
            )
            done = repo.complete_job(job["job_id"], output={"ok": True})
            assert done is not None
            import serverless.webhooks as wh

            monkeypatch.setattr(wh, "validate_webhook_url", _allow_local)
            assert deliver_job_webhook(repo, done) is True
            refreshed = repo.get_job(job["job_id"])
            assert refreshed is not None
            assert refreshed.get("webhook_status") == "delivered"
            assert len(received) == 1
            assert received[0]["status"] == "COMPLETED"
        finally:
            server.shutdown()
            repo.soft_delete_endpoint(ep["endpoint_id"], owner)


class TestQueueBackpressure:
    def test_queue_full_returns_429(self, user_headers):
        import serverless.limits as limits

        limits._RATE_BUCKETS.clear()
        repo = ServerlessRepo()
        ep_resp = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": f"queue-{uuid.uuid4().hex[:6]}",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
            },
        )
        assert ep_resp.status_code == 200
        endpoint_id = ep_resp.json()["endpoint"]["endpoint_id"]
        ep = repo.get_endpoint(endpoint_id)
        assert ep is not None
        repo.patch_endpoint(endpoint_id, str(ep["owner_id"]), {"max_queue_size": 1})

        created = client.post(
            f"/api/v2/serverless/endpoints/{endpoint_id}/keys",
            headers=user_headers,
            json={"name": "queue-test", "rate_limit_rpm": 1000},
        )
        assert created.status_code == 200
        headers = {"Authorization": f"Bearer {created.json()['key']['api_key']}"}

        r1 = client.post(
            f"/v1/serverless/{endpoint_id}/run",
            headers=headers,
            json={"input": {"a": 1}},
        )
        assert r1.status_code == 200, r1.text[:200]
        r2 = client.post(
            f"/v1/serverless/{endpoint_id}/run",
            headers=headers,
            json={"input": {"b": 2}},
        )
        assert r2.status_code == 429, r2.text[:300]
        assert _error_code(r2) == "queue_full", r2.text[:300]