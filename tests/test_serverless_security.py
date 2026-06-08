"""Phase 13 — serverless security hardening."""

import json
import os
import time
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from api import app
from db import UserStore
from serverless.env_secrets import (
    decrypt_env_for_worker,
    encrypt_env_for_storage,
    payload_byte_size,
    redact_env_for_api,
)
from serverless.keys import KEY_PREFIX, create_endpoint_key
from serverless.repo import EndpointCreate, ServerlessRepo
from serverless.service import ServerlessService
from serverless.webhooks import WEBHOOK_MAX_BODY_BYTES, build_webhook_payload, deliver_job_webhook

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
    email = f"slsec-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Sec"},
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


class TestEnvSecrets:
    def test_sensitive_values_encrypted_roundtrip(self):
        stored = encrypt_env_for_storage(
            {"HF_TOKEN": "hf_secret", "MODEL": "llama", "API_KEY": "k"}
        )
        assert stored["MODEL"] == "llama"
        assert stored["HF_TOKEN"].startswith("enc:")
        assert stored["API_KEY"].startswith("enc:")
        plain = decrypt_env_for_worker(stored)
        assert plain["HF_TOKEN"] == "hf_secret"
        assert plain["API_KEY"] == "k"

    def test_redact_env_for_api_masks_secrets(self):
        stored = encrypt_env_for_storage({"HF_TOKEN": "hf_secret", "FOO": "bar"})
        redacted = redact_env_for_api(stored)
        assert redacted["FOO"] == "bar"
        assert redacted["HF_TOKEN"] == "[REDACTED]"


class TestPayloadSize:
    def test_payload_byte_size_counts_json(self):
        assert payload_byte_size({"a": "x" * 100}) > 100


class TestRunPayloadLimit:
    def test_oversized_run_returns_413(self, user_headers):
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": "payload-limit",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
                "max_request_bytes": 1024,
            },
        )
        assert created.status_code == 200
        endpoint_id = created.json()["endpoint"]["endpoint_id"]
        key_resp = client.post(
            f"/api/v2/serverless/endpoints/{endpoint_id}/keys",
            headers=user_headers,
            json={"name": "run-key"},
        )
        assert key_resp.status_code == 200
        raw_key = key_resp.json()["key"]["api_key"]
        assert raw_key.startswith(KEY_PREFIX)
        r = client.post(
            f"/v1/serverless/{endpoint_id}/run",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={"input": {"blob": "x" * 2000}},
        )
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "payload_too_large"


class TestKeyMismatch404:
    def test_key_for_other_endpoint_returns_404(self, user_headers):
        ep1 = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": "ep-a",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
            },
        ).json()["endpoint"]["endpoint_id"]
        ep2 = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": "ep-b",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
            },
        ).json()["endpoint"]["endpoint_id"]
        key = client.post(
            f"/api/v2/serverless/endpoints/{ep1}/keys",
            headers=user_headers,
            json={"name": "scoped"},
        ).json()["key"]["api_key"]
        r = client.post(
            f"/v1/serverless/{ep2}/run",
            headers={"Authorization": f"Bearer {key}"},
            json={"input": {"n": 1}},
        )
        assert r.status_code == 404


class TestQueueTTL:
    def test_stale_queued_job_failed_by_reaper(self):
        repo = ServerlessRepo()
        owner = f"cust-{uuid.uuid4().hex[:12]}"
        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner,
                name="queue-ttl",
                mode="custom",
                image_ref="img",
            )
        )
        job = repo.enqueue_job(ep["endpoint_id"], owner, {"x": 1})
        with repo._conn() as conn:
            conn.execute(
                "UPDATE serverless_jobs SET queued_at = %s WHERE job_id = %s",
                (time.time() - 7200, job["job_id"]),
            )
        svc = ServerlessService(repo)
        with patch.object(svc.repo, "list_workers_stale_heartbeat", return_value=[]), patch.object(
            svc.repo, "list_jobs_past_request_timeout", return_value=[]
        ), patch.object(svc.repo, "list_stuck_booting_workers", return_value=[]), patch.object(
            svc.repo, "list_booting_workers", return_value=[]
        ):
            from serverless.reaper import reap_all

            stats = reap_all(svc)
        assert stats["stale_queued"] >= 1
        refreshed = repo.get_job(job["job_id"])
        assert refreshed is not None
        assert refreshed["status"] == "FAILED"
        repo.soft_delete_endpoint(ep["endpoint_id"], owner)


class TestWebhookBodyCap:
    def test_large_output_truncated_before_post(self, monkeypatch):
        repo = ServerlessRepo()
        owner = f"cust-{uuid.uuid4().hex[:12]}"
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner, name="wh", mode="custom", image_ref="img")
        )
        posted: list[bytes] = []

        import urllib.request

        class _FakeResp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _capture_urlopen(req, timeout=10):
            posted.append(req.data)
            return _FakeResp()

        monkeypatch.setattr(urllib.request, "urlopen", _capture_urlopen)
        job = repo.enqueue_job(ep["endpoint_id"], owner, {"x": 1}, webhook_url="https://example.com/h")
        big = {"data": "z" * WEBHOOK_MAX_BODY_BYTES}
        done = repo.complete_job(job["job_id"], output=big)
        assert done is not None
        deliver_job_webhook(repo, done)
        assert posted
        assert len(posted[0]) <= WEBHOOK_MAX_BODY_BYTES
        repo.soft_delete_endpoint(ep["endpoint_id"], owner)