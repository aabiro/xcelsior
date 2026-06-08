"""Phase 12 — OpenAI-compatible proxy (chat, capability gates, streaming)."""

import json
import os
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from api import app
from db import UserStore
from serverless.openai_proxy import (
    OpenAIProxyError,
    capability_gate,
    normalize_route,
    proxy_chat_completions,
)
from serverless.repo import EndpointCreate, ServerlessRepo, WORKER_STATE_READY

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
    email = f"sloc-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "OpenAI"},
    )
    assert reg.status_code == 200, reg.text[:300]
    reg_body = reg.json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text[:200]
    body = login.json()
    user = reg_body.get("user") or body.get("user") or {}
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    deposit = client.post(
        f"/api/billing/wallet/{user['customer_id']}/deposit",
        json={"amount_cad": 50.0, "description": "openai compat credits"},
        headers=headers,
    )
    assert deposit.status_code == 200, deposit.text[:200]
    assert UserStore.get_user(email) is not None
    return headers


@pytest.fixture(scope="module")
def user_headers():
    return _register_and_fund()


def _error_code(response) -> str | None:
    body = response.json()
    detail = body.get("detail")
    if isinstance(detail, dict):
        err = detail.get("error")
        if isinstance(err, dict):
            return err.get("code")
    if isinstance(body.get("error"), dict):
        return body["error"].get("code")
    return None


class TestOpenAIProxyHelpers:
    def test_normalize_route_strips_prefix(self):
        assert normalize_route("/openai/v1/chat/completions") == "chat/completions"
        assert normalize_route("v1/models") == "models"

    def test_embeddings_capability_gated(self):
        ep = {"mode": "preset", "model_ref": "meta-llama/Llama-3.1-8B-Instruct"}
        with pytest.raises(OpenAIProxyError) as exc:
            capability_gate("embeddings", ep)
        assert exc.value.status_code == 501
        assert exc.value.code == "capability_not_available"

    def test_custom_endpoint_rejects_chat_proxy(self):
        ep = {"mode": "custom", "image_ref": "img"}
        with pytest.raises(OpenAIProxyError) as exc:
            capability_gate("chat/completions", ep)
        assert exc.value.status_code == 400
        assert exc.value.code == "preset_only"

    def test_vision_inputs_rejected(self):
        ep = {"mode": "preset", "model_ref": "m"}
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}],
                }
            ]
        }
        with pytest.raises(OpenAIProxyError) as exc:
            capability_gate("chat/completions", ep, body)
        assert exc.value.status_code == 400
        assert exc.value.code == "vision_not_supported"

    def test_unsupported_route_404(self):
        ep = {"mode": "preset", "model_ref": "m"}
        with pytest.raises(OpenAIProxyError) as exc:
            capability_gate("fine_tuning/jobs", ep)
        assert exc.value.status_code == 404
        assert exc.value.code == "route_not_supported"


class TestOpenAIChatRoute:
    def test_chat_completions_success(self, user_headers, monkeypatch):
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": f"oai-{uuid.uuid4().hex[:6]}",
                "mode": "preset",
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "min_workers": 0,
                "max_workers": 1,
            },
        )
        assert created.status_code == 200, created.text[:300]
        endpoint_id = created.json()["endpoint"]["endpoint_id"]

        warm_worker = {
            "worker_id": "swk-mock",
            "host_ip": "10.0.0.9",
            "job_payload": {"http_ports": {"8080": 18080}},
        }

        import routes.serverless as sl_routes

        monkeypatch.setattr(sl_routes, "_warm_worker_row", lambda _eid: warm_worker)
        monkeypatch.setattr(
            sl_routes,
            "proxy_chat_completions",
            lambda *_a, **_k: {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            },
        )

        r = client.post(
            f"/v1/serverless/{endpoint_id}/openai/v1/chat/completions",
            headers=user_headers,
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200, r.text[:300]
        assert r.json()["choices"][0]["message"]["content"] == "hello"

        client.delete(
            f"/api/v2/serverless/endpoints/{endpoint_id}",
            headers=user_headers,
        )

    def test_custom_endpoint_openai_chat_400(self, user_headers):
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": f"custom-{uuid.uuid4().hex[:6]}",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
            },
        )
        assert created.status_code == 200
        endpoint_id = created.json()["endpoint"]["endpoint_id"]
        r = client.post(
            f"/v1/serverless/{endpoint_id}/openai/v1/chat/completions",
            headers=user_headers,
            json={
                "model": "any",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 400
        assert "preset" in r.text.lower()

        client.delete(
            f"/api/v2/serverless/endpoints/{endpoint_id}",
            headers=user_headers,
        )

    def test_stream_emits_done_chunk(self, user_headers, monkeypatch):
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": f"stream-{uuid.uuid4().hex[:6]}",
                "mode": "preset",
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            },
        )
        assert created.status_code == 200
        endpoint_id = created.json()["endpoint"]["endpoint_id"]

        warm_worker = {"worker_id": "swk-stream", "host_ip": "10.0.0.9", "job_payload": {}}

        async def _fake_lines(*_a, **_k):
            chunk = {
                "choices": [{"delta": {"content": "Hi"}, "index": 0}],
            }
            yield f"data: {json.dumps(chunk)}"

        import routes.serverless as sl_routes

        monkeypatch.setattr(sl_routes, "_warm_worker_row", lambda _eid: warm_worker)
        monkeypatch.setattr(sl_routes, "async_proxy_stream_lines", _fake_lines)

        with client.stream(
            "POST",
            f"/v1/serverless/{endpoint_id}/openai/v1/chat/completions",
            headers=user_headers,
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in (resp.headers.get("content-type") or "")
            body = "".join(resp.iter_text())

        assert "Hi" in body
        assert "[DONE]" in body

        client.delete(
            f"/api/v2/serverless/endpoints/{endpoint_id}",
            headers=user_headers,
        )

    def test_stream_upstream_error_surfaces_chunk(self, user_headers, monkeypatch):
        created = client.post(
            "/api/v2/serverless/endpoints",
            headers=user_headers,
            json={
                "name": f"err-{uuid.uuid4().hex[:6]}",
                "mode": "preset",
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            },
        )
        endpoint_id = created.json()["endpoint"]["endpoint_id"]
        warm_worker = {"worker_id": "swk-err", "host_ip": "10.0.0.9", "job_payload": {}}

        async def _raise(*_a, **_k):
            raise OpenAIProxyError(502, "upstream_http_error", "worker blew up")
            yield  # pragma: no cover

        import routes.serverless as sl_routes

        monkeypatch.setattr(sl_routes, "_warm_worker_row", lambda _eid: warm_worker)
        monkeypatch.setattr(sl_routes, "async_proxy_stream_lines", _raise)

        with client.stream(
            "POST",
            f"/v1/serverless/{endpoint_id}/openai/v1/chat/completions",
            headers=user_headers,
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            body = "".join(resp.iter_text())

        assert "upstream_http_error" in body

        client.delete(
            f"/api/v2/serverless/endpoints/{endpoint_id}",
            headers=user_headers,
        )


class TestOpenAIProxyUnit:
    def test_proxy_chat_parses_upstream_json(self, monkeypatch):
        worker = {"host_ip": "10.0.0.1", "job_payload": {}}
        ep = {"mode": "preset", "http_port": 8080, "request_timeout_sec": 30}

        class FakeResp:
            status_code = 200

            @staticmethod
            def json():
                return {"object": "chat.completion", "choices": []}

        monkeypatch.setattr(
            "serverless.openai_proxy.proxy_request",
            lambda *_a, **_k: FakeResp(),
        )
        out = proxy_chat_completions(worker, ep, {"model": "m", "messages": []})
        assert out["object"] == "chat.completion"