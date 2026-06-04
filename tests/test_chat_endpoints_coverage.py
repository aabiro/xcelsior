"""Smoke coverage for routes/chat.py (UNTESTED_ENDPOINTS.md)."""

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("FEATURE_AI_ASSISTANT", "true")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def user_headers():
    email = f"chatcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Chat Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


# ── Chat widget routes ───────────────────────────────────────────────────


def test_chat_conversations_requires_auth():
    r = client.get("/api/chat/conversations")
    assert r.status_code == 401


def test_chat_conversations_authenticated(user_headers):
    r = client.get("/api/chat/conversations", headers=user_headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert isinstance(r.json().get("conversations"), list)


def test_chat_feedback_validation(user_headers):
    r = client.post(
        "/api/chat/feedback",
        headers=user_headers,
        json={"message_id": "msg-cov-1", "vote": "sideways"},
    )
    assert r.status_code == 400


def test_chat_feedback_ok(user_headers):
    r = client.post(
        "/api/chat/feedback",
        headers=user_headers,
        json={"message_id": "msg-cov-2", "vote": "up"},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_chat_post_without_llm_key(user_headers):
    """Route must return 503 when chat LLM is not configured — never an unhandled 500."""
    import chat
    import routes.chat as routes_chat

    original = chat.CHAT_API_KEY
    routes_chat.CHAT_API_KEY = ""
    chat.CHAT_API_KEY = ""
    try:
        r = client.post(
            "/api/chat",
            headers=user_headers,
            json={"message": "hello"},
        )
        assert r.status_code == 503
    finally:
        chat.CHAT_API_KEY = original
        routes_chat.CHAT_API_KEY = original


# ── AI assistant routes ──────────────────────────────────────────────────


def test_ai_suggestions_requires_auth():
    r = client.get("/api/ai/suggestions")
    assert r.status_code == 401


def test_ai_suggestions_authenticated(user_headers):
    r = client.get("/api/ai/suggestions", headers=user_headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert isinstance(r.json().get("suggestions"), list)


def test_ai_list_conversations(user_headers):
    r = client.get("/api/ai/conversations", headers=user_headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_ai_get_conversation_not_found(user_headers):
    r = client.get(
        "/api/ai/conversations/nonexistent-conv-id",
        headers=user_headers,
    )
    assert r.status_code == 404


def test_ai_delete_conversation_not_found(user_headers):
    r = client.delete(
        "/api/ai/conversations/nonexistent-conv-id",
        headers=user_headers,
    )
    assert r.status_code == 404


@pytest.fixture
def ai_conversation_id(user_headers, monkeypatch):
    import routes.chat as chat_routes

    async def _fake_stream(message, conversation_id, user, page_context=""):
        yield 'data: {"type":"done"}\n\n'

    monkeypatch.setattr(chat_routes, "stream_ai_response", _fake_stream)

    r = client.post(
        "/api/ai/chat",
        headers=user_headers,
        json={"message": "list my instances"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in (r.headers.get("content-type") or "")
    from ai_assistant import list_conversations

    user_id = (
        client.get("/api/auth/me", headers=user_headers).json()["user"]["user_id"]
    )
    convs = list_conversations(user_id, limit=1)
    assert convs, "expected a conversation after ai chat"
    return convs[0]["conversation_id"]


def test_ai_get_and_delete_conversation(user_headers, ai_conversation_id):
    cid = ai_conversation_id
    r = client.get(f"/api/ai/conversations/{cid}", headers=user_headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True

    r_del = client.delete(f"/api/ai/conversations/{cid}", headers=user_headers)
    assert r_del.status_code == 200
    assert r_del.json().get("ok") is True

    r_gone = client.get(f"/api/ai/conversations/{cid}", headers=user_headers)
    assert r_gone.status_code == 404


def test_ai_confirm_unknown_id(user_headers):
    r = client.post(
        "/api/ai/confirm",
        headers=user_headers,
        json={"confirmation_id": "no-such-confirmation", "approved": False},
    )
    assert r.status_code == 200
    assert "text/event-stream" in (r.headers.get("content-type") or "")


def test_ai_analytics_chat_mocked(user_headers, monkeypatch):
    import routes.chat as chat_routes

    async def _fake_stream(message, conversation_id, user, page_context=""):
        yield 'data: {"type":"done"}\n\n'

    monkeypatch.setattr(chat_routes, "stream_ai_response", _fake_stream)

    r = client.post(
        "/api/ai/analytics",
        headers=user_headers,
        json={
            "message": "What changed this week?",
            "analytics_summary": "jobs=3 revenue=10",
        },
    )
    assert r.status_code == 200
    assert "text/event-stream" in (r.headers.get("content-type") or "")


def test_ai_chat_mocked(user_headers, monkeypatch):
    import routes.chat as chat_routes

    async def _fake_stream(message, conversation_id, user, page_context=""):
        yield 'data: {"type":"token","content":"ok"}\n\n'
        yield 'data: {"type":"done"}\n\n'

    monkeypatch.setattr(chat_routes, "stream_ai_response", _fake_stream)

    r = client.post(
        "/api/ai/chat",
        headers=user_headers,
        json={"message": "hello"},
    )
    assert r.status_code == 200
    assert r.status_code not in (500,)
    assert "text/event-stream" in (r.headers.get("content-type") or "")