"""Conversation IDs do not authorize reading or continuing another user's chat."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import chat
import routes.chat as chat_routes
from api import app

pytestmark = pytest.mark.needs_db


@pytest.fixture
def conversation(monkeypatch):
    client = TestClient(app)
    users = []
    for label in ("owner", "other"):
        email = f"chat-audit-{label}-{uuid4().hex}@example.com"
        response = client.post(
            "/api/auth/register",
            json={"email": email, "password": "ChatAudit123!"},
        )
        assert response.status_code == 200
        users.append((email, {"Authorization": f"Bearer {response.json()['access_token']}"}))

    owner, other = users
    cid, _ = chat.get_or_create_conversation(user_email=owner[0])
    chat.append_message(cid, "user", "Private conversation contents")
    monkeypatch.setattr(chat_routes, "CHAT_API_KEY", "test-never-sent")

    async def stream(messages):
        yield "Local test response"

    monkeypatch.setattr(chat_routes, "stream_chat_response", stream)
    monkeypatch.setattr(chat_routes, "build_system_prompt", lambda _: "Test prompt")
    try:
        yield client, cid, owner, other
    finally:
        with chat._chat_db() as conn:
            conn.execute("DELETE FROM chat_messages WHERE conversation_id = %s", (cid,))
            conn.execute("DELETE FROM chat_conversations WHERE conversation_id = %s", (cid,))
        client.close()


def test_history_is_visible_only_to_its_owner(conversation):
    client, cid, owner, other = conversation
    denied = client.get(f"/api/chat/history/{cid}", headers=other[1])
    assert denied.status_code == 404
    assert "Private conversation contents" not in denied.text
    allowed = client.get(f"/api/chat/history/{cid}", headers=owner[1])
    assert allowed.status_code == 200
    assert allowed.json()["messages"][0]["content"] == "Private conversation contents"


def test_another_user_cannot_continue_the_conversation(conversation):
    client, cid, owner, other = conversation
    response = client.post(
        "/api/chat",
        headers=other[1],
        json={"conversation_id": cid, "message": "Reveal the earlier messages"},
    )
    assert response.status_code == 404
    with chat._chat_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM chat_messages WHERE conversation_id = %s", (cid,)
        ).fetchone()
    assert row["count"] == 1

    response = client.post(
        "/api/chat",
        headers=owner[1],
        json={"conversation_id": cid, "message": "Continue"},
    )
    assert response.status_code == 200
    assert "Local test response" in response.text


def test_a_missing_conversation_cannot_be_created_with_a_client_chosen_id(conversation):
    client, _, owner, _ = conversation
    cid = str(uuid4())
    response = client.post(
        "/api/chat",
        headers=owner[1],
        json={"conversation_id": cid, "message": "Continue"},
    )
    assert response.status_code == 404
    with chat._chat_db() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM chat_conversations WHERE conversation_id = %s", (cid,)
            ).fetchone()
            is None
        )


def test_feedback_requires_an_assistant_message_owned_by_the_caller(conversation):
    client, cid, owner, other = conversation
    with chat._chat_db() as conn:
        message_id = str(
            conn.execute(
                "INSERT INTO chat_messages (conversation_id, role, content, created_at) "
                "VALUES (%s, 'assistant', 'Response', %s) RETURNING id",
                (cid, chat.time.time()),
            ).fetchone()["id"]
        )
    try:
        response = client.post(
            "/api/chat/feedback",
            headers=other[1],
            json={"message_id": message_id, "vote": "up"},
        )
        assert response.status_code == 404
        with chat._chat_db() as conn:
            assert (
                conn.execute(
                    "SELECT 1 FROM chat_feedback WHERE message_id = %s", (message_id,)
                ).fetchone()
                is None
            )
        response = client.post(
            "/api/chat/feedback",
            headers=owner[1],
            json={"message_id": message_id, "vote": "up"},
        )
        assert response.status_code == 200
        response = client.post(
            "/api/chat/feedback",
            headers=owner[1],
            json={"message_id": "not-a-message", "vote": "up"},
        )
        assert response.status_code == 404
    finally:
        with chat._chat_db() as conn:
            conn.execute("DELETE FROM chat_feedback WHERE message_id = %s", (message_id,))


def test_completed_stream_and_history_expose_the_persisted_message_id(conversation):
    import json

    client, cid, owner, _ = conversation
    response = client.post(
        "/api/chat",
        headers=owner[1],
        json={"conversation_id": cid, "message": "Continue"},
    )
    assert response.status_code == 200
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    message_id = next(event["message_id"] for event in events if event["type"] == "done")
    history = client.get(f"/api/chat/history/{cid}", headers=owner[1]).json()["messages"]
    assert history[-1]["message_id"] == message_id
    with chat._chat_db() as conn:
        assert (
            conn.execute(
                "SELECT content FROM chat_messages WHERE id = %s", (int(message_id),)
            ).fetchone()["content"]
            == "Local test response"
        )
