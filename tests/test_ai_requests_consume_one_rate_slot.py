"""The route and stream must share one admission decision per request."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import ai_assistant as ai
from api import app

pytestmark = pytest.mark.needs_db


@pytest.mark.parametrize("path", ["/api/ai/chat", "/api/ai/analytics"])
def test_one_message_uses_one_slot_and_the_next_is_refused(path, monkeypatch):
    monkeypatch.setattr(ai, "AI_RATE_LIMIT", 1)
    monkeypatch.setattr(ai, "AI_ENABLE_LIVE_CALLS", False)
    monkeypatch.setattr(ai, "build_ai_system_prompt", lambda *a, **k: "Test prompt")
    with TestClient(app) as client:
        response = client.post(
            "/api/auth/register",
            json={"email": f"ai-quota-{uuid4().hex}@example.com", "password": "QuotaAudit123!"},
        )
        assert response.status_code == 200
        headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
        first = client.post(path, headers=headers, json={"message": "Hello"})
        assert first.status_code == 200
        assert '"type": "done"' in first.text
        assert "Rate limit exceeded" not in first.text
        assert (
            client.post(path, headers=headers, json={"message": "Hello again"}).status_code == 429
        )
