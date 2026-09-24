"""A signed approval authorizes one action, even when workers race."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest

import ai_assistant as ai

pytestmark = pytest.mark.needs_db


@pytest.fixture
def confirmation():
    user_id = f"confirmation-audit-{uuid4().hex}"
    conversation_id = ai.create_conversation(user_id)
    token = ai.create_confirmation(conversation_id, user_id, "stop_job", {"job_id": "audit"})
    confirmation_id = ai._verify_confirmation_token(token)
    try:
        yield token, confirmation_id, user_id
    finally:
        with ai._ai_db() as conn:
            conn.execute(
                "DELETE FROM ai_confirmations WHERE confirmation_id = %s", (confirmation_id,)
            )
            conn.execute(
                "DELETE FROM ai_conversations WHERE conversation_id = %s", (conversation_id,)
            )


@pytest.mark.parametrize("approved", [True, False])
def test_resolution_is_persisted_and_cannot_be_replayed(confirmation, approved):
    token, confirmation_id, user_id = confirmation
    assert ai.resolve_confirmation(token, user_id, approved) is not None
    with ai._ai_db() as conn:
        row = conn.execute(
            "SELECT status, resolved_at FROM ai_confirmations WHERE confirmation_id = %s",
            (confirmation_id,),
        ).fetchone()
    assert row["status"] == ("approved" if approved else "rejected")
    assert row["resolved_at"] > 0
    assert ai.resolve_confirmation(token, user_id, True) is None
    assert ai.resolve_confirmation(token, user_id, False) is None


def test_wrong_user_cannot_consume_the_owners_confirmation(confirmation):
    token, _, user_id = confirmation
    assert ai.resolve_confirmation(token, "another-user", True) is None
    assert ai.resolve_confirmation(token, user_id, True) is not None


def test_concurrent_workers_cannot_both_claim_an_approval(confirmation):
    token, _, user_id = confirmation
    barrier = Barrier(2)

    def resolve():
        barrier.wait(timeout=5)
        return ai.resolve_confirmation(token, user_id, True)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(resolve) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert sum(result is not None for result in results) == 1


@pytest.mark.asyncio
async def test_replaying_the_stream_does_not_execute_the_tool_twice(confirmation, monkeypatch):
    token, _, user_id = confirmation
    calls = []

    async def execute(tool_name, tool_args, user):
        calls.append((tool_name, tool_args))
        return {"ok": True}

    monkeypatch.setattr(ai, "_exec_tool", execute)
    for _ in range(2):
        events = [
            event async for event in ai.execute_confirmed_action(token, {"user_id": user_id}, True)
        ]
        assert events
    assert calls == [("stop_job", {"job_id": "audit"})]


def test_expired_confirmation_returns_only_one_shared_slot(confirmation, monkeypatch):
    token, confirmation_id, user_id = confirmation
    monkeypatch.setattr(ai, "AI_RATE_LIMIT", 2)
    assert ai.check_ai_rate_limit(user_id)
    assert ai.check_ai_rate_limit(user_id)
    assert not ai.check_ai_rate_limit(user_id)
    with ai._ai_db() as conn:
        conn.execute(
            "UPDATE ai_confirmations SET created_at = %s WHERE confirmation_id = %s",
            (ai.time.time() - ai.CONFIRMATION_TTL_SEC - 1, confirmation_id),
        )
    assert ai.resolve_confirmation(token, user_id, True) is None
    assert ai.check_ai_rate_limit(user_id)
    assert not ai.check_ai_rate_limit(user_id)
    assert ai.resolve_confirmation(token, user_id, True) is None
    assert not ai.check_ai_rate_limit(user_id)


def test_expiry_is_committed_before_the_refund_takes_another_connection(confirmation, monkeypatch):
    token, confirmation_id, user_id = confirmation
    with ai._ai_db() as conn:
        conn.execute(
            "UPDATE ai_confirmations SET created_at = 0 WHERE confirmation_id = %s",
            (confirmation_id,),
        )
    refunds = []

    def refund(owner):
        # A separate connection only sees expired after the first transaction
        # commits. Refunding while that transaction holds a pool connection
        # can deadlock when every worker is expiring a confirmation at once.
        with ai._ai_db() as conn:
            row = conn.execute(
                "SELECT status FROM ai_confirmations WHERE confirmation_id = %s", (confirmation_id,)
            ).fetchone()
        assert row["status"] == "expired"
        refunds.append(owner)

    monkeypatch.setattr(ai, "_refund_ai_rate_slot", refund)
    assert ai.resolve_confirmation(token, user_id, True) is None
    assert refunds == [user_id]
