"""Durable, honest cross-store privacy deletion workflow."""

from __future__ import annotations

import concurrent.futures
import uuid

import pytest

from privacy_deletion import (
    FINAL_SINK_STATUSES,
    SINK_ORDER,
    PrivacyDeletionAccessDenied,
    SinkOutcome,
    _claim_due_requests,
    create_deletion_request,
    get_deletion_status,
    process_deletion_requests_task,
)

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _conn:
        _has_schema = (
            _conn.execute(
                "SELECT to_regclass('privacy_deletion_requests')"
            ).fetchone()[0]
            is not None
        )
except Exception as _exc:  # pragma: no cover - environment gate
    pytestmark = pytest.mark.skip(f"no PostgreSQL privacy schema: {_exc}")
    _pool = None
else:
    if not _has_schema:  # pragma: no cover
        pytestmark = pytest.mark.skip("privacy schema missing; upgrade to >= 081")


@pytest.fixture
def made_requests(monkeypatch):
    monkeypatch.setenv(
        "XCELSIOR_PRIVACY_REFERENCE_SECRET",
        "unit-test-privacy-reference-secret-with-enough-entropy",
    )
    request_ids: list[str] = []
    yield request_ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        if request_ids:
            conn.execute(
                """
                DELETE FROM privacy_deletion_sink_status
                 WHERE request_id = ANY(%s::uuid[])
                """,
                (request_ids,),
            )
            conn.execute(
                """
                DELETE FROM privacy_deletion_requests
                 WHERE request_id = ANY(%s::uuid[])
                """,
                (request_ids,),
            )
            conn.execute(
                """
                DELETE FROM outbox_events
                 WHERE aggregate_type = 'privacy_deletion'
                   AND aggregate_id = ANY(%s)
                """,
                (request_ids,),
            )
        conn.commit()


def _create(made_requests, *, suffix: str = ""):
    unique = uuid.uuid4().hex
    receipt = create_deletion_request(
        user_id=f"privacy-user-{unique}",
        email=f"privacy-{unique}@example.test",
        customer_ids=[f"privacy-customer-{unique}"],
        idempotency_key=f"privacy-test-{unique}{suffix}",
        requested_by=f"privacy-user-{unique}",
    )
    made_requests.append(receipt.request_id)
    return receipt


def _completed_handlers(
    calls: list[str] | None = None,
    *,
    artifact_status: str = "completed",
):
    handlers = {}
    for sink in SINK_ORDER:
        status = artifact_status if sink == "artifacts" else "completed"

        def _handler(_request, _current, *, _sink=sink, _status=status):
            if calls is not None:
                calls.append(_sink)
            return SinkOutcome(_status, {"sink": _sink, "verified": True})

        handlers[sink] = _handler
    return handlers


def test_request_is_idempotent_and_rotates_tracking_token(made_requests):
    unique = uuid.uuid4().hex
    args = {
        "user_id": f"privacy-user-{unique}",
        "email": f"privacy-{unique}@example.test",
        "customer_ids": [f"privacy-customer-{unique}"],
        "idempotency_key": f"privacy-idem-{unique}",
        "requested_by": f"privacy-user-{unique}",
    }
    first = create_deletion_request(**args)
    made_requests.append(first.request_id)
    second = create_deletion_request(**args)

    assert first.request_id == second.request_id
    assert first.already_existed is False
    assert second.already_existed is True
    assert first.status_token != second.status_token

    with pytest.raises(PrivacyDeletionAccessDenied):
        get_deletion_status(
            first.request_id, status_token=first.status_token
        )
    status = get_deletion_status(
        second.request_id, status_token=second.status_token
    )
    assert status["state"] == "requested"
    assert [sink["sink"] for sink in status["sinks"]] == list(SINK_ORDER)
    assert {sink["status"] for sink in status["sinks"]} == {"pending"}


def test_different_key_reuses_the_one_active_subject_request(made_requests):
    unique = uuid.uuid4().hex
    common = {
        "user_id": f"privacy-user-{unique}",
        "email": f"privacy-{unique}@example.test",
        "customer_ids": [f"privacy-customer-{unique}"],
        "requested_by": f"privacy-user-{unique}",
    }
    first = create_deletion_request(
        **common, idempotency_key=f"privacy-a-{unique}"
    )
    made_requests.append(first.request_id)
    second = create_deletion_request(
        **common, idempotency_key=f"privacy-b-{unique}"
    )
    assert second.request_id == first.request_id
    assert second.already_existed is True


def test_worker_completes_only_after_every_sink_is_terminal(made_requests):
    receipt = _create(made_requests)
    calls: list[str] = []
    result = process_deletion_requests_task(
        handlers=_completed_handlers(calls),
        worker_id=f"test-worker-{uuid.uuid4().hex}",
    )
    assert result["completed"] == 1
    assert calls == list(SINK_ORDER)

    status = get_deletion_status(
        receipt.request_id, status_token=receipt.status_token
    )
    assert status["state"] == "completed"
    assert status["completed_at"] is not None
    assert all(
        sink["status"] in FINAL_SINK_STATUSES for sink in status["sinks"]
    )
    with _pool.connection() as conn:
        row = conn.execute(
            """
            SELECT subject_user_id, subject_email, subject_customer_ids
              FROM privacy_deletion_requests
             WHERE request_id = %s
            """,
            (receipt.request_id,),
        ).fetchone()
    assert row[0] is None
    assert row[1] is None
    assert row[2] == []


def test_legal_hold_is_visible_and_does_not_block_other_sinks(made_requests):
    receipt = _create(made_requests)
    result = process_deletion_requests_task(
        handlers=_completed_handlers(artifact_status="legal_hold"),
        worker_id=f"test-worker-{uuid.uuid4().hex}",
    )
    assert result["completed"] == 1
    status = get_deletion_status(
        receipt.request_id, status_token=receipt.status_token
    )
    artifacts = next(
        sink for sink in status["sinks"] if sink["sink"] == "artifacts"
    )
    assert artifacts["status"] == "legal_hold"
    assert artifacts["evidence"]["verified"] is True


def test_failed_sink_is_not_reported_as_completed_and_retries(made_requests):
    receipt = _create(made_requests)
    attempts = {"redis": 0}
    handlers = _completed_handlers()

    def flaky(_request, _current):
        attempts["redis"] += 1
        if attempts["redis"] == 1:
            return SinkOutcome(
                "failed",
                {"dependency": "redis"},
                retry_after_sec=5,
                error="redis unavailable",
            )
        return SinkOutcome("completed", {"dependency": "redis"})

    handlers["redis"] = flaky
    first = process_deletion_requests_task(
        handlers=handlers,
        worker_id=f"test-worker-{uuid.uuid4().hex}",
    )
    assert first["failed"] == 1
    status = get_deletion_status(
        receipt.request_id, status_token=receipt.status_token
    )
    assert status["state"] == "processing"
    redis_status = next(
        sink for sink in status["sinks"] if sink["sink"] == "redis"
    )
    assert redis_status["status"] == "failed"
    assert redis_status["last_error"] == "redis unavailable"

    with _pool.connection() as conn:
        conn.execute(
            """
            UPDATE privacy_deletion_requests
               SET next_attempt_at = clock_timestamp()
             WHERE request_id = %s
            """,
            (receipt.request_id,),
        )
        conn.commit()
    second = process_deletion_requests_task(
        handlers=handlers,
        worker_id=f"test-worker-{uuid.uuid4().hex}",
    )
    assert second["completed"] == 1
    assert attempts["redis"] == 2


def test_skip_locked_claim_allows_only_one_worker(made_requests):
    _create(made_requests)

    def claim(worker: str):
        return _claim_due_requests(
            worker_id=worker, limit=1, claim_ttl_sec=180
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        batches = list(
            executor.map(
                claim,
                [f"privacy-worker-{uuid.uuid4().hex}" for _ in range(2)],
            )
        )
    assert sorted(len(batch) for batch in batches) == [0, 1]


def test_missed_deadline_is_failed_never_silently_completed(made_requests):
    receipt = _create(made_requests)
    with _pool.connection() as conn:
        conn.execute(
            """
            UPDATE privacy_deletion_requests
               SET deadline_at = clock_timestamp() - interval '1 second',
                   next_attempt_at = clock_timestamp()
             WHERE request_id = %s
            """,
            (receipt.request_id,),
        )
        conn.execute(
            """
            UPDATE privacy_deletion_sink_status
               SET deadline_at = clock_timestamp() - interval '1 second'
             WHERE request_id = %s
            """,
            (receipt.request_id,),
        )
        conn.commit()

    result = process_deletion_requests_task(
        handlers=_completed_handlers(),
        worker_id=f"test-worker-{uuid.uuid4().hex}",
    )
    assert result["failed"] == 1
    status = get_deletion_status(
        receipt.request_id, status_token=receipt.status_token
    )
    assert status["state"] == "failed"
    assert status["last_error"] == "privacy deletion deadline missed"
    assert all(
        sink["status"] == "failed" for sink in status["sinks"]
    )
