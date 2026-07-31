"""Durable cross-store account deletion orchestration.

The HTTP layer only creates and inspects requests.  A background worker owns
the destructive work, records an outcome for every configured sink, and never
reports completion while a dependency is failed or still deleting data.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence

from psycopg.types.json import Jsonb

log = logging.getLogger("xcelsior.privacy.deletion")

SINK_ORDER: tuple[str, ...] = (
    "authority",
    "redis",
    "artifacts",
    "retrieval",
    "analytics",
    "posthog",
    "verification",
)
FINAL_SINK_STATUSES = frozenset(
    {"completed", "not_applicable", "legal_hold"}
)
MAX_ERROR_LENGTH = 2_000


class PrivacyDeletionError(RuntimeError):
    """Base class for request and worker errors."""


class PrivacyDeletionAccessDenied(PrivacyDeletionError):
    """The caller cannot inspect the requested workflow."""


class PrivacyDeletionNotFound(PrivacyDeletionError):
    """No deletion workflow exists for the supplied id."""


@dataclass(frozen=True)
class DeletionReceipt:
    request_id: str
    state: str
    deadline_at: datetime
    status_token: str
    already_existed: bool


@dataclass(frozen=True)
class SinkOutcome:
    status: str
    evidence: Mapping[str, Any]
    retry_after_sec: int = 300
    external_reference: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        allowed = {
            "pending",
            "completed",
            "not_applicable",
            "legal_hold",
            "failed",
        }
        if self.status not in allowed:
            raise ValueError(f"invalid privacy sink outcome: {self.status}")


SinkHandler = Callable[[Mapping[str, Any], Mapping[str, Any]], SinkOutcome]


def _pool():
    from db import _get_pg_pool

    return _get_pg_pool()


def _row_value(row: Any, name: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return row[index]


def _row_mapping(row: Any, columns: Sequence[str]) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    return dict(zip(columns, row, strict=True))


_REQUEST_COLUMNS = (
    "request_id",
    "subject_reference_hash",
    "subject_user_id",
    "subject_email",
    "subject_customer_ids",
    "requested_by",
    "request_source",
    "legal_basis",
    "idempotency_key",
    "status_token_hash",
    "state",
    "deadline_at",
    "claim_owner",
    "claim_token",
    "claim_expires_at",
    "attempt_count",
    "next_attempt_at",
    "last_error",
    "evidence",
    "request_event_id",
    "created_at",
    "updated_at",
    "validated_at",
    "completed_at",
)

_SINK_COLUMNS = (
    "request_id",
    "sink",
    "status",
    "attempt_count",
    "deadline_at",
    "next_attempt_at",
    "last_error",
    "external_reference",
    "evidence",
    "started_at",
    "completed_at",
    "updated_at",
)


def _reference_secret() -> bytes:
    explicit = os.environ.get("XCELSIOR_PRIVACY_REFERENCE_SECRET", "").strip()
    if explicit:
        return explicit.encode()
    # The OAuth signing secret is already mandatory in deployed profiles and
    # is a safe compatibility bridge while the dedicated secret is rolled out.
    oauth_secret = os.environ.get("XCELSIOR_OAUTH_JWT_SECRET", "").strip()
    if oauth_secret:
        return oauth_secret.encode()
    if os.environ.get("XCELSIOR_ENV", "").strip().lower() in {
        "prod",
        "production",
        "staging",
    }:
        raise PrivacyDeletionError(
            "XCELSIOR_PRIVACY_REFERENCE_SECRET is required in deployed environments"
        )
    return b"xcelsior-development-privacy-reference-only"


def subject_reference(user_id: str, email: str) -> str:
    identity = f"{str(user_id).strip()}\x1f{str(email).strip().lower()}"
    return hmac.new(_reference_secret(), identity.encode(), hashlib.sha256).hexdigest()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _deadline_days() -> int:
    try:
        configured = int(os.environ.get("XCELSIOR_PRIVACY_DELETION_DEADLINE_DAYS", "30"))
    except ValueError as exc:
        raise PrivacyDeletionError(
            "XCELSIOR_PRIVACY_DELETION_DEADLINE_DAYS must be an integer"
        ) from exc
    if configured < 1 or configured > 45:
        raise PrivacyDeletionError(
            "privacy deletion deadline must be between 1 and 45 days"
        )
    return configured


def _normalize_customer_ids(values: Sequence[str] | None) -> list[str]:
    return sorted(
        {
            str(value).strip()
            for value in (values or ())
            if str(value).strip()
        }
    )


def _append_privacy_event(
    conn: Any,
    *,
    request_id: str,
    event_type: str,
    reference: str,
    deadline_at: datetime,
    aggregate_version: int = 0,
) -> str | None:
    from control_plane.outbox import append_event

    return append_event(
        conn,
        aggregate_type="privacy_deletion",
        aggregate_id=request_id,
        aggregate_version=aggregate_version,
        event_type=event_type,
        payload={
            "request_id": request_id,
            "subject_reference_hash": reference,
            "deadline_at": deadline_at.isoformat(),
        },
        headers={"classification": "pii"},
        idempotency_key=f"{event_type}:{request_id}:{aggregate_version}",
    )


def create_deletion_request(
    *,
    user_id: str,
    email: str,
    customer_ids: Sequence[str] | None,
    idempotency_key: str,
    requested_by: str,
    request_source: str = "self_service",
    legal_basis: str = "user_request",
) -> DeletionReceipt:
    """Create, or return, the one active request for this subject.

    Repeating an authenticated request rotates the unguessable status token.
    That lets a user recover tracking access without storing the raw token in
    PostgreSQL.
    """
    user_id = str(user_id).strip()
    email = str(email).strip().lower()
    idempotency_key = str(idempotency_key).strip()
    if not user_id or not email:
        raise PrivacyDeletionError("a user id and email are required")
    if not 8 <= len(idempotency_key) <= 128:
        raise PrivacyDeletionError("Idempotency-Key must be 8 to 128 characters")

    reference = subject_reference(user_id, email)
    status_token = secrets.token_urlsafe(32)
    status_token_hash = _token_hash(status_token)
    deadline_at = datetime.now(timezone.utc) + timedelta(days=_deadline_days())
    normalized_customers = _normalize_customer_ids(customer_ids)

    pool = _pool()
    with pool.connection() as conn:
        # Serialize requests for one pseudonymous subject so two different
        # idempotency keys cannot race the partial unique index.
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"privacy-deletion:{reference}",),
        )
        existing = conn.execute(
            """
            SELECT *
              FROM privacy_deletion_requests
             WHERE subject_reference_hash = %s
               AND state NOT IN ('completed', 'cancelled')
             ORDER BY created_at DESC
             LIMIT 1
               FOR UPDATE
            """,
            (reference,),
        ).fetchone()

        if existing is not None:
            request = _row_mapping(existing, _REQUEST_COLUMNS)
            conn.execute(
                """
                UPDATE privacy_deletion_requests
                   SET status_token_hash = %s,
                       updated_at = clock_timestamp()
                 WHERE request_id = %s
                """,
                (status_token_hash, request["request_id"]),
            )
            conn.commit()
            return DeletionReceipt(
                request_id=str(request["request_id"]),
                state=str(request["state"]),
                deadline_at=request["deadline_at"],
                status_token=status_token,
                already_existed=True,
            )

        row = conn.execute(
            """
            INSERT INTO privacy_deletion_requests (
                subject_reference_hash, subject_user_id, subject_email,
                subject_customer_ids, requested_by, request_source, legal_basis,
                idempotency_key, status_token_hash, deadline_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING request_id, state, deadline_at
            """,
            (
                reference,
                user_id,
                email,
                Jsonb(normalized_customers),
                str(requested_by)[:256],
                str(request_source)[:64],
                str(legal_basis)[:64],
                idempotency_key,
                status_token_hash,
                deadline_at,
            ),
        ).fetchone()
        request_id = str(_row_value(row, "request_id", 0))
        state = str(_row_value(row, "state", 1))
        stored_deadline = _row_value(row, "deadline_at", 2)
        conn.execute(
            """
            INSERT INTO privacy_deletion_sink_status
                (request_id, sink, deadline_at)
            SELECT %s, sink, %s
              FROM unnest(%s::text[]) AS sink
            """,
            (request_id, stored_deadline, list(SINK_ORDER)),
        )
        event_id = _append_privacy_event(
            conn,
            request_id=request_id,
            event_type="privacy.v1.deletion_requested",
            reference=reference,
            deadline_at=stored_deadline,
        )
        if event_id:
            conn.execute(
                """
                UPDATE privacy_deletion_requests
                   SET request_event_id = %s
                 WHERE request_id = %s
                """,
                (event_id, request_id),
            )
        conn.commit()

    return DeletionReceipt(
        request_id=request_id,
        state=state,
        deadline_at=stored_deadline,
        status_token=status_token,
        already_existed=False,
    )


def get_deletion_status(
    request_id: str,
    *,
    caller_user_id: str | None = None,
    status_token: str | None = None,
    is_admin: bool = False,
) -> dict[str, Any]:
    pool = _pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT * FROM privacy_deletion_requests WHERE request_id = %s",
            (request_id,),
        ).fetchone()
        if row is None:
            raise PrivacyDeletionNotFound("deletion request not found")
        request = _row_mapping(row, _REQUEST_COLUMNS)
        authorized = bool(is_admin)
        if caller_user_id and request["subject_user_id"]:
            authorized = authorized or hmac.compare_digest(
                str(caller_user_id), str(request["subject_user_id"])
            )
        if status_token:
            authorized = authorized or hmac.compare_digest(
                _token_hash(status_token), str(request["status_token_hash"])
            )
        if not authorized:
            # Deliberately indistinguishable from a missing UUID to callers.
            raise PrivacyDeletionAccessDenied("deletion request not found")

        rows = conn.execute(
            """
            SELECT *
              FROM privacy_deletion_sink_status
             WHERE request_id = %s
             ORDER BY array_position(%s::text[], sink)
            """,
            (request_id, list(SINK_ORDER)),
        ).fetchall()
        conn.rollback()

    sinks = []
    for raw in rows:
        sink = _row_mapping(raw, _SINK_COLUMNS)
        sinks.append(
            {
                "sink": sink["sink"],
                "status": sink["status"],
                "attempt_count": sink["attempt_count"],
                "deadline_at": sink["deadline_at"],
                "last_error": sink["last_error"],
                "evidence": sink["evidence"] or {},
                "updated_at": sink["updated_at"],
                "completed_at": sink["completed_at"],
            }
        )
    return {
        "request_id": str(request["request_id"]),
        "state": request["state"],
        "deadline_at": request["deadline_at"],
        "created_at": request["created_at"],
        "updated_at": request["updated_at"],
        "completed_at": request["completed_at"],
        "last_error": request["last_error"],
        "sinks": sinks,
    }


def _worker_identity() -> str:
    return f"privacy-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:10]}"


def _claim_due_requests(
    *,
    worker_id: str,
    limit: int,
    claim_ttl_sec: int,
) -> list[dict[str, Any]]:
    pool = _pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            WITH due AS (
                SELECT request_id
                  FROM privacy_deletion_requests
                 WHERE state NOT IN ('completed', 'cancelled', 'failed')
                   AND next_attempt_at <= clock_timestamp()
                   AND (
                       claim_expires_at IS NULL
                       OR claim_expires_at < clock_timestamp()
                   )
                 ORDER BY deadline_at, created_at
                 LIMIT %s
                   FOR UPDATE SKIP LOCKED
            )
            UPDATE privacy_deletion_requests r
               SET claim_owner = %s,
                   claim_token = gen_random_uuid(),
                   claim_expires_at = clock_timestamp()
                       + make_interval(secs => %s),
                   attempt_count = r.attempt_count + 1,
                   updated_at = clock_timestamp()
              FROM due
             WHERE r.request_id = due.request_id
            RETURNING r.*
            """,
            (limit, worker_id, claim_ttl_sec),
        ).fetchall()
        conn.commit()
    return [_row_mapping(row, _REQUEST_COLUMNS) for row in rows]


def _load_sink_rows(request_id: str) -> list[dict[str, Any]]:
    pool = _pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT *
              FROM privacy_deletion_sink_status
             WHERE request_id = %s
             ORDER BY array_position(%s::text[], sink)
            """,
            (request_id, list(SINK_ORDER)),
        ).fetchall()
        conn.rollback()
    return [_row_mapping(row, _SINK_COLUMNS) for row in rows]


def _mark_sink_started(request_id: str, sink: str, claim_token: str) -> bool:
    pool = _pool()
    with pool.connection() as conn:
        result = conn.execute(
            """
            UPDATE privacy_deletion_sink_status s
               SET status = 'in_progress',
                   attempt_count = s.attempt_count + 1,
                   started_at = COALESCE(s.started_at, clock_timestamp()),
                   updated_at = clock_timestamp(),
                   last_error = NULL
              FROM privacy_deletion_requests r
             WHERE s.request_id = r.request_id
               AND s.request_id = %s
               AND s.sink = %s
               AND r.claim_token = %s
               AND r.claim_expires_at > clock_timestamp()
            """,
            (request_id, sink, claim_token),
        )
        conn.commit()
        return result.rowcount == 1


def _record_sink_outcome(
    request_id: str,
    sink: str,
    claim_token: str,
    outcome: SinkOutcome,
) -> bool:
    now = datetime.now(timezone.utc)
    retry_at = now + timedelta(seconds=max(5, outcome.retry_after_sec))
    completed = outcome.status in FINAL_SINK_STATUSES
    pool = _pool()
    with pool.connection() as conn:
        result = conn.execute(
            """
            UPDATE privacy_deletion_sink_status s
               SET status = %s,
                   evidence = %s,
                   external_reference = %s,
                   last_error = %s,
                   next_attempt_at = %s,
                   completed_at = CASE
                       WHEN %s THEN clock_timestamp()
                       ELSE NULL
                   END,
                   updated_at = clock_timestamp()
              FROM privacy_deletion_requests r
             WHERE s.request_id = r.request_id
               AND s.request_id = %s
               AND s.sink = %s
               AND r.claim_token = %s
               AND r.claim_expires_at > clock_timestamp()
            """,
            (
                outcome.status,
                Jsonb(dict(outcome.evidence)),
                outcome.external_reference,
                (outcome.error or "")[:MAX_ERROR_LENGTH] or None,
                retry_at,
                completed,
                request_id,
                sink,
                claim_token,
            ),
        )
        if result.rowcount == 1 and not completed:
            conn.execute(
                """
                UPDATE privacy_deletion_requests
                   SET next_attempt_at = %s,
                       last_error = %s,
                       updated_at = clock_timestamp()
                 WHERE request_id = %s
                   AND claim_token = %s
                """,
                (
                    retry_at,
                    (outcome.error or f"{sink} is still pending")[
                        :MAX_ERROR_LENGTH
                    ],
                    request_id,
                    claim_token,
                ),
            )
        conn.commit()
        return result.rowcount == 1


def _release_claim(
    request_id: str,
    claim_token: str,
    *,
    state: str | None = None,
    next_attempt_sec: int = 30,
    error: str | None = None,
) -> None:
    pool = _pool()
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE privacy_deletion_requests
               SET state = COALESCE(%s, state),
                   claim_owner = NULL,
                   claim_token = NULL,
                   claim_expires_at = NULL,
                   next_attempt_at = clock_timestamp()
                       + make_interval(secs => %s),
                   last_error = %s,
                   updated_at = clock_timestamp()
             WHERE request_id = %s
               AND claim_token = %s
            """,
            (
                state,
                max(5, next_attempt_sec),
                error[:MAX_ERROR_LENGTH] if error else None,
                request_id,
                claim_token,
            ),
        )
        conn.commit()


def _complete_request(request: Mapping[str, Any], claim_token: str) -> bool:
    request_id = str(request["request_id"])
    pool = _pool()
    with pool.connection() as conn:
        statuses = conn.execute(
            """
            SELECT sink, status
              FROM privacy_deletion_sink_status
             WHERE request_id = %s
            """,
            (request_id,),
        ).fetchall()
        if len(statuses) != len(SINK_ORDER) or any(
            str(_row_value(row, "status", 1)) not in FINAL_SINK_STATUSES
            for row in statuses
        ):
            conn.rollback()
            return False

        event_id = _append_privacy_event(
            conn,
            request_id=request_id,
            event_type="privacy.v1.deletion_completed",
            reference=str(request["subject_reference_hash"]),
            deadline_at=request["deadline_at"],
            aggregate_version=int(request["attempt_count"]),
        )
        result = conn.execute(
            """
            UPDATE privacy_deletion_requests
               SET state = 'completed',
                   subject_user_id = NULL,
                   subject_email = NULL,
                   subject_customer_ids = '[]'::jsonb,
                   claim_owner = NULL,
                   claim_token = NULL,
                   claim_expires_at = NULL,
                   last_error = NULL,
                   evidence = evidence || %s,
                   completed_at = clock_timestamp(),
                   updated_at = clock_timestamp()
             WHERE request_id = %s
               AND claim_token = %s
            """,
            (
                Jsonb(
                    {
                        "completion_event_id": event_id,
                        "identifiers_scrubbed": True,
                    }
                ),
                request_id,
                claim_token,
            ),
        )
        conn.commit()
        return result.rowcount == 1


def _fail_expired_request(request: Mapping[str, Any], claim_token: str) -> None:
    request_id = str(request["request_id"])
    pool = _pool()
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE privacy_deletion_sink_status
               SET status = CASE
                       WHEN status IN ('completed', 'not_applicable', 'legal_hold')
                           THEN status
                       ELSE 'failed'
                   END,
                   last_error = CASE
                       WHEN status IN ('completed', 'not_applicable', 'legal_hold')
                           THEN last_error
                       ELSE 'deletion deadline missed'
                   END,
                   updated_at = clock_timestamp()
             WHERE request_id = %s
            """,
            (request_id,),
        )
        conn.execute(
            """
            UPDATE privacy_deletion_requests
               SET state = 'failed',
                   claim_owner = NULL,
                   claim_token = NULL,
                   claim_expires_at = NULL,
                   last_error = 'privacy deletion deadline missed',
                   updated_at = clock_timestamp()
             WHERE request_id = %s
               AND claim_token = %s
            """,
            (request_id, claim_token),
        )
        conn.commit()
    log.error("privacy deletion %s missed its compliance deadline", request_id)


def _default_handlers() -> dict[str, SinkHandler]:
    from privacy_sinks import (
        delete_analytics_subject,
        delete_artifact_subject,
        delete_authoritative_subject,
        delete_posthog_subject,
        delete_redis_subject,
        delete_retrieval_subject,
        verify_subject_absence,
    )

    return {
        "authority": delete_authoritative_subject,
        "redis": delete_redis_subject,
        "artifacts": delete_artifact_subject,
        "retrieval": delete_retrieval_subject,
        "analytics": delete_analytics_subject,
        "posthog": delete_posthog_subject,
        "verification": verify_subject_absence,
    }


def _process_claimed_request(
    request: Mapping[str, Any],
    *,
    handlers: Mapping[str, SinkHandler],
) -> str:
    request_id = str(request["request_id"])
    claim_token = str(request["claim_token"])
    if request["deadline_at"] <= datetime.now(timezone.utc):
        _fail_expired_request(request, claim_token)
        return "deadline_failed"

    # Validation is deliberately durable, even though today's validation is
    # structural.  Future legal-hold and organization-ownership checks can add
    # evidence here without changing the workflow states.
    if request["state"] == "requested":
        pool = _pool()
        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE privacy_deletion_requests
                   SET state = 'validated',
                       validated_at = clock_timestamp(),
                       updated_at = clock_timestamp()
                 WHERE request_id = %s
                   AND claim_token = %s
                """,
                (request_id, claim_token),
            )
            conn.commit()

    sink_rows = _load_sink_rows(request_id)
    by_name = {str(row["sink"]): row for row in sink_rows}
    for sink in SINK_ORDER:
        current = by_name[sink]
        if str(current["status"]) in FINAL_SINK_STATUSES:
            continue
        if sink == "verification":
            prior = [
                by_name[name]
                for name in SINK_ORDER
                if name != "verification"
            ]
            if any(
                str(row["status"]) not in FINAL_SINK_STATUSES
                for row in prior
            ):
                _release_claim(
                    request_id,
                    claim_token,
                    state="processing",
                    next_attempt_sec=30,
                )
                return "waiting_for_sinks"

        if not _mark_sink_started(request_id, sink, claim_token):
            return "lost_claim"
        try:
            outcome = handlers[sink](request, current)
        except Exception as exc:
            log.exception(
                "privacy deletion %s sink %s failed", request_id, sink
            )
            outcome = SinkOutcome(
                status="failed",
                evidence={"exception_type": type(exc).__name__},
                retry_after_sec=300,
                error=str(exc),
            )
        if not _record_sink_outcome(
            request_id, sink, claim_token, outcome
        ):
            return "lost_claim"
        by_name[sink] = {**current, "status": outcome.status}
        if outcome.status not in FINAL_SINK_STATUSES:
            _release_claim(
                request_id,
                claim_token,
                state="processing",
                next_attempt_sec=outcome.retry_after_sec,
                error=outcome.error,
            )
            return outcome.status

    if _complete_request(request, claim_token):
        return "completed"
    _release_claim(
        request_id,
        claim_token,
        state="verifying",
        next_attempt_sec=30,
    )
    return "waiting_for_sinks"


def process_deletion_requests_task(
    *,
    limit: int = 10,
    claim_ttl_sec: int = 180,
    handlers: Mapping[str, SinkHandler] | None = None,
    worker_id: str | None = None,
) -> dict[str, int]:
    """Claim and advance a bounded batch of privacy workflows."""
    if limit < 1 or limit > 100:
        raise ValueError("privacy deletion batch limit must be 1 to 100")
    if claim_ttl_sec < 30 or claim_ttl_sec > 3_600:
        raise ValueError("privacy deletion claim TTL must be 30 to 3600 seconds")
    active_handlers = dict(handlers or _default_handlers())
    missing = set(SINK_ORDER) - set(active_handlers)
    if missing:
        raise ValueError(f"missing privacy deletion handlers: {sorted(missing)}")

    claimed = _claim_due_requests(
        worker_id=worker_id or _worker_identity(),
        limit=limit,
        claim_ttl_sec=claim_ttl_sec,
    )
    stats = {
        "claimed": len(claimed),
        "completed": 0,
        "pending": 0,
        "failed": 0,
        "lost_claim": 0,
    }
    for request in claimed:
        outcome = _process_claimed_request(request, handlers=active_handlers)
        if outcome == "completed":
            stats["completed"] += 1
        elif outcome in {"failed", "deadline_failed"}:
            stats["failed"] += 1
        elif outcome == "lost_claim":
            stats["lost_claim"] += 1
        else:
            stats["pending"] += 1
    return stats
