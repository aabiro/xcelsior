"""Concrete privacy sink behavior against the migrated PostgreSQL schema."""

from __future__ import annotations

import time
import uuid

import pytest

from privacy_deletion import (
    _REQUEST_COLUMNS,
    _row_mapping,
    create_deletion_request,
)
from privacy_sinks import (
    delete_authoritative_subject,
    delete_retrieval_subject,
    verify_subject_absence,
)

try:
    from db import UserStore, _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _conn:
        _has_schema = (
            _conn.execute(
                "SELECT to_regclass('privacy_deletion_requests')"
            ).fetchone()[0]
            is not None
        )
except Exception as _exc:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"privacy sink database unavailable: {_exc}")
    _pool = None
else:
    if not _has_schema:  # pragma: no cover
        pytestmark = pytest.mark.skip("privacy schema missing; upgrade to >= 081")


@pytest.fixture
def subject(monkeypatch):
    monkeypatch.setenv(
        "XCELSIOR_PRIVACY_REFERENCE_SECRET",
        "sink-test-privacy-reference-secret-with-enough-entropy",
    )
    suffix = uuid.uuid4().hex
    user_id = f"privacy-sink-user-{suffix}"
    email = f"privacy-sink-{suffix}@example.test"
    customer_id = f"privacy-sink-customer-{suffix}"
    UserStore.create_user(
        {
            "user_id": user_id,
            "email": email,
            "customer_id": customer_id,
            "name": "Sensitive Name",
            "password_hash": "sensitive-password-hash",
            "salt": "sensitive-salt",
            "role": "submitter",
        }
    )
    receipt = create_deletion_request(
        user_id=user_id,
        email=email,
        customer_ids=[customer_id],
        idempotency_key=f"privacy-sink-test-{suffix}",
        requested_by=user_id,
    )
    with _pool.connection() as conn:
        request = _row_mapping(
            conn.execute(
                """
                SELECT * FROM privacy_deletion_requests
                 WHERE request_id = %s
                """,
                (receipt.request_id,),
            ).fetchone(),
            _REQUEST_COLUMNS,
        )
        conn.rollback()

    context = {
        "user_id": user_id,
        "email": email,
        "customer_id": customer_id,
        "request": request,
        "receipt": receipt,
    }
    yield context

    with _pool.connection() as conn:
        conn.execute(
            """
            DELETE FROM chat_feedback
             WHERE message_id IN (
                 SELECT m.id::text
                   FROM chat_messages m
                   JOIN chat_conversations c
                     ON c.conversation_id = m.conversation_id
                  WHERE c.user_email IN (%s, %s)
             )
            """,
            (email, f"erased+{request['subject_reference_hash'][:24]}@deleted.invalid"),
        )
        conn.execute(
            """
            DELETE FROM chat_messages
             WHERE conversation_id IN (
                 SELECT conversation_id FROM chat_conversations
                  WHERE user_email IN (%s, %s)
             )
            """,
            (email, f"erased+{request['subject_reference_hash'][:24]}@deleted.invalid"),
        )
        conn.execute(
            "DELETE FROM chat_conversations WHERE user_email IN (%s, %s)",
            (email, f"erased+{request['subject_reference_hash'][:24]}@deleted.invalid"),
        )
        conn.execute(
            "DELETE FROM ai_confirmations WHERE user_id = %s", (user_id,)
        )
        conn.execute(
            "DELETE FROM ai_conversations WHERE user_id = %s", (user_id,)
        )
        conn.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))
        conn.execute("DELETE FROM casl_consent WHERE user_id = %s", (user_id,))
        conn.execute(
            "DELETE FROM consent_records WHERE entity_id = ANY(%s)",
            ([user_id, email, customer_id],),
        )
        conn.execute(
            "DELETE FROM retention_records WHERE entity_id IN (%s, %s, %s)",
            (user_id, email, request["subject_reference_hash"]),
        )
        conn.execute(
            "DELETE FROM user_encryption_keys WHERE user_id = %s", (user_id,)
        )
        conn.execute(
            "DELETE FROM wallet_holds WHERE customer_id = %s", (customer_id,)
        )
        conn.execute(
            "DELETE FROM wallet_transactions WHERE customer_id = %s",
            (customer_id,),
        )
        conn.execute(
            "DELETE FROM payment_intents WHERE customer_id = %s", (customer_id,)
        )
        conn.execute("DELETE FROM invoices WHERE customer_id = %s", (customer_id,))
        conn.execute("DELETE FROM wallets WHERE customer_id = %s", (customer_id,))
        conn.execute(
            "DELETE FROM privacy_deletion_sink_status WHERE request_id = %s",
            (receipt.request_id,),
        )
        conn.execute(
            "DELETE FROM privacy_deletion_requests WHERE request_id = %s",
            (receipt.request_id,),
        )
        conn.execute(
            """
            DELETE FROM outbox_events
             WHERE aggregate_type = 'privacy_deletion'
               AND aggregate_id = %s
            """,
            (receipt.request_id,),
        )
        conn.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
        conn.commit()


def test_authority_revokes_access_anonymizes_identity_and_retains_finance(subject):
    from billing import get_billing_engine

    user_id = subject["user_id"]
    email = subject["email"]
    customer_id = subject["customer_id"]
    UserStore.create_session(
        {
            "token": f"privacy-session-{uuid.uuid4().hex}",
            "email": email,
            "user_id": user_id,
            "role": "submitter",
            "name": "Sensitive Name",
            "created_at": time.time(),
            "expires_at": time.time() + 3_600,
        }
    )
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO casl_consent (
                consent_id, user_id, tenant_id, consent_type, purpose,
                granted_at, expires_at, withdrawn_at, source, ip_address, active
            ) VALUES (
                %s, %s, %s, 'express', 'newsletter', to_timestamp(%s),
                NULL, NULL, 'test', '203.0.113.10', true
            )
            """,
            (f"consent-{uuid.uuid4().hex}", user_id, user_id, time.time()),
        )
        conn.execute(
            """
            INSERT INTO user_encryption_keys (user_id, tenant_id, fernet_key)
            VALUES (%s, %s, 'test-fernet-key')
            """,
            (user_id, user_id),
        )
        conn.commit()
    get_billing_engine().deposit(
        customer_id,
        10,
        description="privacy retention fixture",
        idempotency_key=f"privacy-deposit-{uuid.uuid4().hex}",
    )

    outcome = delete_authoritative_subject(subject["request"], {})
    assert outcome.status == "completed"
    assert outcome.evidence["identity_anonymized"] is True
    assert outcome.evidence["finance_records"].startswith("retained")

    with _pool.connection() as conn:
        user = conn.execute(
            """
            SELECT email, name, provider_id, notifications_enabled,
                   email_verified, is_admin
              FROM users WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()
        sessions = conn.execute(
            "SELECT count(*) FROM sessions WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
        consents = conn.execute(
            "SELECT count(*) FROM casl_consent WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
        key = conn.execute(
            """
            SELECT active, fernet_key FROM user_encryption_keys
             WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()
        wallet = conn.execute(
            """
            SELECT balance_micros, auto_topup_enabled,
                   stripe_payment_method_id
              FROM wallets WHERE customer_id = %s
            """,
            (customer_id,),
        ).fetchone()
    assert user[0].endswith("@deleted.invalid")
    assert user[0] != email
    assert user[1] == "Deleted account"
    assert user[2] is None
    assert not bool(user[3])
    assert not bool(user[4])
    assert int(user[5]) == 0
    assert sessions == 0
    assert consents == 0
    assert key[0] is False and key[1] == "DESTROYED"
    assert int(wallet[0]) == 10_000_000
    assert wallet[1] is False
    assert wallet[2] is None


def test_retrieval_sink_deletes_conversations_and_verifier_passes(subject):
    conversation_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO ai_conversations (
                conversation_id, user_id, title, created_at, updated_at
            ) VALUES (%s, %s, 'Sensitive conversation', %s, %s)
            """,
            (
                conversation_id,
                subject["user_id"],
                time.time(),
                time.time(),
            ),
        )
        conn.execute(
            """
            INSERT INTO ai_messages (
                message_id, conversation_id, role, content, created_at
            ) VALUES (%s, %s, 'user', 'sensitive prompt', %s)
            """,
            (str(uuid.uuid4()), conversation_id, time.time()),
        )
        conn.execute(
            """
            INSERT INTO chat_conversations (
                conversation_id, ip_hash, user_email, created_at, updated_at
            ) VALUES (%s, 'hashed-ip', %s, %s, %s)
            """,
            (chat_id, subject["email"], time.time(), time.time()),
        )
        conn.execute(
            """
            INSERT INTO chat_messages (
                conversation_id, role, content, created_at
            ) VALUES (%s, 'user', 'sensitive support message', %s)
            """,
            (chat_id, time.time()),
        )
        conn.commit()

    authority = delete_authoritative_subject(subject["request"], {})
    retrieval = delete_retrieval_subject(subject["request"], {})
    verification = verify_subject_absence(subject["request"], {})

    assert authority.status == "completed"
    assert retrieval.status == "completed"
    assert retrieval.evidence["rows_deleted_or_anonymized"]["ai_conversations"] == 1
    assert retrieval.evidence["rows_deleted_or_anonymized"]["chat_conversations"] == 1
    assert verification.status == "completed"
    assert verification.evidence["residual_counts"] == {}
