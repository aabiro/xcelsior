"""Concrete sink implementations for the privacy deletion workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Mapping

from psycopg.types.json import Jsonb

from cache_keys import opaque
from privacy_deletion import SinkOutcome

log = logging.getLogger("xcelsior.privacy.sinks")


def _pool():
    from db import _get_pg_pool

    return _get_pg_pool()


def _value(row: Any, name: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return row[index]


def _identifiers(request: Mapping[str, Any]) -> tuple[str, str, list[str]]:
    user_id = str(request.get("subject_user_id") or "").strip()
    email = str(request.get("subject_email") or "").strip().lower()
    raw_customers = request.get("subject_customer_ids") or []
    if isinstance(raw_customers, str):
        raw_customers = json.loads(raw_customers)
    customer_ids = sorted({str(value).strip() for value in raw_customers if str(value).strip()})
    return user_id, email, customer_ids


def _anonymized_email(reference: str) -> str:
    return f"erased+{reference[:24]}@deleted.invalid"


def _append_event(
    conn: Any,
    request: Mapping[str, Any],
    event_type: str,
    evidence: Mapping[str, Any],
) -> None:
    from control_plane.outbox import append_event

    request_id = str(request["request_id"])
    append_event(
        conn,
        aggregate_type="privacy_deletion",
        aggregate_id=request_id,
        aggregate_version=int(request.get("attempt_count") or 0),
        event_type=event_type,
        payload={
            "request_id": request_id,
            "subject_reference_hash": request["subject_reference_hash"],
            "evidence": dict(evidence),
        },
        headers={"classification": "pii"},
        idempotency_key=(f"{event_type}:{request_id}:{int(request.get('attempt_count') or 0)}"),
    )


def _stop_subject_workloads(conn: Any, customer_ids: list[str]) -> tuple[int, int]:
    if not customer_ids:
        return 0, 0
    result = conn.execute(
        """
        UPDATE jobs
           SET desired_state = 'stopped',
               generation = generation + 1,
               updated_at = clock_timestamp(),
               reason_code = 'privacy_deletion_requested',
               reason_details = jsonb_build_object(
                   'category', 'user_requested_deletion'
               )
         WHERE owner_id = ANY(%s)
           AND COALESCE(phase, '') NOT IN ('succeeded', 'failed', 'stopped')
           AND COALESCE(desired_state, 'running') <> 'stopped'
        """,
        (customer_ids,),
    )
    remaining = conn.execute(
        """
        SELECT count(*)
          FROM jobs
         WHERE owner_id = ANY(%s)
           AND COALESCE(phase, '') NOT IN ('succeeded', 'failed', 'stopped')
        """,
        (customer_ids,),
    ).fetchone()
    return result.rowcount, int(_value(remaining, "count", 0))


def delete_authoritative_subject(
    request: Mapping[str, Any], _sink: Mapping[str, Any]
) -> SinkOutcome:
    """Revoke identity access and anonymize authoritative relational data.

    Finance rows remain as legally governed records, but their direct mapping
    to the login identity is severed and payment methods/automatic charges are
    disabled.
    """
    user_id, email, request_customer_ids = _identifiers(request)
    reference = str(request["subject_reference_hash"])
    if not user_id or not email:
        return SinkOutcome(
            "completed",
            {"already_anonymized": True, "identifiers_present": False},
        )

    pool = _pool()
    with pool.connection() as conn:
        user = conn.execute(
            """
            SELECT user_id, email, customer_id, provider_id
              FROM users
             WHERE user_id = %s OR lower(email) = %s
             ORDER BY CASE WHEN user_id = %s THEN 0 ELSE 1 END
             LIMIT 1
               FOR UPDATE
            """,
            (user_id, email, user_id),
        ).fetchone()
        if user is None:
            conn.rollback()
            return SinkOutcome(
                "completed",
                {"already_anonymized": True, "identity_row_found": False},
            )

        stored_user_id = str(_value(user, "user_id", 0) or user_id)
        stored_email = str(_value(user, "email", 1) or email).lower()
        customer_id = str(_value(user, "customer_id", 2) or "").strip()
        provider_id = str(_value(user, "provider_id", 3) or "").strip()
        customer_ids = sorted(set(request_customer_ids + ([customer_id] if customer_id else [])))

        # Account deletion explicitly withdraws compute authorization.  The
        # worker waits for the scheduler/reconciler to observe the stop before
        # destroying credentials or storage.
        stop_requested, active_jobs = _stop_subject_workloads(conn, customer_ids)
        if active_jobs:
            conn.commit()
            return SinkOutcome(
                "pending",
                {
                    "active_jobs": active_jobs,
                    "stop_requests_written": stop_requested,
                },
                retry_after_sec=30,
                error="waiting for active workloads to stop",
            )

        # A team with other members cannot be orphaned.  The deletion remains
        # open and visible until ownership is transferred by an operator/user.
        owned_teams = conn.execute(
            """
            SELECT t.team_id,
                   count(tm.email) FILTER (
                       WHERE lower(tm.email) <> %s
                   ) AS other_members
              FROM teams t
              LEFT JOIN team_members tm ON tm.team_id = t.team_id
             WHERE lower(t.owner_email) = %s
             GROUP BY t.team_id
            """,
            (stored_email, stored_email),
        ).fetchall()
        blocked_teams = [
            str(_value(row, "team_id", 0))
            for row in owned_teams
            if int(_value(row, "other_members", 1) or 0) > 0
        ]
        if blocked_teams:
            conn.rollback()
            return SinkOutcome(
                "failed",
                {
                    "ownership_transfer_required": len(blocked_teams),
                    "team_reference_hashes": [
                        hashlib.sha256(team.encode()).hexdigest() for team in blocked_teams
                    ],
                },
                retry_after_sec=86_400,
                error="team ownership must be transferred before account deletion",
            )

        anonymous_email = _anonymized_email(reference)
        counts: dict[str, int] = {}

        # Credentials and live authorization first.
        delete_statements = (
            (
                "oauth_refresh_tokens",
                """
                DELETE FROM oauth_refresh_tokens
                 WHERE user_id = %s OR lower(email) = %s
                """,
                (stored_user_id, stored_email),
            ),
            (
                "sessions",
                "DELETE FROM sessions WHERE user_id = %s OR lower(email) = %s",
                (stored_user_id, stored_email),
            ),
            (
                "api_keys",
                "DELETE FROM api_keys WHERE user_id = %s OR lower(email) = %s",
                (stored_user_id, stored_email),
            ),
            (
                "oauth_clients",
                """
                DELETE FROM oauth_clients
                 WHERE lower(created_by_email) = %s
                   AND COALESCE(is_system_managed, 0) = 0
                """,
                (stored_email,),
            ),
            (
                "mfa_backup_codes",
                "DELETE FROM mfa_backup_codes WHERE lower(email) = %s",
                (stored_email,),
            ),
            (
                "mfa_challenges",
                "DELETE FROM mfa_challenges WHERE lower(email) = %s",
                (stored_email,),
            ),
            (
                "mfa_methods",
                "DELETE FROM mfa_methods WHERE lower(email) = %s",
                (stored_email,),
            ),
            (
                "ssh_keys",
                """
                DELETE FROM user_ssh_keys
                 WHERE user_id = %s OR lower(email) = %s
                """,
                (stored_user_id, stored_email),
            ),
            (
                "avatars",
                "DELETE FROM user_avatars WHERE user_id = %s",
                (stored_user_id,),
            ),
            (
                "push_subscriptions",
                """
                DELETE FROM web_push_subscriptions
                 WHERE lower(user_email) = %s
                """,
                (stored_email,),
            ),
            (
                "notifications",
                "DELETE FROM notifications WHERE lower(user_email) = %s",
                (stored_email,),
            ),
            (
                "casl_consents",
                "DELETE FROM casl_consent WHERE user_id = %s",
                (stored_user_id,),
            ),
            (
                "consent_records",
                """
                DELETE FROM consent_records
                 WHERE entity_id = ANY(%s)
                """,
                ([stored_user_id, stored_email, *customer_ids],),
            ),
        )
        for name, statement, params in delete_statements:
            counts[name] = conn.execute(statement, params).rowcount

        # Revoke serverless entry points while retaining metering/ledger rows.
        if customer_ids:
            endpoint_ids = [
                str(_value(row, "endpoint_id", 0))
                for row in conn.execute(
                    """
                    SELECT endpoint_id
                      FROM serverless_endpoints
                     WHERE owner_id = ANY(%s)
                       AND deleted_at = 0
                    """,
                    (customer_ids,),
                ).fetchall()
            ]
            if endpoint_ids:
                counts["serverless_keys"] = conn.execute(
                    """
                    UPDATE serverless_api_keys
                       SET revoked_at = COALESCE(
                           revoked_at, extract(epoch FROM clock_timestamp())
                       )
                     WHERE endpoint_id = ANY(%s)
                       AND revoked_at IS NULL
                    """,
                    (endpoint_ids,),
                ).rowcount
                counts["serverless_endpoints"] = conn.execute(
                    """
                    UPDATE serverless_endpoints
                       SET status = 'deleted',
                           deleted_at = extract(epoch FROM clock_timestamp()),
                           updated_at = extract(epoch FROM clock_timestamp())
                     WHERE endpoint_id = ANY(%s)
                       AND deleted_at = 0
                    """,
                    (endpoint_ids,),
                ).rowcount

            counts["job_logs"] = conn.execute(
                """
                DELETE FROM job_logs
                 WHERE job_id IN (
                     SELECT job_id FROM jobs WHERE owner_id = ANY(%s)
                 )
                """,
                (customer_ids,),
            ).rowcount
            counts["job_payloads"] = conn.execute(
                """
                UPDATE jobs
                       SET payload = jsonb_build_object(
                           'privacy_deleted', true,
                           'subject_reference_hash', %s::text
                       ),
                       spec = jsonb_build_object('privacy_deleted', true),
                       reason_details = COALESCE(
                           reason_details, '{}'::jsonb
                       ) || jsonb_build_object('privacy_deleted', true)
                 WHERE owner_id = ANY(%s)
                """,
                (reference, customer_ids),
            ).rowcount

            # Preserve wallet/finance reconciliation, but make any future
            # automatic debit impossible.
            counts["wallets_hardened"] = conn.execute(
                """
                UPDATE wallets
                   SET auto_topup_enabled = false,
                       stripe_payment_method_id = NULL,
                       auto_topup_failures = 0
                 WHERE customer_id = ANY(%s)
                """,
                (customer_ids,),
            ).rowcount

        if provider_id:
            counts["offers_delisted"] = conn.execute(
                """
                UPDATE gpu_offers
                   SET available = false,
                       updated_at = extract(epoch FROM clock_timestamp())
                 WHERE provider_id = %s
                   AND available
                """,
                (provider_id,),
            ).rowcount
            counts["hosts_drained"] = conn.execute(
                """
                UPDATE hosts
                   SET administrative_state = 'draining',
                       availability_state = 'unavailable',
                       drain_reason = 'provider_account_deleted',
                       generation = generation + 1
                 WHERE provider_id = %s
                   AND administrative_state <> 'decommissioned'
                """,
                (provider_id,),
            ).rowcount
            # Financial/tax rows are retained, with direct contact identity
            # removed. Legal names/rail IDs remain governed finance evidence.
            conn.execute(
                """
                UPDATE provider_accounts
                   SET status = 'deactivated',
                       email = %s
                 WHERE provider_id = %s
                """,
                (anonymous_email, provider_id),
            )

        counts["connect_contacts_anonymized"] = conn.execute(
            """
            UPDATE connect_accounts
               SET contact_email = %s,
                   display_name = 'Deleted provider'
             WHERE lower(contact_email) = %s
            """,
            (anonymous_email, stored_email),
        ).rowcount

        counts["retention_records"] = conn.execute(
            """
            UPDATE retention_records
               SET entity_id = %s,
                   purged_at = extract(epoch FROM clock_timestamp()),
                   purge_reason = 'right_to_erasure',
                   metadata = COALESCE(metadata, '{}'::jsonb)
                       || jsonb_build_object(
                           'privacy_request_id', %s::text
                       )
             WHERE entity_id = ANY(%s)
            """,
            (
                reference,
                str(request["request_id"]),
                [stored_user_id, stored_email, *customer_ids],
            ),
        ).rowcount

        # Destroy the per-user key only after every identifier needed by this
        # transaction has been captured.
        counts["encryption_keys_destroyed"] = conn.execute(
            """
            UPDATE user_encryption_keys
               SET active = false,
                   destroyed_at = clock_timestamp(),
                   fernet_key = 'DESTROYED'
             WHERE user_id = %s
               AND active
            """,
            (stored_user_id,),
        ).rowcount

        # Solo teams may remain as finance workspaces; remove the old address.
        counts["team_members_anonymized"] = conn.execute(
            """
            UPDATE team_members
               SET email = %s
             WHERE lower(email) = %s
            """,
            (anonymous_email, stored_email),
        ).rowcount
        counts["teams_anonymized"] = conn.execute(
            """
            UPDATE teams
               SET owner_email = %s
             WHERE lower(owner_email) = %s
            """,
            (anonymous_email, stored_email),
        ).rowcount

        password_tombstone = hashlib.sha256(f"{reference}:{time.time_ns()}".encode()).hexdigest()
        counts["users_anonymized"] = conn.execute(
            """
            UPDATE users
               SET email = %s,
                   name = 'Deleted account',
                   password_hash = %s,
                   salt = %s,
                   role = 'submitter',
                   provider_id = NULL,
                   country = '',
                   province = '',
                   oauth_provider = '',
                   team_id = NULL,
                   reset_token = NULL,
                   reset_token_expires = NULL,
                   -- These five are INTEGER flags on `users`, not booleans;
                   -- assigning SQL false here aborted the whole anonymization
                   -- with a DatatypeMismatch, so no identity was ever erased.
                   -- (wallets.auto_topup_enabled and casl_consent.active are
                   -- real booleans and are assigned as such elsewhere.)
                   notifications_enabled = 0,
                   preferences = '{}'::jsonb,
                   mfa_enabled = 0,
                   email_verified = 0,
                   email_verification_token = NULL,
                   email_verification_expires = NULL,
                   is_admin = 0,
                   max_concurrent_instances = 0,
                   pending_email = NULL,
                   email_change_token = NULL,
                   email_change_expires = NULL
             WHERE user_id = %s
            """,
            (
                anonymous_email,
                password_tombstone,
                password_tombstone,
                stored_user_id,
            ),
        ).rowcount

        evidence = {
            "identity_anonymized": counts["users_anonymized"] == 1,
            "credentials_revoked": sum(
                counts.get(name, 0)
                for name in (
                    "oauth_refresh_tokens",
                    "sessions",
                    "api_keys",
                    "oauth_clients",
                    "mfa_backup_codes",
                    "mfa_challenges",
                    "mfa_methods",
                    "ssh_keys",
                )
            ),
            "rows_affected": counts,
            "finance_records": "retained_under_legal_and_reconciliation_policy",
            "payment_methods_disabled": True,
        }
        _append_event(conn, request, "privacy.v1.authority_anonymized", evidence)
        conn.commit()
    return SinkOutcome("completed", evidence)


def delete_redis_subject(request: Mapping[str, Any], _sink: Mapping[str, Any]) -> SinkOutcome:
    """Invalidate identifier-derived keys in each configured Redis database."""
    user_id, email, customer_ids = _identifiers(request)
    urls = {
        value.strip()
        for value in (
            os.environ.get("XCELSIOR_AUTH_REDIS_URL", ""),
            os.environ.get("XCELSIOR_SERVERLESS_REDIS_URL", ""),
            os.environ.get("XCELSIOR_MCP_REDIS_URL", ""),
            os.environ.get("MCP_REDIS_URL", ""),
        )
        if value and value.strip()
    }
    if not urls:
        return SinkOutcome("not_applicable", {"configured_redis_databases": 0})

    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("redis client is required for cache invalidation") from exc

    digests = {
        opaque(identifier)
        for identifier in (
            user_id,
            email,
            *customer_ids,
            str(request["subject_reference_hash"]),
        )
        if identifier
    }
    deleted = 0
    scanned_databases = 0
    for url in sorted(urls):
        client = redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        scanned_databases += 1
        keys: set[str] = set()
        for digest in digests:
            # Key names contain only hashes, never the raw subject identifiers.
            for key in client.scan_iter(match=f"*{digest}*", count=500):
                keys.add(str(key))
                if len(keys) >= 5_000:
                    break
        if keys:
            deleted += int(client.unlink(*sorted(keys)))

    return SinkOutcome(
        "completed",
        {
            "configured_redis_databases": scanned_databases,
            "identifier_derived_keys_deleted": deleted,
            "database_authorization_revoked": True,
        },
    )


def delete_artifact_subject(request: Mapping[str, Any], _sink: Mapping[str, Any]) -> SinkOutcome:
    user_id, _email, customer_ids = _identifiers(request)
    request_id = str(request["request_id"])
    pool = _pool()
    with pool.connection() as conn:
        artifacts = conn.execute(
            """
            SELECT artifact_id, state, legal_hold, retain_until
              FROM storage.artifacts
             WHERE (
                    owner_user_id = %s
                    OR tenant_id = ANY(%s)
                   )
               AND state <> 'deleted'
             ORDER BY artifact_id
            """,
            (user_id, customer_ids or ["__none__"]),
        ).fetchall()
        held = 0
        pending = 0
        for row in artifacts:
            artifact_id = str(_value(row, "artifact_id", 0))
            legal_hold = bool(_value(row, "legal_hold", 2))
            retain_until = _value(row, "retain_until", 3)
            if legal_hold:
                held += 1
                continue
            if retain_until and retain_until > datetime.now(timezone.utc):
                held += 1
                continue
            conn.execute(
                """
                INSERT INTO storage.artifact_deletion_jobs (
                    artifact_id, reason, requested_by, state
                )
                SELECT %s, 'privacy deletion request', %s, 'requested'
                 WHERE NOT EXISTS (
                     SELECT 1
                       FROM storage.artifact_deletion_jobs
                      WHERE artifact_id = %s
                        AND state IN (
                            'requested', 'claimed', 'delete_pending',
                            'delete_failed'
                        )
                 )
                """,
                (artifact_id, f"privacy:{request_id}", artifact_id),
            )
            conn.execute(
                """
                UPDATE storage.artifacts
                   SET state = 'delete_pending',
                       version = version + 1
                 WHERE artifact_id = %s
                   AND state NOT IN ('deleted', 'delete_pending')
                """,
                (artifact_id,),
            )
            pending += 1

        open_jobs = conn.execute(
            """
            SELECT count(*)
              FROM storage.artifact_deletion_jobs j
              JOIN storage.artifacts a ON a.artifact_id = j.artifact_id
             WHERE (
                    a.owner_user_id = %s
                    OR a.tenant_id = ANY(%s)
                   )
               AND NOT a.legal_hold
               AND j.state <> 'completed'
            """,
            (user_id, customer_ids or ["__none__"]),
        ).fetchone()
        remaining_jobs = int(_value(open_jobs, "count", 0))
        conn.commit()

    # Persistent NFS volumes are also user artifacts, but they have their own
    # fenced storage engine and cannot be deleted by SQL alone.
    volume_deleted = 0
    volume_pending = 0
    if customer_ids:
        from volumes import get_volume_engine

        engine = get_volume_engine()
        for volume in engine.list_volumes_for_owner_ids(customer_ids):
            if str(volume.get("status")) == "deleted":
                continue
            if volume.get("attached_to"):
                volume_pending += 1
                continue
            try:
                engine.delete_volume(str(volume["volume_id"]), str(volume["owner_id"]))
                volume_deleted += 1
            except (RuntimeError, ValueError):
                volume_pending += 1

    evidence = {
        "catalog_artifacts_seen": len(artifacts),
        "catalog_deletions_enqueued": pending,
        "catalog_deletions_remaining": remaining_jobs,
        "artifacts_under_hold_or_retention": held,
        "volumes_deleted": volume_deleted,
        "volumes_remaining": volume_pending,
    }
    if remaining_jobs or volume_pending:
        return SinkOutcome(
            "pending",
            evidence,
            retry_after_sec=60,
            error="artifact deletion is still in progress",
        )
    if held:
        return SinkOutcome("legal_hold", evidence)
    return SinkOutcome("completed", evidence)


def delete_retrieval_subject(request: Mapping[str, Any], _sink: Mapping[str, Any]) -> SinkOutcome:
    user_id, email, customer_ids = _identifiers(request)
    pool = _pool()
    with pool.connection() as conn:
        counts: dict[str, int] = {}
        counts["ai_confirmations"] = conn.execute(
            "DELETE FROM ai_confirmations WHERE user_id = %s", (user_id,)
        ).rowcount
        # ai_messages cascade from ai_conversations.
        counts["ai_conversations"] = conn.execute(
            "DELETE FROM ai_conversations WHERE user_id = %s", (user_id,)
        ).rowcount

        chat_ids = [
            str(_value(row, "conversation_id", 0))
            for row in conn.execute(
                """
                SELECT conversation_id
                  FROM chat_conversations
                 WHERE lower(user_email) = %s
                """,
                (email,),
            ).fetchall()
        ]
        if chat_ids:
            message_ids = [
                str(_value(row, "id", 0))
                for row in conn.execute(
                    """
                    SELECT id FROM chat_messages
                     WHERE conversation_id = ANY(%s)
                    """,
                    (chat_ids,),
                ).fetchall()
            ]
            if message_ids:
                counts["chat_feedback"] = conn.execute(
                    "DELETE FROM chat_feedback WHERE message_id = ANY(%s::text[])",
                    (message_ids,),
                ).rowcount
            counts["chat_messages"] = conn.execute(
                "DELETE FROM chat_messages WHERE conversation_id = ANY(%s)",
                (chat_ids,),
            ).rowcount
            counts["chat_conversations"] = conn.execute(
                "DELETE FROM chat_conversations WHERE conversation_id = ANY(%s)",
                (chat_ids,),
            ).rowcount

        if customer_ids:
            endpoint_ids = [
                str(_value(row, "endpoint_id", 0))
                for row in conn.execute(
                    """
                    SELECT endpoint_id FROM serverless_endpoints
                     WHERE owner_id = ANY(%s)
                    """,
                    (customer_ids,),
                ).fetchall()
            ]
            if endpoint_ids:
                counts["semantic_cache"] = conn.execute(
                    """
                    DELETE FROM serverless_semantic_cache
                     WHERE endpoint_id = ANY(%s)
                    """,
                    (endpoint_ids,),
                ).rowcount
                counts["serverless_payloads"] = conn.execute(
                    """
                    UPDATE serverless_jobs
                       SET payload = '{}'::jsonb,
                           output = NULL,
                           error = NULL,
                           webhook_url = '',
                           idempotency_key = encode(
                               digest(job_id || %s, 'sha256'), 'hex'
                           )
                     WHERE owner_id = ANY(%s)
                    """,
                    (str(request["subject_reference_hash"]), customer_ids),
                ).rowcount

            inference_ids = [
                str(_value(row, "job_id", 0))
                for row in conn.execute(
                    """
                    SELECT job_id FROM inference_jobs
                     WHERE customer_id = ANY(%s)
                    """,
                    (customer_ids,),
                ).fetchall()
            ]
            if inference_ids:
                counts["inference_results"] = conn.execute(
                    """
                    UPDATE inference_results
                       SET outputs = '{}'::jsonb
                     WHERE job_id = ANY(%s)
                    """,
                    (inference_ids,),
                ).rowcount
                counts["inference_inputs"] = conn.execute(
                    """
                    UPDATE inference_jobs
                       SET inputs = '{}'::jsonb
                     WHERE job_id = ANY(%s)
                    """,
                    (inference_ids,),
                ).rowcount
        conn.commit()
    return SinkOutcome(
        "completed",
        {
            "rows_deleted_or_anonymized": counts,
            "shared_system_document_index": "not_subject_addressable",
        },
    )


def delete_analytics_subject(_request: Mapping[str, Any], _sink: Mapping[str, Any]) -> SinkOutcome:
    configured = any(
        os.environ.get(name, "").strip()
        for name in (
            "XCELSIOR_BIGQUERY_PROJECT",
            "XCELSIOR_BIGQUERY_DATASET",
            "GOOGLE_CLOUD_PROJECT",
        )
    )
    if not configured:
        return SinkOutcome(
            "not_applicable",
            {"warehouse_projection_configured": False},
        )
    # A configured warehouse without a deletion delivery/verification client
    # is a compliance failure, never an implicit success.
    return SinkOutcome(
        "failed",
        {"warehouse_projection_configured": True},
        retry_after_sec=3_600,
        error="warehouse deletion client is not configured",
    )


def _posthog_request(
    method: str,
    path: str,
    *,
    api_key: str,
    base_url: str,
    body: Mapping[str, Any] | None = None,
    query: Mapping[str, str] | None = None,
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = json.dumps(dict(body)).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(1_000).decode(errors="replace")
        raise RuntimeError(f"PostHog API returned HTTP {exc.code}: {detail}") from exc
    return json.loads(payload) if payload else {}


def delete_posthog_subject(request: Mapping[str, Any], sink: Mapping[str, Any]) -> SinkOutcome:
    tracking_enabled = any(
        os.environ.get(name, "").strip()
        for name in (
            "NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN",
            "XCELSIOR_MCP_POSTHOG_PROJECT_API_KEY",
            "POSTHOG_PROJECT_API_KEY",
        )
    )
    if not tracking_enabled:
        return SinkOutcome("not_applicable", {"posthog_tracking_configured": False})

    api_key = os.environ.get("XCELSIOR_POSTHOG_PERSONAL_API_KEY", "").strip()
    project_id = os.environ.get("XCELSIOR_POSTHOG_PROJECT_ID", "").strip()
    # Private person/deletion endpoints live on the PostHog app host, not the
    # public *.i.posthog.com ingestion host used by browser and MCP capture.
    base_url = (
        os.environ.get("XCELSIOR_POSTHOG_API_HOST", "").strip()
        or "https://us.posthog.com"
    )
    if not api_key or not project_id:
        return SinkOutcome(
            "failed",
            {"posthog_tracking_configured": True},
            retry_after_sec=3_600,
            error=(
                "PostHog deletion requires XCELSIOR_POSTHOG_PERSONAL_API_KEY "
                "and XCELSIOR_POSTHOG_PROJECT_ID"
            ),
        )

    prior_evidence = sink.get("evidence") or {}
    if isinstance(prior_evidence, str):
        prior_evidence = json.loads(prior_evidence)
    person_uuids = [str(value) for value in prior_evidence.get("person_uuids", [])]
    if person_uuids:
        pending = 0
        completed = 0
        for person_uuid in person_uuids:
            response = _posthog_request(
                "GET",
                f"/api/projects/{project_id}/persons/deletion_status/",
                api_key=api_key,
                base_url=base_url,
                query={"person_uuid": person_uuid, "status": "all"},
            )
            statuses = response.get("results", response if isinstance(response, list) else [])
            if statuses and all(str(item.get("status")) == "completed" for item in statuses):
                completed += 1
            else:
                pending += 1
        evidence = {
            "posthog_tracking_configured": True,
            "person_uuids": person_uuids,
            "event_deletions_completed": completed,
            "event_deletions_pending": pending,
        }
        if pending:
            return SinkOutcome(
                "pending",
                evidence,
                retry_after_sec=300,
                error="PostHog event deletion remains asynchronous",
            )
        return SinkOutcome("completed", evidence)

    user_id, email, customer_ids = _identifiers(request)
    distinct_ids = sorted({value for value in (user_id, email, *customer_ids) if value})
    found_uuids: set[str] = set()
    for distinct_id in distinct_ids:
        response = _posthog_request(
            "GET",
            f"/api/projects/{project_id}/persons/",
            api_key=api_key,
            base_url=base_url,
            query={"distinct_id": distinct_id},
        )
        for person in response.get("results", []):
            if person.get("uuid"):
                found_uuids.add(str(person["uuid"]))
            elif person.get("id"):
                found_uuids.add(str(person["id"]))

    response = _posthog_request(
        "POST",
        f"/api/projects/{project_id}/persons/bulk_delete/",
        api_key=api_key,
        base_url=base_url,
        body={
            "distinct_ids": distinct_ids,
            "delete_events": True,
            "delete_recordings": True,
            "keep_person": False,
        },
    )
    evidence = {
        "posthog_tracking_configured": True,
        "person_uuids": sorted(found_uuids),
        "persons_found": int(response.get("persons_found", 0)),
        "persons_deleted": int(response.get("persons_deleted", 0)),
        "events_queued_for_deletion": bool(response.get("events_queued_for_deletion", False)),
        "recordings_queued_for_deletion": bool(
            response.get("recordings_queued_for_deletion", False)
        ),
        "deletion_errors": len(response.get("deletion_errors", [])),
    }
    if response.get("deletion_errors"):
        return SinkOutcome(
            "failed",
            evidence,
            retry_after_sec=3_600,
            error="PostHog rejected one or more person deletions",
        )
    if not found_uuids:
        return SinkOutcome("completed", evidence)
    return SinkOutcome(
        "pending",
        evidence,
        retry_after_sec=300,
        error="PostHog event deletion remains asynchronous",
    )


def verify_subject_absence(request: Mapping[str, Any], _sink: Mapping[str, Any]) -> SinkOutcome:
    """Count what remains of a subject, and report whether they are gone.

    ## The append-only tables are outside this check, and that is now decided

    The checks below are a **hand-enumerated literal**. Any table not named here
    is invisible to a function whose name asserts *absence* — so it can return a
    clean verdict while rows persist. Everywhere else in the erasure path a
    missing table is an omission; here it is an affirmative claim the code makes
    on the reader's behalf, which is why the note lives at this function rather
    than beside any one table.

    Three tables carry an append-only (WORM) trigger — currently
    `audit_events_v2` (072), `audit_checkpoints` (075) and `placement_decisions`
    (105/106) — and **none of them is reachable from here or from any delete
    sink above**. The trigger rejects DELETE unconditionally, and partitioning
    prunes by time, not by tenant, so a per-subject erasure cannot reach them by
    any existing mechanism.

    **This predates all three tables and is not a defect introduced by any of
    them.** It is the standing treatment of immutable audit data in this
    repository. Audit tables legitimately resolve this in one of two ways —
    pseudonymise the subject's identifiers at erasure time, or record a
    documented retention basis for keeping them.

    **The ruling, 2026-08-11 by Aaryn Biro: retained under a documented basis.**
    Not pseudonymisation. Placement and access decisions are a standard
    legitimate-interest / legal-obligation retention under GDPR Art. 17(3) and
    the equivalent carve-outs in other privacy regimes, and partition-dropping
    already supplied the mechanism, so the cost was a policy statement rather
    than a rewrite.

    A retention basis is not free, and the three things it owes are now real
    rather than intended:

    - **A period.** `WORM_RETENTION_MONTHS = 24`.
    - **Disclosure.** `docs/audit-retention.md`, and the privacy-policy line it
      carries. Silence was the only genuinely untenable option.
    - **Enforcement.** `drop_expired_partitions`, scheduled daily. Partitions
      were created ahead of time and never dropped until then, so a stated
      period would have been a claim with nothing behind it.

    So the enumeration below stays as it is, and the verdict **names the
    exception** in `evidence["append_only_records"]` rather than implying the
    subject is gone from everywhere. A reader of a clean outcome is entitled to
    know what it does not cover.

    If a contract or regulator later demands attributable erasure, the technical
    answer is crypto-shredding — hold the tenant identifier encrypted under a
    per-tenant key stored outside the WORM table and delete the key, which makes
    rows non-attributable with no UPDATE or DELETE. Recorded as the escape
    hatch; deliberately not built, because nothing requires it today.

    `tests/test_worm_tables_have_an_erasure_decision.py` derives both sides —
    the WORM set from `pg_trigger`, the reachable set from this function's own
    source — so a **new** append-only table cannot join the unresolved set
    silently.
    """
    user_id, email, customer_ids = _identifiers(request)
    pool = _pool()
    with pool.connection() as conn:
        checks = {
            "live_login_email": conn.execute(
                "SELECT count(*) FROM users WHERE lower(email) = %s",
                (email,),
            ).fetchone(),
            "sessions": conn.execute(
                """
                SELECT count(*) FROM sessions
                 WHERE user_id = %s OR lower(email) = %s
                """,
                (user_id, email),
            ).fetchone(),
            "oauth_refresh_tokens": conn.execute(
                """
                SELECT count(*) FROM oauth_refresh_tokens
                 WHERE user_id = %s OR lower(email) = %s
                """,
                (user_id, email),
            ).fetchone(),
            "mfa_methods": conn.execute(
                "SELECT count(*) FROM mfa_methods WHERE lower(email) = %s",
                (email,),
            ).fetchone(),
            "ssh_keys": conn.execute(
                """
                SELECT count(*) FROM user_ssh_keys
                 WHERE user_id = %s OR lower(email) = %s
                """,
                (user_id, email),
            ).fetchone(),
            "notifications": conn.execute(
                "SELECT count(*) FROM notifications WHERE lower(user_email) = %s",
                (email,),
            ).fetchone(),
            "ai_conversations": conn.execute(
                "SELECT count(*) FROM ai_conversations WHERE user_id = %s",
                (user_id,),
            ).fetchone(),
            "chat_conversations": conn.execute(
                """
                SELECT count(*) FROM chat_conversations
                 WHERE lower(user_email) = %s
                """,
                (email,),
            ).fetchone(),
            "team_members": conn.execute(
                "SELECT count(*) FROM team_members WHERE lower(email) = %s",
                (email,),
            ).fetchone(),
            "team_owners": conn.execute(
                "SELECT count(*) FROM teams WHERE lower(owner_email) = %s",
                (email,),
            ).fetchone(),
        }
        if customer_ids:
            checks["active_jobs"] = conn.execute(
                """
                SELECT count(*) FROM jobs
                 WHERE owner_id = ANY(%s)
                   AND COALESCE(phase, '') NOT IN (
                       'succeeded', 'failed', 'stopped'
                   )
                """,
                (customer_ids,),
            ).fetchone()
        residuals = {
            name: int(_value(row, "count", 0))
            for name, row in checks.items()
            if int(_value(row, "count", 0)) > 0
        }
        conn.rollback()

    from control_plane.audit_partitions import (
        PARTITIONED_TABLES,
        WORM_RETENTION_MONTHS,
    )

    evidence = {
        "checks_run": len(checks),
        "residual_counts": residuals,
        "finance_and_audit_records": "retained_under_governed_policy",
        # The exception, stated rather than implied. This function's name is a
        # claim of absence, and the claim is not true of the append-only tables:
        # their trigger rejects DELETE, so no erasure path reaches them and none
        # is intended to. Saying so in the evidence is the difference between a
        # documented retention basis and a silent gap — a reader of this outcome
        # would otherwise conclude the subject is gone from everywhere.
        #
        # Derived from `PARTITIONED_TABLES`, so a new WORM table appears here
        # without an edit. `tests/test_worm_tables_have_an_erasure_decision.py`
        # holds the other half: it derives the WORM set from `pg_trigger` and
        # fails when one joins without a decision.
        "append_only_records": {
            "tables": sorted(PARTITIONED_TABLES),
            "erased": False,
            "basis": "legitimate_interest_and_legal_obligation",
            "authority": "GDPR Art. 17(3) and equivalent provisions elsewhere",
            "retention_months": WORM_RETENTION_MONTHS,
            "enforced_by": "control_plane.audit_partitions.drop_expired_partitions",
            "decided": "2026-08-11 by Aaryn Biro",
            "disclosure": "docs/audit-retention.md",
        },
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    if residuals:
        return SinkOutcome(
            "failed",
            evidence,
            retry_after_sec=300,
            error="subject identifiers remain in deletion-governed stores",
        )
    return SinkOutcome("completed", evidence)
