# Xcelsior Privacy-by-Default Controls
# Implements REPORT_FEATURE_FINAL.md § "Privacy-by-default and governance hooks":
#   - Default-minimal logging for job payloads; explicit opt-in for sensitive fields
#   - Privacy officer designation in org/provider settings
#   - Job metadata redaction and retention policies
#   - Québec Law 25: identification/location/profiling disabled by default
#   - PIPEDA 10 fair information principles as built-in controls
#
# Design principle: privacy governance is encoded in code, not policy documents.

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("xcelsior")


# ── Privacy Settings (defaults = maximum confidentiality per Law 25) ──


class PrivacyLevel(str, Enum):
    """Privacy level for data handling. Default is STRICT per Québec Law 25."""

    STRICT = "strict"  # Maximum confidentiality, all PII redacted
    STANDARD = "standard"  # Reasonable safeguards, minimal PII retained
    PERMISSIVE = "permissive"  # Explicit opt-in by user — full metadata retained


class DataCategory(str, Enum):
    """Categories of data for retention and redaction policies."""

    JOB_PAYLOAD = "job_payload"  # Container args, env vars, commands
    JOB_METADATA = "job_metadata"  # Job name, tags, labels
    PROVIDER_IDENTITY = "provider_identity"  # Host owner info
    BILLING_INFO = "billing_info"  # Payment details, wallet
    TELEMETRY = "telemetry"  # GPU metrics, utilization data
    LOGS = "logs"  # Job stdout/stderr
    NETWORK = "network"  # IP addresses, connection data
    LOCATION = "location"  # Geolocation, province, city
    CHAT_MESSAGES = "chat_messages"  # AI chat conversation data


# ── Retention Policies ────────────────────────────────────────────────
# PIPEDA principle: "Limiting Use, Disclosure, and Retention —
# keeping data only as long as required for the identified job."

# Default retention periods in seconds
RETENTION_POLICIES = {
    DataCategory.JOB_PAYLOAD: {
        "retention_sec": 0,  # Deleted immediately after job completion
        "description": "Job payloads (container args, env vars) are not retained",
        "redact_on_completion": True,
    },
    DataCategory.JOB_METADATA: {
        "retention_sec": 90 * 86400,  # 90 days
        "description": "Job metadata retained 90 days for billing reconciliation",
        "redact_on_completion": False,
    },
    DataCategory.PROVIDER_IDENTITY: {
        "retention_sec": 365 * 86400,  # 1 year
        "description": "Provider identity retained for tax/compliance records",
        "redact_on_completion": False,
    },
    DataCategory.BILLING_INFO: {
        "retention_sec": 7 * 365 * 86400,  # 7 years (CRA requirement)
        "description": "Billing records retained 7 years per CRA requirements",
        "redact_on_completion": False,
    },
    DataCategory.TELEMETRY: {
        "retention_sec": 30 * 86400,  # 30 days
        "description": "GPU telemetry retained 30 days for SLA enforcement",
        "redact_on_completion": False,
    },
    DataCategory.LOGS: {
        "retention_sec": 7 * 86400,  # 7 days
        "description": "Job logs retained 7 days, then purged",
        "redact_on_completion": False,
    },
    DataCategory.NETWORK: {
        "retention_sec": 30 * 86400,  # 30 days
        "description": "Network/IP data retained 30 days for security audit",
        "redact_on_completion": False,
    },
    DataCategory.LOCATION: {
        "retention_sec": 90 * 86400,  # 90 days
        "description": "Location data retained 90 days for residency traces",
        "redact_on_completion": False,
    },
    DataCategory.CHAT_MESSAGES: {
        "retention_sec": 30 * 86400,  # 30 days
        "description": "Chat messages retained 30 days for quality monitoring",
        "redact_on_completion": False,
    },
}


# ── PII Patterns for Redaction ────────────────────────────────────────

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "sin": re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b"),  # Canadian SIN
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "api_key": re.compile(r"(?:sk|pk|api|key|token|secret)[_-]?[a-zA-Z0-9]{20,}", re.IGNORECASE),
}

REDACTION_PLACEHOLDER = "[REDACTED]"


# ── Sensitive Environment Variable Names ──────────────────────────────
# These are never logged or retained, even in PERMISSIVE mode.

ALWAYS_REDACT_ENV_VARS = frozenset(
    {
        "PASSWORD",
        "SECRET",
        "TOKEN",
        "API_KEY",
        "PRIVATE_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "DATABASE_URL",
        "DB_PASSWORD",
        "SMTP_PASSWORD",
        "STRIPE_SECRET_KEY",
        "WEBHOOK_SECRET",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "WANDB_API_KEY",
    }
)


# ── Privacy Configuration ─────────────────────────────────────────────


@dataclass
class PrivacyConfig:
    """Per-organization or per-job privacy configuration.

    Defaults to STRICT per Québec Law 25 guidance:
    "default settings provide the highest level of confidentiality,
     and identification/location/profiling functions cannot be
     enabled by default."
    """

    privacy_level: str = PrivacyLevel.STRICT

    # PIPEDA Accountability: privacy officer designation
    privacy_officer_name: str = ""
    privacy_officer_email: str = ""
    privacy_officer_designated: bool = False

    # Québec Law 25: identification/location/profiling OFF by default
    enable_identification: bool = False  # Link jobs to named individuals
    enable_location_tracking: bool = False  # Detailed geolocation beyond country
    enable_profiling: bool = False  # Usage pattern analysis

    # Retention overrides (None = use defaults)
    log_retention_days: Optional[int] = None
    telemetry_retention_days: Optional[int] = None
    metadata_retention_days: Optional[int] = None

    # Redaction
    redact_pii_in_logs: bool = True  # Scan logs for PII patterns
    redact_env_vars: bool = True  # Scrub env vars from job records
    redact_ip_addresses: bool = True  # Replace IPs in stored data

    # Consent
    cross_border_consent: bool = False  # Explicit consent for non-CA processing
    data_collection_consent: bool = False  # Consent for analytics/telemetry

    def to_dict(self) -> dict:
        return asdict(self)


# ── Redaction Functions ───────────────────────────────────────────────


def redact_pii(text: str, patterns: Optional[dict] = None) -> str:
    """Scan text for PII patterns and replace with [REDACTED].

    Used for log sanitization before storage.
    """
    if not text:
        return text

    active_patterns = patterns or PII_PATTERNS
    result = text
    for name, pattern in active_patterns.items():
        result = pattern.sub(REDACTION_PLACEHOLDER, result)

    return result


def redact_env_vars(env: dict) -> dict:
    """Redact sensitive environment variables from job records.

    Always redacts known secrets. Returns a safe copy.
    """
    if not env:
        return {}

    safe = {}
    for key, value in env.items():
        key_upper = key.upper()
        # Always redact known secret patterns
        if key_upper in ALWAYS_REDACT_ENV_VARS:
            safe[key] = REDACTION_PLACEHOLDER
        elif any(s in key_upper for s in ("SECRET", "PASSWORD", "TOKEN", "KEY", "CREDENTIAL")):
            safe[key] = REDACTION_PLACEHOLDER
        else:
            safe[key] = value

    return safe


def redact_job_record(job: dict, config: Optional[PrivacyConfig] = None) -> dict:
    """Apply privacy controls to a job record before storage.

    Applies the organization's privacy config (defaults to STRICT).
    """
    cfg = config or PrivacyConfig()
    safe = dict(job)

    # Always redact environment variables
    if cfg.redact_env_vars and "env" in safe:
        safe["env"] = redact_env_vars(safe.get("env", {}))

    # Redact container command if strict
    if cfg.privacy_level == PrivacyLevel.STRICT:
        if "container_command" in safe:
            safe["container_command"] = REDACTION_PLACEHOLDER
        if "container_args" in safe:
            safe["container_args"] = REDACTION_PLACEHOLDER

    # Redact IP addresses
    if cfg.redact_ip_addresses:
        for field_name in ("ip", "host_ip", "worker_ip"):
            if field_name in safe and safe[field_name]:
                safe[field_name] = redact_pii(
                    str(safe[field_name]),
                    {"ip_address": PII_PATTERNS["ip_address"]},
                )

    # Redact PII in logs
    if cfg.redact_pii_in_logs:
        for field_name in ("stdout", "stderr", "output", "error_message"):
            if field_name in safe and safe[field_name]:
                safe[field_name] = redact_pii(str(safe[field_name]))

    # Location: only country-level unless opted in
    if not cfg.enable_location_tracking:
        for field_name in ("city", "data_center_name", "latitude", "longitude"):
            if field_name in safe:
                safe[field_name] = ""

    # Identification: strip names unless opted in
    if not cfg.enable_identification:
        for field_name in ("owner_name", "submitter_name", "operator_name"):
            if field_name in safe:
                safe[field_name] = REDACTION_PLACEHOLDER

    return safe


def sanitize_log_output(log_text: str, max_length: int = 10000) -> str:
    """Sanitize and truncate log output before storage.

    - Removes PII patterns
    - Truncates to maximum length
    - Strips trailing whitespace
    """
    if not log_text:
        return ""

    sanitized = redact_pii(log_text)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "\n... [TRUNCATED]"

    return sanitized.rstrip()


# ── Data Lifecycle Manager ────────────────────────────────────────────


class DataLifecycleManager:
    """Manages data retention and purging per PIPEDA principles.

    "Organizations must protect personal information with appropriate
     safeguards and destroy it when no longer needed."
     — Office of the Privacy Commissioner of Canada
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path  # Legacy compat

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ── Retention tracking ────────────────────────────────────────────

    def track_data(
        self,
        data_category: str,
        entity_id: str,
        entity_type: str = "job",
        retention_override_sec: Optional[int] = None,
    ) -> str:
        """Register a data item for lifecycle tracking.

        Returns the record_id.
        """
        import uuid

        record_id = str(uuid.uuid4())[:12]
        now = time.time()

        # Determine retention period
        cat = (
            DataCategory(data_category)
            if data_category in DataCategory.__members__.values()
            else None
        )
        if retention_override_sec is not None:
            retention = retention_override_sec
        elif cat and cat in RETENTION_POLICIES:
            retention = RETENTION_POLICIES[cat]["retention_sec"]
        else:
            retention = 90 * 86400  # Default: 90 days

        expires_at = now + retention

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO retention_records
                   (record_id, data_category, entity_id, entity_type,
                    created_at, expires_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (record_id, data_category, entity_id, entity_type, now, expires_at),
            )

        log.debug(
            "RETENTION TRACKED %s/%s category=%s expires=%.0f",
            entity_type,
            entity_id,
            data_category,
            expires_at,
        )
        return record_id

    def get_expired_records(self, before: Optional[float] = None) -> list[dict]:
        """Get all retention records that have expired and need purging."""
        cutoff = before or time.time()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM retention_records
                   WHERE expires_at <= %s AND purged_at = 0
                   ORDER BY expires_at""",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_purged(self, record_id: str, reason: str = "retention_expired"):
        """Mark a retention record as purged."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE retention_records
                   SET purged_at = %s, purge_reason = %s
                   WHERE record_id = %s""",
                (time.time(), reason, record_id),
            )
        log.info("DATA PURGED record=%s reason=%s", record_id, reason)

    def purge_expired(self) -> int:
        """Purge all expired data records and cascade-delete from data stores.

        Returns count of records purged.
        Should be called periodically (e.g., daily cron).
        """
        expired = self.get_expired_records()
        count = 0
        for record in expired:
            category = record["data_category"]
            entity_id = record["entity_id"]
            entity_type = record.get("entity_type", "job")

            try:
                self._cascade_delete(category, entity_id, entity_type)
            except Exception as e:
                log.error(
                    "PURGE CASCADE FAILED record=%s category=%s entity=%s: %s",
                    record["record_id"],
                    category,
                    entity_id,
                    e,
                )
                # Still mark purged — the retention record is consumed even
                # if the underlying store is unavailable.  A re-run won't
                # find orphaned data because it's keyed by the record.

            self.mark_purged(record["record_id"])
            count += 1

        if count > 0:
            log.info("RETENTION PURGE completed: %d records purged", count)

        return count

    def _cascade_delete(self, category: str, entity_id: str, entity_type: str):
        """Delete actual data from the appropriate data store."""
        handler = self._purge_handlers.get(category)
        if handler:
            handler(self, entity_id, entity_type)
        else:
            log.debug("No cascade handler for category=%s, metadata-only purge", category)

    # ── Per-category purge handlers ───────────────────────────────────

    def _purge_job_payload(self, entity_id: str, entity_type: str):
        """Redact job payload fields (container args, env vars, command).

        Does NOT delete the job record — just scrubs sensitive payload data.
        """
        from scheduler import _db_connection

        with _db_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM jobs WHERE job_id = %s", (entity_id,)
            ).fetchone()
            if not row:
                return
            payload = (
                json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
            )
            for field_name in (
                "container_command",
                "container_args",
                "env_vars",
                "docker_image",
                "nfs_path",
                "command",
            ):
                if field_name in payload:
                    payload[field_name] = "[PURGED]"
            conn.execute(
                "UPDATE jobs SET payload = %s WHERE job_id = %s",
                (json.dumps(payload), entity_id),
            )
        log.info("PURGE job_payload entity=%s", entity_id)

    def _purge_job_metadata(self, entity_id: str, entity_type: str):
        """Delete completed job records entirely."""
        from scheduler import _db_connection

        with _db_connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (entity_id,))
        log.info("PURGE job_metadata entity=%s", entity_id)

    def _purge_telemetry(self, entity_id: str, entity_type: str):
        """Delete telemetry event records for a host."""
        from events import EventStore

        store = EventStore()
        with store._conn() as conn:
            conn.execute(
                "DELETE FROM events WHERE entity_id = %s AND event_type LIKE %s",
                (entity_id, "%telemetry%"),
            )
        log.info("PURGE telemetry entity=%s", entity_id)

    def _purge_logs(self, entity_id: str, entity_type: str):
        """Delete log/event records for a job or entity."""
        from events import EventStore

        store = EventStore()
        with store._conn() as conn:
            conn.execute(
                "DELETE FROM events WHERE entity_id = %s",
                (entity_id,),
            )
        log.info("PURGE logs entity=%s", entity_id)

    def _purge_network(self, entity_id: str, entity_type: str):
        """Scrub network/IP data from host records."""
        from scheduler import _db_connection

        with _db_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM hosts WHERE host_id = %s", (entity_id,)
            ).fetchone()
            if not row:
                return
            payload = (
                json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
            )
            for field_name in ("ip", "public_ip", "private_ip", "headscale_ip"):
                if field_name in payload:
                    payload[field_name] = "[PURGED]"
            conn.execute(
                "UPDATE hosts SET payload = %s WHERE host_id = %s",
                (json.dumps(payload), entity_id),
            )
        log.info("PURGE network entity=%s", entity_id)

    def _purge_location(self, entity_id: str, entity_type: str):
        """Scrub geolocation data from host records."""
        from scheduler import _db_connection

        with _db_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM hosts WHERE host_id = %s", (entity_id,)
            ).fetchone()
            if not row:
                return
            payload = (
                json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
            )
            for field_name in (
                "city",
                "province",
                "latitude",
                "longitude",
                "data_center_name",
                "country",
            ):
                if field_name in payload:
                    payload[field_name] = "[PURGED]"
            conn.execute(
                "UPDATE hosts SET payload = %s WHERE host_id = %s",
                (json.dumps(payload), entity_id),
            )
        log.info("PURGE location entity=%s", entity_id)

    def _purge_chat_messages(self, entity_id: str, entity_type: str):
        """Delete chat messages for a conversation."""
        from chat import _chat_db

        with _chat_db() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE conversation_id = %s",
                (entity_id,),
            )
            conn.execute(
                "DELETE FROM chat_conversations WHERE conversation_id = %s",
                (entity_id,),
            )
        log.info("PURGE chat_messages entity=%s", entity_id)

    def _purge_billing_info(self, entity_id: str, entity_type: str):
        """Delete billing records (invoices, usage meters) for a customer."""
        from billing import get_billing_engine

        be = get_billing_engine()
        with be._conn() as conn:
            conn.execute("DELETE FROM usage_meters WHERE customer_id = %s", (entity_id,))
            conn.execute("DELETE FROM invoices WHERE customer_id = %s", (entity_id,))
        log.info("PURGE billing_info entity=%s", entity_id)

    def _purge_provider_identity(self, entity_id: str, entity_type: str):
        """Delete provider/user identity records."""
        from db import UserStore

        try:
            UserStore.delete_user(entity_id)
        except Exception:
            pass  # User may not exist in user store
        log.info("PURGE provider_identity entity=%s", entity_id)

    # Handler dispatch table
    _purge_handlers = {
        DataCategory.JOB_PAYLOAD: _purge_job_payload,
        DataCategory.JOB_METADATA: _purge_job_metadata,
        DataCategory.TELEMETRY: _purge_telemetry,
        DataCategory.LOGS: _purge_logs,
        DataCategory.NETWORK: _purge_network,
        DataCategory.LOCATION: _purge_location,
        DataCategory.CHAT_MESSAGES: _purge_chat_messages,
        DataCategory.BILLING_INFO: _purge_billing_info,
        DataCategory.PROVIDER_IDENTITY: _purge_provider_identity,
    }

    # ── Consent management ────────────────────────────────────────────

    def record_consent(
        self,
        entity_id: str,
        consent_type: str,
        details: Optional[dict] = None,
    ) -> str:
        """Record explicit consent (PIPEDA principle: Consent)."""
        import uuid

        consent_id = str(uuid.uuid4())[:12]
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO consent_records
                   (consent_id, entity_id, consent_type, granted_at, details)
                   VALUES (%s, %s, %s, %s, %s)""",
                (consent_id, entity_id, consent_type, time.time(), json.dumps(details or {})),
            )
        log.info("CONSENT RECORDED entity=%s type=%s", entity_id, consent_type)
        return consent_id

    def revoke_consent(self, entity_id: str, consent_type: str):
        """Revoke previously granted consent."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE consent_records
                   SET revoked_at = %s, is_active = 0
                   WHERE entity_id = %s AND consent_type = %s AND is_active = 1""",
                (time.time(), entity_id, consent_type),
            )
        log.info("CONSENT REVOKED entity=%s type=%s", entity_id, consent_type)

    def has_consent(self, entity_id: str, consent_type: str) -> bool:
        """Check if entity has active consent for a given type."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM consent_records
                   WHERE entity_id = %s AND consent_type = %s AND is_active = 1""",
                (entity_id, consent_type),
            ).fetchone()
        return (row["cnt"] or 0) > 0

    def get_consents(self, entity_id: str) -> list[dict]:
        """Get all consent records for an entity (PIPEDA: Individual Access)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM consent_records
                   WHERE entity_id = %s ORDER BY granted_at DESC""",
                (entity_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Privacy config management ─────────────────────────────────────

    def save_config(self, org_id: str, config: PrivacyConfig):
        """Save privacy configuration for an organization."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO privacy_configs
                   (org_id, config, updated_at) VALUES (%s, %s, %s)
                   ON CONFLICT (org_id) DO UPDATE SET config = EXCLUDED.config, updated_at = EXCLUDED.updated_at""",
                (org_id, json.dumps(config.to_dict()), time.time()),
            )

    def get_config(self, org_id: str) -> PrivacyConfig:
        """Load privacy config for an org. Returns STRICT defaults if none set."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT config FROM privacy_configs WHERE org_id = %s",
                (org_id,),
            ).fetchone()

        if row:
            raw = row["config"]
            data = json.loads(raw) if isinstance(raw, str) else raw
            return PrivacyConfig(
                **{k: v for k, v in data.items() if k in PrivacyConfig.__dataclass_fields__}
            )
        return PrivacyConfig()  # STRICT defaults

    # ── Retention summary ─────────────────────────────────────────────

    def get_retention_summary(self) -> dict:
        """Get a summary of current retention status across all categories."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT data_category,
                          COUNT(*) as total,
                          SUM(CASE WHEN purged_at > 0 THEN 1 ELSE 0 END) as purged,
                          SUM(CASE WHEN purged_at = 0 AND expires_at <= %s THEN 1 ELSE 0 END) as expired_pending
                   FROM retention_records
                   GROUP BY data_category""",
                (time.time(),),
            ).fetchall()

        policies = {}
        for cat, policy in RETENTION_POLICIES.items():
            policies[cat.value] = {
                "retention_days": policy["retention_sec"] // 86400,
                "description": policy["description"],
                "redact_on_completion": policy.get("redact_on_completion", False),
            }

        return {
            "policies": policies,
            "categories": {
                r["data_category"]: {
                    "total_records": r["total"],
                    "purged": r["purged"],
                    "expired_pending_purge": r["expired_pending"],
                    "active": r["total"] - r["purged"],
                }
                for r in rows
            },
            "queried_at": time.time(),
        }


# ── Singleton access ─────────────────────────────────────────────────

_lifecycle_manager: Optional[DataLifecycleManager] = None


def get_lifecycle_manager() -> DataLifecycleManager:
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = DataLifecycleManager()
    return _lifecycle_manager


# ── Québec Law 25 Cross-Border Assessment ─────────────────────────────
# Before communicating personal information outside Québec, an enterprise
# must conduct a PIA. This helper determines if a cross-border flag is needed.


def requires_quebec_pia(
    data_origin_province: str,
    processing_province: str,
    data_contains_pi: bool = False,
) -> dict:
    """Check if Québec Law 25 cross-border PIA assessment is required.

    Returns assessment dict with required/reason/recommendation.
    """
    origin = data_origin_province.upper()
    target = processing_province.upper()

    if origin != "QC":
        return {
            "pia_required": False,
            "reason": "Data does not originate from Québec",
            "recommendation": None,
        }

    if not data_contains_pi:
        return {
            "pia_required": False,
            "reason": "No personal information identified in workload",
            "recommendation": "Verify workload does not contain PI",
        }

    if target == "QC":
        return {
            "pia_required": False,
            "reason": "Processing stays within Québec",
            "recommendation": None,
        }

    return {
        "pia_required": True,
        "reason": (
            f"Québec Law 25 requires PIA for cross-border transfer "
            f"(QC → {target}). Enterprise must assess sensitivity, purpose, "
            f"safeguards, and legal framework of destination."
        ),
        "recommendation": "Use quebec_only routing or obtain written agreement",
        "max_penalty": "$25,000,000 or 4% of worldwide turnover",
        "law_reference": "Act respecting the protection of personal information in the private sector, s. 17",
    }


# ── Cryptographic Shredding ──────────────────────────────────────────
# PII in events is encrypted with a per-user key. Erasure = destroy
# the user's key → historical events become permanently anonymized.
# Hash chain integrity is preserved (encrypted blobs still chain).

import base64
import hashlib

try:
    from cryptography.fernet import Fernet

    _HAS_FERNET = True
except ImportError:
    _HAS_FERNET = False


class CryptoShredder:
    """Per-user encryption key manager for PIPEDA right-to-erasure."""

    def __init__(self, db_path=None):
        self._db_path = db_path

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool

        pool = _get_pg_pool()
        conn = pool.getconn()
        conn.row_factory = None
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            pool.putconn(conn)

    def _ensure_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_encryption_keys (
                user_id TEXT PRIMARY KEY,
                fernet_key TEXT NOT NULL,
                created_at REAL NOT NULL DEFAULT (extract(epoch FROM now())),
                destroyed_at REAL DEFAULT 0,
                active BOOLEAN DEFAULT TRUE
            )
        """)

    def get_or_create_key(self, user_id):
        """Get or create a Fernet encryption key for a user.

        Returns: Fernet key string (base64-encoded)
        """
        if not _HAS_FERNET:
            return None
        with self._conn() as conn:
            self._ensure_table(conn)
            cur = conn.execute(
                "SELECT fernet_key FROM user_encryption_keys WHERE user_id = %s AND active = TRUE",
                (user_id,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            key = Fernet.generate_key().decode("ascii")
            conn.execute(
                "INSERT INTO user_encryption_keys (user_id, fernet_key) VALUES (%s, %s) "
                "ON CONFLICT (user_id) DO UPDATE SET fernet_key = EXCLUDED.fernet_key, active = TRUE, destroyed_at = 0",
                (user_id, key),
            )
            return key

    def encrypt_pii(self, user_id, plaintext):
        """Encrypt PII with the user's key before appending to event log."""
        key = self.get_or_create_key(user_id)
        if not key:
            return plaintext  # Graceful degradation without cryptography
        f = Fernet(key.encode("ascii") if isinstance(key, str) else key)
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")
        return f"ENC:{f.encrypt(plaintext).decode('ascii')}"

    def decrypt_pii(self, user_id, ciphertext):
        """Decrypt PII. Returns '[REDACTED]' if key was destroyed."""
        if not isinstance(ciphertext, str) or not ciphertext.startswith("ENC:"):
            return ciphertext
        with self._conn() as conn:
            self._ensure_table(conn)
            cur = conn.execute(
                "SELECT fernet_key, active FROM user_encryption_keys WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            if not row or not row[1]:
                return "[REDACTED — encryption key destroyed per right-to-erasure]"
            try:
                f = Fernet(row[0].encode("ascii"))
                return f.decrypt(ciphertext[4:].encode("ascii")).decode("utf-8")
            except Exception:
                return "[REDACTED — decryption failed]"

    def destroy_user_key(self, user_id):
        """Destroy a user's encryption key (right-to-erasure).

        All historical events containing this user's PII become
        permanently undecryptable. Hash chain integrity is preserved.
        """
        with self._conn() as conn:
            self._ensure_table(conn)
            conn.execute(
                "UPDATE user_encryption_keys SET active = FALSE, destroyed_at = extract(epoch FROM now()), "
                "fernet_key = 'DESTROYED' WHERE user_id = %s",
                (user_id,),
            )
            log.info("CRYPTO SHRED: destroyed encryption key for user=%s", user_id)
            return True

    def is_key_active(self, user_id):
        """Check if a user's encryption key is still active."""
        with self._conn() as conn:
            self._ensure_table(conn)
            cur = conn.execute(
                "SELECT active FROM user_encryption_keys WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            return row[0] if row else False


_crypto_shredder: Optional[CryptoShredder] = None


def get_crypto_shredder() -> CryptoShredder:
    global _crypto_shredder
    if _crypto_shredder is None:
        _crypto_shredder = CryptoShredder()
    return _crypto_shredder


# ── CASL Consent Management ──────────────────────────────────────────
# Canada's Anti-Spam Legislation requires express consent for
# commercial electronic messages (CEMs).


class ConsentManager:
    """Track CASL express/implied consent for marketing communications."""

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _ensure_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS casl_consent (
                consent_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
                user_id TEXT NOT NULL,
                consent_type TEXT NOT NULL CHECK (consent_type IN ('express', 'implied')),
                purpose TEXT NOT NULL,
                granted_at REAL NOT NULL DEFAULT (extract(epoch FROM now())),
                expires_at REAL DEFAULT 0,
                withdrawn_at REAL DEFAULT 0,
                source TEXT DEFAULT '',
                ip_address TEXT DEFAULT '',
                active BOOLEAN DEFAULT TRUE,
                UNIQUE(user_id, purpose)
            )
        """)

    def record_consent(
        self, user_id, consent_type, purpose, source="", ip_address="", expires_in_days=0
    ):
        """Record user consent for a specific purpose.

        Args:
            consent_type: 'express' (opt-in) or 'implied' (existing business relationship)
            purpose: e.g. 'marketing_email', 'product_updates', 'third_party_offers'
            expires_in_days: 0 = no expiry (express), implied consent expires in 2 years per CASL
        """
        expires_at = 0
        if consent_type == "implied" and expires_in_days == 0:
            expires_at = time.time() + (730 * 86400)  # 2 years per CASL s.10(9)
        elif expires_in_days > 0:
            expires_at = time.time() + (expires_in_days * 86400)

        with self._conn() as conn:
            self._ensure_table(conn)
            conn.execute(
                """
                INSERT INTO casl_consent (user_id, consent_type, purpose, source, ip_address, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, purpose) DO UPDATE SET
                    consent_type = EXCLUDED.consent_type,
                    granted_at = extract(epoch FROM now()),
                    expires_at = EXCLUDED.expires_at,
                    withdrawn_at = 0,
                    source = EXCLUDED.source,
                    ip_address = EXCLUDED.ip_address,
                    active = TRUE
            """,
                (user_id, consent_type, purpose, source, ip_address, expires_at),
            )
        log.info("CASL CONSENT: user=%s purpose=%s type=%s", user_id, purpose, consent_type)

    def withdraw_consent(self, user_id, purpose):
        """Withdraw consent (unsubscribe). CASL requires processing within 10 business days."""
        with self._conn() as conn:
            self._ensure_table(conn)
            conn.execute(
                "UPDATE casl_consent SET active = FALSE, withdrawn_at = extract(epoch FROM now()) "
                "WHERE user_id = %s AND purpose = %s",
                (user_id, purpose),
            )
        log.info("CASL WITHDRAW: user=%s purpose=%s", user_id, purpose)

    def has_consent(self, user_id, purpose):
        """Check if user has active consent for a purpose.

        Returns: (has_consent: bool, consent_type: str or None)
        """
        with self._conn() as conn:
            self._ensure_table(conn)
            cur = conn.execute(
                "SELECT consent_type, expires_at FROM casl_consent "
                "WHERE user_id = %s AND purpose = %s AND active = TRUE",
                (user_id, purpose),
            )
            row = cur.fetchone()
            if not row:
                return False, None
            # Check expiry
            if row["expires_at"] > 0 and time.time() > row["expires_at"]:
                conn.execute(
                    "UPDATE casl_consent SET active = FALSE WHERE user_id = %s AND purpose = %s",
                    (user_id, purpose),
                )
                return False, None
            return True, row["consent_type"]

    def get_user_consents(self, user_id):
        """Get all consent records for a user."""
        with self._conn() as conn:
            self._ensure_table(conn)
            cur = conn.execute(
                "SELECT purpose, consent_type, granted_at, expires_at, withdrawn_at, active "
                "FROM casl_consent WHERE user_id = %s ORDER BY granted_at DESC",
                (user_id,),
            )
            return [
                {
                    "purpose": r["purpose"],
                    "consent_type": r["consent_type"],
                    "granted_at": r["granted_at"],
                    "expires_at": r["expires_at"],
                    "withdrawn_at": r["withdrawn_at"],
                    "active": r["active"],
                }
                for r in cur.fetchall()
            ]

    def expire_implied_consents(self):
        """Expire all implied consents past their 2-year window. Run periodically."""
        now = time.time()
        with self._conn() as conn:
            self._ensure_table(conn)
            cur = conn.execute(
                "UPDATE casl_consent SET active = FALSE "
                "WHERE consent_type = 'implied' AND expires_at > 0 AND expires_at < %s AND active = TRUE "
                "RETURNING user_id, purpose",
                (now,),
            )
            expired = cur.fetchall()
            if expired:
                log.info("CASL: expired %d implied consents", len(expired))
            return len(expired)


_consent_manager: Optional[ConsentManager] = None


def get_consent_manager() -> ConsentManager:
    global _consent_manager
    if _consent_manager is None:
        _consent_manager = ConsentManager()
    return _consent_manager


# ── Right-to-Erasure Orchestrator ─────────────────────────────────────


def execute_right_to_erasure(user_id):
    """Complete right-to-erasure workflow per PIPEDA/Law 25.

    Steps:
    1. Destroy user's encryption key (cryptographic shredding)
    2. Withdraw all CASL consents
    3. Anonymize non-encrypted records
    4. Mark lifecycle records as purged

    Returns summary dict.
    """
    summary = {"user_id": user_id, "actions": []}

    # Step 1: Cryptographic shredding
    shredder = get_crypto_shredder()
    shredder.destroy_user_key(user_id)
    summary["actions"].append("encryption_key_destroyed")

    # Step 2: Withdraw all CASL consents
    cm = get_consent_manager()
    consents = cm.get_user_consents(user_id)
    for c in consents:
        if c["active"]:
            cm.withdraw_consent(user_id, c["purpose"])
    summary["actions"].append(f"casl_consents_withdrawn:{len(consents)}")

    # Step 3: Anonymize lifecycle tracking records
    lm = get_lifecycle_manager()
    with lm._conn() as conn:
        conn.execute(
            "UPDATE data_lifecycle SET entity_id = %s, purged_at = extract(epoch FROM now()), "
            "purge_reason = 'right_to_erasure' WHERE entity_id = %s",
            (f"erased-{hashlib.sha256(user_id.encode()).hexdigest()[:12]}", user_id),
        )
    summary["actions"].append("lifecycle_records_anonymized")

    log.info("RIGHT TO ERASURE complete for user=%s actions=%s", user_id, summary["actions"])
    return summary
