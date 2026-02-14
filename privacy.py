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
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("xcelsior")


# ── Privacy Settings (defaults = maximum confidentiality per Law 25) ──

class PrivacyLevel(str, Enum):
    """Privacy level for data handling. Default is STRICT per Québec Law 25."""
    STRICT = "strict"         # Maximum confidentiality, all PII redacted
    STANDARD = "standard"     # Reasonable safeguards, minimal PII retained
    PERMISSIVE = "permissive" # Explicit opt-in by user — full metadata retained


class DataCategory(str, Enum):
    """Categories of data for retention and redaction policies."""
    JOB_PAYLOAD = "job_payload"           # Container args, env vars, commands
    JOB_METADATA = "job_metadata"         # Job name, tags, labels
    PROVIDER_IDENTITY = "provider_identity"  # Host owner info
    BILLING_INFO = "billing_info"         # Payment details, wallet
    TELEMETRY = "telemetry"              # GPU metrics, utilization data
    LOGS = "logs"                         # Job stdout/stderr
    NETWORK = "network"                   # IP addresses, connection data
    LOCATION = "location"                 # Geolocation, province, city


# ── Retention Policies ────────────────────────────────────────────────
# PIPEDA principle: "Limiting Use, Disclosure, and Retention —
# keeping data only as long as required for the identified job."

# Default retention periods in seconds
RETENTION_POLICIES = {
    DataCategory.JOB_PAYLOAD: {
        "retention_sec": 0,           # Deleted immediately after job completion
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
        "retention_sec": 7 * 86400,   # 7 days
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

ALWAYS_REDACT_ENV_VARS = frozenset({
    "PASSWORD", "SECRET", "TOKEN", "API_KEY", "PRIVATE_KEY",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "DATABASE_URL", "DB_PASSWORD", "SMTP_PASSWORD",
    "STRIPE_SECRET_KEY", "WEBHOOK_SECRET",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "WANDB_API_KEY",
})


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
    enable_identification: bool = False    # Link jobs to named individuals
    enable_location_tracking: bool = False # Detailed geolocation beyond country
    enable_profiling: bool = False         # Usage pattern analysis

    # Retention overrides (None = use defaults)
    log_retention_days: Optional[int] = None
    telemetry_retention_days: Optional[int] = None
    metadata_retention_days: Optional[int] = None

    # Redaction
    redact_pii_in_logs: bool = True        # Scan logs for PII patterns
    redact_env_vars: bool = True           # Scrub env vars from job records
    redact_ip_addresses: bool = True       # Replace IPs in stored data

    # Consent
    cross_border_consent: bool = False     # Explicit consent for non-CA processing
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
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "xcelsior_privacy.db"
        )
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS retention_records (
                    record_id TEXT PRIMARY KEY,
                    data_category TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT DEFAULT 'job',
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    purged_at REAL DEFAULT 0,
                    purge_reason TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_retention_expiry
                    ON retention_records(expires_at);
                CREATE INDEX IF NOT EXISTS idx_retention_entity
                    ON retention_records(entity_id);
                CREATE INDEX IF NOT EXISTS idx_retention_category
                    ON retention_records(data_category);

                CREATE TABLE IF NOT EXISTS consent_records (
                    consent_id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    consent_type TEXT NOT NULL,
                    granted_at REAL NOT NULL,
                    revoked_at REAL DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    details TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_consent_entity
                    ON consent_records(entity_id);

                CREATE TABLE IF NOT EXISTS privacy_configs (
                    org_id TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

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
        cat = DataCategory(data_category) if data_category in DataCategory.__members__.values() else None
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
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (record_id, data_category, entity_id, entity_type,
                 now, expires_at),
            )

        log.debug("RETENTION TRACKED %s/%s category=%s expires=%.0f",
                   entity_type, entity_id, data_category, expires_at)
        return record_id

    def get_expired_records(self, before: Optional[float] = None) -> list[dict]:
        """Get all retention records that have expired and need purging."""
        cutoff = before or time.time()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM retention_records
                   WHERE expires_at <= ? AND purged_at = 0
                   ORDER BY expires_at""",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_purged(self, record_id: str, reason: str = "retention_expired"):
        """Mark a retention record as purged."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE retention_records
                   SET purged_at = ?, purge_reason = ?
                   WHERE record_id = ?""",
                (time.time(), reason, record_id),
            )
        log.info("DATA PURGED record=%s reason=%s", record_id, reason)

    def purge_expired(self) -> int:
        """Purge all expired data records. Returns count of records purged.

        This should be called periodically (e.g., daily cron).
        In production, this would also delete the actual data from
        event stores, log stores, etc.
        """
        expired = self.get_expired_records()
        count = 0
        for record in expired:
            self.mark_purged(record["record_id"])
            count += 1

        if count > 0:
            log.info("RETENTION PURGE completed: %d records purged", count)

        return count

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
                   VALUES (?, ?, ?, ?, ?)""",
                (consent_id, entity_id, consent_type, time.time(),
                 json.dumps(details or {})),
            )
        log.info("CONSENT RECORDED entity=%s type=%s", entity_id, consent_type)
        return consent_id

    def revoke_consent(self, entity_id: str, consent_type: str):
        """Revoke previously granted consent."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE consent_records
                   SET revoked_at = ?, is_active = 0
                   WHERE entity_id = ? AND consent_type = ? AND is_active = 1""",
                (time.time(), entity_id, consent_type),
            )
        log.info("CONSENT REVOKED entity=%s type=%s", entity_id, consent_type)

    def has_consent(self, entity_id: str, consent_type: str) -> bool:
        """Check if entity has active consent for a given type."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM consent_records
                   WHERE entity_id = ? AND consent_type = ? AND is_active = 1""",
                (entity_id, consent_type),
            ).fetchone()
        return (row["cnt"] or 0) > 0

    def get_consents(self, entity_id: str) -> list[dict]:
        """Get all consent records for an entity (PIPEDA: Individual Access)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM consent_records
                   WHERE entity_id = ? ORDER BY granted_at DESC""",
                (entity_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Privacy config management ─────────────────────────────────────

    def save_config(self, org_id: str, config: PrivacyConfig):
        """Save privacy configuration for an organization."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO privacy_configs
                   (org_id, config, updated_at) VALUES (?, ?, ?)""",
                (org_id, json.dumps(config.to_dict()), time.time()),
            )

    def get_config(self, org_id: str) -> PrivacyConfig:
        """Load privacy config for an org. Returns STRICT defaults if none set."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT config FROM privacy_configs WHERE org_id = ?",
                (org_id,),
            ).fetchone()

        if row:
            data = json.loads(row["config"])
            return PrivacyConfig(**{
                k: v for k, v in data.items()
                if k in PrivacyConfig.__dataclass_fields__
            })
        return PrivacyConfig()  # STRICT defaults

    # ── Retention summary ─────────────────────────────────────────────

    def get_retention_summary(self) -> dict:
        """Get a summary of current retention status across all categories."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT data_category,
                          COUNT(*) as total,
                          SUM(CASE WHEN purged_at > 0 THEN 1 ELSE 0 END) as purged,
                          SUM(CASE WHEN purged_at = 0 AND expires_at <= ? THEN 1 ELSE 0 END) as expired_pending
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
            "categories": {r["data_category"]: {
                "total_records": r["total"],
                "purged": r["purged"],
                "expired_pending_purge": r["expired_pending"],
                "active": r["total"] - r["purged"],
            } for r in rows},
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
