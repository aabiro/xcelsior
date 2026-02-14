"""Tests for Xcelsior privacy module — PII redaction, retention lifecycle, consent, Law 25."""

import json
import os
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from privacy import (
    ALWAYS_REDACT_ENV_VARS,
    PII_PATTERNS,
    REDACTION_PLACEHOLDER,
    RETENTION_POLICIES,
    DataCategory,
    DataLifecycleManager,
    PrivacyConfig,
    PrivacyLevel,
    redact_env_vars,
    redact_job_record,
    redact_pii,
    requires_quebec_pia,
    sanitize_log_output,
)


# ── PII Redaction ─────────────────────────────────────────────────────


class TestRedactPII:
    """Verify PII pattern detection and redaction across all 6 pattern types."""

    def test_redacts_email(self):
        text = "Contact user at alice@example.com for details"
        assert "[REDACTED]" in redact_pii(text)
        assert "alice@example.com" not in redact_pii(text)

    def test_redacts_phone(self):
        text = "Call 555-123-4567 or 5551234567"
        result = redact_pii(text)
        assert "555-123-4567" not in result
        assert "5551234567" not in result

    def test_redacts_canadian_sin(self):
        text = "SIN: 123-456-789"
        assert "123-456-789" not in redact_pii(text)

    def test_redacts_credit_card(self):
        text = "Card: 4111-1111-1111-1111"
        assert "4111-1111-1111-1111" not in redact_pii(text)

    def test_redacts_ip_address(self):
        text = "Host IP: 192.168.1.100"
        result = redact_pii(text)
        assert "192.168.1.100" not in result

    def test_redacts_api_key(self):
        text = "Using api_key_abcdefghijklmnopqrstuvwx"
        result = redact_pii(text)
        assert "api_key_abcdef" not in result

    def test_empty_string_unchanged(self):
        assert redact_pii("") == ""

    def test_none_returns_none(self):
        assert redact_pii(None) is None

    def test_no_pii_unchanged(self):
        text = "GPU utilization at 95% for RTX 4090"
        assert redact_pii(text) == text

    def test_multiple_pii_all_redacted(self):
        text = "User alice@test.com called from 555-111-2222 with IP 10.0.0.1"
        result = redact_pii(text)
        assert "alice@test.com" not in result
        assert "555-111-2222" not in result
        assert "10.0.0.1" not in result

    def test_custom_patterns(self):
        custom = {"email": PII_PATTERNS["email"]}
        text = "alice@test.com and IP 10.0.0.1"
        result = redact_pii(text, patterns=custom)
        assert "alice@test.com" not in result
        # IP should NOT be redacted with only email pattern
        assert "10.0.0.1" in result


# ── Environment Variable Redaction ────────────────────────────────────


class TestRedactEnvVars:
    """Verify env var scrubbing for known secret patterns."""

    def test_known_secrets_redacted(self):
        env = {
            "API_KEY": "sk-12345",
            "DATABASE_URL": "postgres://user:pass@host/db",
            "NORMAL_VAR": "safe_value",
        }
        safe = redact_env_vars(env)
        assert safe["API_KEY"] == REDACTION_PLACEHOLDER
        assert safe["DATABASE_URL"] == REDACTION_PLACEHOLDER
        assert safe["NORMAL_VAR"] == "safe_value"

    def test_keyword_pattern_redaction(self):
        env = {
            "MY_SECRET_VALUE": "hidden",
            "CUSTOM_PASSWORD": "pass123",
            "CUSTOM_TOKEN_XYZ": "tok-abc",
            "JOB_NAME": "llama3",
        }
        safe = redact_env_vars(env)
        assert safe["MY_SECRET_VALUE"] == REDACTION_PLACEHOLDER
        assert safe["CUSTOM_PASSWORD"] == REDACTION_PLACEHOLDER
        assert safe["CUSTOM_TOKEN_XYZ"] == REDACTION_PLACEHOLDER
        assert safe["JOB_NAME"] == "llama3"

    def test_empty_env_returns_empty(self):
        assert redact_env_vars({}) == {}
        assert redact_env_vars(None) == {}

    def test_all_always_redact_names(self):
        """Every name in ALWAYS_REDACT_ENV_VARS should be redacted."""
        env = {name: "value" for name in ALWAYS_REDACT_ENV_VARS}
        safe = redact_env_vars(env)
        for name in ALWAYS_REDACT_ENV_VARS:
            assert safe[name] == REDACTION_PLACEHOLDER, f"{name} was not redacted"

    def test_case_insensitive_keyword_match(self):
        env = {"my_credential_id": "cred-123"}
        safe = redact_env_vars(env)
        assert safe["my_credential_id"] == REDACTION_PLACEHOLDER


# ── Job Record Redaction ──────────────────────────────────────────────


class TestRedactJobRecord:
    """Verify full job record redaction with privacy config."""

    def test_strict_redacts_commands(self):
        job = {
            "job_id": "j1",
            "name": "llama3",
            "container_command": "/run.sh --secret-key=abc123",
            "container_args": "--model=gpt --api-key=sk-xxx",
            "env": {"API_KEY": "secret"},
        }
        safe = redact_job_record(job, PrivacyConfig(privacy_level=PrivacyLevel.STRICT))
        assert safe["container_command"] == REDACTION_PLACEHOLDER
        assert safe["container_args"] == REDACTION_PLACEHOLDER
        assert safe["env"]["API_KEY"] == REDACTION_PLACEHOLDER

    def test_permissive_retains_commands(self):
        job = {
            "job_id": "j1",
            "container_command": "/run.sh",
            "container_args": "--model=gpt",
        }
        safe = redact_job_record(
            job,
            PrivacyConfig(privacy_level=PrivacyLevel.PERMISSIVE),
        )
        assert safe["container_command"] == "/run.sh"
        assert safe["container_args"] == "--model=gpt"

    def test_ip_address_redacted(self):
        job = {"ip": "192.168.1.50", "host_ip": "10.0.0.1"}
        safe = redact_job_record(job)
        assert "192.168.1.50" not in safe["ip"]
        assert "10.0.0.1" not in safe["host_ip"]

    def test_pii_in_logs_redacted(self):
        job = {"stdout": "Error for user alice@test.com at IP 10.0.0.1"}
        safe = redact_job_record(job)
        assert "alice@test.com" not in safe["stdout"]

    def test_location_stripped_when_disabled(self):
        job = {"city": "Toronto", "latitude": "43.6", "longitude": "-79.3"}
        cfg = PrivacyConfig(enable_location_tracking=False)
        safe = redact_job_record(job, cfg)
        assert safe["city"] == ""
        assert safe["latitude"] == ""

    def test_location_kept_when_enabled(self):
        job = {"city": "Toronto"}
        cfg = PrivacyConfig(enable_location_tracking=True)
        safe = redact_job_record(job, cfg)
        assert safe["city"] == "Toronto"

    def test_identification_stripped_by_default(self):
        job = {"owner_name": "Alice", "submitter_name": "Bob"}
        safe = redact_job_record(job)
        assert safe["owner_name"] == REDACTION_PLACEHOLDER
        assert safe["submitter_name"] == REDACTION_PLACEHOLDER

    def test_identification_kept_when_enabled(self):
        job = {"owner_name": "Alice"}
        cfg = PrivacyConfig(enable_identification=True)
        safe = redact_job_record(job, cfg)
        assert safe["owner_name"] == "Alice"


# ── Sanitize Log Output ──────────────────────────────────────────────


class TestSanitizeLogOutput:
    def test_truncation(self):
        long_log = "A" * 20000
        result = sanitize_log_output(long_log, max_length=100)
        assert len(result) <= 120  # 100 + truncation marker
        assert "TRUNCATED" in result

    def test_pii_removed(self):
        text = "Error: contact admin@company.com"
        result = sanitize_log_output(text)
        assert "admin@company.com" not in result

    def test_empty_returns_empty(self):
        assert sanitize_log_output("") == ""
        assert sanitize_log_output(None) == ""

    def test_short_log_no_truncation(self):
        text = "Job completed successfully"
        result = sanitize_log_output(text)
        assert result == text
        assert "TRUNCATED" not in result


# ── Privacy Config ────────────────────────────────────────────────────


class TestPrivacyConfig:
    """Verify Law 25 default settings."""

    def test_defaults_are_strict(self):
        cfg = PrivacyConfig()
        assert cfg.privacy_level == PrivacyLevel.STRICT
        assert cfg.enable_identification is False
        assert cfg.enable_location_tracking is False
        assert cfg.enable_profiling is False
        assert cfg.redact_pii_in_logs is True
        assert cfg.redact_env_vars is True
        assert cfg.redact_ip_addresses is True
        assert cfg.cross_border_consent is False

    def test_to_dict(self):
        cfg = PrivacyConfig()
        d = cfg.to_dict()
        assert d["privacy_level"] == "strict"
        assert isinstance(d, dict)


# ── Retention Policies ────────────────────────────────────────────────


class TestRetentionPolicies:
    def test_all_categories_have_policies(self):
        for cat in DataCategory:
            assert cat in RETENTION_POLICIES, f"Missing policy for {cat.value}"

    def test_job_payload_immediate_deletion(self):
        policy = RETENTION_POLICIES[DataCategory.JOB_PAYLOAD]
        assert policy["retention_sec"] == 0
        assert policy["redact_on_completion"] is True

    def test_billing_7_year_retention(self):
        policy = RETENTION_POLICIES[DataCategory.BILLING_INFO]
        assert policy["retention_sec"] == 7 * 365 * 86400

    def test_logs_7_day_retention(self):
        policy = RETENTION_POLICIES[DataCategory.LOGS]
        assert policy["retention_sec"] == 7 * 86400


# ── Data Lifecycle Manager ────────────────────────────────────────────


class TestDataLifecycleManager:
    """Test retention tracking, purging, and consent with SQLite backend."""

    @pytest.fixture(autouse=True)
    def lifecycle_mgr(self, tmp_path):
        self.mgr = DataLifecycleManager(
            db_path=str(tmp_path / "privacy_test.db")
        )

    def test_track_data_returns_record_id(self):
        rid = self.mgr.track_data("job_metadata", "job-1")
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_track_data_uses_default_retention(self):
        rid = self.mgr.track_data("job_metadata", "job-1")
        records = self.mgr.get_expired_records(before=time.time() + 91 * 86400)
        found = [r for r in records if r["record_id"] == rid]
        assert len(found) == 1

    def test_track_data_with_override_retention(self):
        rid = self.mgr.track_data("job_metadata", "job-2", retention_override_sec=1)
        time.sleep(1.1)
        expired = self.mgr.get_expired_records()
        ids = [r["record_id"] for r in expired]
        assert rid in ids

    def test_get_expired_records_empty(self):
        expired = self.mgr.get_expired_records()
        assert expired == []

    def test_mark_purged(self):
        rid = self.mgr.track_data("logs", "job-3", retention_override_sec=0)
        self.mgr.mark_purged(rid, reason="test")
        # Should no longer appear as expired
        expired = self.mgr.get_expired_records(
            before=time.time() + 86400
        )
        ids = [r["record_id"] for r in expired]
        assert rid not in ids

    def test_purge_expired(self):
        # Create records that expire immediately
        for i in range(5):
            self.mgr.track_data("telemetry", f"job-{i}", retention_override_sec=0)
        count = self.mgr.purge_expired()
        assert count == 5
        # Second call should purge 0
        assert self.mgr.purge_expired() == 0


# ── Consent Management ───────────────────────────────────────────────


class TestConsent:
    @pytest.fixture(autouse=True)
    def lifecycle_mgr(self, tmp_path):
        self.mgr = DataLifecycleManager(
            db_path=str(tmp_path / "consent_test.db")
        )

    def test_record_consent(self):
        cid = self.mgr.record_consent("user-1", "data_collection")
        assert isinstance(cid, str)
        assert self.mgr.has_consent("user-1", "data_collection") is True

    def test_no_consent_by_default(self):
        assert self.mgr.has_consent("user-x", "analytics") is False

    def test_revoke_consent(self):
        self.mgr.record_consent("user-2", "cross_border")
        assert self.mgr.has_consent("user-2", "cross_border") is True
        self.mgr.revoke_consent("user-2", "cross_border")
        assert self.mgr.has_consent("user-2", "cross_border") is False

    def test_get_consents(self):
        self.mgr.record_consent("user-3", "analytics")
        self.mgr.record_consent("user-3", "cross_border")
        consents = self.mgr.get_consents("user-3")
        assert len(consents) == 2
        types = [c["consent_type"] for c in consents]
        assert "analytics" in types
        assert "cross_border" in types

    def test_consent_with_details(self):
        details = {"purpose": "improve service", "version": "1.0"}
        self.mgr.record_consent("user-4", "analytics", details=details)
        consents = self.mgr.get_consents("user-4")
        stored_details = json.loads(consents[0]["details"])
        assert stored_details["purpose"] == "improve service"


# ── Privacy Config Persistence ────────────────────────────────────────


class TestConfigPersistence:
    @pytest.fixture(autouse=True)
    def lifecycle_mgr(self, tmp_path):
        self.mgr = DataLifecycleManager(
            db_path=str(tmp_path / "config_test.db")
        )

    def test_save_and_load_config(self):
        cfg = PrivacyConfig(
            privacy_level=PrivacyLevel.PERMISSIVE,
            enable_identification=True,
            enable_location_tracking=True,
        )
        self.mgr.save_config("org-1", cfg)
        loaded = self.mgr.get_config("org-1")
        assert loaded.privacy_level == PrivacyLevel.PERMISSIVE
        assert loaded.enable_identification is True
        assert loaded.enable_location_tracking is True

    def test_unknown_org_returns_strict_defaults(self):
        cfg = self.mgr.get_config("nonexistent-org")
        assert cfg.privacy_level == PrivacyLevel.STRICT
        assert cfg.enable_identification is False

    def test_overwrite_config(self):
        self.mgr.save_config("org-2", PrivacyConfig())
        self.mgr.save_config(
            "org-2",
            PrivacyConfig(privacy_level=PrivacyLevel.STANDARD),
        )
        loaded = self.mgr.get_config("org-2")
        assert loaded.privacy_level == PrivacyLevel.STANDARD


# ── Retention Summary ─────────────────────────────────────────────────


class TestRetentionSummary:
    @pytest.fixture(autouse=True)
    def lifecycle_mgr(self, tmp_path):
        self.mgr = DataLifecycleManager(
            db_path=str(tmp_path / "summary_test.db")
        )

    def test_summary_with_data(self):
        self.mgr.track_data("logs", "j-1", retention_override_sec=0)
        self.mgr.track_data("logs", "j-2", retention_override_sec=86400)
        summary = self.mgr.get_retention_summary()
        assert "policies" in summary
        assert "categories" in summary
        assert "queried_at" in summary

    def test_summary_shows_policies(self):
        summary = self.mgr.get_retention_summary()
        assert "logs" in summary["policies"]
        assert summary["policies"]["logs"]["retention_days"] == 7


# ── Québec Law 25 Cross-Border Assessment ─────────────────────────────


class TestQuebecPIA:
    def test_non_qc_origin_no_pia(self):
        result = requires_quebec_pia("ON", "BC", data_contains_pi=True)
        assert result["pia_required"] is False

    def test_qc_origin_no_pi_no_pia(self):
        result = requires_quebec_pia("QC", "ON", data_contains_pi=False)
        assert result["pia_required"] is False
        assert result["recommendation"] is not None

    def test_qc_to_qc_no_pia(self):
        result = requires_quebec_pia("QC", "QC", data_contains_pi=True)
        assert result["pia_required"] is False

    def test_qc_to_on_with_pi_requires_pia(self):
        result = requires_quebec_pia("QC", "ON", data_contains_pi=True)
        assert result["pia_required"] is True
        assert "Law 25" in result["reason"]
        assert "max_penalty" in result
        assert "law_reference" in result

    def test_case_insensitive(self):
        result = requires_quebec_pia("qc", "on", data_contains_pi=True)
        assert result["pia_required"] is True
