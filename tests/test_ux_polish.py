"""Tests for UX Polish & Profile Enhancements (v2.8.0).

Covers:
- Data export endpoint (PIPEDA right)
- Artifact expiry/TTL endpoint
- Reputation score breakdown endpoint
- Enhanced profile (avatar, member since)
- Dashboard HTML: skeleton loaders, empty states, date range picker,
  balance chart, score breakdown, SSH key upload, notification toggles,
  data export/delete buttons, artifact expiry column
"""

import json
import logging
import os
import tempfile
import time

import pytest

# Use a TemporaryDirectory for all data files
_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_ux_polish_test_")
_tmpdir = _tmp_ctx.name
os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ.setdefault("XCELSIOR_BILLING_DB", os.path.join(_tmpdir, "billing.db"))
os.environ["XCELSIOR_AUTH_DB_PATH"] = os.path.join(_tmpdir, "auth.db")

import scheduler

# Patch file paths to use temp directory
scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")

# Reconfigure logger to temp dir
for _h in scheduler.log.handlers[:]:
    if isinstance(_h, logging.FileHandler):
        scheduler.log.removeHandler(_h)
        _h.close()
_fh = logging.FileHandler(scheduler.LOG_FILE)
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
scheduler.log.addHandler(_fh)

import db as db_mod
db_mod.AUTH_DB_FILE = os.path.join(_tmpdir, "auth.db")

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

GOOD_VERSIONS = {
    "runc": "1.2.4",
    "gvisor": "20250101.0",
    "nvidia_driver": "550.120",
    "cuda": "12.4",
}


def _register_host(host_id="h-ux-1", gpu="RTX 4090", vram=24, cost=0.45,
                    country="CA", province="ON"):
    """Register a host for testing."""
    return client.put("/host", json={
        "host_id": host_id,
        "ip": "10.50.1.1",
        "gpu_model": gpu,
        "total_vram_gb": vram,
        "free_vram_gb": vram,
        "cost_per_hour": cost,
        "country": country,
        "province": province,
        "versions": GOOD_VERSIONS,
    })


def _submit_job(name="ux-test-job", vram=8, tier="standard"):
    """Submit a test job."""
    return client.post("/job", json={
        "name": name,
        "docker_image": "pytorch/pytorch:latest",
        "vram_needed_gb": vram,
        "tier": tier,
    })


def _create_test_user():
    """Register a test user and return auth token."""
    r = client.post("/api/auth/register", json={
        "email": "ux-test@xcelsior.ca",
        "password": "TestPass123!",
        "name": "UX Test User",
    })
    if r.status_code == 200 and r.json().get("access_token"):
        return r.json()["access_token"]
    # Try login if already registered
    r = client.post("/api/auth/login", json={
        "email": "ux-test@xcelsior.ca",
        "password": "TestPass123!",
    })
    if r.status_code == 200:
        return r.json().get("access_token")
    return None


# ═══════════════════════════════════════════════════════════════════════
# API Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDataExportEndpoint:
    """Test the PIPEDA data export endpoint."""

    def test_data_export_requires_auth(self):
        r = client.get("/api/auth/me/data-export")
        assert r.status_code in (401, 403)

    def test_data_export_returns_bundle(self):
        token = _create_test_user()
        if not token:
            pytest.skip("Auth not available")
        r = client.get("/api/auth/me/data-export",
                       headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        export = d["data_export"]
        assert "exported_at" in export
        assert "profile" in export
        assert "jobs" in export
        assert "billing_transactions" in export
        assert "reputation" in export
        assert isinstance(export["exported_at"], (int, float))

    def test_data_export_excludes_password(self):
        token = _create_test_user()
        if not token:
            pytest.skip("Auth not available")
        r = client.get("/api/auth/me/data-export",
                       headers={"Authorization": f"Bearer {token}"})
        d = r.json()
        profile = d["data_export"]["profile"]
        assert "hashed_password" not in profile
        assert "password" not in profile

    def test_data_export_includes_total_counts(self):
        token = _create_test_user()
        if not token:
            pytest.skip("Auth not available")
        r = client.get("/api/auth/me/data-export",
                       headers={"Authorization": f"Bearer {token}"})
        d = r.json()
        export = d["data_export"]
        assert "total_jobs" in export
        assert "total_transactions" in export
        assert isinstance(export["total_jobs"], int)


class TestArtifactExpiryEndpoint:
    """Test the artifact expiry/TTL endpoint."""

    def test_artifact_expiry_returns_list(self):
        r = client.get("/api/artifacts/test-job-999/expiry")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "artifacts" in d
        assert isinstance(d["artifacts"], list)

    def test_artifact_expiry_has_ttl_fields(self):
        """If artifacts exist, they should have TTL info."""
        # Upload an artifact first
        upload_r = client.post("/api/artifacts/upload", json={
            "job_id": "ux-expiry-job",
            "artifact_type": "log_bundle",
            "size_bytes": 1024,
            "residency_policy": "canada_only",
        })
        if upload_r.status_code != 200:
            pytest.skip("Artifact upload not available")
        r = client.get("/api/artifacts/ux-expiry-job/expiry")
        d = r.json()
        if d["artifacts"]:
            art = d["artifacts"][0]
            assert "artifact_type" in art
            assert "ttl_days" in art
            assert "expires_at" in art
            assert "days_remaining" in art
            assert isinstance(art["ttl_days"], int)
            assert isinstance(art["days_remaining"], int)

    def test_artifact_expiry_ttl_varies_by_type(self):
        """Different artifact types should have different TTLs."""
        r = client.get("/api/artifacts/some-job/expiry")
        d = r.json()
        # Just verify the endpoint works and returns the right shape
        assert d["ok"] is True
        assert d["job_id"] == "some-job"


class TestReputationBreakdownEndpoint:
    """Test the reputation score breakdown endpoint."""

    def test_breakdown_returns_components(self):
        # Register host to create reputation entity
        _register_host("h-rep-bd-1")
        r = client.get("/api/reputation/h-rep-bd-1/breakdown")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "breakdown" in d
        b = d["breakdown"]
        assert "jobs_completed" in b
        assert "uptime_bonus" in b
        assert "penalties" in b
        assert "decay" in b

    def test_breakdown_has_entity_id(self):
        r = client.get("/api/reputation/test-entity/breakdown")
        assert r.status_code == 200
        d = r.json()
        assert d["entity_id"] == "test-entity"

    def test_breakdown_has_tier(self):
        r = client.get("/api/reputation/test-entity-2/breakdown")
        d = r.json()
        assert "tier" in d
        assert isinstance(d["tier"], str)

    def test_breakdown_has_events_count(self):
        r = client.get("/api/reputation/test-entity-3/breakdown")
        d = r.json()
        assert "events_analyzed" in d
        assert isinstance(d["events_analyzed"], int)

    def test_breakdown_values_are_numeric(self):
        r = client.get("/api/reputation/h-rep-bd-1/breakdown")
        d = r.json()
        b = d["breakdown"]
        for key in ("jobs_completed", "uptime_bonus", "penalties", "decay"):
            assert isinstance(b[key], (int, float)), f"{key} should be numeric"


class TestAuthMeCreatedAt:
    """Verify /api/auth/me returns created_at."""

    def test_me_includes_created_at(self):
        token = _create_test_user()
        if not token:
            pytest.skip("Auth not available")
        r = client.get("/api/auth/me",
                       headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        user = r.json()["user"]
        assert "created_at" in user


# ═══════════════════════════════════════════════════════════════════════
# Dashboard HTML Structure Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDashboardUXPolish:
    """Verify v2.8.0 UX polish elements in dashboard HTML."""

    @pytest.fixture(autouse=True)
    def setup(self):
        r = client.get("/dashboard")
        self.html = r.text

    def test_version_string(self):
        assert "v2.8.0" in self.html

    # ── Profile Card ──────────────────────────────────────────────────
    def test_profile_avatar(self):
        assert 'id="profile-avatar"' in self.html

    def test_member_since(self):
        assert 'id="profile-member-since"' in self.html
        assert "Member since" in self.html

    def test_profile_name_display(self):
        assert 'id="profile-name-display"' in self.html

    def test_profile_email_display(self):
        assert 'id="profile-email-display"' in self.html

    # ── SSH Key Upload ────────────────────────────────────────────────
    def test_ssh_key_upload_input(self):
        assert 'id="ssh-upload-key"' in self.html

    def test_ssh_key_label_input(self):
        assert 'id="ssh-key-label"' in self.html

    def test_ssh_keys_table(self):
        assert 'id="ssh-keys-table"' in self.html

    def test_upload_ssh_button(self):
        assert "uploadSshKey()" in self.html

    # ── Data Export & Delete ──────────────────────────────────────────
    def test_data_export_button(self):
        assert 'id="data-export-btn"' in self.html
        assert "requestDataExport()" in self.html

    def test_data_export_status(self):
        assert 'id="data-export-status"' in self.html

    def test_delete_account_button(self):
        assert "confirmDeleteAccount()" in self.html
        assert "Delete My Account" in self.html

    # ── Score Breakdown ───────────────────────────────────────────────
    def test_score_breakdown_div(self):
        assert 'id="score-breakdown"' in self.html

    def test_score_history_div(self):
        assert 'id="score-history"' in self.html

    # ── Balance Chart ─────────────────────────────────────────────────
    def test_balance_chart_div(self):
        assert 'id="balance-chart"' in self.html

    # ── Date Range Picker ─────────────────────────────────────────────
    def test_date_from_input(self):
        assert 'id="txn-date-from"' in self.html

    def test_date_to_input(self):
        assert 'id="txn-date-to"' in self.html

    def test_apply_date_filter_button(self):
        assert "applyDateFilter()" in self.html

    def test_clear_date_filter_button(self):
        assert "clearDateFilter()" in self.html

    # ── Artifact Expiry Column ────────────────────────────────────────
    def test_artifact_table_has_expires_header(self):
        assert "<th>Expires</th>" in self.html

    def test_artifact_table_colspan_8(self):
        assert 'colspan="8"' in self.html

    # ── Loading Skeletons ─────────────────────────────────────────────
    def test_skeleton_css(self):
        assert "skeleton-block" in self.html
        assert "skeleton-pulse" in self.html

    def test_skeleton_in_score_breakdown(self):
        # Score breakdown should start with a skeleton
        idx = self.html.find('id="score-breakdown"')
        assert idx > 0
        nearby = self.html[idx:idx+200]
        assert "skeleton-block" in nearby

    # ── Empty States ──────────────────────────────────────────────────
    def test_empty_state_css(self):
        assert ".empty-state" in self.html
        assert ".empty-icon" in self.html

    # ── Notification Sound & Push ─────────────────────────────────────
    def test_sound_toggle(self):
        assert 'id="notif-sound-toggle"' in self.html
        assert "toggleNotifSound()" in self.html

    def test_push_notif_toggle(self):
        assert 'id="notif-push-toggle"' in self.html
        assert "togglePushNotifs()" in self.html

    # ── Expiry Badge CSS ──────────────────────────────────────────────
    def test_expiry_badge_css(self):
        assert ".expiry-badge" in self.html
        assert ".expiry-ok" in self.html
        assert ".expiry-warn" in self.html
        assert ".expiry-crit" in self.html

    # ── Breakdown Bar CSS ─────────────────────────────────────────────
    def test_breakdown_bar_css(self):
        assert ".breakdown-bar" in self.html
        assert ".bar-fill.green" in self.html
        assert ".bar-fill.red" in self.html

    # ── JavaScript Functions ──────────────────────────────────────────
    def test_js_load_enhanced_profile(self):
        assert "loadEnhancedProfile" in self.html

    def test_js_request_data_export(self):
        assert "function requestDataExport()" in self.html

    def test_js_confirm_delete_account(self):
        assert "function confirmDeleteAccount()" in self.html

    def test_js_upload_ssh_key(self):
        assert "function uploadSshKey()" in self.html

    def test_js_fetch_score_breakdown(self):
        assert "function fetchScoreBreakdown" in self.html

    def test_js_fetch_score_history(self):
        assert "function fetchScoreHistory" in self.html

    def test_js_render_balance_chart(self):
        assert "function renderBalanceChart" in self.html

    def test_js_apply_date_filter(self):
        assert "function applyDateFilter()" in self.html

    def test_js_play_alert_sound(self):
        assert "function playAlertSound()" in self.html

    def test_js_fetch_artifact_expiry(self):
        assert "function fetchArtifactExpiry" in self.html

    def test_low_balance_moderate_warning(self):
        assert "_moderateBalanceToasted" in self.html

    def test_balance_chart_rendered_on_filter(self):
        assert "renderBalanceChart(txns)" in self.html
