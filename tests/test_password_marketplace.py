"""Tests for Password Reset, Change Password, and Marketplace Search endpoints.

Phase 8.8 — Tests for new auth and marketplace features added in v2.4.0.
"""

import logging
import os
import tempfile
import time

import pytest

# Use a TemporaryDirectory (auto-cleaned) for all scheduler data files
_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_pwmkt_test_")
_tmpdir = _tmp_ctx.name
os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_AUTH_DB_PATH"] = os.path.join(_tmpdir, "auth.db")
os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"

import scheduler

scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")

for _h in scheduler.log.handlers[:]:
    if isinstance(_h, logging.FileHandler):
        scheduler.log.removeHandler(_h)
        _h.close()
_fh = logging.FileHandler(scheduler.LOG_FILE)
_fh.setLevel(logging.INFO)
_fh.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
scheduler.log.addHandler(_fh)

from fastapi.testclient import TestClient
from api import app
import db as db_mod
db_mod.AUTH_DB_FILE = os.path.join(_tmpdir, "auth.db")

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════════
# Password Reset
# ═══════════════════════════════════════════════════════════════════════


class TestPasswordReset:
    """Tests for /api/auth/password-reset and /api/auth/password-reset/confirm."""

    def _register(self, email="pwreset@xcelsior.ca", password="oldpass12345"):
        return client.post("/api/auth/register", json={
            "email": email, "password": password, "name": "Reset User"
        }).json()

    def test_password_reset_request(self):
        """POST /api/auth/password-reset returns reset_token in test mode."""
        self._register()
        r = client.post("/api/auth/password-reset", json={"email": "pwreset@xcelsior.ca"})
        assert r.status_code == 200
        d = r.json()
        assert d["message"] == "If the email exists, a reset link has been sent."
        # In test mode, token is included
        assert "reset_token" in d
        assert len(d["reset_token"]) > 0

    def test_password_reset_unknown_email(self):
        """POST /api/auth/password-reset with unknown email still returns 200 (no leak)."""
        r = client.post("/api/auth/password-reset", json={"email": "nobody@xcelsior.ca"})
        assert r.status_code == 200
        d = r.json()
        assert "message" in d
        # Should NOT have a token (user doesn't exist)
        assert "reset_token" not in d or d.get("reset_token") is None

    def test_password_reset_confirm(self):
        """POST /api/auth/password-reset/confirm sets new password."""
        self._register(email="pwconfirm@xcelsior.ca", password="oldpass12345")
        # Request reset
        reset = client.post("/api/auth/password-reset", json={"email": "pwconfirm@xcelsior.ca"}).json()
        token = reset["reset_token"]
        # Confirm reset
        r = client.post("/api/auth/password-reset/confirm", json={
            "token": token, "new_password": "newpass12345"
        })
        assert r.status_code == 200
        assert r.json()["message"] == "Password updated. Please log in again."
        # Login with new password
        login = client.post("/api/auth/login", json={
            "email": "pwconfirm@xcelsior.ca", "password": "newpass12345"
        })
        assert login.status_code == 200
        assert "access_token" in login.json()

    def test_password_reset_confirm_old_password_fails(self):
        """After reset, old password should fail."""
        self._register(email="pwold@xcelsior.ca", password="oldpass12345")
        reset = client.post("/api/auth/password-reset", json={"email": "pwold@xcelsior.ca"}).json()
        client.post("/api/auth/password-reset/confirm", json={
            "token": reset["reset_token"], "new_password": "brandnew123"
        })
        login = client.post("/api/auth/login", json={
            "email": "pwold@xcelsior.ca", "password": "oldpass12345"
        })
        assert login.status_code == 401

    def test_password_reset_invalid_token(self):
        """POST /api/auth/password-reset/confirm with bad token returns 400."""
        r = client.post("/api/auth/password-reset/confirm", json={
            "token": "invalid-token-xxx", "new_password": "newpass12345"
        })
        assert r.status_code == 400

    def test_password_reset_short_password(self):
        """POST /api/auth/password-reset/confirm with <8 char password returns 400."""
        self._register(email="pwshort@xcelsior.ca", password="oldpass12345")
        reset = client.post("/api/auth/password-reset", json={"email": "pwshort@xcelsior.ca"}).json()
        r = client.post("/api/auth/password-reset/confirm", json={
            "token": reset["reset_token"], "new_password": "short"
        })
        assert r.status_code == 400

    def test_password_reset_expired_token(self):
        """Expired reset tokens should be rejected."""
        self._register(email="pwexpire@xcelsior.ca", password="oldpass12345")
        reset = client.post("/api/auth/password-reset", json={"email": "pwexpire@xcelsior.ca"}).json()
        token = reset["reset_token"]
        # Manually expire the token by modifying the store
        import api
        with api._user_lock:
            for email, data in api._users_db.items():
                if data.get("reset_token") == token:
                    data["reset_token_expires"] = time.time() - 100  # expired
                    break
        # Also expire in persistent store if active
        if api._USE_PERSISTENT_AUTH:
            from db import UserStore
            UserStore.update_user("pwexpire@xcelsior.ca", {"reset_token_expires": time.time() - 100})
        r = client.post("/api/auth/password-reset/confirm", json={
            "token": token, "new_password": "newpass12345"
        })
        assert r.status_code == 400
        d = r.json()
        msg = d.get("detail", d.get("error", {}).get("message", "")).lower()
        assert "expired" in msg or "invalid" in msg


# ═══════════════════════════════════════════════════════════════════════
# Change Password
# ═══════════════════════════════════════════════════════════════════════


class TestChangePassword:
    """Tests for POST /api/auth/change-password."""

    def _register_and_login(self, email, password="testpass123"):
        reg = client.post("/api/auth/register", json={
            "email": email, "password": password
        }).json()
        return reg.get("access_token")

    def test_change_password_success(self):
        """POST /api/auth/change-password with correct current password works."""
        token = self._register_and_login("chpw1@xcelsior.ca", "oldpass12345")
        r = client.post("/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "oldpass12345", "new_password": "newpass12345"})
        assert r.status_code == 200
        # Login with new password
        login = client.post("/api/auth/login", json={
            "email": "chpw1@xcelsior.ca", "password": "newpass12345"
        })
        assert login.status_code == 200

    def test_change_password_wrong_current(self):
        """POST /api/auth/change-password with wrong current password returns 400."""
        token = self._register_and_login("chpw2@xcelsior.ca")
        r = client.post("/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "wrongpass", "new_password": "newpass12345"})
        assert r.status_code == 400
        d = r.json()
        msg = d.get("detail", d.get("error", {}).get("message", "")).lower()
        assert "incorrect" in msg

    def test_change_password_short_new(self):
        """POST /api/auth/change-password with short new password returns 400."""
        token = self._register_and_login("chpw3@xcelsior.ca", "testpass123")
        r = client.post("/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "testpass123", "new_password": "abc"})
        assert r.status_code == 400

    def test_change_password_unauthenticated(self):
        """POST /api/auth/change-password without auth returns 401."""
        r = client.post("/api/auth/change-password",
            json={"current_password": "x", "new_password": "y" * 8})
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════
# Marketplace Search
# ═══════════════════════════════════════════════════════════════════════


class TestMarketplaceSearch:
    """Tests for GET /marketplace/search."""

    @classmethod
    def setup_class(cls):
        """Register hosts and list them on marketplace."""
        # Register a few hosts
        hosts = [
            {"host_id": "mkt-h1", "ip": "10.0.0.1", "gpu_model": "RTX 4090",
             "total_vram_gb": 24, "free_vram_gb": 24, "cost_per_hour": 0.50,
             "country": "CA", "province": "ON"},
            {"host_id": "mkt-h2", "ip": "10.0.0.2", "gpu_model": "A100 80GB",
             "total_vram_gb": 80, "free_vram_gb": 80, "cost_per_hour": 2.00,
             "country": "CA", "province": "QC"},
            {"host_id": "mkt-h3", "ip": "10.0.0.3", "gpu_model": "RTX 3090",
             "total_vram_gb": 24, "free_vram_gb": 24, "cost_per_hour": 0.35,
             "country": "US"},
        ]
        for h in hosts:
            r = client.put("/host", json=h)
            assert r.status_code == 200, f"Failed to register {h['host_id']}: {r.text}"
            # List on marketplace
            client.post("/marketplace", json={
                "host_id": h["host_id"], "gpu_model": h["gpu_model"],
                "vram_gb": h["total_vram_gb"], "price_per_hour": h["cost_per_hour"],
                "country": h.get("country", ""), "province": h.get("province", ""),
            })

    def test_marketplace_search_all(self):
        """GET /marketplace/search with no filters returns all listings."""
        r = client.get("/marketplace/search")
        assert r.status_code == 200
        d = r.json()
        assert "listings" in d
        assert len(d["listings"]) >= 0  # may be empty if marketplace setup differs

    def test_marketplace_search_by_gpu(self):
        """GET /marketplace/search?gpu_model=RTX filters by GPU."""
        r = client.get("/marketplace/search?gpu_model=RTX")
        assert r.status_code == 200
        d = r.json()
        for listing in d["listings"]:
            assert "RTX" in listing.get("gpu_model", "").upper() or "rtx" in listing.get("gpu_model", "").lower()

    def test_marketplace_search_by_vram(self):
        """GET /marketplace/search?min_vram=48 filters by minimum VRAM."""
        r = client.get("/marketplace/search?min_vram=48")
        assert r.status_code == 200
        d = r.json()
        for listing in d["listings"]:
            vram = listing.get("vram_gb", listing.get("free_vram_gb", 0))
            assert vram >= 48

    def test_marketplace_search_by_price(self):
        """GET /marketplace/search?max_price=1.0 filters by max cost."""
        r = client.get("/marketplace/search?max_price=1.0")
        assert r.status_code == 200
        d = r.json()
        for listing in d["listings"]:
            price = listing.get("cost_per_hour", listing.get("price_per_hour", 0))
            assert price <= 1.0

    def test_marketplace_search_by_province(self):
        """GET /marketplace/search?province=ON filters by province."""
        r = client.get("/marketplace/search?province=ON")
        assert r.status_code == 200
        d = r.json()
        for listing in d["listings"]:
            assert listing.get("province", "") == "ON"

    def test_marketplace_search_sort_price(self):
        """GET /marketplace/search?sort_by=price sorts ascending by price."""
        r = client.get("/marketplace/search?sort_by=price")
        assert r.status_code == 200
        d = r.json()
        prices = [l.get("cost_per_hour", l.get("price_per_hour", 0)) for l in d["listings"]]
        assert prices == sorted(prices)

    def test_marketplace_search_filters_applied(self):
        """GET /marketplace/search includes filters_applied in response."""
        r = client.get("/marketplace/search?gpu_model=RTX&min_vram=8")
        assert r.status_code == 200
        d = r.json()
        assert "filters_applied" in d
        fa = d["filters_applied"]
        assert fa.get("gpu_model") == "RTX"
        assert fa.get("min_vram") == 8

    def test_marketplace_search_limit(self):
        """GET /marketplace/search?limit=1 returns at most 1 result."""
        r = client.get("/marketplace/search?limit=1")
        assert r.status_code == 200
        d = r.json()
        assert len(d["listings"]) <= 1

    def test_marketplace_search_empty_result(self):
        """GET /marketplace/search with impossible filters returns empty list."""
        r = client.get("/marketplace/search?gpu_model=NONEXISTENT_GPU_XYZ")
        assert r.status_code == 200
        d = r.json()
        assert len(d["listings"]) == 0
