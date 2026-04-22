"""P3/B1 — agent auth bypass rules.

Rules (first match wins, see routes/agent.py::_require_agent_auth):
  1. XCELSIOR_ENV=production → NEVER bypass (hard-fail escape hatches).
  2. XCELSIOR_ENV=test → bypass (test pattern, unchanged legacy behavior).
  3. XCELSIOR_ALLOW_UNAUTH_AGENT=1 (non-prod only) → bypass with warning.
  4. Otherwise → 401.
"""

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/agent/commands/host-1",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def _env(env: str | None = None, allow: str | None = None):
    new = {k: v for k, v in os.environ.items()
           if k not in ("XCELSIOR_ENV", "XCELSIOR_ALLOW_UNAUTH_AGENT")}
    if env is not None:
        new["XCELSIOR_ENV"] = env
    if allow is not None:
        new["XCELSIOR_ALLOW_UNAUTH_AGENT"] = allow
    return new


class TestAgentAuthBypass:
    def test_production_never_bypasses_even_with_allow_flag(self):
        """Rule 1: production hard-fails unauth even when ALLOW_UNAUTH=1."""
        from routes.agent import _require_agent_auth

        with patch.dict(os.environ, _env(env="production", allow="1"), clear=True), \
             patch("routes.agent._get_current_user", return_value=None):
            with pytest.raises(HTTPException) as ei:
                _require_agent_auth(_fake_request())
            assert ei.value.status_code == 401

    def test_production_authed_user_passes(self):
        from routes.agent import _require_agent_auth

        fake_user = {"user_id": "u1", "email": "a@b.com", "is_admin": True}
        with patch.dict(os.environ, _env(env="production"), clear=True), \
             patch("routes.agent._get_current_user", return_value=fake_user):
            assert _require_agent_auth(_fake_request()) == fake_user

    def test_env_test_bypasses(self):
        """Rule 2: XCELSIOR_ENV=test alone is a bypass (test pattern)."""
        from routes.agent import _require_agent_auth

        with patch.dict(os.environ, _env(env="test"), clear=True), \
             patch("routes.agent._get_current_user", return_value=None):
            result = _require_agent_auth(_fake_request())
            assert result.get("unauth") is True
            assert result.get("test") is True

    def test_dev_with_allow_flag_bypasses_with_warning(self):
        """Rule 3: non-prod + ALLOW_UNAUTH_AGENT=1 → bypass."""
        from routes.agent import _require_agent_auth

        with patch.dict(os.environ, _env(env="dev", allow="1"), clear=True), \
             patch("routes.agent._get_current_user", return_value=None):
            result = _require_agent_auth(_fake_request())
            assert result.get("unauth") is True

    def test_dev_no_flags_requires_auth(self):
        """Rule 4: dev without any bypass flag → 401."""
        from routes.agent import _require_agent_auth

        with patch.dict(os.environ, _env(env="dev"), clear=True), \
             patch("routes.agent._get_current_user", return_value=None):
            with pytest.raises(HTTPException) as ei:
                _require_agent_auth(_fake_request())
            assert ei.value.status_code == 401

    def test_production_ignores_allow_flag_but_staging_respects_it(self):
        """Audit intent: a production VPS MUST not silently allow unauth
        even if XCELSIOR_ALLOW_UNAUTH_AGENT is accidentally set. Staging
        and other non-test non-prod envs honor the escape hatch."""
        from routes.agent import _require_agent_auth

        with patch.dict(os.environ, _env(env="production", allow="true"), clear=True), \
             patch("routes.agent._get_current_user", return_value=None):
            with pytest.raises(HTTPException):
                _require_agent_auth(_fake_request())

        with patch.dict(os.environ, _env(env="staging", allow="1"), clear=True), \
             patch("routes.agent._get_current_user", return_value=None):
            result = _require_agent_auth(_fake_request())
            assert result.get("unauth") is True
