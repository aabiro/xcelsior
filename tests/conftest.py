"""Shared pytest configuration for Xcelsior test suite.

Ensures the project root is on sys.path so test files can import
source modules (api, scheduler, billing, etc.) directly.

Loads .env.test so tests always use the test database and config.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so `import scheduler`, `from api import app`, etc. work
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load test environment BEFORE any module imports touch os.environ
from dotenv import load_dotenv

_env_test = os.path.join(PROJECT_ROOT, ".env.test")
# Never override env vars already set (GitHub Actions sets sqlite backend, etc.).
if os.path.exists(_env_test):
    load_dotenv(_env_test, override=False)
else:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

# CI job env must win over .env.test (workflow sets postgres + limits).
if os.environ.get("CI"):
    os.environ["XCELSIOR_DB_BACKEND"] = "postgres"
    os.environ["XCELSIOR_BG_TASKS"] = "false"

# .env.test is gitignored — GitHub Actions has no local secrets file. These defaults
# must be set before any test module imports api (e.g. test_auto_launch), otherwise
# cookies use secure+production domain (BASE_URL defaults to https://xcelsior.ca) and
# TestClient never sends session cookies; Stripe/OAuth/feature flags stay off.
_TEST_ENV_DEFAULTS = {
    "XCELSIOR_BASE_URL": "http://localhost:9501",
    "XCELSIOR_API_TOKEN": "test-token-not-for-production",
    "FEATURE_AI_ASSISTANT": "true",
    "GOOGLE_CLIENT_ID": "test-google-client-id",
    "GOOGLE_CLIENT_SECRET": "test-google-client-secret",
    "GITHUB_CLIENT_ID": "test-github-client-id",
    "GITHUB_CLIENT_SECRET": "test-github-client-secret",
    "HUGGINGFACE_CLIENT_ID": "test-hf-client-id",
    "HUGGINGFACE_CLIENT_SECRET": "test-hf-client-secret",
    # Enables STRIPE_ENABLED; retrieve/detach map Stripe errors to 404 in tests.
    "XCELSIOR_STRIPE_SECRET_KEY": "sk_test_ci_placeholder_not_for_production",
    "XCELSIOR_MAX_TOTAL_STORAGE_GB": "100",
    "XCELSIOR_MAX_VOLUME_GB": "2000",
}
for _key, _val in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _val)

# Empty string from CI env blocks setdefault — treat as unset for optional secrets.
if not (os.environ.get("XCELSIOR_STRIPE_SECRET_KEY") or "").strip():
    os.environ["XCELSIOR_STRIPE_SECRET_KEY"] = _TEST_ENV_DEFAULTS["XCELSIOR_STRIPE_SECRET_KEY"]

# B1 — agent auth bypass is now an explicit opt-in (see routes/agent.py).
# Tests that hit /agent/* endpoints without a bearer token need this flag
# set regardless of which .env was loaded. Also pin XCELSIOR_ENV=test so
# helpers that branch on it (e.g. logging) still behave.
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_ALLOW_UNAUTH_AGENT", "1")
# Avoid api lifespan background threads during TestClient runs (reduces CI deadlocks/timeouts).
os.environ.setdefault("XCELSIOR_BG_TASKS", "false")

# Exclude live E2E test scripts from pytest collection
collect_ignore = ["test_e2e_live.py"]


import pytest


@pytest.fixture(autouse=True)
def _pin_test_auth_env(monkeypatch):
    """Keep auth flags consistent when tests temporarily rewrite os.environ."""
    import routes._deps as deps

    monkeypatch.setenv("XCELSIOR_ENV", "test")
    monkeypatch.setattr(deps, "XCELSIOR_ENV", "test")
    monkeypatch.setattr(deps, "AUTH_REQUIRED", False)
    # test_bitcoin.py sets sqlite at import; CI must stay on migrated Postgres.
    if os.environ.get("CI"):
        monkeypatch.setenv("XCELSIOR_DB_BACKEND", "postgres")


@pytest.fixture(autouse=True)
def _clear_module_test_client_cookies():
    """Prevent session cookies from one test bleeding into the next (shared TestClient)."""
    import sys

    yield
    for mod in sys.modules.values():
        client = getattr(mod, "client", None)
        if client is not None and hasattr(client, "cookies"):
            try:
                client.cookies.clear()
            except Exception:
                pass
