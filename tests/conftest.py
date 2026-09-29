"""Shared pytest configuration for Xcelsior test suite.

Ensures the project root is on sys.path so test files can import
source modules (api, scheduler, billing, etc.) directly.

Loads .env.test so tests always use the test database and config.
"""

import os
import sys
import tempfile
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
# Always override local .env production flags before any test module imports api.
_TEST_ENV_FORCE = {
    "XCELSIOR_ENV": "test",
    "XCELSIOR_NFS_REQUIRED": "false",
    # Parity with CI: api's lifespan background threads (scheduler tick,
    # failover monitor, reconcilers) must not run inside the test process
    # — a live process_queue loop assigns leftover queued jobs onto test
    # fixture hosts mid-test, corrupting capacity assertions.
    "XCELSIOR_BG_TASKS": "false",
    # https BASE_URL from .env makes session cookies secure+domain-scoped; TestClient won't store them.
    "XCELSIOR_BASE_URL": "http://localhost:9501",
    "XCELSIOR_SCHEDULER_URL": "http://localhost:9501",
}
for _key, _val in _TEST_ENV_FORCE.items():
    os.environ[_key] = _val

_TEST_ENV_DEFAULTS = {
    "XCELSIOR_BASE_URL": "http://localhost:9501",
    "XCELSIOR_SCHEDULER_URL": "http://localhost:9501",
    "XCELSIOR_API_TOKEN": "test-token-not-for-production",
    "FEATURE_AI_ASSISTANT": "true",
    "GOOGLE_CLIENT_ID": "test-google-client-id",
    "GOOGLE_CLIENT_SECRET": "test-google-client-secret",
    "GITHUB_CLIENT_ID": "test-github-client-id",
    "GITHUB_CLIENT_SECRET": "test-github-client-secret",
    "HUGGINGFACE_CLIENT_ID": "test-hf-client-id",
    "HUGGINGFACE_CLIENT_SECRET": "test-hf-client-secret",
    "FACEBOOK_CLIENT_ID": "test-facebook-client-id",
    "FACEBOOK_CLIENT_SECRET": "test-facebook-client-secret",
    # Enables STRIPE_ENABLED; retrieve/detach map Stripe errors to 404 in tests.
    "XCELSIOR_STRIPE_SECRET_KEY": "sk_test_ci_placeholder_not_for_production",
    "XCELSIOR_MAX_TOTAL_STORAGE_GB": "100",
    "XCELSIOR_MAX_VOLUME_GB": "2000",
    "XCELSIOR_SERVERLESS_ENABLED": "true",
}
for _key, _val in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _val)

# Scheduler/serverless integration tests must never append generated host scores
# to a tracked repository fixture.
_TEST_STATE_DIR = tempfile.mkdtemp(prefix="xcelsior_pytest_state_")
os.environ["XCELSIOR_COMPUTE_SCORES_FILE"] = os.path.join(
    _TEST_STATE_DIR, "compute_scores.json"
)

# Tests must not depend on a local Redis service for OAuth/device auth cache.
os.environ["XCELSIOR_AUTH_CACHE_BACKEND"] = "memory"

# Empty string from CI env blocks setdefault — treat as unset for optional secrets.
if not (os.environ.get("XCELSIOR_STRIPE_SECRET_KEY") or "").strip():
    os.environ["XCELSIOR_STRIPE_SECRET_KEY"] = _TEST_ENV_DEFAULTS["XCELSIOR_STRIPE_SECRET_KEY"]

# B1 — agent auth bypass is now an explicit opt-in (see routes/agent.py).
os.environ.setdefault("XCELSIOR_ALLOW_UNAUTH_AGENT", "1")
# Avoid api lifespan background threads during TestClient runs (reduces CI deadlocks/timeouts).
os.environ.setdefault("XCELSIOR_BG_TASKS", "false")

# Exclude live E2E test scripts from pytest collection
collect_ignore = ["test_e2e_live.py"]

# Baseline DB backend intended for this run (env already loaded above). Legacy
# modules must not be able to flip it for everyone at import time.
_INTENDED_DB_BACKEND = os.environ.get("XCELSIOR_DB_BACKEND")


import pytest


def pytest_collection_finish(session):
    """Re-assert the canonical test env after ALL test modules have imported.

    Collection imports every test module in one process; legacy modules write
    to os.environ at import time (e.g. a stray ``XCELSIOR_ENV = "dev"``).
    The per-test ``_pin_test_auth_env`` monkeypatch undoes itself on teardown,
    so *module/session-scoped fixtures* — which run between tests, outside the
    per-test pins — would otherwise execute under whichever module's
    import-time env won collection. That poisoned logins minted by
    module-scoped fixtures and broke ~150 unrelated tests in full-suite runs.
    """
    for _key, _val in _TEST_ENV_FORCE.items():
        os.environ[_key] = _val
    os.environ["XCELSIOR_AUTH_CACHE_BACKEND"] = "memory"
    if _INTENDED_DB_BACKEND:
        os.environ["XCELSIOR_DB_BACKEND"] = _INTENDED_DB_BACKEND


#: Every runtime namespace that holds a rolling rate-limit or lease window.
#: These live in the `state` table, so unlike the per-process dicts a test
#: fixture clears they are *durable* — they outlive the pytest process and leak
#: into the next run.
_SHARED_RUNTIME_NAMESPACES = (
    "runtime.auth_rate_limit",
    "runtime.billing_payment_rate_limit",
    "runtime.chat_rate_limit",
    "runtime.ai_rate_limit",
    "runtime.snapshot_rate_limit",
    "runtime.ws_connect_rate_limit",
    "runtime.ws_tickets",
    "runtime.terminal_session_slots",
)


@pytest.fixture(autouse=True)
def _reset_shared_runtime_limits():
    """Clear shared rate-limit state around every test.

    The limiters count in shared state and keep their per-process dict only as
    a fallback, so a fixture that clears the dict resets the half that was not
    being used. Ten-odd test files do exactly that, and they were written when
    the dict *was* the whole story.

    The leak is worse than it looks because the state is durable. Windows that
    prune on read hid it: the WS connect and terminal namespaces roll over in
    60 and 120 seconds, so a suite that takes longer than that never noticed.
    The snapshot limiter's window is an hour, and it failed on the first call
    of the following run — which is how this was found.

    Runs before *and* after, so a suite left dirty by an earlier run still
    starts clean.
    """
    _clear_shared_runtime_limits()
    yield
    _clear_shared_runtime_limits()


def _clear_shared_runtime_limits() -> None:
    """Blank every namespace in one transaction.

    Resetting each namespace separately would require two transactions per
    namespace per test. Batch the resets to avoid that overhead.
    """
    from db import DatabaseOps, get_engine

    empty = {"buckets": {}, "sessions": {}, "updated_at": 0.0}
    try:
        engine = get_engine()
        with engine.transaction() as (conn, backend):
            for namespace in _SHARED_RUNTIME_NAMESPACES:
                DatabaseOps.upsert_state(conn, namespace, empty, backend=backend)
    except Exception:
        # A test that never touches shared state should not fail because the
        # database is not up for it. The limiters degrade the same way.
        pass


@pytest.fixture(autouse=True)
def _pin_test_auth_env(monkeypatch):
    """Keep auth flags consistent when tests temporarily rewrite os.environ."""
    import routes._deps as deps
    import routes.auth as auth_mod
    import oauth_service as oauth_mod

    monkeypatch.setenv("XCELSIOR_ENV", "test")
    monkeypatch.setenv("XCELSIOR_NFS_REQUIRED", "false")
    monkeypatch.setenv("XCELSIOR_AUTH_CACHE_BACKEND", "memory")
    monkeypatch.setenv("XCELSIOR_BG_TASKS", "false")
    monkeypatch.setattr(deps, "XCELSIOR_ENV", "test")
    monkeypatch.setattr(auth_mod, "XCELSIOR_ENV", "test")
    monkeypatch.setattr(oauth_mod, "AUTH_CACHE_BACKEND", "memory")
    monkeypatch.setattr(deps, "AUTH_REQUIRED", False)

    # The auth rate limiter is 10 requests per 5 minutes **per client IP**, and
    # every test shares one TestClient IP. A full run performs far more than ten
    # registers and logins in five minutes, so past that point `/auth/register`
    # and `/auth/login` answer 429 and every fixture doing `reg["user"]` or
    # `login.json()["access_token"]` raises KeyError — nowhere near the code
    # under test, and only in a *full* run, never in isolation.
    #
    # `tests/test_api.py` already set `XCELSIOR_AUTH_RATE_LIMIT_REQUESTS=5000`
    # at import for this reason, and it could not be relied on:
    # `routes/_deps.py` reads that variable **at import time**, so the override
    # worked only when test_api.py happened to be imported before `routes._deps`
    # — a property of collection order, not of configuration. That is why the
    # failures moved around: `test_unlist_rig` answering 404, a billing accrual
    # off by a round number, `KeyError: 'user'`, `KeyError: 'access_token'`.
    #
    # Patched on the module attribute instead, which is what the limiter
    # actually reads at call time, so it holds regardless of import order. The
    # limiter itself is untouched in production; a test that wants to exercise
    # it can monkeypatch this back down, as `auth_enforced` does for
    # AUTH_REQUIRED in test_compliance_surfaces_require_auth.py.
    monkeypatch.setattr(deps, "_AUTH_RATE_LIMIT_REQUESTS", 100_000)

    # test_bitcoin.py sets sqlite at import; CI must stay on migrated Postgres.
    if os.environ.get("CI"):
        monkeypatch.setenv("XCELSIOR_DB_BACKEND", "postgres")


@pytest.fixture(scope="module")
def persistent_auth_module():
    """Pin ``_USE_PERSISTENT_AUTH=True`` for module-scoped register/login fixtures.

    Several modules pin persistent auth per-test with a function-scoped autouse
    monkeypatch, but their *module-scoped* user fixtures (register + login +
    fund) run outside that pin. They used to work only because one module
    leaked a raw un-restored ``_USE_PERSISTENT_AUTH = True`` into the rest of
    the run. Depend on this fixture from any module-scoped fixture that needs
    users written to the persistent (PostgreSQL) store; it undoes itself at
    module teardown.
    """
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth_mod

    mp = pytest.MonkeyPatch()
    mp.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    mp.setattr(auth_mod, "_USE_PERSISTENT_AUTH", True)
    mp.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)
    yield
    mp.undo()


@pytest.fixture
def fake_vllm_port(monkeypatch):
    """Official test upstream seam — sets XCELSIOR_TEST_FAKE_VLLM_PORT for proxy routes."""
    from tests.fixtures.fake_vllm_upstream import start_fake_vllm

    server, port, thread = start_fake_vllm()
    monkeypatch.setenv("XCELSIOR_TEST_FAKE_VLLM_PORT", str(port))
    yield port
    monkeypatch.delenv("XCELSIOR_TEST_FAKE_VLLM_PORT", raising=False)
    server.shutdown()
    thread.join(timeout=2)


@pytest.fixture
def mac_reachable_api(fake_vllm_port):
    """Expose FastAPI on Tailscale/LAN so Mac SSH can POST real inference requests."""
    import socket
    import threading
    import time
    import urllib.error
    import urllib.request

    import uvicorn

    from api import app

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    port = sock.getsockname()[1]
    sock.close()
    api_host = os.environ.get("XCELSIOR_MAC_INFERENCE_API_HOST", "100.64.0.6")
    os.environ["XCELSIOR_BG_TASKS"] = "false"
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(80):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=0.5)
            break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    yield f"http://{api_host}:{port}"
    server.should_exit = True
    thread.join(timeout=8)


@pytest.fixture(autouse=True)
def _clear_module_test_client_cookies():
    """Prevent session cookies from one test bleeding into the next (shared TestClient)."""
    import sys

    yield
    for name, mod in list(sys.modules.items()):
        if not (name == "__main__" or name.startswith("tests") or name.startswith("xcelsior")):
            continue
        try:
            client = getattr(mod, "client", None)
            if client is not None and hasattr(client, "cookies"):
                client.cookies.clear()
        except Exception:
            pass
