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
    # Measurement switch: XCELSIOR_TEST_ENFORCE_AUTH=1 pins the control on so
    # the true blast radius of inverting this default can be counted rather
    # than guessed. Default stays False until the ratchet lands.
    monkeypatch.setattr(
        deps, "AUTH_REQUIRED", os.environ.get("XCELSIOR_TEST_ENFORCE_AUTH") == "1"
    )
    # test_bitcoin.py sets sqlite at import; CI must stay on migrated Postgres.
    if os.environ.get("CI"):
        monkeypatch.setenv("XCELSIOR_DB_BACKEND", "postgres")


@pytest.fixture
def auth_enforced(_pin_test_auth_env):
    """Run a test with authentication actually enforced.

    `_pin_test_auth_env` is autouse and sets `AUTH_REQUIRED = False` for every
    test in the suite, so `_require_auth` hands an anonymous caller a synthetic
    principal with `is_admin: True`. Any authorization test that does not undo
    that is measuring the fixture, not the endpoint — it cannot fail.

    Declaring `_pin_test_auth_env` as a dependency is load-bearing: it forces
    this fixture to run *after* the autouse one. A plugin-level fixture that
    merely set the flag was silently reverted, and a run reported as "verified
    under enforced auth" had in fact executed with auth off.

    Use this on every test whose subject is authentication or authorization.
    """
    import routes._deps as deps

    original = deps.AUTH_REQUIRED
    deps.AUTH_REQUIRED = True
    try:
        yield
    finally:
        deps.AUTH_REQUIRED = original


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


# ── Precondition enforcement ─────────────────────────────────────────────
#
# Three consecutive measurements in this codebase passed for reasons unrelated
# to what they claimed to check: a fixture silently reverted by an autouse one,
# a log truncated below the evidence, and a parametrized function that accepted
# a value and evaluated the ambient environment instead.
#
# The one thing that worked was a test asserting its own precondition in its own
# body, after every fixture had run. This generalises that, and the polarity is
# the point: **enforcement is the default**, and relaxation is an explicit,
# countable opt-out.
#
# An opt-in marker cannot ratchet — there is no denominator, so a new test can
# silently join the unenforced majority and nothing notices. With the default
# inverted, the opt-out list *is* the measurement, and it only goes down.
#
# A hook rather than a fixture, because a fixture can be reordered by an autouse
# one; `pytest_runtest_call` runs immediately before the test body and nothing
# reorders it.

#: The relaxations the suite may grant, each with the probe that says whether it
#: is currently in force. Keyed by the name used in `relaxed_auth(...)`.
_RELAXATIONS = {
    # Anonymous callers receive a synthetic principal with is_admin=True.
    "auth_required": lambda: __import__(
        "routes._deps", fromlist=["AUTH_REQUIRED"]
    ).AUTH_REQUIRED,
    # The committed dev JWT secret and symmetric signing are reachable.
    "asymmetric_signing": lambda: not __import__("env_config").is_relaxed_env(),
    # The deterministic Fernet key is reachable.
    "secrets_key": lambda: bool(os.environ.get("XCELSIOR_SECRETS_KEY", "").strip()),
    # The production configuration gate degrades to log lines.
    "startup_gate": lambda: __import__(
        "control_plane.startup_validation", fromlist=["enforcement_enabled"]
    ).enforcement_enabled(),
}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "relaxed_auth(*controls, reason=...): this test needs one or more "
        "security relaxations. Every use is counted by the ratchet in "
        "tests/test_enforcement_ratchet.py and may only decrease.",
    )
    config.addinivalue_line(
        "markers",
        "enforced_auth: legacy alias; enforcement is now the default and this "
        "marker is a no-op kept so existing tests still read as intentional.",
    )


#: Controls the whole suite currently runs with relaxed. **This is the ratchet.**
#:
#: Per-test markers were the wrong granularity: these relaxations come from
#: `_pin_test_auth_env` and `XCELSIOR_ENV=test`, so every one of ~4,700 tests
#: experiences all of them. Annotating each would produce 4,700 identical
#: markers carrying no information and no way to tell progress from noise.
#:
#: Counted and pinned by `tests/test_enforcement_ratchet.py`, which allows this
#: set to shrink and never to grow. Removing an entry is the unit of progress:
#: it means the suite now runs with that control genuinely on.
#:
#: `secrets_key` is deliberately absent — `.env.test` provides a real Fernet
#: key, so that control is already enforcing.
SUITE_RELAXATIONS = frozenset({
    "auth_required",       # _pin_test_auth_env sets AUTH_REQUIRED = False
    "asymmetric_signing",  # XCELSIOR_ENV=test permits the symmetric dev secret
    "startup_gate",        # the production configuration gate only warns
})


def pytest_runtest_call(item):
    """Refuse a test that experiences a relaxation nobody declared.

    The suite-wide baseline is `SUITE_RELAXATIONS`. A test may declare more with
    `@pytest.mark.relaxed_auth(...)`, but anything outside both is a control
    that quietly went off — which is how the anonymous-admin principal survived
    unnoticed across the whole suite.
    """
    marker = item.get_closest_marker("relaxed_auth")
    allowed = set(marker.args) if marker else set()
    allowed |= SUITE_RELAXATIONS
    unknown = allowed - set(_RELAXATIONS)
    if unknown:
        raise AssertionError(
            f"{item.nodeid} declares unknown relaxation(s) {sorted(unknown)}; "
            f"valid names: {sorted(_RELAXATIONS)}"
        )
    violations = [
        name
        for name, probe in _RELAXATIONS.items()
        if name not in allowed and not probe()
    ]
    if violations:
        raise AssertionError(
            f"{item.nodeid} runs with these controls relaxed but does not "
            f"declare them: {violations}. Either enforce them, or mark the "
            f"test `@pytest.mark.relaxed_auth({', '.join(repr(v) for v in violations)}, "
            'reason="...")` — which adds to a ratchet that may only decrease.'
        )
