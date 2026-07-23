"""Redis/Valkey key contract (Track B B9.1a, companion §5.4).

Before this landed, `oauth_service._cache_key` interpolated the credential
itself into the key name: `xcelsior:oauth:access_token:<live access token>`,
and likewise for authorization codes and device codes. A Redis key name is
not private — `SCAN`, `MONITOR`, the slowlog, RDB/AOF files on disk,
managed-service support tooling, and any keyspace exporter see it. The
serverless limiter had the same shape with `dashboard-test:{owner_id}`,
where owner ids are frequently email addresses.

`hash_token()` already existed in `oauth_service` and was already applied
to refresh tokens, so the correct treatment was present and applied
inconsistently.

Companion §5.4; Track B §B9.1a.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import cache_keys
import oauth_service
from cache_keys import cache_key, environment, opaque

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────── key shape ───────────────────────────


def test_key_carries_namespace_environment_and_version(monkeypatch):
    monkeypatch.setenv("XCELSIOR_ENV", "prod")
    key = cache_key("oauth", "access_token", secret="tok")
    assert key.startswith("xc:prod:v1:oauth:access_token:")


def test_environment_defaults_and_normalizes(monkeypatch):
    monkeypatch.delenv("XCELSIOR_ENV", raising=False)
    assert environment() == "dev"
    monkeypatch.setenv("XCELSIOR_ENV", "  PROD  ")
    assert environment() == "prod"


def test_two_environments_do_not_collide(monkeypatch):
    monkeypatch.setenv("XCELSIOR_ENV", "prod")
    prod = cache_key("oauth", "access_token", secret="same-token")
    monkeypatch.setenv("XCELSIOR_ENV", "test")
    test = cache_key("oauth", "access_token", secret="same-token")
    assert prod != test, (
        "two deployments sharing one Redis instance must not read each "
        "other's auth state"
    )


def test_key_without_a_secret_is_allowed():
    assert cache_key("presence", "fleet") == f"xc:{environment()}:v1:presence:fleet"


# ─────────────────────── secrets never appear ───────────────────────


@pytest.mark.parametrize(
    "secret",
    [
        "xoa_live_access_token_material",
        "authcode-abc123",
        "user@example.com",
        "https://bucket.example/artifact.bin?sig=deadbeef",
        "a prompt containing customer data",
    ],
)
def test_secret_never_appears_in_the_key(secret):
    key = cache_key("oauth", "access_token", secret=secret)
    assert secret not in key
    assert opaque(secret) in key


def test_hashing_is_deterministic_and_distinct():
    assert opaque("tok-a") == opaque("tok-a")
    assert opaque("tok-a") != opaque("tok-b")
    assert re.fullmatch(r"[0-9a-f]{64}", opaque("tok-a"))


@pytest.mark.parametrize(
    "bad_segment",
    [
        "user@example.com",   # an email smuggled into a public position
        "has:colon",          # would forge a segment boundary
        "has space",
        "",
        "x" * 65,
    ],
)
def test_public_segments_reject_freeform_data(bad_segment):
    """A public segment is a fixed token, not a place to put data.

    This is the guard that stops a future call site from doing
    `cache_key("ratelimit", owner_email)` and reintroducing the leak.
    """
    with pytest.raises(ValueError):
        cache_key("oauth", bad_segment)


def test_invalid_family_rejected():
    with pytest.raises(ValueError):
        cache_key("bad family")


# ───────────────── oauth_service uses the contract ─────────────────


@pytest.mark.parametrize(
    "kind,identifier",
    [
        ("access_token", "xoa_live_token_value"),
        ("authorization_code", "code-live-value"),
        ("device_code", "device-live-value"),
        ("device_user_code", "WDJB-MJHT"),
        ("social_oauth", "state-value"),
    ],
)
def test_auth_cache_key_hashes_the_credential(kind, identifier):
    """The regression: live credentials in Redis key names."""
    key = oauth_service._cache_key(kind, identifier)
    assert identifier not in key, (
        f"{kind} key still contains the raw credential: {key!r}. A key name "
        f"is visible to SCAN, MONITOR, the slowlog, and RDB/AOF on disk."
    )
    assert key.startswith(f"xc:{environment()}:v1:oauth:{kind}:")
    assert opaque(identifier) in key


def test_auth_cache_key_round_trips_through_the_cache(monkeypatch):
    """Hashing must be transparent to reads — deterministic, not random."""
    monkeypatch.setattr(oauth_service, "AUTH_CACHE_BACKEND", "memory")
    monkeypatch.setattr(oauth_service, "_cache_instance", None)

    assert oauth_service._cache_set_json("access_token", "tok-1", {"sub": "u1"}, 60)
    assert oauth_service._cache_get_json("access_token", "tok-1") == {"sub": "u1"}
    assert oauth_service._cache_get_json("access_token", "tok-2") is None


def test_retired_prefix_env_var_is_gone():
    """A per-deployment prefix override would defeat the contract."""
    assert not hasattr(oauth_service, "AUTH_CACHE_PREFIX"), (
        "XCELSIOR_AUTH_CACHE_PREFIX is retired; the namespace is owned by "
        "cache_keys.cache_key so env/version/hashing apply uniformly"
    )


# ───────────────── TTL is set with the key ─────────────────


class _RecordingPipeline:
    def __init__(self, sink):
        self.sink = sink

    def incr(self, key):
        self.sink.append(("incr", key))
        return self

    def expire(self, key, ttl, nx=False):
        self.sink.append(("expire", key, ttl, nx))
        return self

    def execute(self):
        return [1, True]


class _RecordingRedis:
    def __init__(self):
        self.calls: list[tuple] = []

    def pipeline(self, transaction=False):
        self.calls.append(("pipeline", transaction))
        return _RecordingPipeline(self.calls)


def test_incr_sets_ttl_in_one_transaction():
    """Companion §5.4: TTL is set in the same atomic operation.

    The previous implementation issued INCR and then a conditional
    EXPIRE. A crash between them left an immortal key — and for an abuse
    counter that means a principal stays locked out until someone deletes
    it by hand.
    """
    cache = object.__new__(oauth_service.RedisAuthCache)
    fake = _RecordingRedis()
    cache._client = fake  # type: ignore[attr-defined]

    assert cache.incr("k", 30) == 1

    assert ("pipeline", True) in fake.calls, "the pair must run in one MULTI"
    ops = [c[0] for c in fake.calls if c[0] in ("incr", "expire")]
    assert ops == ["incr", "expire"]
    expire_call = next(c for c in fake.calls if c[0] == "expire")
    assert expire_call[3] is True, (
        "EXPIRE must use NX so later increments do not extend a window"
    )


# ───────────────── no direct key construction remains ─────────────────

_KEY_LITERAL_RE = re.compile(
    r"""["']((?:xcelsior|serverless|xc)[:][^"']*)["']""", re.IGNORECASE
)

# Files allowed to mention a raw key-shaped literal (the contract module
# defines the namespace; tests assert on it).
_ALLOWED = {"cache_keys.py"}


def _redis_using_modules() -> list[Path]:
    found = []
    for path in PROJECT_ROOT.glob("**/*.py"):
        parts = set(path.parts)
        if parts & {".venv", "tests", "migrations", "node_modules", "build"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "redis" in text.lower():
            found.append(path)
    return found


def test_no_module_builds_a_redis_key_by_hand():
    """Every ephemeral key goes through the one contract helper.

    Without this, the next feature adds `f"xcelsior:thing:{token}"` and
    the leak returns in a place nobody re-reviews.
    """
    offenders: list[str] = []
    for path in _redis_using_modules():
        if path.name in _ALLOWED:
            continue
        for match in _KEY_LITERAL_RE.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {match.group(1)!r}")
    assert not offenders, (
        "raw Redis key literals found; build keys with "
        "cache_keys.cache_key(family, *public, secret=...) so the "
        "environment, version, and secret hashing are applied "
        f"(companion §5.4):\n  " + "\n  ".join(offenders)
    )


def test_rate_limit_store_hashes_its_bucket_key():
    """`dashboard-test:{owner_id}` composites often contain an email."""
    source = (PROJECT_ROOT / "serverless" / "rate_limit_store.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "check_key_rate_limit_redis"
    )
    calls = [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "cache_key"
    ]
    assert calls, (
        "check_key_rate_limit_redis must build its bucket key through "
        "cache_keys.cache_key"
    )
    assert any(
        kw.arg == "secret" for call in calls for kw in call.keywords
    ), "the rate-limit identifier must be passed as secret= so it is hashed"


def test_contract_module_documents_the_format():
    text = (PROJECT_ROOT / "cache_keys.py").read_text(encoding="utf-8")
    assert "companion §5.4" in text
    assert cache_keys.NAMESPACE == "xc"
    assert cache_keys.KEY_VERSION == "v1"
