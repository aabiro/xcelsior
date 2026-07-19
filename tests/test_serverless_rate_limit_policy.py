"""Companion §2.3/§5.5 — serverless rate-limit degradation policy.

The Redis limiter is the only *global* rate limiter; process-local deque
buckets grant one quota per replica. In production, the global limiter
being unavailable must NOT silently degrade to per-process quotas — the
behavior is an explicit policy, and an undefined production policy is
rejected (fail closed).
"""

from unittest import mock

import pytest

from serverless import limits
from serverless.limits import (
    RateLimitExceeded,
    RateLimiterUnavailable,
    RateLimitPolicy,
    RateLimitPolicyError,
    check_key_rate_limit,
    rate_limit_policy,
    validate_rate_limit_policy,
)


@pytest.fixture(autouse=True)
def _clear_buckets():
    limits._RATE_BUCKETS.clear()
    yield
    limits._RATE_BUCKETS.clear()


class TestPolicyResolution:
    def test_valid_values_case_insensitive(self, monkeypatch):
        for val, expected in [
            ("strict-deny", RateLimitPolicy.STRICT_DENY),
            ("upstream-enforced", RateLimitPolicy.UPSTREAM_ENFORCED),
            ("disabled-for-development", RateLimitPolicy.DISABLED_FOR_DEVELOPMENT),
            ("STRICT-DENY", RateLimitPolicy.STRICT_DENY),
        ]:
            monkeypatch.setenv("XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", val)
            assert rate_limit_policy() is expected

    def test_production_undefined_raises(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_ENV", "production")
        monkeypatch.delenv("XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", raising=False)
        with pytest.raises(RateLimitPolicyError):
            rate_limit_policy()
        with pytest.raises(RateLimitPolicyError):
            validate_rate_limit_policy()

    def test_production_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_ENV", "production")
        monkeypatch.setenv("XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", "guess")
        with pytest.raises(RateLimitPolicyError):
            rate_limit_policy()

    def test_nonproduction_defaults_to_dev(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_ENV", "test")
        monkeypatch.delenv("XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", raising=False)
        assert rate_limit_policy() is RateLimitPolicy.DISABLED_FOR_DEVELOPMENT
        validate_rate_limit_policy()  # does not raise outside production


class TestCheckBehaviorWhenGlobalLimiterDown:
    # In the test env the Redis limiter is not in effect (returns None),
    # so these exercise the degradation-policy branch directly.

    def test_dev_uses_local_deque(self, monkeypatch):
        monkeypatch.setenv(
            "XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", "disabled-for-development"
        )
        check_key_rate_limit("k-dev", rpm=2)
        check_key_rate_limit("k-dev", rpm=2)
        with pytest.raises(RateLimitExceeded):
            check_key_rate_limit("k-dev", rpm=2)  # third over the local limit

    def test_strict_deny_fails_closed(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", "strict-deny")
        # Even far under the nominal limit, a missing global limiter denies.
        with pytest.raises(RateLimiterUnavailable):
            check_key_rate_limit("k-sd", rpm=1000)
        # It is a RateLimitExceeded, so existing 429 handlers still work.
        try:
            check_key_rate_limit("k-sd", rpm=1000)
        except RateLimitExceeded as exc:
            assert exc.info.remaining == 0
        else:  # pragma: no cover
            pytest.fail("expected fail-closed denial")

    def test_upstream_enforced_passes_through(self, monkeypatch):
        monkeypatch.setenv(
            "XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", "upstream-enforced"
        )
        # No per-process quota is fabricated: many calls, never denied.
        for _ in range(50):
            info = check_key_rate_limit("k-up", rpm=1)
            assert info.limit == 1

    def test_production_undefined_policy_denies_at_check(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_ENV", "production")
        monkeypatch.delenv("XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", raising=False)
        # No silent per-process fallback: the check raises rather than admit.
        with pytest.raises(RateLimitPolicyError):
            check_key_rate_limit("k-prod", rpm=60)


class TestRedisTimeoutBound:
    """Companion §5.5: a Redis outage must fail fast (bounded socket
    timeout) so it never holds the request open or triggers retry storms."""

    @pytest.fixture(autouse=True)
    def _reset_redis_cache(self):
        import serverless.rate_limit_store as rls
        rls._REDIS_TRIED = False
        rls._REDIS_CLIENT = None
        yield
        rls._REDIS_TRIED = False
        rls._REDIS_CLIENT = None

    def test_timeout_default_floor_and_bad_value(self, monkeypatch):
        import serverless.rate_limit_store as rls
        monkeypatch.delenv("XCELSIOR_SERVERLESS_REDIS_TIMEOUT_SEC", raising=False)
        assert rls._redis_timeout_sec() == 2.0
        monkeypatch.setenv("XCELSIOR_SERVERLESS_REDIS_TIMEOUT_SEC", "0.01")
        assert rls._redis_timeout_sec() == 0.1  # floored
        monkeypatch.setenv("XCELSIOR_SERVERLESS_REDIS_TIMEOUT_SEC", "5")
        assert rls._redis_timeout_sec() == 5.0
        monkeypatch.setenv("XCELSIOR_SERVERLESS_REDIS_TIMEOUT_SEC", "garbage")
        assert rls._redis_timeout_sec() == 2.0  # fail-safe default

    def test_get_redis_passes_bounded_timeouts(self, monkeypatch):
        import serverless.rate_limit_store as rls
        monkeypatch.setenv("XCELSIOR_SERVERLESS_REDIS_RATE_LIMITS", "true")
        monkeypatch.setenv("XCELSIOR_SERVERLESS_REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("XCELSIOR_SERVERLESS_REDIS_TIMEOUT_SEC", "1.5")

        captured: dict = {}
        fake_client = mock.Mock()

        def fake_from_url(url, **kw):
            captured.update(kw)
            return fake_client

        fake_redis = mock.Mock()
        fake_redis.from_url = fake_from_url
        with mock.patch.dict("sys.modules", {"redis": fake_redis}):
            client = rls._get_redis()

        assert client is fake_client
        assert captured["socket_timeout"] == 1.5
        assert captured["socket_connect_timeout"] == 1.5
        assert captured["retry_on_timeout"] is False
        fake_client.ping.assert_called_once()
