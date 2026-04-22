"""P3/C3 — PII log scrubbing."""
import logging

import pytest


def test_email_redacted():
    from log_pii_filter import _scrub

    out = _scrub("User alice@example.com logged in")
    assert "alice@example.com" not in out
    assert "<email:" in out


def test_same_email_hashes_identically():
    """Correlation: two log lines with the same email must share the same tag."""
    from log_pii_filter import _scrub

    a = _scrub("login alice@example.com")
    b = _scrub("logout alice@example.com")
    tag_a = a.split("<email:")[1].split(">")[0]
    tag_b = b.split("<email:")[1].split(">")[0]
    assert tag_a == tag_b


def test_different_emails_hash_differently():
    from log_pii_filter import _scrub

    a = _scrub("alice@example.com")
    b = _scrub("bob@example.com")
    assert a != b
    assert "<email:" in a and "<email:" in b


def test_stripe_customer_id_redacted():
    from log_pii_filter import _scrub

    out = _scrub("Charge customer cus_Abc123XYZ45678 $5.00")
    assert "cus_Abc123XYZ45678" not in out
    assert "<cus:Abc1...>" in out


def test_jwt_redacted():
    from log_pii_filter import _scrub

    jwt = "eyJhbGciOiJIUzI1NiJ9." \
          "eyJ1c2VyIjoiYWxpY2UifQ." \
          "abcdef1234567890abcdef12"
    out = _scrub(f"bad token: {jwt}")
    assert jwt not in out
    assert "<jwt:redacted>" in out


def test_bearer_token_redacted():
    from log_pii_filter import _scrub

    out = _scrub("Authorization: Bearer abcdef1234567890ABCDEF.xyz")
    assert "abcdef1234567890ABCDEF" not in out
    assert "<token:redacted>" in out


def test_filter_rewrites_log_record():
    from log_pii_filter import PIIScrubFilter

    f = PIIScrubFilter()
    rec = logging.LogRecord(
        name="xcelsior",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Customer %s charged cus_Abc123XYZ45678",
        args=("alice@example.com",),
        exc_info=None,
    )
    assert f.filter(rec) is True
    assert "alice@example.com" not in rec.getMessage()
    assert "cus_Abc123XYZ45678" not in rec.getMessage()


def test_scrubbing_preserves_non_pii():
    from log_pii_filter import _scrub

    msg = "GET /api/v2/instances 200 in 45ms"
    assert _scrub(msg) == msg


def test_install_disabled_by_env_var(monkeypatch):
    monkeypatch.setenv("XCELSIOR_PII_SCRUB", "0")
    # Reset the installed flag.
    import importlib
    import log_pii_filter
    importlib.reload(log_pii_filter)
    log_pii_filter.install("test-logger-no-scrub")
    # No filter should be attached.
    logger = logging.getLogger("test-logger-no-scrub")
    assert not any(isinstance(f, log_pii_filter.PIIScrubFilter) for f in logger.filters)
    # Cleanup: restore scrubbing for subsequent tests.
    monkeypatch.setenv("XCELSIOR_PII_SCRUB", "1")
    importlib.reload(log_pii_filter)


def test_install_is_idempotent():
    import importlib
    import log_pii_filter
    importlib.reload(log_pii_filter)
    log_pii_filter.install("xcelsior.test-idempotent")
    log_pii_filter.install("xcelsior.test-idempotent")
    # Should only attach once at most per logger tree.
    logger = logging.getLogger("xcelsior")
    count = sum(1 for f in logger.filters if isinstance(f, log_pii_filter.PIIScrubFilter))
    assert count <= 1
