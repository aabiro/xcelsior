"""P3/B4 — per-user snapshot rate limit regression tests."""
import pytest
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def _reset_buckets():
    from routes import instances as mod

    mod._SNAPSHOT_RATE_BUCKETS.clear()
    yield
    mod._SNAPSHOT_RATE_BUCKETS.clear()


def test_rate_limit_allows_under_quota():
    from routes.instances import _check_snapshot_rate_limit

    # Default quota is 5/hour. Fire 5 — all must pass.
    for _ in range(5):
        _check_snapshot_rate_limit("user-a")


def test_rate_limit_rejects_6th_call():
    from routes.instances import _check_snapshot_rate_limit

    for _ in range(5):
        _check_snapshot_rate_limit("user-a")
    with pytest.raises(HTTPException) as exc:
        _check_snapshot_rate_limit("user-a")
    assert exc.value.status_code == 429
    assert "rate limit" in exc.value.detail.lower()


def test_rate_limit_is_per_user():
    from routes.instances import _check_snapshot_rate_limit

    for _ in range(5):
        _check_snapshot_rate_limit("user-a")
    # Different user starts fresh.
    for _ in range(5):
        _check_snapshot_rate_limit("user-b")


def test_rate_limit_bypass_when_disabled(monkeypatch):
    from routes import instances as mod

    monkeypatch.setattr(mod, "_SNAPSHOT_RATE_LIMIT", 0)
    # Should never raise.
    for _ in range(50):
        mod._check_snapshot_rate_limit("user-a")


def test_rate_limit_window_expires(monkeypatch):
    """After the sliding window elapses, older entries are evicted."""
    from routes import instances as mod

    # Start with 5 entries at t=0.
    base = 1_000_000.0
    monkeypatch.setattr(mod.time, "time", lambda: base)
    for _ in range(5):
        mod._check_snapshot_rate_limit("user-a")
    # 6th should fail at t=0.
    with pytest.raises(HTTPException):
        mod._check_snapshot_rate_limit("user-a")
    # Jump past the window. 6th should now succeed.
    monkeypatch.setattr(mod.time, "time", lambda: base + mod._SNAPSHOT_RATE_WINDOW_SEC + 1)
    mod._check_snapshot_rate_limit("user-a")


def test_rate_limit_snapshot_endpoint_calls_helper():
    """Source grep: the snapshot endpoint must call the rate limiter."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "routes" / "instances.py").read_text()
    idx = src.find("def api_snapshot_instance(")
    assert idx >= 0
    body_end = src.find("\n@router", idx)
    body = src[idx: body_end if body_end > 0 else len(src)]
    assert "_check_snapshot_rate_limit(owner_id)" in body, (
        "api_snapshot_instance must call _check_snapshot_rate_limit(owner_id)"
    )
