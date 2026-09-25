"""P3/B4 — per-user snapshot rate limit regression tests."""

import pytest
from fastapi import HTTPException


def _clear_all_snapshot_buckets() -> None:
    """Reset both halves of the limiter's state.

    The quota is enforced through the shared runtime state and falls back to
    the per-process dict, so clearing only the dict leaves the real counters
    behind. Worse, the shared state is *durable* — it outlives the pytest
    process — so a suite that clears half of it starts the next run already
    over quota and fails on the first call.
    """
    from routes import instances as mod
    from routes._deps import _shared_state_update

    mod._SNAPSHOT_RATE_BUCKETS.clear()
    _shared_state_update(
        mod._SNAPSHOT_RATE_STATE_NAMESPACE,
        lambda: {"buckets": {}},
        lambda _state: ({"buckets": {}, "updated_at": 0.0}, None),
    )


@pytest.fixture(autouse=True)
def _reset_buckets():
    _clear_all_snapshot_buckets()
    yield
    _clear_all_snapshot_buckets()


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
    body = src[idx : body_end if body_end > 0 else len(src)]
    assert (
        "_check_snapshot_rate_limit(owner_id)" in body
    ), "api_snapshot_instance must call _check_snapshot_rate_limit(owner_id)"


def test_the_quota_is_not_multiplied_by_the_worker_count():
    """The limit has to hold across workers, not per worker.

    `_SNAPSHOT_RATE_BUCKETS` is a dict in one worker's memory and the default
    deployment runs two, so a user's requests round-robin and each worker
    counted only the ones it happened to serve. The code said 5 per hour and
    the system allowed 5 × GUNICORN_WORKERS — 10 by default. Snapshots are a
    docker commit plus a registry push, so the quota bounds real disk and
    bandwidth; being twice as loose as it reads is not a rounding error.

    Clearing only the per-process dict is exactly what a second worker looks
    like: same user, same shared state, an empty local bucket. Before the fix
    the 6th call sailed through on the fresh dict.
    """
    from routes import instances as mod

    for _ in range(5):
        mod._check_snapshot_rate_limit("user-a")

    mod._SNAPSHOT_RATE_BUCKETS.clear()  # the next request lands on worker 2

    with pytest.raises(HTTPException) as exc:
        mod._check_snapshot_rate_limit("user-a")
    assert exc.value.status_code == 429, (
        "a second worker granted a fresh quota to a user who had already spent "
        "theirs, so the effective limit is the stated one times the worker count"
    )


def test_the_shared_path_is_the_one_being_used():
    """Otherwise the test above proves nothing.

    If shared state were unavailable the limiter would fall back to the local
    dict, `_SNAPSHOT_RATE_BUCKETS.clear()` really would reset the quota, and
    the test above would fail for a reason that has nothing to do with the
    defect. This asserts the precondition directly rather than inferring it.
    """
    from routes import instances as mod
    from routes._deps import _shared_state_update

    mod._check_snapshot_rate_limit("user-shared-probe")
    shared_ok, state = _shared_state_update(
        mod._SNAPSHOT_RATE_STATE_NAMESPACE,
        lambda: {"buckets": {}},
        lambda s: (s, s.get("buckets", {})),
    )
    assert shared_ok, (
        "shared runtime state is unavailable here, so the limiter is running on "
        "its per-process fallback and the cross-worker test cannot be trusted"
    )
    assert "user-shared-probe" in (state or {}), (
        f"the call was not recorded in shared state; buckets={state}"
    )
