"""Cross-replica MCP spend-counter policy gates."""

import pytest

from control_plane.launch import spend_counters


@pytest.fixture(autouse=True)
def memory_backend(monkeypatch):
    monkeypatch.setenv("MCP_RATE_LIMIT_BACKEND", "memory")
    spend_counters.reset_for_tests()
    yield
    spend_counters.reset_for_tests()


def _reserve(plan: str, amount: int = 400):
    return spend_counters.reserve(
        plan_id=plan,
        client_id="client-1",
        tenant_id="tenant-1",
        amount_micros=amount,
        hourly_limit_micros=1_000,
        daily_limit_micros=1_500,
        now=1_800_000_000.0,
    )


def test_reservation_is_idempotent_per_plan():
    first = _reserve("plan-1")
    replay = _reserve("plan-1")
    assert first is not None and not first.replay
    assert replay is not None and replay.replay
    _reserve("plan-2", 600)
    with pytest.raises(spend_counters.SpendLimitExceeded) as exc:
        _reserve("plan-3", 1)
    assert exc.value.window == "hourly"


def test_daily_ceiling_is_independent():
    _reserve("plan-1", 800)
    with pytest.raises(spend_counters.SpendLimitExceeded) as exc:
        spend_counters.reserve(
            plan_id="plan-2",
            client_id="client-1",
            tenant_id="tenant-1",
            amount_micros=800,
            hourly_limit_micros=2_000,
            daily_limit_micros=1_500,
            now=1_800_000_000.0,
        )
    assert exc.value.window == "daily"


def test_failed_execution_can_release_reservation():
    reservation = _reserve("plan-1", 900)
    spend_counters.release(reservation)
    _reserve("plan-2", 900)
