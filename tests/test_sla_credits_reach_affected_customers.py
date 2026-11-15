"""An SLA credit goes to the customers a host's outage affected, sized by their own spend.

Before this, the path existed only on paper:

- Nothing wrote ``sla_downtime``. ``record_downtime_start`` had no caller, so
  every host measured 100% uptime and owed nothing.
- ``sla_monthly`` rows came only from an admin call with the customer's spend
  typed in by hand.
- When a credit was paid, it went to the alphabetically first customer who had
  *ever* completed a job on the host, in any month.
- The credit bands started below 99.0% for every tier, so a host sold at 99.9%
  that delivered 99.2% owed nothing.
"""

import os
import time
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest

import scheduler
from billing import PERIODIC_BILLING_INTERVAL_SEC, get_billing_engine
from sla import MONITORING_STARTED, SLAEngine, compute_credit_pct, month_bounds, month_of

DAY = 86_400.0


def _closed_month() -> str:
    """A month long closed, unique enough that hosts never collide across runs."""
    return f"20{10 + uuid.uuid4().int % 10:02d}-{1 + uuid.uuid4().int % 12:02d}"


def _host() -> str:
    return f"h-sla-{uuid.uuid4().hex[:10]}"


def _customer() -> str:
    cid = f"cust-sla-{uuid.uuid4().hex[:10]}"
    get_billing_engine().get_wallet(cid)  # creates the wallet
    return cid


def _balance(cid: str) -> float:
    return get_billing_engine().get_wallet(cid)["balance_cad"]


def _cycle(customer_id: str, host_id: str, start: float, end: float, micros: int) -> None:
    """One charged billing period, the record the billing loop writes every five minutes."""
    with get_billing_engine()._conn() as conn:
        conn.execute(
            """INSERT INTO billing_cycles
               (cycle_id, job_id, customer_id, host_id, resource_type,
                period_start, period_end, duration_seconds, rate_per_hour,
                gpu_model, tier, amount_micros, status, created_at)
               VALUES (%s, %s, %s, %s, 'gpu', %s, %s, %s, 0, 'RTX 3060', '',
                       %s, 'charged', %s)""",
            (
                f"BC-sla-{uuid.uuid4().hex[:12]}",
                f"j-sla-{uuid.uuid4().hex[:10]}",
                customer_id,
                host_id,
                start,
                end,
                end - start,
                micros,
                end,
            ),
        )


def _outage(engine: SLAEngine, host_id: str, start: float, end: float) -> None:
    """The host dies at ``start`` and is seen alive again at ``end``, as check_hosts reports it."""
    engine.sync_host_liveness(observed=[host_id], down={host_id: start}, up=[], now=start)
    engine.sync_host_liveness(observed=[host_id], down={}, up=[host_id], now=end)


def _downtime_rows(host_id: str) -> list[dict]:
    with SLAEngine()._conn() as conn:
        return conn.execute(
            "SELECT start_ts, end_ts, resolved, reason FROM sla_downtime "
            "WHERE host_id = %s ORDER BY start_ts, id",
            (host_id,),
        ).fetchall()


# ── Credit bands follow the tier's own target ─────────────────────────


def test_a_breach_of_the_tiers_own_target_owes_a_credit():
    assert compute_credit_pct(99.2, "community") == 0.0
    assert compute_credit_pct(99.2, "secure") == 10.0
    assert compute_credit_pct(99.2, "dedicated") == 10.0
    assert compute_credit_pct(99.6, "secure") == 0.0
    assert compute_credit_pct(99.6, "dedicated") == 10.0
    assert compute_credit_pct(99.95, "dedicated") == 0.0


def test_an_unknown_tier_is_held_to_the_community_target():
    assert compute_credit_pct(99.2, "platinum") == compute_credit_pct(99.2, "community")


# ── The health check writes the downtime log ──────────────────────────


def _check(monkeypatch, host: dict, *, first_miss: float | None = None) -> str:
    """Run check_hosts against one host with the hosts table stubbed, SLA tables real."""
    monkeypatch.setattr(scheduler, "ping_host", lambda ip: False)
    monkeypatch.setattr(scheduler, "load_hosts", lambda active_only=False: [host])

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(scheduler, "_atomic_mutation", lambda: _Conn())
    monkeypatch.setattr(scheduler, "_get_host_by_id_conn", lambda conn, hid: dict(host))
    monkeypatch.setattr(scheduler, "_upsert_host_row", lambda conn, row: None)
    monkeypatch.setattr(scheduler, "alert_host_dead", lambda *a: None)
    monkeypatch.setattr(scheduler, "_requeue_dead_host_jobs", lambda *a: None)
    misses = {} if first_miss is None else {host["host_id"]: first_miss}
    monkeypatch.setattr(scheduler, "_host_first_miss", misses)
    return scheduler.check_hosts()[host["host_id"]]


def test_a_host_that_dies_opens_downtime_from_its_first_missed_check(monkeypatch):
    host_id = _host()
    went_quiet = time.time() - scheduler.HOST_DEATH_GRACE_SEC - 60
    host = {
        "host_id": host_id,
        "ip": "100.64.0.9",
        "status": "active",
        "last_agent_heartbeat": went_quiet - 30,
    }

    assert _check(monkeypatch, host, first_miss=went_quiet) == "dead"

    outages = [r for r in _downtime_rows(host_id) if r["reason"] != MONITORING_STARTED]
    assert len(outages) == 1
    assert outages[0]["resolved"] == 0
    assert outages[0]["start_ts"] == pytest.approx(went_quiet)


def test_a_revived_host_closes_its_downtime_and_monitoring_starts_once(monkeypatch):
    host_id = _host()
    went_quiet = time.time() - scheduler.HOST_DEATH_GRACE_SEC - 60
    dead = {"host_id": host_id, "ip": "", "status": "active", "last_agent_heartbeat": 1.0}
    assert _check(monkeypatch, dead, first_miss=went_quiet) == "dead"
    assert _check(monkeypatch, {**dead, "status": "dead"}, first_miss=went_quiet) == "dead"

    alive = {**dead, "status": "dead", "last_agent_heartbeat": time.time()}
    assert _check(monkeypatch, alive) == "alive"

    rows = _downtime_rows(host_id)
    markers = [r for r in rows if r["reason"] == MONITORING_STARTED]
    outages = [r for r in rows if r["reason"] != MONITORING_STARTED]
    assert len(markers) == 1, "three checks must not start monitoring three times"
    assert len(outages) == 1, "a host still dead on the next check keeps one outage open"
    assert outages[0]["resolved"] == 1
    assert outages[0]["end_ts"] > outages[0]["start_ts"]


def test_a_host_inside_its_grace_window_is_not_down(monkeypatch):
    host_id = _host()
    host = {"host_id": host_id, "ip": "", "status": "active", "last_agent_heartbeat": 1.0}
    _check(monkeypatch, host, first_miss=time.time() - 5)

    assert [r["reason"] for r in _downtime_rows(host_id)] == [MONITORING_STARTED]


# ── Monthly records measure only monitored time ───────────────────────


def test_time_before_monitoring_started_is_not_counted_as_uptime():
    engine = SLAEngine()
    host_id, month = _host(), _closed_month()
    start, end = month_bounds(month)
    began = start + 10 * DAY
    engine.sync_host_liveness(observed=[host_id], down={}, up=[host_id], now=began)

    record = engine.enforce_monthly(host_id, "community", month)

    assert record.total_seconds == pytest.approx(end - began)
    assert record.uptime_pct == 100.0


def test_a_month_with_no_monitoring_measures_nothing():
    record = SLAEngine().enforce_monthly(_host(), "community", _closed_month())
    assert record.total_seconds == 0
    assert record.credit_pct == 0.0


def test_a_dedicated_host_below_its_own_target_owes_a_credit():
    engine = SLAEngine()
    host_id, month = _host(), _closed_month()
    start, end = month_bounds(month)
    engine.sync_host_liveness(observed=[host_id], down={}, up=[host_id], now=start)
    # 0.4% of the month down: 99.6% uptime, which meets Secure but breaches Dedicated.
    _outage(engine, host_id, start + DAY, start + DAY + (end - start) * 0.004)

    assert engine.enforce_monthly(host_id, "secure", month).credit_pct == 0.0
    assert engine.enforce_monthly(host_id, "dedicated", month).credit_pct == 10.0


# ── Settlement credits the affected customers, once ───────────────────


def test_the_credit_goes_to_customers_on_the_host_during_the_outage_by_their_own_spend():
    engine = SLAEngine()
    host_id, other_host, month = _host(), _host(), _closed_month()
    start, end = month_bounds(month)
    engine.sync_host_liveness(observed=[host_id], down={}, up=[host_id], now=start)
    down_at = start + 15 * DAY
    _outage(engine, host_id, down_at, down_at + 2 * DAY)  # ~93% uptime → 25% credit

    on_host = _customer()  # running when the host went quiet
    _cycle(on_host, host_id, start + 2 * DAY, start + 2 * DAY + 3600, 6_000_000)
    _cycle(on_host, host_id, down_at - PERIODIC_BILLING_INTERVAL_SEC, down_at - 10, 4_000_000)
    earlier = _customer()  # used the host in week one only
    _cycle(earlier, host_id, start + DAY, start + DAY + 3600, 20_000_000)
    elsewhere = _customer()  # running during the outage, on a different host
    _cycle(elsewhere, other_host, down_at, down_at + 3600, 8_000_000)
    before = {c: _balance(c) for c in (on_host, earlier, elsewhere)}

    issued = engine.settle_month(month)

    assert [i["customer_id"] for i in issued] == [on_host]
    assert _balance(on_host) == pytest.approx(before[on_host] + 2.5)  # 25% of $10 on this host
    assert _balance(earlier) == pytest.approx(before[earlier])
    assert _balance(elsewhere) == pytest.approx(before[elsewhere])

    with get_billing_engine()._conn() as conn:
        tx = conn.execute(
            "SELECT tx_type FROM wallet_transactions WHERE idempotency_key = %s",
            (f"sla:{host_id}:{month}:{on_host}",),
        ).fetchone()
    assert tx["tx_type"] == "credit", "a service credit is not money received"

    row = engine.get_host_sla(host_id, month)
    assert row is not None and row.credit_pct == 25.0


def test_settling_twice_credits_once():
    engine = SLAEngine()
    host_id, month = _host(), _closed_month()
    start, _ = month_bounds(month)
    engine.sync_host_liveness(observed=[host_id], down={}, up=[host_id], now=start)
    _outage(engine, host_id, start + 5 * DAY, start + 9 * DAY)  # under 90% in any month
    cid = _customer()
    _cycle(cid, host_id, start + 5 * DAY - 60, start + 5 * DAY + 120, 10_000_000)
    before = _balance(cid)

    first = engine.settle_month(month)
    second = engine.settle_month(month)

    assert len(first) == 1 and second == []
    assert _balance(cid) == pytest.approx(before + 10.0)  # <90% uptime → 100%


def test_an_open_month_is_never_settled():
    engine = SLAEngine()
    host_id = _host()
    now = time.time()
    month = month_of(now)
    start, _ = month_bounds(month)
    engine.sync_host_liveness(observed=[host_id], down={}, up=[host_id], now=start)
    engine.sync_host_liveness(observed=[host_id], down={host_id: start + 1}, up=[], now=start + 1)
    cid = _customer()
    _cycle(cid, host_id, start, start + 300, 5_000_000)
    before = _balance(cid)

    assert engine.settle_month(month, now=now) == []
    assert _balance(cid) == pytest.approx(before)
