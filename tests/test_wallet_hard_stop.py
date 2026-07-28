"""Wallet hard-stop and low-balance warning behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def billing_engine(monkeypatch):
    from billing import BillingEngine

    be = BillingEngine()
    wallet = {
        "customer_id": "cust-test",
        "balance_cad": 10.0,
        "status": "active",
        "grace_until": 0,
        "auto_topup_threshold_cad": 5.0,
        "stripe_customer_id": "",
    }

    def get_wallet(cid):
        return dict(wallet)

    monkeypatch.setattr(be, "get_wallet", get_wallet)
    monkeypatch.setattr(be, "_ensure_wallet_table", lambda: None)
    return be, wallet


def test_charge_hard_stops_when_insufficient(billing_engine, monkeypatch):
    be, wallet = billing_engine
    wallet["balance_cad"] = 1.0

    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute = MagicMock()
    monkeypatch.setattr(be, "_conn", lambda: conn)

    with patch("db.NotificationStore.create"):
        result = be.charge("cust-test", 5.0, job_id="job-1", description="GPU")

    assert result["charged"] is False
    assert result["reason"] == "insufficient_balance"
    assert result["action"] in ("hard_stop", "account_suspended")


def test_charge_zero_balance_suspends(billing_engine, monkeypatch):
    be, wallet = billing_engine
    wallet["balance_cad"] = 0.0

    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute = MagicMock()
    monkeypatch.setattr(be, "_conn", lambda: conn)

    with patch("db.NotificationStore.create"):
        result = be.charge("cust-test", 0.5, job_id="job-2")

    assert result["charged"] is False
    assert result["action"] == "account_suspended"


def test_successful_charge_enqueues_meter(billing_engine, monkeypatch):
    be, wallet = billing_engine
    wallet["balance_cad"] = 20.0

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            m = MagicMock()
            m.fetchone.return_value = {"balance_micros": 19_000_000, "balance_cad": 19.0}
            return m

    monkeypatch.setattr(be, "_conn", lambda: _Conn())
    enqueued = {}

    def fake_enqueue(**kwargs):
        enqueued.update(kwargs)
        return {"enqueued": True}

    monkeypatch.setattr(
        "stripe_meters.enqueue_usage_from_charge",
        fake_enqueue,
        raising=False,
    )
    # charge imports inside try — patch module path used at call site
    import stripe_meters

    monkeypatch.setattr(stripe_meters, "enqueue_usage_from_charge", fake_enqueue)

    with patch.object(be, "maybe_warn_low_balance", return_value={"warned": False}):
        result = be.charge("cust-test", 1.0, job_id="job-3", description="Serverless worker (60s @ 0.5/hr)")

    assert result["charged"] is True
    assert enqueued.get("customer_id") == "cust-test"
    assert enqueued.get("amount_cad") == 1.0


def test_meter_infer_serverless():
    from stripe_meters import EVENT_SERVERLESS_SECOND, _infer_event

    name, value = _infer_event("Serverless worker final: ep (60s @ 0.5/hr)", "slvr-x")
    assert name == EVENT_SERVERLESS_SECOND
    assert value == 60.0
