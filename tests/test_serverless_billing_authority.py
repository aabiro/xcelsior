"""Actual native execution and a captured quote are required to debit a wallet."""

import time
from unittest import mock

import pytest

from billing import get_billing_engine
from serverless.metering import charge_serverless_execution
from tests.test_serverless_native_lifecycle import native, start, report  # noqa: F401


def test_serverless_price_and_gpu_count_are_frozen_at_admission(native, monkeypatch):
    start(native)
    billing = get_billing_engine()
    billing.deposit(native.owner, 100)
    allocated = time.time() - 180
    native.repo.update_worker(native.wid, allocated_at=allocated)
    monkeypatch.setattr(
        "serverless.metering.get_gpu_rate_per_hour",
        mock.Mock(side_effect=AssertionError("billing reconstructed a live price")),
    )
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE serverless_endpoints SET gpu_count=2 WHERE endpoint_id=%s",
            (native.ep["endpoint_id"],),
        )
    for seconds in (60, 120):
        result = charge_serverless_execution(
            billing,
            native.repo,
            native.repo.get_worker(native.wid),
            native.ep,
            period_end=allocated + seconds,
            final=True,
        )
        assert result["charged"] and result["amount_cad"] == pytest.approx(0.06)
    assert billing.get_wallet(native.owner)["balance_cad"] == pytest.approx(99.88)


def test_unclaimed_worker_cannot_be_billed_from_an_observed_allocation_time(native):
    native.reserve()
    billing = get_billing_engine()
    billing.deposit(native.owner, 100)
    native.repo.update_worker(native.wid, allocated_at=time.time() - 180)
    result = charge_serverless_execution(
        billing,
        native.repo,
        native.repo.get_worker(native.wid),
        native.ep,
        final=True,
    )
    assert result == {"charged": False, "reason": "native_running_evidence_missing"}
    assert billing.get_wallet(native.owner)["balance_cad"] == 100


def test_missing_worker_cannot_gain_billing_authority_from_caller_fields(native):
    billing = get_billing_engine()
    billing.deposit(native.owner, 100)
    result = charge_serverless_execution(
        billing,
        native.repo,
        {
            "worker_id": "missing",
            "scheduler_job_id": native.jid,
            "allocated_at": 1000,
            "host_id": native.hid,
        },
        native.ep,
        final=True,
    )
    assert result == {"charged": False, "reason": "native_worker_missing"}
    assert billing.get_wallet(native.owner)["balance_cad"] == 100


def test_serverless_invoice_does_not_charge_the_provider_usage_meter_again(native):
    start(native)
    billing = get_billing_engine()
    billing.deposit(native.owner, 100)
    allocated = time.time() - 120
    native.repo.update_worker(native.wid, allocated_at=allocated)
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE usage_meters SET started_at=%s WHERE job_id=%s", (allocated, native.jid)
        )
        conn.execute(
            "UPDATE job_attempts SET started_at=to_timestamp(%s) WHERE attempt_id=%s",
            (allocated, native.r.attempt_id),
        )
    report(native, "succeeded")
    with native.pool.connection() as conn:
        assert conn.execute(
            "UPDATE usage_meters SET total_cost_micros=7000000 WHERE job_id=%s RETURNING meter_id",
            (native.jid,),
        ).fetchone()
    worker = native.repo.get_worker(native.wid)
    result = charge_serverless_execution(
        billing,
        native.repo,
        worker,
        native.ep,
        final=True,
    )
    assert result["charged"]
    invoice = billing.generate_invoice(
        native.owner,
        "Test",
        allocated - 1,
        time.time() + 1,
        tax_rate=0,
        persist=False,
    )
    assert invoice.line_items and all(
        line["line_type"] == "serverless" for line in invoice.line_items
    )
    assert len(invoice.line_items) == 1
    assert invoice.line_items[0]["subtotal_cad"] == pytest.approx(result["amount_cad"])
    assert invoice.subtotal_cad == round(result["amount_cad"], 2)
