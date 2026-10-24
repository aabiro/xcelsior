"""Actual native execution and a captured quote are required to debit a wallet."""

import time
from unittest import mock

import pytest

from billing import get_billing_engine
from serverless.metering import charge_serverless_execution
from tests.test_serverless_native_lifecycle import native, start, report  # noqa: F401


def backdate_execution(x, allocated):
    """Age the native allocation/start evidence together, never observations alone."""
    with x.pool.connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET reserved_at=to_timestamp(%s),started_at=to_timestamp(%s) "
            "WHERE attempt_id=%s",
            (allocated, allocated, x.r.attempt_id),
        )
        allocated = conn.execute(
            "SELECT EXTRACT(EPOCH FROM started_at)::float8 FROM job_attempts WHERE attempt_id=%s",
            (x.r.attempt_id,),
        ).fetchone()[0]
        conn.execute("UPDATE usage_meters SET started_at=%s WHERE job_id=%s", (allocated, x.jid))
    x.repo.update_worker(x.wid, allocated_at=allocated)
    return allocated


def test_serverless_future_period_is_capped_at_server_execution(native):
    start(native)
    billing = get_billing_engine()
    billing.deposit(native.owner, 100)
    backdate_execution(native, time.time() - 120)
    before = billing.get_wallet(native.owner)["balance_micros"]
    result = charge_serverless_execution(
        billing,
        native.repo,
        native.repo.get_worker(native.wid),
        native.ep,
        period_end=time.time() + 3600,
        final=True,
    )
    assert result["charged"]
    assert 120_000 <= before - billing.get_wallet(native.owner)["balance_micros"] < 180_000
    with native.pool.connection() as conn:
        assert conn.execute(
            "SELECT max(period_end)<=EXTRACT(EPOCH FROM clock_timestamp()) FROM billing_cycles WHERE job_id=%s",
            (native.jid,),
        ).fetchone() == (True,)


def test_serverless_cannot_bill_a_corrupt_allocation_before_native_reservation(native):
    start(native)
    billing = get_billing_engine()
    billing.deposit(native.owner, 100)
    native.repo.update_worker(native.wid, allocated_at=time.time() - 180)
    result = charge_serverless_execution(
        billing, native.repo, native.repo.get_worker(native.wid), native.ep, final=True
    )
    assert result == {"charged": False, "reason": "native_running_evidence_missing"}
    assert billing.get_wallet(native.owner)["balance_cad"] == 100


def test_serverless_expired_lease_has_no_periodic_billing_authority(native):
    start(native)
    billing = get_billing_engine()
    billing.deposit(native.owner, 100)
    backdate_execution(native, time.time() - 120)
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 second' WHERE attempt_id=%s",
            (native.r.attempt_id,),
        )
    result = charge_serverless_execution(
        billing, native.repo, native.repo.get_worker(native.wid), native.ep, final=True
    )
    assert result == {"charged": False, "reason": "native_running_evidence_missing"}
    assert billing.get_wallet(native.owner)["balance_cad"] == 100


def test_serverless_price_and_gpu_count_are_frozen_at_admission(native, monkeypatch):
    start(native)
    billing = get_billing_engine()
    billing.deposit(native.owner, 100)
    allocated = time.time() - 180
    allocated = backdate_execution(native, allocated)
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
    allocated = backdate_execution(native, allocated)
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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, "wrong_resource"])
def test_invalid_slice_cannot_debit_or_advance_serverless_billing(native, bad):
    start(native)
    billing = get_billing_engine()
    billing.deposit(native.owner, 100)
    backdate_execution(native, time.time() - 120)
    kwargs = {"resource_type": "gpu"} if bad == "wrong_resource" else {"period_end": bad}
    with pytest.raises(ValueError, match="invalid native serverless billing"):
        charge_serverless_execution(
            billing,
            native.repo,
            native.repo.get_worker(native.wid),
            native.ep,
            final=True,
            **kwargs,
        )
    assert billing.get_wallet(native.owner)["balance_cad"] == 100
    with native.pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM billing_cycles WHERE job_id=%s", (native.jid,)
        ).fetchone() == (0,)
