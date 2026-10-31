"""Stripe usage meters receive the resource and quantity that was actually billed.

The native GPU and storage paths both projected their charges with the fixed
description "Native periodic billing". The meter mapping inferred the event
from that text: storage charges went to the GPU-hour meter, and every
five-minute GPU slice — any length — reported one full GPU-hour. Serverless
projected "Serverless worker", which the mapping read as one second. The drain
then rounded every value up to a whole unit.

Each billing path now names its resource and quantity: GPU-hours (seconds /
3600 × GPUs), storage GB-months (the storage meter's unit) and serverless
worker seconds, and the drain sends the exact decimal.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

import stripe_meters
from billing import get_billing_engine
from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance


@pytest.fixture
def captured(monkeypatch):
    events: list[dict] = []

    def backend(op, **kwargs):
        if op == "enqueue":
            events.append(kwargs)
            return {"enqueued": True}
        return {"sent": 0, "failed": 0, "skipped": 0}

    monkeypatch.setenv("XCELSIOR_STRIPE_METERS_ENABLED", "1")
    stripe_meters.set_outbox_backend(backend)
    yield events
    stripe_meters.set_outbox_backend(None)


def test_a_five_minute_gpu_slice_reports_a_twelfth_of_a_gpu_hour(captured):
    owner = f"meter-gpu-{uuid.uuid4().hex[:10]}"
    engine = get_billing_engine()
    engine.deposit(owner, 100)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=600) as native:
        with _get_pg_pool().connection() as conn:
            started = conn.execute(
                "SELECT EXTRACT(EPOCH FROM started_at)::float8 FROM job_attempts WHERE job_id=%s",
                (native["job_id"],),
            ).fetchone()[0]
        assert engine.bill_running_period(native["job_id"], period_end=started + 300)["billed"]
    mine = [e for e in captured if e["payload"].get("job_id") == native["job_id"]]
    assert len(mine) == 1
    assert mine[0]["event_name"] == stripe_meters.EVENT_GPU_HOUR
    assert mine[0]["value"] == pytest.approx(
        300 / 3600
    ), "a five-minute slice of one GPU is 1/12 GPU-hour, not a whole one"


def test_storage_reports_gb_months_to_the_storage_meter(captured, monkeypatch):
    from volumes import VOLUME_PRICE_PER_GB_MONTH_CAD, VolumeEngine

    owner = f"meter-vol-{uuid.uuid4().hex[:10]}"
    engine = get_billing_engine()
    engine.deposit(owner, 100)
    volumes = VolumeEngine()
    monkeypatch.setattr(volumes, "_provision_volume_storage", lambda *a, **kw: True)
    monkeypatch.setattr(volumes, "_emit_event", lambda *a, **kw: None)
    volume = volumes.create_volume(owner, "meter-vol", 100)
    try:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE volumes SET created_at=EXTRACT(EPOCH FROM clock_timestamp())-7300 "
                "WHERE volume_id=%s",
                (volume["volume_id"],),
            )
        engine.auto_billing_cycle()
        mine = [e for e in captured if e["payload"].get("job_id") == volume["volume_id"]]
        assert len(mine) == 1
        assert mine[0]["event_name"] == stripe_meters.EVENT_STORAGE_GB_MONTH
        with _get_pg_pool().connection() as conn:
            seconds, micros = conn.execute(
                "SELECT duration_seconds,amount_micros FROM billing_cycles WHERE job_id=%s",
                (volume["volume_id"],),
            ).fetchone()
        gb_months = 100 * seconds / (730 * 3600)
        assert mine[0]["value"] == pytest.approx(gb_months)
        # The quantity prices back to the charge at the catalog rate.
        assert gb_months * VOLUME_PRICE_PER_GB_MONTH_CAD == pytest.approx(micros / 1e6, abs=1e-4)
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (volume["volume_id"],))
            conn.execute("DELETE FROM volumes WHERE volume_id=%s", (volume["volume_id"],))


def test_serverless_reports_the_seconds_it_billed(captured):
    from serverless.metering import bill_active_serverless_workers
    from serverless.repo import EndpointCreate, ServerlessRepo
    from tests.fixtures.native_serverless import running_serverless

    owner = f"meter-slvr-{uuid.uuid4().hex[:10]}"
    engine = get_billing_engine()
    engine.deposit(owner, 100)
    repo = ServerlessRepo()
    endpoint = repo.create_endpoint(
        EndpointCreate(owner_id=owner, name="meter-slvr", gpu_tier="RTX 4090", min_workers=0)
    )
    try:
        with running_serverless(repo, endpoint["endpoint_id"], age_sec=300) as worker:
            bill_active_serverless_workers(engine)
            job_id = worker["scheduler_job_id"]
            mine = [e for e in captured if e["payload"].get("job_id") == job_id]
            assert len(mine) == 1
            assert mine[0]["event_name"] == stripe_meters.EVENT_SERVERLESS_SECOND
            with _get_pg_pool().connection() as conn:
                billed = conn.execute(
                    "SELECT duration_seconds FROM billing_cycles WHERE job_id=%s "
                    "AND status='charged'",
                    (job_id,),
                ).fetchone()[0]
            assert billed >= 299
            assert mine[0]["value"] == billed, "serverless reported 1 s instead of the slice"
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "DELETE FROM serverless_endpoints WHERE endpoint_id=%s", (endpoint["endpoint_id"],)
            )


def test_the_drain_sends_the_exact_quantity_not_a_whole_unit(monkeypatch):
    rows = [
        {
            "event_id": "e1",
            "customer_id": "c",
            "event_name": stripe_meters.EVENT_GPU_HOUR,
            "value": 300 / 3600,
            "idempotency_key": "k1",
            "attempts": 0,
        }
    ]

    class _Conn:
        row_factory = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *args, **kwargs):
            result = MagicMock()
            result.fetchall.return_value = rows
            return result

    class _Pool:
        def connection(self):
            return _Conn()

    import db
    import stripe_connect

    monkeypatch.setattr(db, "_get_pg_pool", lambda: _Pool())
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True)
    monkeypatch.setattr(stripe_meters, "meters_enabled", lambda: True)
    monkeypatch.setattr(stripe_meters, "_stripe_customer_id", lambda cid: "cus_test")
    monkeypatch.setattr(stripe_meters, "_mark", lambda *a, **kw: None)
    client = MagicMock()
    stats = stripe_meters.drain_meter_outbox(limit=10, stripe_mod=client)
    assert stats["sent"] == 1
    payload = client.billing.MeterEvent.create.call_args.kwargs["payload"]
    assert payload["value"] == "0.083333"


def test_the_meter_mapping_refuses_an_unknown_resource():
    with pytest.raises(ValueError):
        stripe_meters.enqueue_metered_usage(
            customer_id="c",
            resource_type="gpu-hours-ish",
            quantity=1,
            amount_cad=1,
            job_id="j",
            tx_id="t",
        )
