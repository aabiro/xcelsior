"""Phase 4 — spot jobs billed at locked spot_rate_cad, not host on-demand rate."""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from billing import BillingEngine, resolve_compute_rate_cad


def _engine() -> BillingEngine:
    return BillingEngine()


def _unique_owner() -> str:
    return f"spot-bill-{uuid.uuid4().hex[:10]}@xcelsior.ca"


class TestResolveComputeRate:
    def test_spot_uses_locked_rate_times_gpus(self):
        job = {"pricing_mode": "spot", "spot_rate_cad": 0.22, "num_gpus": 2}
        rate, mode = resolve_compute_rate_cad(job, {"cost_per_hour": 0.55})
        assert mode == "spot"
        assert rate == pytest.approx(0.44, rel=1e-4)

    def test_on_demand_uses_host_rate(self):
        job = {"pricing_mode": "on_demand", "num_gpus": 2}
        rate, mode = resolve_compute_rate_cad(job, {"cost_per_hour": 0.55})
        assert mode == "on_demand"
        assert rate == pytest.approx(0.55, rel=1e-4)


class TestMeterJobSpot:
    def test_spot_meter_uses_locked_rate_not_host(self):
        eng = _engine()
        customer = _unique_owner()
        eng.deposit(customer, 50.0, description="test deposit")
        now = time.time()
        locked = 0.18
        host_rate = 0.55

        meter = eng.meter_job(
            {
                "job_id": f"spot-meter-{uuid.uuid4().hex[:8]}",
                "owner": customer,
                "pricing_mode": "spot",
                "spot_rate_cad": locked,
                "num_gpus": 1,
                "started_at": now - 1800,
                "completed_at": now,
                "vram_needed_gb": 24,
            },
            {
                "host_id": "h-spot-1",
                "gpu_model": "RTX 4090",
                "cost_per_hour": host_rate,
                "country": "CA",
            },
        )

        expected = round((1800 / 3600) * locked, 4)
        assert meter.pricing_mode == "spot"
        assert meter.spot_discount == 0.0
        assert meter.base_rate_per_hour == pytest.approx(locked, rel=1e-4)
        assert meter.total_cost_cad == pytest.approx(expected, rel=1e-4)
        assert meter.total_cost_cad < round((1800 / 3600) * host_rate, 4)