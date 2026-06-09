"""Phase 10 — staging checklist E2E validation (§10.1).

Maps integration-plan checklist items to automated tests runnable in CI
without a live staging cluster. Live smoke: ``scripts/spot_staging_smoke.py``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_spot_e2e_")
_tmpdir = _tmp_ctx.name

os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ.setdefault("XCELSIOR_BILLING_DB", os.path.join(_tmpdir, "billing.db"))

import scheduler

scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")

for _h in scheduler.log.handlers[:]:
    if isinstance(_h, logging.FileHandler):
        scheduler.log.removeHandler(_h)
        _h.close()

ROOT = Path(__file__).resolve().parents[1]

try:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as _conn:
        _conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'jobs' AND column_name = 'pricing_mode'"
        )
    _PG_AVAILABLE = True
except Exception:
    _PG_AVAILABLE = False

pg = pytest.mark.skipif(not _PG_AVAILABLE, reason="PostgreSQL not available")


def _admit(host_id: str, **extra):
    scheduler._set_host_fields(host_id, admitted=True, **extra)


def _host(
    host_id: str,
    *,
    spot_enabled: bool = True,
    spot_gpu_slots: int = 1,
    spot_min_cents: int = 0,
    gpu_count: int = 1,
    cost_per_hour: float = 0.55,
):
    scheduler.register_host(host_id, "10.0.0.1", "RTX 4090", 24, 24, cost_per_hour=cost_per_hour)
    _admit(
        host_id,
        spot_enabled=spot_enabled,
        spot_gpu_slots=spot_gpu_slots,
        spot_min_cents=spot_min_cents,
        gpu_count=gpu_count,
    )


@pytest.fixture(autouse=True)
def _clean_data():
    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM hosts")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM state")
    for f in (
        scheduler.HOSTS_FILE,
        scheduler.JOBS_FILE,
        scheduler.BILLING_FILE,
        scheduler.MARKETPLACE_FILE,
        os.environ["XCELSIOR_DB_PATH"],
    ):
        if os.path.exists(f):
            os.remove(f)
    yield


class TestStagingChecklistMigration040:
    """10.1 — Migration 040 applied (schema + bid retirement)."""

    def test_migration_040_module_present(self):
        path = ROOT / "migrations" / "versions" / "040_retire_spot_bidding.py"
        assert path.is_file()
        text = path.read_text()
        assert 'revision = "040"' in text
        assert "pricing_mode" in text
        assert "spot_rate_cad" in text
        assert "max_bid" in text

    @pg
    def test_jobs_table_has_spot_columns(self):
        pool = _get_pg_pool()
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'jobs'
                  AND column_name IN ('pricing_mode', 'spot_rate_cad')
                """
            ).fetchall()
        cols = {r["column_name"] if isinstance(r, dict) else r[0] for r in rows}
        assert cols == {"pricing_mode", "spot_rate_cad"}


class TestStagingChecklistSpotMetering:
    """10.1 — Spot launch meters wallet at locked spot rate."""

    def test_spot_lock_and_meter_below_on_demand(self):
        from billing import BillingEngine, resolve_compute_rate_cad

        _host("bill-h")
        job = scheduler.submit_job("spot-bill", 8, pricing_mode="spot", gpu_model="RTX 4090")
        assigned = scheduler.process_queue()
        assert len(assigned) == 1
        refreshed = scheduler.get_job(job["job_id"])
        locked = float(refreshed["spot_rate_cad"])
        assert locked > 0
        assert locked < 0.55

        host = scheduler.list_hosts()[0]
        rate, mode = resolve_compute_rate_cad(refreshed, host)
        assert mode == "spot"
        assert rate == pytest.approx(locked, rel=1e-4)

        eng = BillingEngine()
        customer = f"spot-e2e-{uuid.uuid4().hex[:8]}@xcelsior.ca"
        eng.deposit(customer, 100.0, description="staging e2e")
        now = time.time()
        refreshed["owner"] = customer
        refreshed["started_at"] = now - 3600
        refreshed["completed_at"] = now
        meter = eng.meter_job(refreshed, host)
        expected = round((3600 / 3600) * locked, 4)
        assert meter.total_cost_cad == pytest.approx(expected, rel=1e-4)
        assert meter.total_cost_cad < round(0.55, 4)


class TestStagingChecklistPreemption:
    """10.1 — On-demand preempts spot; spot requeues and reschedules."""

    def test_on_demand_preempts_running_spot(self):
        _host("pre-h", gpu_count=1)
        spot = scheduler.submit_job("spot-run", 8, pricing_mode="spot", gpu_model="RTX 4090")
        scheduler.process_queue()
        scheduler.update_job_status(spot["job_id"], "running", host_id="pre-h")

        od = scheduler.submit_job("od-contend", 8, pricing_mode="on_demand", gpu_model="RTX 4090")
        assigned = scheduler.process_queue()
        assert len(assigned) == 1
        assert assigned[0][0]["job_id"] == od["job_id"]

        spot_after = scheduler.get_job(spot["job_id"])
        assert spot_after["status"] == "queued"
        assert spot_after.get("preemption_count") == 1
        assert spot_after.get("pricing_mode") == "spot"

    def test_spot_reschedules_after_on_demand_releases_gpu(self):
        _host("resched-h", gpu_count=1)
        spot = scheduler.submit_job("spot-resched", 8, pricing_mode="spot", gpu_model="RTX 4090")
        scheduler.process_queue()
        scheduler.update_job_status(spot["job_id"], "running", host_id="resched-h")

        od = scheduler.submit_job("od-temp", 8, pricing_mode="on_demand", gpu_model="RTX 4090")
        scheduler.process_queue()
        scheduler.update_job_status(od["job_id"], "running", host_id="resched-h")

        scheduler.update_job_status(od["job_id"], "stopped", host_id=None)
        reassigned = scheduler.process_queue()
        assert len(reassigned) == 1
        assert reassigned[0][0]["job_id"] == spot["job_id"]
        assert reassigned[0][1]["host_id"] == "resched-h"
        assert reassigned[0][0].get("spot_rate_cad") is not None


class TestStagingChecklistMarketplace:
    """10.1 — Marketplace spot_enabled=false and provider floor."""

    def test_scheduler_skips_spot_disabled_host(self):
        _host("spot-off", spot_enabled=False)
        _host("spot-on", spot_enabled=True)
        job = scheduler.submit_job("mkt-spot", 8, pricing_mode="spot", gpu_model="RTX 4090")
        picked = scheduler.allocate(job, scheduler.list_hosts())
        assert picked is not None
        assert picked["host_id"] == "spot-on"

    def test_marketplace_rejects_spot_when_offer_disabled(self):
        from marketplace import MarketplaceEngine

        engine = MarketplaceEngine.__new__(MarketplaceEngine)
        offer = {
            "offer_id": "off-1",
            "available": True,
            "gpu_count_available": 1,
            "ask_cents_per_hour": 55,
            "gpu_model": "RTX 4090",
            "spot_enabled": False,
            "spot_multiplier": 0.6,
            "spot_min_cents": 0,
        }
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [offer, None]
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch.object(MarketplaceEngine, "_conn", return_value=mock_cm):
            result = engine.allocate_gpu("off-1", "job-1", allocation_type="spot")
        assert result is None

    def test_marketplace_spot_price_respects_provider_floor(self, monkeypatch):
        from marketplace import MarketplaceEngine
        import spot_pricing as sp

        monkeypatch.setattr(
            sp,
            "compute_live_spot_quote",
            lambda gpu_model, supply=0, demand=0: sp.SpotQuote(
                gpu_model=gpu_model,
                rate_cad=0.10,
                on_demand_cad=0.55,
                savings_pct=80,
                supply=supply,
                demand=demand,
                spot_cents=10,
                provider_floor_cents=0,
                as_of=0.0,
            ),
        )
        engine = MarketplaceEngine.__new__(MarketplaceEngine)
        offer = {
            "offer_id": "off-floor",
            "available": True,
            "gpu_count_available": 1,
            "ask_cents_per_hour": 55,
            "gpu_model": "RTX 4090",
            "spot_enabled": True,
            "spot_multiplier": 0.6,
            "spot_min_cents": 25,
        }
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [offer, None]
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch.object(MarketplaceEngine, "_conn", return_value=mock_cm):
            result = engine.allocate_gpu("off-floor", "job-floor", allocation_type="spot")

        assert result is not None
        assert result["price_cents_per_hour"] >= 25


class TestStagingChecklistApiSurface:
    """10.1 — Public API: spot endpoints and no bid fields."""

    def test_spot_enabled_endpoint(self):
        from fastapi.testclient import TestClient
        from api import app

        with TestClient(app) as client:
            r = client.get("/api/pricing/spot-enabled")
        assert r.status_code == 200
        body = r.json()
        assert body.get("ok") is True
        assert "enabled" in body

    def test_spot_prices_endpoint(self):
        from fastapi.testclient import TestClient
        from api import app

        with TestClient(app) as client:
            r = client.get("/spot-prices")
        assert r.status_code == 200
        body = r.json()
        assert "prices" in body or "spot_prices" in body

    def test_openapi_job_in_has_no_max_bid(self):
        spec = json.loads((ROOT / "public" / "openapi.json").read_text())
        job_in = spec["components"]["schemas"].get("JobIn", {})
        props = job_in.get("properties", {})
        assert "max_bid" not in props
        assert "pricing_mode" in props

    def test_spot_launch_rejects_max_bid_field(self):
        from fastapi.testclient import TestClient
        from api import app

        _host("api-h")
        with TestClient(app) as client:
            r = client.post(
                "/instance",
                json={
                    "name": "spot-no-bid",
                    "vram_needed_gb": 8.0,
                    "pricing_mode": "spot",
                    "max_bid": 0.99,
                },
            )
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            job = scheduler.get_job(r.json()["instance"]["job_id"])
            assert "max_bid" not in (job or {})