"""Phase 8 — spot feature flag and observability."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")


class TestSpotFeatureFlag:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("XCELSIOR_SPOT_ENABLED", raising=False)
        from spot.feature import spot_feature_status, spot_global_enabled

        assert spot_global_enabled() is True
        status = spot_feature_status()
        assert status["enabled"] is True
        assert status["message"] is None

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SPOT_ENABLED", "false")
        import importlib

        import spot.feature as sf

        importlib.reload(sf)
        assert sf.spot_global_enabled() is False
        assert "unavailable" in (sf.spot_feature_status().get("message") or "").lower()


class TestSpotMetrics:
    def test_metrics_snapshot_includes_spot(self):
        import scheduler
        from spot_metrics import get_spot_metrics_snapshot, record_spot_preemption

        scheduler.register_host("m1", "127.0.0.1", "RTX 4090", 24, 24)
        scheduler._set_host_fields("m1", admitted=True)
        job = scheduler.submit_job("spot-m", 8, pricing_mode="spot")
        scheduler.update_job_status(job["job_id"], "running", host_id="m1")

        snap = get_spot_metrics_snapshot(scheduler.list_jobs())
        assert snap["jobs_running"] >= 1

        before = snap["preemptions_total"]
        record_spot_preemption()
        snap2 = get_spot_metrics_snapshot(scheduler.list_jobs())
        assert snap2["preemptions_total"] == before + 1

    def test_scheduler_metrics_snapshot_has_spot_key(self):
        import scheduler

        snap = scheduler.get_metrics_snapshot()
        assert "spot" in snap
        assert "jobs_running" in snap["spot"]