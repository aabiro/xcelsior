"""Unit tests for Phase 15 predictive scaling hints."""

from serverless.scaling_predictive import (
    forecast_queue_depth,
    headroom_workers,
    predictive_scaling_enabled,
)


def test_forecast_queue_depth_linear():
    samples = [(0.0, 2), (10.0, 4)]
    assert forecast_queue_depth(samples, horizon_sec=10.0) == 6


def test_forecast_single_sample():
    assert forecast_queue_depth([(1.0, 3)]) == 3


def test_headroom_disabled_by_default(monkeypatch):
    monkeypatch.delenv("XCELSIOR_SERVERLESS_PREDICTIVE_SCALING", raising=False)
    assert headroom_workers(
        queue_depth=10, active_workers=1, max_concurrency=4, forecast_depth=20
    ) == 0


def test_headroom_when_enabled(monkeypatch):
    monkeypatch.setenv("XCELSIOR_SERVERLESS_PREDICTIVE_SCALING", "true")
    assert predictive_scaling_enabled()
    extra = headroom_workers(
        queue_depth=8, active_workers=1, max_concurrency=4, forecast_depth=12
    )
    assert extra >= 1