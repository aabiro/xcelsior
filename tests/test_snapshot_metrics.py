"""P3/C1 — Prometheus counter wiring for snapshot pipeline."""
import pytest
from fastapi import HTTPException


def _get_value(counter, **labels) -> float:
    """Read a labeled prometheus Counter's current value."""
    return counter.labels(**labels)._value.get()


@pytest.fixture(autouse=True)
def _reset_buckets():
    from routes import instances as mod
    mod._SNAPSHOT_RATE_BUCKETS.clear()
    yield
    mod._SNAPSHOT_RATE_BUCKETS.clear()


def test_c1_rate_limit_increments_counter():
    from routes import instances as mod

    before = _get_value(mod._snapshot_requests_total, outcome="rate_limited")
    # Fill the bucket, then trigger.
    for _ in range(mod._SNAPSHOT_RATE_LIMIT):
        mod._check_snapshot_rate_limit("c1-user-1")
    with pytest.raises(HTTPException) as exc:
        mod._check_snapshot_rate_limit("c1-user-1")
    assert exc.value.status_code == 429
    after = _get_value(mod._snapshot_requests_total, outcome="rate_limited")
    assert after == before + 1


def test_c1_rate_limit_success_does_not_increment():
    """Successful calls must NOT touch the rate_limited counter."""
    from routes import instances as mod

    before = _get_value(mod._snapshot_requests_total, outcome="rate_limited")
    mod._check_snapshot_rate_limit("c1-user-2")
    after = _get_value(mod._snapshot_requests_total, outcome="rate_limited")
    assert after == before


def test_c1_completion_counter_labels_exposed():
    """Counters exist and accept ready/failed labels without raising."""
    from routes import instances as mod

    before_ready = _get_value(mod._snapshot_completions_total, status="ready")
    before_failed = _get_value(mod._snapshot_completions_total, status="failed")
    mod._snapshot_completions_total.labels(status="ready").inc()
    mod._snapshot_completions_total.labels(status="failed").inc()
    assert _get_value(mod._snapshot_completions_total, status="ready") == before_ready + 1
    assert _get_value(mod._snapshot_completions_total, status="failed") == before_failed + 1


def test_c1_source_wires_counters_in_endpoint():
    """Source-grep guard: api_snapshot_instance wires all required outcomes."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "routes" / "instances.py").read_text()
    idx = src.find("def api_snapshot_instance(")
    end = src.find("\n@router", idx)
    body = src[idx:end]
    for outcome in ("not_found", "forbidden", "bad_state", "duplicate_name", "enqueued"):
        assert f'outcome="{outcome}"' in body, (
            f"api_snapshot_instance missing outcome={outcome} counter"
        )


def test_c1_source_wires_completion_counter():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "routes" / "instances.py").read_text()
    idx = src.find("def api_user_image_complete(")
    end = src.find("\n@router", idx)
    if end < 0:
        end = src.find("\ndef ", idx + 1)
    body = src[idx:end]
    assert "_snapshot_completions_total.labels(status=body.status).inc()" in body
