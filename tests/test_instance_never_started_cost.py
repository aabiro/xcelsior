"""An instance that ended without starting has a compute estimate of zero, not "unknown".

The console showed "Est. Compute Cost: Unavailable" for an instance whose image
pull failed. Compute is billed from started_at, so the server knows the answer.
"""

from routes.instances import _enrich_instance


def _enriched(**job) -> dict:
    j = {"job_id": "j1", "submitted_at": 1000.0, **job}
    _enrich_instance(j, {})
    return j


def test_failed_before_start_costs_nothing():
    j = _enriched(status="failed", completed_at=1014.0)
    assert j["cost_cad"] == 0.0
    assert j["wait_elapsed_sec"] == 14.0


def test_cancelled_before_start_costs_nothing():
    assert _enriched(status="cancelled", completed_at=1005.0)["cost_cad"] == 0.0


def test_still_waiting_has_no_cost_yet():
    assert _enriched(status="queued").get("cost_cad") is None


def test_an_instance_that_ran_keeps_its_computed_cost():
    j = _enriched(status="failed", started_at=1010.0, completed_at=4610.0)
    assert j["cost_cad"] > 0


def test_a_recorded_cost_is_never_overwritten():
    assert _enriched(status="failed", completed_at=1014.0, cost_cad=1.25)["cost_cad"] == 1.25
