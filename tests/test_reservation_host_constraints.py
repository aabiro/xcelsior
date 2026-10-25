"""A scored host must still satisfy canonical constraints under the row lock."""

import time

import pytest
from psycopg.types.json import Jsonb

from control_plane.scheduler.filters import FilterContext, evaluate_host, price
from control_plane.scheduler.reservation import HostNotEligible
from control_plane.scheduler.snapshot import read_fleet
from tests.test_control_plane_reservation import (  # noqa: F401
    _claim,
    _counts,
    _mkhost,
    _mkjob,
    _pool,
    _reserve,
    fleet,
)


def scored_pair(fleet):
    job = _mkjob(fleet, admission_fields={"region": "ca-on", "max_price_per_hour": 1})
    host = _mkhost(fleet)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET region='ca-on',payload=payload||%s WHERE host_id=%s",
            (
                Jsonb(
                    {
                        "gpu_model": job,
                        "gpu_count": 2,
                        "region": "ca-on",
                        "cost_per_hour": 0.5,
                        "capabilities": {"egress_isolation": True},
                        "last_seen": time.time(),
                    }
                ),
                host,
            ),
        )
        snapshot = read_fleet(conn)
        current = next(h for h in snapshot.hosts if h["host_id"] == host)
        spec = conn.execute("SELECT spec FROM jobs WHERE job_id=%s", (job,)).fetchone()[0]
        assert evaluate_host(spec, current, FilterContext(snapshot.stale_host_ids)) == []
    return job, host, _claim(job)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("capabilities='{}'::jsonb", "egress_isolation_unavailable"),
        ("capabilities='{\"egress_isolation\":false}'::jsonb", "egress_isolation_unavailable"),
        ("region='ca-qc'", "region_mismatch"),
        ("region=NULL", "region_mismatch"),
        ("status='inactive'", "host_not_ready"),
        ("payload=payload||'{\"cost_per_hour\":2}'::jsonb", "price_exceeds_approved"),
        ("payload=payload-'cost_per_hour'", "host_price_unqualified"),
        ('payload=payload||\'{"cost_per_hour":"NaN"}\'::jsonb', "host_price_unqualified"),
    ],
)
def test_changed_constraint_refuses_reservation_without_partial_authority(fleet, mutation, reason):
    from psycopg import sql

    job, host, claimed = scored_pair(fleet)
    with _pool.connection() as conn:
        # The mutation is selected only from the literal parametrization above.
        conn.execute(
            sql.SQL("UPDATE hosts SET {} WHERE host_id=%s").format(sql.SQL(mutation)), (host,)
        )
    with pytest.raises(HostNotEligible) as rejected:
        _reserve(claimed, host, expected_inventory_generation=1)
    assert rejected.value.details["filter_reason"] == reason
    assert not any(_counts(job).values())


def test_payload_cannot_revoke_or_override_canonical_constraint_evidence(fleet):
    job, host, claimed = scored_pair(fleet)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET payload=payload||%s WHERE host_id=%s",
            (Jsonb({"region": "ca-qc", "capabilities": {}, "status": "dead"}), host),
        )
    reservation = _reserve(claimed, host, expected_inventory_generation=1)
    assert reservation.host_id == host
    assert _counts(job)["attempts"] == 1


@pytest.mark.parametrize("cost", [None, True, False, "bad", "NaN", "Infinity", -1])
def test_unknown_host_price_cannot_pass_an_approved_ceiling(cost):
    reason = price({"max_price_per_hour": 1}, {"cost_per_hour": cost}, FilterContext())
    assert reason is not None and reason.code == "host_price_unqualified"


@pytest.mark.parametrize("maximum", [True, "bad", "NaN", "Infinity", -1])
def test_invalid_price_approval_cannot_become_an_unbounded_admission(maximum):
    reason = price({"max_price_per_hour": maximum}, {"cost_per_hour": 1}, FilterContext())
    assert reason is not None and reason.code == "price_approval_unqualified"
