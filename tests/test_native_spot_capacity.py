"""Native spot policy cannot be bypassed by stale scores or simultaneous claims."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from psycopg.types.json import Jsonb

from control_plane.scheduler.reservation import HostNotEligible
from tests.test_control_plane_reservation import (  # noqa: F401
    _mkhost,
    _mkjob,
    _claim,
    _reserve,
    _counts,
    _pool,
    fleet,
)


def policy(host, enabled=True, slots=1):
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET payload=payload||%s WHERE host_id=%s",
            (Jsonb({"spot_enabled": enabled, "spot_gpu_slots": slots}), host),
        )


def spot(fleet):
    return _mkjob(fleet, admission_fields={"pricing_mode": "spot", "spot_rate_cad": 0.5})


def test_disabled_host_refuses_spot_at_reservation_without_partial_rows(fleet):
    host, job = _mkhost(fleet), spot(fleet)
    policy(host, enabled=False)
    with pytest.raises(HostNotEligible) as rejected:
        _reserve(_claim(job), host)
    assert rejected.value.details["spot_policy_reason"] == "spot_disabled"
    assert not any(_counts(job).values())


def test_spot_pool_change_after_scoring_is_revalidated_under_lock(fleet):
    host, job = _mkhost(fleet), spot(fleet)
    policy(host, slots=1)
    claimed = _claim(job)
    policy(host, slots=0)
    with pytest.raises(HostNotEligible) as rejected:
        _reserve(claimed, host)
    assert rejected.value.details["spot_policy_reason"] == "spot_capacity_exhausted"
    assert not any(_counts(job).values())


def test_concurrent_spot_reservations_do_not_overcommit_pool(fleet):
    host = _mkhost(fleet, gpus=2)
    policy(host, slots=1)
    jobs = [spot(fleet), spot(fleet)]
    claimed = [_claim(job) for job in jobs]

    def reserve(claim):
        try:
            return _reserve(claim, host)
        except HostNotEligible as rejected:
            assert rejected.details["spot_policy_reason"] == "spot_capacity_exhausted"
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(reserve, claimed))
    assert sum(value is not None for value in result) == 1
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE host_id=%s AND released_at IS NULL",
            (host,),
        ).fetchone() == (1,)
    # An on-demand job can use the other concrete device.
    ondemand = _mkjob(fleet)
    assert _reserve(_claim(ondemand), host).host_id == host


def test_spot_admission_locks_quote_once_and_replay_does_not_reprice(fleet, monkeypatch):
    import uuid
    from types import SimpleNamespace
    import scheduler
    import spot_pricing

    monkeypatch.setattr(
        spot_pricing,
        "compute_live_spot_quote",
        lambda model, **kwargs: SimpleNamespace(rate_cad=0.25),
    )
    job_id = "spot-admit-" + uuid.uuid4().hex[:10]
    fields = {
        "name": "admitted-spot",
        "vram_needed_gb": 8,
        "gpu_model": "RTX 4090",
        "pricing_mode": "spot",
        "job_id": job_id,
    }
    first = scheduler.submit_job(**fields)
    fleet["jobs"].append(job_id)
    with _pool.connection() as conn:
        before = conn.execute(
            "SELECT spec_hash,spec->'spot_rate_cad',spot_rate_micros FROM jobs WHERE job_id=%s",
            (job_id,),
        ).fetchone()
    assert before[1:] == (0.25, 250000)
    assert first["spot_rate_cad"] == 0.25
    monkeypatch.setattr(
        spot_pricing,
        "compute_live_spot_quote",
        lambda model, **kwargs: pytest.fail("committed admission was repriced"),
    )
    assert scheduler.submit_job(**fields)["job_id"] == job_id
    with _pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT spec_hash,spec->'spot_rate_cad',spot_rate_micros FROM jobs WHERE job_id=%s",
                (job_id,),
            ).fetchone()
            == before
        )


@pytest.mark.parametrize("model", ["", "unknown-model-with-no-price"])
def test_unqualified_spot_price_never_creates_admission(fleet, model):
    import uuid
    from scheduler import submit_job
    from control_plane.spot_admission import SpotAdmissionRejected

    jid = "spot-refused-" + uuid.uuid4().hex[:10]
    with pytest.raises(SpotAdmissionRejected):
        submit_job("unquoted-spot", 8, gpu_model=model, pricing_mode="spot", job_id=jid)
    with _pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM jobs WHERE job_id=%s", (jid,)).fetchone() == (0,)
        assert conn.execute(
            "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s", (jid,)
        ).fetchone() == (0,)


def test_strict_admission_quote_cannot_use_a_compatibility_rate(monkeypatch):
    from spot_pricing import compute_live_spot_quote
    import spot_pricing

    monkeypatch.setattr(spot_pricing, "get_platform_spot_base", lambda *a, **k: None)
    monkeypatch.setattr(spot_pricing, "get_platform_on_demand_base", lambda *a, **k: 0.55)
    with pytest.raises(ValueError, match="active catalog prices"):
        compute_live_spot_quote("RTX 4090", supply=1, demand=1, require_catalog=True)
