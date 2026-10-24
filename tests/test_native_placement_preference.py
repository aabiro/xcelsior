"""Admitted constraints cannot disappear between preview, claim and binding."""

import time

import pytest
from psycopg.types.json import Jsonb

from control_plane.db import run_transaction
from control_plane.scheduler.reservation import reserve_and_bind, HostNotEligible
from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.service import SchedulerService
from control_plane.scheduler.native_preference import choose_preferred
from tests.test_control_plane_reservation import fleet, _mkjob, _mkhost, _claim, _pool  # noqa: F401


def setup(fleet, preference):
    jid = _mkjob(fleet, admission_fields=dict(placement_preference=preference))
    hosts = [_mkhost(fleet, gpus=1) for _ in range(2)]
    with _pool.connection() as conn:
        for i, hid in enumerate(hosts):
            conn.execute(
                "UPDATE hosts SET payload=payload||%s,last_observed_at=statement_timestamp() WHERE host_id=%s",
                (
                    Jsonb(
                        dict(
                            gpu_model=jid,
                            gpu_count=1,
                            cost_per_hour=1 + i,
                            last_seen=time.time(),
                            free_vram_gb=24,
                            capabilities=dict(egress_isolation=True),
                        )
                    ),
                    hid,
                ),
            )
    return jid, hosts


def verify(hid, *, state="verified"):
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO host_verifications(host_id,verification_id,state,last_check_at)
            VALUES(%s,%s,%s,%s) ON CONFLICT(host_id) DO UPDATE SET state=EXCLUDED.state,last_check_at=EXCLUDED.last_check_at""",
            (hid, "verify-" + hid, state, time.time()),
        )


def reserve(claimed, hid):
    return run_transaction(
        lambda c: reserve_and_bind(
            c,
            job_id=claimed.job_id,
            claim_token=claimed.claim_token,
            host_id=hid,
            replica_id="preference-test",
        ),
        isolation_level="SERIALIZABLE",
    )


def tick(jid, hosts):
    return SchedulerService(
        SchedulerConfig(
            mode=SchedulerMode.CANARY,
            replica_id="preference-test",
            canary_gpu_models=frozenset({jid}),
            canary_host_ids=frozenset(hosts),
            tick_max_placements=1,
        )
    ).tick()


def test_verification_constraint_selects_verified_host_and_records_evidence(fleet):
    jid, hosts = setup(fleet, dict(require_verified=True))
    verify(hosts[1])
    report = tick(jid, hosts)
    assert len(report.placements) == 1 and report.placements[0].host_id == hosts[1]
    with _pool.connection() as conn:
        evidence = conn.execute(
            "SELECT placement_explanation FROM job_attempts WHERE job_id=%s", (jid,)
        ).fetchone()[0]["placement_preference"]
    assert evidence["host_id"] == hosts[1]
    assert evidence["evidence"]["verification_state"] == "verified"


def test_unsatisfiable_constraint_never_places_and_payload_cannot_remove_it(fleet):
    jid, hosts = setup(fleet, dict(require_verified=True))
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=payload-'placement_preference' WHERE job_id=%s", (jid,)
        )
    report = tick(jid, hosts)
    assert report.placements == []
    assert report.released[0][1] == "placement_preference_verification_unsatisfiable"
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM job_attempts WHERE job_id=%s", (jid,)
        ).fetchone() == (0,)


def test_premium_is_relative_to_cheapest_hard_eligible_host(fleet):
    jid, hosts = setup(fleet, dict(require_verified=True, max_premium_pct=50))
    verify(hosts[1])
    report = tick(jid, hosts)
    assert report.placements == [] and report.released == [
        (jid, "placement_preference_premium_exceeded")
    ]
    with _pool.connection() as conn:
        detail = conn.execute("SELECT reason_details FROM jobs WHERE job_id=%s", (jid,)).fetchone()[
            0
        ]
    assert detail["asked"] == 50 and detail["best_available"] == 100


def test_missing_reliability_history_refuses_clearly(fleet):
    jid, hosts = setup(fleet, dict(min_uptime_pct=99.5))
    report = tick(jid, hosts)
    assert report.placements == [] and report.released == [
        (jid, "placement_preference_insufficient_history")
    ]


def test_revocation_after_claim_is_revalidated_before_binding(fleet):
    jid, hosts = setup(fleet, dict(require_verified=True))
    verify(hosts[1])
    selected = _claim(jid)
    verify(hosts[1], state="deverified")
    with pytest.raises(HostNotEligible, match="preference refused"):
        reserve(selected, hosts[1])
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE job_id=%s", (jid,)
        ).fetchone() == (0,)


def test_price_change_after_claim_revalidates_the_premium_baseline(fleet):
    jid, hosts = setup(fleet, dict(require_verified=True, max_premium_pct=100))
    verify(hosts[1])
    selected = _claim(jid)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET payload=payload||%s WHERE host_id=%s",
            (Jsonb(dict(cost_per_hour=0.5)), hosts[0]),
        )
    with pytest.raises(HostNotEligible, match="preference refused"):
        reserve(selected, hosts[1])


def test_raw_reservation_cannot_bypass_the_transactional_preference_gate(fleet):
    jid, hosts = setup(fleet, dict(require_verified=True))
    verify(hosts[1])
    selected = _claim(jid)
    with pytest.raises(HostNotEligible, match="serializable"):
        run_transaction(
            lambda c: reserve_and_bind(
                c,
                job_id=jid,
                claim_token=selected.claim_token,
                host_id=hosts[1],
                replica_id="unqualified-transaction",
            )
        )


def test_preemption_cannot_disrupt_for_an_unsatisfied_constraint(fleet, policies):
    from tests.test_durable_preemption import setup as preemption_setup
    from control_plane.scheduler.preemption import create_plan

    host, active, nominee, _ = preemption_setup(
        fleet, policies, nominee_fields=dict(placement_preference=dict(require_verified=True))
    )
    with pytest.raises(HostNotEligible):
        run_transaction(
            lambda c: create_plan(
                c,
                job_id=nominee.job_id,
                claim_token=nominee.claim_token,
                host_id=host,
                replica_id="preference-preemption",
            ),
            isolation_level="SERIALIZABLE",
        )
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM agent_commands WHERE attempt_id=%s AND command='stop_attempt'",
            (active.attempt_id,),
        ).fetchone() == (0,)


from tests.test_durable_fairness import policies  # noqa: F401
