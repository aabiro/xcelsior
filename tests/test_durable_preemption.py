"""Plans survive scheduler restarts and transfer only settled GPU authority."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from psycopg.errors import LockNotAvailable
from psycopg.types.json import Jsonb

from control_plane.attempts import report_attempt_status
from control_plane.db import run_transaction
from control_plane.leases import claim_lease, FencingViolation
from control_plane.scheduler.preemption import (
    create_plan,
    reconcile_plan,
    PreemptionWaiting,
)
from control_plane.scheduler.reservation import CapacityConflict
from tests.test_control_plane_reservation import (
    fleet,
    _mkjob,
    _mkhost,
    _claim,
    _reserve,
    _pool,
)  # noqa: F401
from tests.test_durable_fairness import policies, policy  # noqa: F401
from tests.test_serverless_worker_attempt_binding import scratch  # noqa: F401


def running(claimed, host):
    reservation = _reserve(claimed, host)

    def start(conn):
        claim_lease(
            conn,
            lease_id=reservation.lease_id,
            job_id=reservation.job_id,
            attempt_id=reservation.attempt_id,
            host_id=host,
            fencing_token=reservation.fencing_token,
            worker_session_id="preemption-test",
        )
        report_attempt_status(
            conn,
            job_id=reservation.job_id,
            attempt_id=reservation.attempt_id,
            host_id=host,
            fencing_token=reservation.fencing_token,
            status="running",
        )

    run_transaction(start)
    return reservation


def setup(fleet, policies, *, enabled=True, force=False, interruptible=True, nominee_fields=None):
    host = _mkhost(fleet, gpus=2)
    victim = _mkjob(fleet, admission_fields=dict(preemptible=interruptible))
    nominee = _mkjob(fleet, num_gpus=2, owner="nominee-" + host, admission_fields=nominee_fields)
    third = _mkjob(fleet)
    policy(policies, "nominee-" + host, preemption_enabled=enabled, force_after_deadline=force)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET last_observed_at=statement_timestamp(),payload=payload||%s WHERE host_id=%s",
            (
                Jsonb(
                    dict(
                        gpu_model=" ".join([victim, nominee, third]),
                        gpu_count=2,
                        capabilities=dict(egress_isolation=True),
                    )
                ),
                host,
            ),
        )
        conn.execute("UPDATE jobs SET priority=10 WHERE job_id=%s", (nominee,))
    active = running(_claim(victim), host)
    return host, active, _claim(nominee), third


def plan(claimed, host):
    return run_transaction(
        lambda c: create_plan(
            c,
            job_id=claimed.job_id,
            claim_token=claimed.claim_token,
            host_id=host,
            replica_id="preemption-test",
        )
    )


def settle(active, *, intent=None):
    def transaction(conn):
        if intent:
            conn.execute(
                "UPDATE jobs SET desired_state='stopped',lifecycle_intent=%s WHERE job_id=%s",
                (intent, active.job_id),
            )
        return report_attempt_status(
            conn,
            job_id=active.job_id,
            attempt_id=active.attempt_id,
            host_id=active.host_id,
            fencing_token=active.fencing_token,
            status="stopped",
        )

    return run_transaction(transaction)


def test_restart_and_new_reservation_wait_for_exact_terminal_proof(fleet, policies):
    host, active, nominee, third = setup(fleet, policies)
    pid = plan(nominee, host)
    assert pid and plan(nominee, host) == pid
    assert run_transaction(lambda c: reconcile_plan(c, pid)) == "stopping"
    with pytest.raises(PreemptionWaiting):
        _reserve(nominee, host)
    other = _claim(third)
    with pytest.raises(CapacityConflict):
        _reserve(other, host)
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND released_at IS NULL",
            (active.attempt_id,),
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT count(*) FROM preemption_plan_victims WHERE plan_id=%s", (pid,)
        ).fetchone() == (1,)
        generation = conn.execute(
            "SELECT generation FROM jobs WHERE job_id=%s", (active.job_id,)
        ).fetchone()[0]
    settle(active)
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT phase,desired_state,active_attempt_id,generation,reason_code FROM jobs WHERE job_id=%s",
            (active.job_id,),
        ).fetchone() == ("pending", "running", None, generation + 1, "preempted")
    assert run_transaction(lambda c: reconcile_plan(c, pid)) == "ready"
    with pytest.raises(CapacityConflict):
        _reserve(other, host)
    replacement = _reserve(nominee, host)
    assert replacement.fencing_token > active.fencing_token
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT state FROM preemption_plans WHERE plan_id=%s", (pid,)
        ).fetchone() == ("placed",)
        assert conn.execute(
            "SELECT count(*) FROM preemption_plan_devices WHERE plan_id=%s AND held", (pid,)
        ).fetchone() == (0,)
    with pytest.raises(FencingViolation):
        settle(active)


@pytest.mark.parametrize("force", [False, True])
def test_deadline_escalation_requires_policy_and_is_idempotent(fleet, policies, force):
    host, active, nominee, _ = setup(fleet, policies, force=force)
    pid = plan(nominee, host)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE preemption_plans SET grace_deadline=statement_timestamp()-interval '1 second' WHERE plan_id=%s",
            (pid,),
        )
    for _ in range(2):
        assert run_transaction(lambda c: reconcile_plan(c, pid)) == "stopping"
    with _pool.connection() as conn:
        commands = conn.execute(
            "SELECT args FROM agent_commands WHERE attempt_id=%s AND command='stop_attempt'",
            (active.attempt_id,),
        ).fetchall()
        assert len(commands) == 1 + int(force)
        assert sum(r[0].get("force") is True for r in commands) == int(force)
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND released_at IS NULL",
            (active.attempt_id,),
        ).fetchone() == (1,)


@pytest.mark.parametrize("enabled,interruptible", [(False, True), (True, False)])
def test_default_policy_and_protected_workloads_cannot_be_evicted(
    fleet, policies, enabled, interruptible
):
    host, _, nominee, _ = setup(fleet, policies, enabled=enabled, interruptible=interruptible)
    assert plan(nominee, host) is None


def test_user_cancellation_wins_over_preemption_requeue(fleet, policies):
    host, active, nominee, _ = setup(fleet, policies)
    assert plan(nominee, host)
    settle(active, intent="cancel")
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,desired_state FROM jobs WHERE job_id=%s", (active.job_id,)
        ).fetchone() == ("cancelled", "stopped")


def test_cancelled_nomination_releases_holds_without_freeing_victim(fleet, policies):
    host, active, nominee, _ = setup(fleet, policies)
    pid = plan(nominee, host)
    with _pool.connection() as conn:
        conn.execute("UPDATE jobs SET desired_state='stopped' WHERE job_id=%s", (nominee.job_id,))
    assert run_transaction(lambda c: reconcile_plan(c, pid)) == "cancelled"
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM preemption_plan_devices WHERE plan_id=%s AND held", (pid,)
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND released_at IS NULL",
            (active.attempt_id,),
        ).fetchone() == (1,)


def test_two_replicas_create_one_plan_and_one_victim_command(fleet, policies):
    host, active, nominee, _ = setup(fleet, policies)

    def attempt(_):
        try:
            return plan(nominee, host)
        except LockNotAvailable:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, range(2)))
    assert any(results)
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM preemption_plans WHERE nominated_job_id=%s", (nominee.job_id,)
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT count(*) FROM agent_commands WHERE attempt_id=%s AND command='stop_attempt'",
            (active.attempt_id,),
        ).fetchone() == (1,)


def test_scheduler_service_resumes_plan_and_places_only_nominated_host(fleet, policies):
    from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
    from control_plane.scheduler.service import SchedulerService

    host, active, nominee, _ = setup(fleet, policies)
    with _pool.connection() as conn:
        from control_plane.scheduler.claim import release_claim

        release_claim(conn, nominee.job_id, nominee.claim_token, reason_code="test")
    cfg = SchedulerConfig(
        mode=SchedulerMode.CANARY,
        replica_id="plan-service",
        canary_host_ids=frozenset({host}),
        canary_gpu_models=frozenset({nominee.job_id}),
        tick_max_placements=1,
    )
    first = SchedulerService(cfg).tick()
    assert first.released == [(nominee.job_id, "preemption_waiting")]
    assert first.placements == []
    settle(active)
    with _pool.connection() as conn:
        conn.execute("UPDATE jobs SET next_schedule_at=NULL WHERE job_id=%s", (nominee.job_id,))
    second = SchedulerService(cfg).tick()
    assert len(second.placements) == 1
    assert second.placements[0].job_id == nominee.job_id
    assert second.placements[0].host_id == host


def test_current_policy_can_protect_a_running_victim(fleet, policies):
    host, active, nominee, _ = setup(fleet, policies)
    # A later operator policy protects this immutable team identity.
    with _pool.connection() as conn:
        # Internal jobs have isolated keys but cannot receive operator policy;
        # use an explicit policy row to exercise the latest-version join.
        conn.execute(
            """INSERT INTO scheduling_team_policies(team_key,version,
            administrative_class,created_by) VALUES(%s,1,100,'isolated-operator')""",
            ("job:" + active.job_id,),
        )
    policies.append("job:" + active.job_id)
    assert plan(nominee, host) is None


def test_serverless_victim_is_closed_instead_of_requeued(scratch, policies):
    from tests.test_serverless_worker_attempt_binding import (
        _mk_serverless_job,
        _mkhost as serverless_host,
    )
    from serverless.repo import ServerlessRepo, EndpointCreate

    repo = ServerlessRepo()
    endpoint = repo.create_endpoint(
        EndpointCreate(
            owner_id="preempt-test", name="preempt", mode="preset", model_ref="m", min_workers=0
        )
    )
    scratch["endpoints"].append(endpoint["endpoint_id"])
    worker = repo.create_worker(endpoint["endpoint_id"], gpu_count=1)
    scratch["workers"].append(worker["worker_id"])
    # Initial fixture admission records explicit interruptibility.
    victim = _mk_serverless_job(scratch, worker["worker_id"], preemptible=True)
    host = serverless_host(scratch, gpus=1)
    nominee = _mkjob(scratch, owner="serverless-nominee-" + host)
    policy(policies, "serverless-nominee-" + host, preemption_enabled=True)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET last_observed_at=statement_timestamp(),payload=payload||%s WHERE host_id=%s",
            (Jsonb(dict(gpu_model=nominee, capabilities=dict(egress_isolation=True))), host),
        )
        conn.execute("UPDATE jobs SET priority=10 WHERE job_id=%s", (nominee,))
    active = running(_claim(victim), host)
    assert plan(_claim(nominee), host)
    settle(active)
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,desired_state FROM jobs WHERE job_id=%s", (victim,)
        ).fetchone() == ("terminated", "stopped")
        state, released = conn.execute(
            "SELECT state,released_at FROM serverless_workers WHERE worker_id=%s",
            (worker["worker_id"],),
        ).fetchone()
    assert state == "terminated" and released is not None


@pytest.mark.parametrize(
    "limit,allowed",
    [
        (dict(max_concurrent_jobs=1), True),
        (dict(max_allocated_gpus=2), True),
        (dict(max_allocated_gpus=1), False),
    ],
)
def test_same_team_replacement_checks_net_quota_without_early_release(
    fleet, policies, limit, allowed
):
    owner = "replacement-team-" + __import__("uuid").uuid4().hex
    host = _mkhost(fleet, gpus=2)
    victim = _mkjob(fleet, owner=owner, admission_fields=dict(preemptible=True, spot_rate_cad=0))
    nominee = _mkjob(fleet, owner=owner, num_gpus=2)
    policy(policies, owner, preemption_enabled=True, **limit)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET last_observed_at=statement_timestamp(),payload=payload||%s WHERE host_id=%s",
            (
                Jsonb(
                    dict(
                        gpu_model=" ".join([victim, nominee]),
                        gpu_count=2,
                        capabilities=dict(egress_isolation=True),
                    )
                ),
                host,
            ),
        )
        conn.execute("UPDATE jobs SET priority=10 WHERE job_id=%s", (nominee,))
    active = running(_claim(victim), host)
    selected = _claim(nominee)
    pid = plan(selected, host)
    if not allowed:
        assert pid is None
        with _pool.connection() as conn:
            assert conn.execute(
                "SELECT count(*) FROM agent_commands WHERE attempt_id=%s AND command='stop_attempt'",
                (active.attempt_id,),
            ).fetchone() == (0,)
        return
    assert pid
    with pytest.raises(CapacityConflict):
        _reserve(selected, host)
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND released_at IS NULL",
            (active.attempt_id,),
        ).fetchone() == (1,)
    settle(active)
    assert run_transaction(lambda c: reconcile_plan(c, pid)) == "ready"
    placed = _reserve(selected, host)
    assert placed.fencing_token > active.fencing_token


@pytest.mark.parametrize("failure", ["container_exit", "forced_exit"])
def test_preemption_failure_still_readmits_compute_under_a_new_generation(fleet, policies, failure):
    host, active, nominee, _ = setup(fleet, policies, force=True)
    pid = plan(nominee, host)
    with _pool.connection() as conn:
        generation = conn.execute(
            "SELECT generation FROM jobs WHERE job_id=%s", (active.job_id,)
        ).fetchone()[0]
    run_transaction(
        lambda c: report_attempt_status(
            c,
            job_id=active.job_id,
            attempt_id=active.attempt_id,
            host_id=host,
            fencing_token=active.fencing_token,
            status="failed",
            failure_code=failure,
        )
    )
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,phase,generation,active_attempt_id FROM jobs WHERE job_id=%s",
            (active.job_id,),
        ).fetchone() == ("queued", "pending", generation + 1, None)
        assert conn.execute(
            "SELECT status FROM job_attempts WHERE attempt_id=%s", (active.attempt_id,)
        ).fetchone() == ("failed",)
    assert run_transaction(lambda c: reconcile_plan(c, pid)) == "ready"


def test_failed_user_cancel_does_not_inherit_preemption_requeue(fleet, policies):
    host, active, nominee, _ = setup(fleet, policies)
    assert plan(nominee, host)

    def fail(conn):
        conn.execute(
            "UPDATE jobs SET desired_state='stopped',lifecycle_intent='cancel' WHERE job_id=%s",
            (active.job_id,),
        )
        return report_attempt_status(
            conn,
            job_id=active.job_id,
            attempt_id=active.attempt_id,
            host_id=host,
            fencing_token=active.fencing_token,
            status="failed",
        )

    run_transaction(fail)
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,active_attempt_id FROM jobs WHERE job_id=%s", (active.job_id,)
        ).fetchone() == ("failed", None)
