"""Weighted service, quota and replica ordering against actual reservations."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import uuid

import psycopg
import pytest

from control_plane.db import run_transaction
from control_plane.scheduler.claim import claim_next_job, release_claim
from control_plane.scheduler.fairness import TeamPolicy, publish_policy
from control_plane.scheduler.reservation import (
    CapacityConflict,
    TeamQuotaConflict,
    QueuePolicyChanged,
    reserve_and_bind,
)
from tests.test_control_plane_reservation import fleet, _mkjob, _mkhost, _pool  # noqa: F401


def claim(ids):
    with _pool.connection() as conn:
        return claim_next_job(conn, replica_id="fairness-test", scope_gpu_models=ids)


def reserve(claimed, host):
    return run_transaction(
        lambda conn: reserve_and_bind(
            conn,
            job_id=claimed.job_id,
            claim_token=claimed.claim_token,
            replica_id="fairness-test",
            host_id=host,
        )
    )


@pytest.fixture
def policies():
    keys = []
    yield keys
    with _pool.connection() as conn:
        conn.execute("DELETE FROM scheduling_team_policies WHERE team_key=ANY(%s)", (keys,))
        conn.execute("DELETE FROM scheduling_team_state WHERE team_key=ANY(%s)", (keys,))


def policy(keys, owner, **fields):
    key = "owner:" + owner
    keys.append(key)
    with _pool.connection() as conn:
        return publish_policy(conn, TeamPolicy(key=key, **fields), created_by="isolated-operator")


def test_weighted_service_survives_replica_recreation(fleet, policies):
    owner = "weighted-" + uuid.uuid4().hex
    other = "other-" + uuid.uuid4().hex
    policy(policies, owner, weight=Decimal(2))
    policy(policies, other)
    ids = [_mkjob(fleet, owner=owner) for _ in range(4)]
    ids.extend(_mkjob(fleet, owner=other) for _ in range(2))
    host = _mkhost(fleet, gpus=6)
    order = []
    for _ in ids:
        chosen = claim(ids)
        assert chosen
        reserve(chosen, host)
        with _pool.connection() as conn:
            data = conn.execute(
                "SELECT queue_components FROM jobs WHERE job_id=%s", (chosen.job_id,)
            ).fetchone()[0]
        order.append(data["team"])
        assert data["policy"] == "fairness/v1" and data["age_boost"] <= 20
    assert order == [
        "owner:" + owner,
        "owner:" + owner,
        "owner:" + other,
        "owner:" + owner,
        "owner:" + owner,
        "owner:" + other,
    ]
    with _pool.connection() as conn:
        states = dict(
            conn.execute(
                "SELECT team_key,virtual_finish FROM scheduling_team_state WHERE team_key=ANY(%s)",
                (policies,),
            ).fetchall()
        )
    assert states["owner:" + owner] == states["owner:" + other]


def test_failed_capacity_does_not_spend_credit(fleet, policies):
    owner = "failed-" + uuid.uuid4().hex
    policy(policies, owner)
    jid = _mkjob(fleet, owner=owner, num_gpus=2)
    host = _mkhost(fleet, gpus=1)
    selected = claim([jid])
    with _pool.connection() as conn:
        before = conn.execute(
            "SELECT virtual_finish FROM scheduling_team_state WHERE team_key=%s",
            ("owner:" + owner,),
        ).fetchone()
    with pytest.raises(CapacityConflict):
        reserve(selected, host)
    with _pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT virtual_finish FROM scheduling_team_state WHERE team_key=%s",
                ("owner:" + owner,),
            ).fetchone()
            == before
        )
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE job_id=%s", (jid,)
        ).fetchone() == (0,)


@pytest.mark.parametrize("limit", [dict(max_concurrent_jobs=1), dict(max_allocated_gpus=1)])
def test_two_replicas_cannot_overrun_team_limits(fleet, policies, limit):
    owner = "quota-" + uuid.uuid4().hex
    policy(policies, owner, **limit)
    ids = [_mkjob(fleet, owner=owner) for _ in range(2)]
    hosts = [_mkhost(fleet, gpus=1) for _ in ids]
    claims = [claim(ids) for _ in ids]
    assert {c.job_id for c in claims} == set(ids)

    def allocate(pair):
        try:
            return reserve(*pair)
        except TeamQuotaConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(allocate, zip(claims, hosts)))
    assert sum(x is not None for x in results) == 1
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE job_id=ANY(%s) AND released_at IS NULL",
            (ids,),
        ).fetchone() == (1,)


def test_age_is_capped_and_administrative_class_has_precedence(fleet, policies):
    owner = "admin-" + uuid.uuid4().hex
    policy(policies, owner, administrative_class=1)
    fresh = _mkjob(fleet, owner=owner)
    old = _mkjob(fleet)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET queued_at=clock_timestamp()-interval '100 days',priority=979 WHERE job_id=%s",
            (old,),
        )
    got = claim([fresh, old])
    assert got.job_id == fresh
    with _pool.connection() as conn:
        release_claim(conn, got.job_id, got.claim_token, reason_code="test", requeue_delay_sec=3600)
    aged = claim([fresh, old])
    assert aged.job_id == old
    with _pool.connection() as conn:
        values = conn.execute(
            "SELECT queue_components,effective_priority FROM jobs WHERE job_id=%s", (old,)
        ).fetchone()
    assert values[0]["age_boost"] == 20 and values[1] == 999


def test_policy_revision_and_immutable_owner_override_observation(fleet, policies):
    owner = "policy-" + uuid.uuid4().hex
    assert policy(policies, owner, weight=Decimal(1)) == 1
    assert policy(policies, owner, weight=Decimal(3), reservation_entitlement=1) == 2
    jid = _mkjob(fleet, owner=owner)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET owner_id='different',payload=payload||'{\"owner\":\"different\"}' WHERE job_id=%s",
            (jid,),
        )
    assert claim([jid]).job_id == jid
    with _pool.connection() as conn:
        components = conn.execute(
            "SELECT queue_components FROM jobs WHERE job_id=%s", (jid,)
        ).fetchone()[0]
    assert components["team"] == "owner:" + owner
    assert (
        components["revision"] == 2 and components["weight"] == 3 and components["entitlement"] == 1
    )


def test_policy_change_after_claim_requires_a_new_queue_decision(fleet, policies):
    owner = "changed-" + uuid.uuid4().hex
    policy(policies, owner)
    job = _mkjob(fleet, owner=owner)
    host = _mkhost(fleet)
    selected = claim([job])
    policy(policies, owner, weight=Decimal(3))
    with pytest.raises(QueuePolicyChanged):
        reserve(selected, host)
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM job_attempts WHERE job_id=%s", (job,)
        ).fetchone() == (0,)
        release_claim(conn, job, selected.claim_token, reason_code="queue_policy_changed")
    assert reserve(claim([job]), host).job_id == job
