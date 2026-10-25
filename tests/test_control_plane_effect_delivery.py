"""Scheduler identity must not dead-letter API-owned billing effects."""

import uuid

import pytest

from billing import get_billing_engine
from control_plane.billing_effects import handle_attempt_settlement, handle_wallet_notice
from control_plane.db import run_transaction
from control_plane.db_roles import service_scope
from control_plane.leases import expire_stale_leases
from control_plane.outbox import OutboxEvent, append_event, claim_batch
from control_plane.outbox_runtime import API_EFFECT_DESTINATIONS
from db import _get_pg_pool
from tests.fixtures.native_runtime import running_instance
from tests.test_db_role_workflows import restricted_logins  # noqa: F401
from tests.test_db_service_roles import provisioned_roles  # noqa: F401


def _notice(owner: str) -> OutboxEvent:
    return OutboxEvent(
        event_id=str(uuid.uuid4()),
        aggregate_type="wallet",
        aggregate_id=owner,
        event_type="billing.v1.wallet_notice_requested",
        payload={
            "customer_id": owner,
            "kind": "billing_low_balance",
            "title": "Low balance",
            "body": "Top up",
            "data": {},
        },
        headers={},
        destination_class="billing_notice",
        idempotency_key="wallet_notice:" + owner,
        attempt_count=0,
        created_at=None,
    )


def test_scheduler_claim_does_not_take_api_effects(restricted_logins):
    owner = "effect-claim-" + uuid.uuid4().hex
    event_id = run_transaction(
        lambda conn: append_event(
            conn,
            aggregate_type="wallet",
            aggregate_id=owner,
            event_type="billing.v1.wallet_notice_requested",
            payload=_notice(owner).payload,
            destination_class="billing_notice",
            idempotency_key="wallet_notice:" + owner,
            tenant_id=owner,
        )
    )
    assert event_id
    with service_scope("scheduler"):
        with _get_pg_pool().connection() as conn:
            stolen = claim_batch(
                conn,
                dispatcher_id="scheduler-effect-test",
                include_destination_classes=("billing_notice",),
                exclude_destination_classes=API_EFFECT_DESTINATIONS,
                limit=20,
            )
            assert stolen == []
            conn.rollback()
    with service_scope("scheduler"):
        with pytest.raises(RuntimeError, match="API identity"):
            handle_wallet_notice(_notice(owner))


def test_scheduler_lease_expiry_survives_its_role_and_the_api_settles(
    restricted_logins,
):
    owner = "lease-role-" + uuid.uuid4().hex
    get_billing_engine().deposit(owner, 20)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=120) as native:
        job_id = native["job_id"]
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 hour' "
                "WHERE job_id=%s AND status='active'",
                (job_id,),
            )
        with service_scope("scheduler"):
            expired = run_transaction(
                lambda conn: expire_stale_leases(conn, grace_sec=0, current_job_id=job_id)
            )
        assert expired and expired[0]["job_id"] == job_id
        with _get_pg_pool().connection() as conn:
            open_meter = conn.execute(
                "SELECT completed_at FROM usage_meters WHERE job_id=%s", (job_id,)
            ).fetchone()
            event = conn.execute(
                "SELECT event_id, payload FROM outbox_events "
                "WHERE aggregate_id=%s AND destination_class='attempt_settlement' "
                "AND published_at IS NULL",
                (job_id,),
            ).fetchone()
        assert open_meter is not None and not open_meter[0]
        assert event and event[1]["attempt_id"]
        assert "wallet_hold_id" in event[1]
        handle_attempt_settlement(
            OutboxEvent(
                event_id=str(event[0]),
                aggregate_type="job",
                aggregate_id=job_id,
                event_type="billing.v1.attempt_settlement_requested",
                payload=event[1],
                headers={},
                destination_class="attempt_settlement",
                idempotency_key="attempt_settlement:" + event[1]["attempt_id"],
                attempt_count=1,
                created_at=None,
            )
        )
        with _get_pg_pool().connection() as conn:
            settled = conn.execute(
                "SELECT completed_at FROM usage_meters WHERE job_id=%s", (job_id,)
            ).fetchone()
        assert settled and settled[0]
