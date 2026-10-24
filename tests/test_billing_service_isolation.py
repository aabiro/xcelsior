"""Real billing login, atomic side-effect intent and fenced API delivery."""

import uuid
from contextlib import ExitStack

import psycopg
from psycopg.errors import InsufficientPrivilege
import pytest

from billing import get_billing_engine
from db import _get_pg_pool
from control_plane.db_roles import service_scope
from control_plane.billing_effects import publish_committed_effects
from tests.test_db_role_workflows import restricted_logins  # noqa: F401
from tests.test_db_service_roles import provisioned_roles  # noqa: F401
from tests.fixtures.native_runtime import running_instance


def test_billing_login_runs_native_periodic_and_terminal_settlement(restricted_logins):
    from tests.test_data_layer_readiness import finish_compute

    owner = "billing-login-" + uuid.uuid4().hex
    billing = get_billing_engine()
    billing.deposit(owner, 4)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=120) as native:
        before = billing.get_wallet(owner)["balance_micros"]
        with service_scope("billing"):
            result = billing.auto_billing_cycle()
            assert result["errors"] == 0
            assert result["billed"] >= 1
            with _get_pg_pool().connection() as conn:
                assert conn.execute("SELECT current_user,session_user").fetchone() == (
                    "xcelsior_billing",
                    "xcelsior_billing",
                )
                assert conn.execute(
                    "SELECT count(*) FROM notifications WHERE user_email=%s", (owner,)
                ).fetchone() == (0,)
                event_id = conn.execute(
                    "SELECT event_id FROM outbox_events WHERE aggregate_id=%s AND destination_class='billing_notice'",
                    (owner,),
                ).fetchone()[0]
        assert billing.get_wallet(owner)["balance_micros"] < before
        publish_committed_effects(notice_ids=["notif-billing-" + str(event_id)])
        publish_committed_effects(notice_ids=["notif-billing-" + str(event_id)])
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT count(*) FROM notifications WHERE user_email=%s", (owner,)
            ).fetchone() == (1,)

        finish_compute(native)
        with service_scope("billing"):
            assert billing.auto_billing_cycle()["errors"] == 0
    with psycopg.connect(restricted_logins["billing"]) as conn:
        for statement in (
            "UPDATE jobs SET status=status",
            "UPDATE volumes SET status=status",
            "UPDATE serverless_workers SET state=state",
            "UPDATE serverless_endpoints SET env=env",
            "UPDATE serverless_endpoints SET total_requests=total_requests",
            "INSERT INTO notifications(id,user_email,type,title) VALUES('forbidden','u','billing','x')",
            "DELETE FROM agent_commands",
        ):
            with pytest.raises(InsufficientPrivilege):
                conn.execute(statement)
            conn.rollback()


def test_notice_outbox_failure_rolls_back_wallet_watermark_and_cycle(
    restricted_logins, monkeypatch
):
    import control_plane.billing_effects as effects

    owner = "billing-rollback-" + uuid.uuid4().hex
    billing = get_billing_engine()
    billing.deposit(owner, 4)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=120) as native:
        before = billing.get_wallet(owner)["balance_micros"]
        monkeypatch.setattr(
            effects,
            "append_event",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("outbox unavailable")),
        )
        with service_scope("billing"):
            assert billing.auto_billing_cycle()["errors"] >= 1
            with _get_pg_pool().connection() as conn:
                assert conn.execute(
                    "SELECT balance_micros,low_balance_warned_at FROM wallets WHERE customer_id=%s",
                    (owner,),
                ).fetchone() == (before, None)
                assert conn.execute(
                    "SELECT count(*) FROM billing_cycles WHERE job_id=%s", (native["job_id"],)
                ).fetchone() == (0,)


def test_committed_notice_is_recovered_by_the_real_api_dispatcher(restricted_logins):
    from control_plane.outbox import OutboxDispatcher
    from control_plane.outbox_runtime import default_handlers

    owner = "billing-delivery-" + uuid.uuid4().hex
    billing = get_billing_engine()
    billing.deposit(owner, 4)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=120):
        with service_scope("billing"):
            assert billing.auto_billing_cycle()["errors"] == 0
        dispatcher = OutboxDispatcher("billing-recovery-" + uuid.uuid4().hex, default_handlers())
        for _ in range(30):
            dispatcher.run_once()
            with _get_pg_pool().connection() as conn:
                delivered = conn.execute(
                    "SELECT published_at FROM outbox_events WHERE aggregate_id=%s AND destination_class='billing_notice'",
                    (owner,),
                ).fetchone()
                if delivered and delivered[0]:
                    assert conn.execute(
                        "SELECT count(*) FROM notifications WHERE user_email=%s", (owner,)
                    ).fetchone() == (1,)
                    break
        else:
            pytest.fail("committed wallet notice did not recover through outbox")


@pytest.mark.parametrize("serverless", [False, True])
def test_uncommitted_terminal_time_cannot_race_financial_settlement(restricted_logins, serverless):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from psycopg.rows import dict_row
    from control_plane.attempts import report_attempt_status
    from serverless.repo import EndpointCreate, ServerlessRepo
    from tests.fixtures.native_serverless import running_serverless

    owner = "billing-terminal-race-" + uuid.uuid4().hex
    billing = get_billing_engine()
    billing.deposit(owner, 100)
    repo = ServerlessRepo()
    with ExitStack() as stack:
        if serverless:
            endpoint = repo.create_endpoint(
                EndpointCreate(
                    owner_id=owner, name="terminal-race", gpu_tier="RTX 4090", min_workers=0
                )
            )
            native = stack.enter_context(running_serverless(repo, endpoint["endpoint_id"]))
            job_id = native["scheduler_job_id"]
        else:
            native = stack.enter_context(
                running_instance(owner=owner, cost_per_hour=3.6, age_sec=120)
            )
            job_id = native["job_id"]
        with _get_pg_pool().connection() as conn:
            conn.row_factory = dict_row
            authority = conn.execute(
                "SELECT a.job_id,a.host_id,a.attempt_id,a.fencing_token FROM jobs j JOIN job_attempts a ON a.attempt_id=j.active_attempt_id WHERE j.job_id=%s",
                (job_id,),
            ).fetchone()
            authority["attempt_id"] = str(authority["attempt_id"])
        ready, release = threading.Event(), threading.Event()

        def terminal():
            with service_scope("api"):
                with _get_pg_pool().connection() as conn:
                    report_attempt_status(conn, **authority, status="succeeded")
                    ready.set()
                    assert release.wait(10)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(terminal)
            try:
                if not ready.wait(10):
                    if future.done():
                        future.result()
                    pytest.fail("terminal transaction did not reach the commit barrier")
                with service_scope("billing"):
                    result = billing.auto_billing_cycle()
                assert result["errors"] == 0
                with _get_pg_pool().connection() as conn:
                    assert conn.execute(
                        "SELECT count(*) FROM billing_cycles WHERE job_id=%s", (job_id,)
                    ).fetchone() == (0,)
            finally:
                release.set()
            future.result(timeout=15)
        with service_scope("billing"):
            assert billing.auto_billing_cycle()["errors"] == 0
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT count(*),bool_and(b.period_end<=EXTRACT(EPOCH FROM a.ended_at)) FROM billing_cycles b JOIN job_attempts a ON a.job_id=b.job_id WHERE b.job_id=%s AND b.status='charged'",
                (job_id,),
            ).fetchone() == (1, True)


def test_billing_login_charges_native_serverless_and_storage(restricted_logins, monkeypatch):
    from serverless.repo import EndpointCreate, ServerlessRepo
    from tests.fixtures.native_serverless import running_serverless
    from volumes import VolumeEngine

    owner = "billing-domains-" + uuid.uuid4().hex
    billing = get_billing_engine()
    billing.deposit(owner, 100)
    repo = ServerlessRepo()
    ep = repo.create_endpoint(
        EndpointCreate(
            owner_id=owner,
            name="billing-role",
            mode="custom",
            image_ref="example/image:v1",
            gpu_tier="RTX 4090",
            min_workers=0,
        )
    )
    engine = VolumeEngine()
    monkeypatch.setattr(engine, "_provision_volume_storage", lambda *a, **kw: True)
    monkeypatch.setattr(engine, "_emit_event", lambda *a, **kw: None)
    volume = engine.create_volume(owner, "billing-role", 100)
    try:
        with ExitStack() as stack:
            stack.enter_context(running_serverless(repo, ep["endpoint_id"]))
            with _get_pg_pool().connection() as conn:
                conn.execute(
                    "UPDATE volumes SET created_at=EXTRACT(EPOCH FROM clock_timestamp())-3600 WHERE volume_id=%s",
                    (volume["volume_id"],),
                )
            with service_scope("billing"):
                result = billing.auto_billing_cycle()
            assert result["errors"] == 0
            assert result["volume_billed"] >= 1
            assert result["inference_billed"] >= 1
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "DELETE FROM volume_attachments WHERE volume_id=%s", (volume["volume_id"],)
            )
            conn.execute("DELETE FROM volumes WHERE volume_id=%s", (volume["volume_id"],))
            conn.execute(
                "DELETE FROM serverless_endpoints WHERE endpoint_id=%s", (ep["endpoint_id"],)
            )


def test_billing_failure_stop_is_durable_and_rechecks_funding(restricted_logins):
    owner = "billing-stop-" + uuid.uuid4().hex
    billing = get_billing_engine()
    billing.deposit(owner, 0.0001)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=120) as native:
        with service_scope("billing"):
            result = billing.auto_billing_cycle()
            assert result["errors"] == 0 and result["suspended"] >= 1
            with _get_pg_pool().connection() as conn:
                event = conn.execute(
                    "SELECT event_id FROM outbox_events WHERE aggregate_id=%s AND destination_class='billing_stop'",
                    (native["job_id"],),
                ).fetchone()
                assert event
                assert conn.execute(
                    "SELECT status FROM jobs WHERE job_id=%s", (native["job_id"],)
                ).fetchone() == ("running",)
        billing.deposit(owner, 10)
        publish_committed_effects(stop_event_ids=[str(event[0])])
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT status FROM jobs WHERE job_id=%s", (native["job_id"],)
            ).fetchone() == ("running",)
            assert conn.execute(
                "SELECT count(*) FROM agent_commands WHERE job_id=%s AND command='stop_attempt'",
                (native["job_id"],),
            ).fetchone() == (0,)


def test_billing_failure_api_delivery_enqueues_one_preserving_stop(restricted_logins):
    owner = "billing-stop-active-" + uuid.uuid4().hex
    billing = get_billing_engine()
    billing.deposit(owner, 0.0001)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=120) as native:
        with service_scope("billing"):
            result = billing.auto_billing_cycle()
            assert result["errors"] == 0
            with _get_pg_pool().connection() as conn:
                event_id = conn.execute(
                    "SELECT event_id FROM outbox_events WHERE aggregate_id=%s AND destination_class='billing_stop'",
                    (native["job_id"],),
                ).fetchone()[0]
        publish_committed_effects(stop_event_ids=[str(event_id)])
        publish_committed_effects(stop_event_ids=[str(event_id)])
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT status FROM jobs WHERE job_id=%s", (native["job_id"],)
            ).fetchone() == ("stopping",)
            assert conn.execute(
                "SELECT count(*) FROM agent_commands WHERE job_id=%s AND command='stop_attempt'",
                (native["job_id"],),
            ).fetchone() == (1,)


def test_exhausted_topup_does_not_stop_a_funded_native_job(restricted_logins):
    owner = "billing-funded-" + uuid.uuid4().hex
    billing = get_billing_engine()
    billing.deposit(owner, 100)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=120) as native:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE wallets SET auto_topup_failures=3,auto_topup_enabled=false WHERE customer_id=%s",
                (owner,),
            )
        billing.stop_jobs_for_suspended_wallets()
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT status FROM jobs WHERE job_id=%s", (native["job_id"],)
            ).fetchone() == ("running",)


def test_suspension_sweep_rechecks_funding_after_its_snapshot(restricted_logins, monkeypatch):
    owner = "billing-refunded-race-" + uuid.uuid4().hex
    billing = get_billing_engine()
    billing.deposit(owner, 100)
    with running_instance(owner=owner, cost_per_hour=3.6, age_sec=120) as native:
        with _get_pg_pool().connection() as conn:
            conn.execute("UPDATE wallets SET status='suspended' WHERE customer_id=%s", (owner,))
        original = billing.stop_instance
        outcomes = []

        def refund_before_stop(job_id, *args, **kwargs):
            if job_id == native["job_id"]:
                billing.deposit(owner, 1)
                assert billing.reactivate_wallet(owner)["status"] == "active"
                outcomes.append(original(job_id, *args, **kwargs))
                return outcomes[-1]
            return original(job_id, *args, **kwargs)

        monkeypatch.setattr(billing, "stop_instance", refund_before_stop)
        billing.stop_jobs_for_suspended_wallets()
        assert outcomes and outcomes[0]["reason"] == "billing_stop_no_longer_required"
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT status FROM jobs WHERE job_id=%s", (native["job_id"],)
            ).fetchone() == ("running",)
            assert conn.execute(
                "SELECT count(*) FROM agent_commands WHERE job_id=%s AND command='stop_attempt'",
                (native["job_id"],),
            ).fetchone() == (0,)
