"""Paid tier contract across imported Prices, holds, placement and exact billing.

Uses isolated per-test GPU variants and real PostgreSQL; no Stripe calls and no
real payments. Price snapshots are append-only, so fixtures deactivate them.
"""

from __future__ import annotations

import importlib.util
import time
import uuid
from pathlib import Path

import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from billing import get_billing_engine, resolve_compute_rate_cad
from db import _get_pg_pool
from service_tier_pricing import (
    TierFundingRequired,
    TierPricingUnavailable,
    quote_tier,
    stripe_amount_micros,
    usage_charge_micros,
)
from control_plane.attempts import report_attempt_status
from control_plane.db import run_transaction
from control_plane.leases import claim_lease
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import CapacityConflict, HostNotEligible, reserve_and_bind


@pytest.fixture
def paid_fleet(monkeypatch):
    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "active")
    marker = uuid.uuid4().hex[:12]
    model = f"Tier GPU {marker}"
    owner = f"tier-{marker}"
    host_id = f"tier-host-{marker}"
    prices = []
    for tier, cents in (
        ("standard", "49"),
        ("premium", "78.4"),
        ("dedicated", "137.2"),
        ("gpu-exclusive", "19.6"),
        ("queue-bid", "1"),
    ):
        prices.append(
            {
                "variant": {
                    "gpu_model": model,
                    "vram_gb": 24,
                    "form_factor": "pcie",
                    "high_frequency": False,
                },
                "mode": "all" if tier == "queue-bid" else "on_demand",
                "tier": tier,
                "price": {
                    "id": f"price_tier_{marker}_{tier}",
                    "product": f"prod_tier_{marker}",
                    "currency": "cad",
                    "billing_scheme": "per_unit",
                    "unit_amount_decimal": cents,
                    "livemode": False,
                    "lookup_key": (
                        f"tier:{marker}:{tier}:all"
                        if tier == "queue-bid"
                        else f"tier:{marker}:{tier}:on_demand"
                    ),
                },
            }
        )
    script = Path(__file__).resolve().parents[1] / "api/scripts/sync_stripe_service_tiers.py"
    spec = importlib.util.spec_from_file_location("tier_catalog_import", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with _get_pg_pool().connection() as conn:
        module.import_catalog(conn, {"catalog_version": f"test:{marker}", "prices": prices})
        payload = {
            "host_id": host_id,
            "gpu_model": model,
            "gpu_count": 2,
            "cost_per_hour": 99,
            "admitted": True,
            "capabilities": {
                "gpu_uuid_binding": True,
                "gpu_exclusivity": True,
                "dedicated_host": True,
            },
            "last_seen": time.time(),
        }
        conn.execute(
            "INSERT INTO hosts(host_id,status,registered_at,payload,capabilities,last_observed_at,admission_state,administrative_state,availability_state) VALUES(%s,'active',%s,%s,%s,clock_timestamp(),'admitted','admitted','ready')",
            (
                host_id,
                time.time(),
                Jsonb(payload),
                Jsonb({**payload["capabilities"], "egress_isolation": True}),
            ),
        )
        for i in range(2):
            conn.execute(
                "INSERT INTO host_gpu_devices(host_id,gpu_uuid,device_index,model,total_vram_mb,allocatable_vram_mb,health) VALUES(%s,%s,%s,%s,24576,24576,'healthy')",
                (host_id, f"GPU-{marker}-{i}", i, model),
            )
    get_billing_engine().deposit(owner, 50, description="isolated service-tier test")
    fleet = {"model": model, "owner": owner, "host": host_id, "jobs": [], "prices": prices}
    yield fleet
    with _get_pg_pool().connection() as conn:
        for jid in fleet["jobs"]:
            conn.execute("DELETE FROM queue_bid_events WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM usage_meters WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
        conn.execute("DELETE FROM gpu_pricing WHERE gpu_model=%s", (model,))
        conn.execute(
            "UPDATE stripe_compute_prices SET active=FALSE WHERE stripe_product_id=%s",
            (f"prod_tier_{marker}",),
        )
        conn.execute("DELETE FROM wallet_holds WHERE customer_id=%s", (owner,))
        conn.execute("DELETE FROM wallet_transactions WHERE customer_id=%s", (owner,))
        conn.execute("DELETE FROM wallets WHERE customer_id=%s", (owner,))


def submit(fleet, tier="standard", **kw):
    from scheduler import submit_job

    job = submit_job(
        f"tier-{uuid.uuid4().hex[:8]}",
        8,
        10,
        gpu_model=fleet["model"],
        owner=fleet["owner"],
        tier=tier,
        **kw,
    )
    fleet["jobs"].append(job["job_id"])
    return job


def claim(fleet):
    return run_transaction(
        lambda conn: claim_next_job(conn, replica_id="paid-test", scope_gpu_models=[fleet["model"]])
    )


def reserve(fleet, claimed, count=1):
    return run_transaction(
        lambda conn: reserve_and_bind(
            conn,
            job_id=claimed.job_id,
            claim_token=claimed.claim_token,
            host_id=fleet["host"],
            replica_id="paid-test",
            num_gpus=count,
        )
    )


def start(fleet, reservation):
    args = dict(
        job_id=reservation.job_id,
        attempt_id=reservation.attempt_id,
        host_id=fleet["host"],
        fencing_token=reservation.fencing_token,
    )
    grant = run_transaction(
        lambda conn: claim_lease(
            conn, lease_id=reservation.lease_id, worker_session_id="tier-test", **args
        )
    )
    run_transaction(lambda conn: report_attempt_status(conn, status="running", **args))
    return grant, args


def test_exact_prices_pin_both_components_and_refuse_changed_ids(paid_fleet):
    model = paid_fleet["model"]
    assert quote_tier(model)["total_rate_micros"] == 490000
    assert quote_tier(model, tier="premium", num_gpus=2)["total_rate_micros"] == 1568000
    dedicated = quote_tier(model, tier="dedicated", num_gpus=2, gpu_exclusive=True)
    assert dedicated["total_rate_micros"] == 2744000 and dedicated["gpu_exclusive_rate_micros"] == 0
    addon = quote_tier(model, tier="premium", gpu_exclusive=True)
    assert addon["total_rate_micros"] == 980000
    with pytest.raises(TierPricingUnavailable, match="changed"):
        quote_tier(model, price_id="price_stale")
    with pytest.raises(TierPricingUnavailable, match="changed"):
        quote_tier(model, gpu_exclusive=True, gpu_exclusive_price_id="price_stale")
    with pytest.raises(ValueError, match="Choose"):
        quote_tier(model, tier="urgent")


def test_paid_priority_then_fifo_and_no_raw_priority_boost(paid_fleet):
    standard = submit(paid_fleet)
    first = submit(paid_fleet, "premium")
    second = submit(paid_fleet, "premium")
    dedicated = submit(paid_fleet, "dedicated", num_gpus=2)
    assert [claim(paid_fleet).job_id for _ in range(4)] == [
        dedicated["job_id"],
        first["job_id"],
        second["job_id"],
        standard["job_id"],
    ]
    assert [standard["priority"], first["priority"], dedicated["priority"]] == [1, 2, 3]
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT spec->'service_tier_quote'->>'stripe_price_id',wallet_hold_id,(spec->>'host_exclusive')::boolean FROM jobs WHERE job_id=%s",
            (dedicated["job_id"],),
        ).fetchone()
        hold = conn.execute(
            "SELECT amount_micros FROM wallet_holds WHERE hold_id=%s", (row[1],)
        ).fetchone()
    assert row[0] == dedicated["service_tier_quote"]["stripe_price_id"] and row[2] is True
    assert hold[0] == 2744000


def test_unfunded_or_expired_priority_cannot_enter_placement(paid_fleet):
    from scheduler import submit_job

    with pytest.raises(TierFundingRequired):
        submit_job(
            "unfunded-premium",
            8,
            gpu_model=paid_fleet["model"],
            owner=f"empty-{uuid.uuid4().hex}",
            tier="premium",
        )
    job = submit(paid_fleet, "premium")
    with _get_pg_pool().connection() as conn:
        conn.execute("UPDATE wallet_holds SET expires_at=0 WHERE job_id=%s", (job["job_id"],))
    assert claim(paid_fleet) is None


def test_billing_renews_expired_queue_funding_without_repricing_or_reordering(paid_fleet):
    from service_tier_pricing import refresh_pending_tier_funding

    job = submit(paid_fleet, "premium")
    with _get_pg_pool().connection() as conn:
        queued = conn.execute(
            "SELECT queued_at FROM jobs WHERE job_id=%s", (job["job_id"],)
        ).fetchone()[0]
        conn.execute("UPDATE wallet_holds SET expires_at=0 WHERE job_id=%s", (job["job_id"],))
        conn.execute(
            "UPDATE stripe_compute_prices SET active=FALSE WHERE stripe_price_id=%s",
            (job["service_tier_price_id"],),
        )
    refresh_pending_tier_funding(scope_gpu_models=[paid_fleet["model"]])
    assert claim(paid_fleet).job_id == job["job_id"]
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT j.queued_at,h.amount_micros FROM jobs j JOIN wallet_holds h ON h.hold_id=j.wallet_hold_id WHERE j.job_id=%s",
            (job["job_id"],),
        ).fetchone()
    assert row[0] == queued and row[1] == 784000


def test_dedicated_reserves_every_gpu_and_blocks_other_tenants(paid_fleet):
    submit(paid_fleet, "dedicated", num_gpus=2)
    reservation = reserve(paid_fleet, claim(paid_fleet), 2)
    grant, _ = start(paid_fleet, reservation)
    assert len(grant.gpu_device_uuids) == 2
    assert set(grant.gpu_device_uuids) == {
        f"GPU-{paid_fleet['host'].removeprefix('tier-host-')}-{i}" for i in range(2)
    }
    submit(paid_fleet)
    with pytest.raises(CapacityConflict, match="exclusively"):
        reserve(paid_fleet, claim(paid_fleet))


def test_dedicated_refuses_partial_or_already_occupied_host(paid_fleet):
    submit(paid_fleet, "dedicated", num_gpus=1)
    with pytest.raises(CapacityConflict, match="every GPU"):
        reserve(paid_fleet, claim(paid_fleet))
    submit(paid_fleet)
    reserve(paid_fleet, claim(paid_fleet))
    submit(paid_fleet, "dedicated", num_gpus=2)
    with pytest.raises(CapacityConflict, match="occupied"):
        reserve(paid_fleet, claim(paid_fleet), 2)


def test_gpu_only_allows_a_different_gpu_on_same_host(paid_fleet):
    submit(paid_fleet, exclusive_gpu=True)
    one = reserve(paid_fleet, claim(paid_fleet))
    submit(paid_fleet)
    two = reserve(paid_fleet, claim(paid_fleet))
    assert set(one.gpu_device_ids).isdisjoint(two.gpu_device_ids)


def test_unqualified_host_is_refused_again_under_reservation_lock(paid_fleet):
    submit(paid_fleet, exclusive_gpu=True)
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE hosts SET capabilities=jsonb_set(capabilities,'{gpu_exclusivity}','false') WHERE host_id=%s",
            (paid_fleet["host"],),
        )
    with pytest.raises(HostNotEligible, match="not qualified"):
        reserve(paid_fleet, claim(paid_fleet))


def test_worker_observation_replaces_slots_with_physical_inventory_before_paid_execution(
    paid_fleet, monkeypatch
):
    from api import app
    from fastapi.testclient import TestClient
    import routes.agent_v2 as routes
    from control_plane.physical_inventory import sync_physical_inventory

    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM host_gpu_devices WHERE host_id=%s", (paid_fleet["host"],))
    submit(paid_fleet, "dedicated", num_gpus=2)
    claimed = claim(paid_fleet)
    with pytest.raises(CapacityConflict):
        reserve(paid_fleet, claimed, 2)
    monkeypatch.setattr(routes, "_require_agent_auth", lambda request, host_id: None)
    inventory = [
        {
            "gpu_uuid": f"GPU-observed-{i}",
            "device_index": i,
            "model": paid_fleet["model"],
            "total_vram_mb": 24576,
            "free_vram_mb": 24576,
        }
        for i in range(2)
    ]
    response = TestClient(app).post(
        "/agent/v2/observations",
        json={
            "host_id": paid_fleet["host"],
            "worker_session_id": "physical-inventory-test",
            "observation_generation": 1,
            "workloads": [],
            "gpu_inventory": inventory,
            "capabilities": {
                "egress_isolation": True,
                "gpu_uuid_binding": True,
                "gpu_exclusivity": True,
                "dedicated_host": True,
            },
        },
    )
    assert response.status_code == 200, response.text
    assert (
        run_transaction(
            lambda conn: sync_physical_inventory(
                conn, host_id=paid_fleet["host"], inventory=inventory
            )
        )
        is False
    )
    reservation = reserve(paid_fleet, claimed, 2)
    grant, _ = start(paid_fleet, reservation)
    # Exactly the observed physical devices. Their order follows the random
    # gpu_device_id the lease sorts by, not device_index.
    assert sorted(grant.gpu_device_uuids) == ["GPU-observed-0", "GPU-observed-1"]
    with _get_pg_pool().connection() as conn:
        devices = conn.execute(
            "SELECT total_vram_mb FROM host_gpu_devices WHERE host_id=%s AND retired_at IS NULL",
            (paid_fleet["host"],),
        ).fetchall()
    assert len(devices) == 2 and all(row[0] == 24576 for row in devices)


def test_unreadable_observed_inventory_removes_paid_isolation_capability(paid_fleet):
    from control_plane.observations import ingest_observation

    run_transaction(
        lambda conn: ingest_observation(
            conn,
            host_id=paid_fleet["host"],
            session_id="empty-driver",
            observation_generation=1,
            workloads=[],
            gpu_inventory=[],
            capabilities={"gpu_uuid_binding": True, "dedicated_host": True},
        )
    )
    submit(paid_fleet, "dedicated", num_gpus=2)
    with pytest.raises(HostNotEligible, match="availability|qualified"):
        reserve(paid_fleet, claim(paid_fleet), 2)


def test_snapshot_and_wallet_cycle_keep_exact_price_after_catalog_retirement(paid_fleet):
    job = submit(paid_fleet, "premium")
    reservation = reserve(paid_fleet, claim(paid_fleet))
    _, args = start(paid_fleet, reservation)
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET started_at=clock_timestamp()-make_interval(secs=>%s) WHERE attempt_id=%s",
            (61, reservation.attempt_id),
        )
        conn.row_factory = dict_row
        snapshot = conn.execute(
            "SELECT placement_explanation->'billing_snapshot' AS s FROM job_attempts WHERE attempt_id=%s",
            (reservation.attempt_id,),
        ).fetchone()["s"]
        started = float(
            conn.execute(
                "SELECT EXTRACT(EPOCH FROM started_at) FROM job_attempts WHERE attempt_id=%s",
                (reservation.attempt_id,),
            ).fetchone()["extract"]
        )
        conn.execute(
            "UPDATE stripe_compute_prices SET active=FALSE WHERE stripe_price_id=%s",
            (job["service_tier_price_id"],),
        )
        conn.execute(
            "UPDATE hosts SET payload=jsonb_set(payload,'{cost_per_hour}','999') WHERE host_id=%s",
            (paid_fleet["host"],),
        )
    assert resolve_compute_rate_cad(snapshot, {"cost_per_hour": 999})[0] == 0.784
    engine = get_billing_engine()
    before = engine.get_wallet(paid_fleet["owner"])["balance_cad"]
    result = engine.bill_running_period(job["job_id"], period_end=started + 61)
    assert result["billed"] is True, result
    expected = usage_charge_micros(784000, 61)
    with _get_pg_pool().connection() as conn:
        cycle = conn.execute(
            "SELECT amount_micros,pricing_snapshot FROM billing_cycles WHERE job_id=%s",
            (job["job_id"],),
        ).fetchone()
        ledger = conn.execute(
            "SELECT amount_micros FROM wallet_transactions WHERE job_id=%s AND tx_type='charge'",
            (job["job_id"],),
        ).fetchone()
    assert cycle[0] == expected == -ledger[0]
    assert cycle[1]["stripe_price_id"] == job["service_tier_price_id"]
    with _get_pg_pool().connection() as conn:
        event = conn.execute(
            "SELECT value FROM stripe_meter_event_outbox WHERE customer_id=%s AND payload_json->>'job_id'=%s",
            (paid_fleet["owner"], job["job_id"]),
        ).fetchone()
    assert event[0] == pytest.approx(61 / 3600)
    assert (
        round((before - engine.get_wallet(paid_fleet["owner"])["balance_cad"]) * 1000000)
        == expected
    )
    # Rollback must undo both money and the cycle; retry must then charge once.
    with pytest.raises(RuntimeError):
        with _get_pg_pool().connection() as conn:
            engine._bill_gpu_period(
                conn,
                {**job, "host_id": paid_fleet["host"]},
                started + 61,
                started + 122,
                min_seconds=0,
            )
            raise RuntimeError("injected cycle transaction failure")
    assert (
        round((before - engine.get_wallet(paid_fleet["owner"])["balance_cad"]) * 1000000)
        == expected
    )
    run_transaction(lambda conn: report_attempt_status(conn, status="succeeded", **args))
    with _get_pg_pool().connection() as conn:
        meter = conn.execute(
            "SELECT pricing_snapshot FROM usage_meters WHERE attempt_id=%s",
            (reservation.attempt_id,),
        ).fetchone()
    assert meter[0]["stripe_price_id"] == job["service_tier_price_id"]


def test_stripe_price_identity_is_immutable_in_the_database(paid_fleet):
    import psycopg

    with pytest.raises(psycopg.errors.RestrictViolation):
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE stripe_compute_prices SET unit_amount_micros=1 WHERE stripe_price_id=%s",
                (paid_fleet["prices"][0]["price"]["id"],),
            )


def test_multi_gpu_invoice_preserves_service_tier_and_stripe_identities(paid_fleet):
    engine = get_billing_engine()
    quote = quote_tier(paid_fleet["model"], tier="premium", num_gpus=2, gpu_exclusive=True)
    now = time.time()
    jid = f"invoice-tier-{uuid.uuid4().hex[:8]}"
    paid_fleet["jobs"].append(jid)
    meter = engine.meter_job(
        {
            "job_id": jid,
            "owner": paid_fleet["owner"],
            "started_at": now - 3600,
            "completed_at": now,
            "service_tier_quote": quote,
            "num_gpus": 2,
        },
        {"host_id": paid_fleet["host"], "gpu_model": paid_fleet["model"], "cost_per_hour": 999},
    )
    assert meter.gpu_seconds == 7200
    assert meter.total_cost_cad == 1.96
    invoice = engine.generate_invoice(
        paid_fleet["owner"], "Tier test", now - 3601, now + 1, tax_rate=0, persist=False
    )
    lines = invoice.line_items
    line = next(row for row in lines if row["line_type"] == "compute")
    assert line["service_tier"] == "premium"
    assert line["quantity"] == 2
    assert line["subtotal_cad"] == 1.96
    assert line["stripe_price_id"] == quote["stripe_price_id"]
    assert line["gpu_exclusive_price_id"] == quote["gpu_exclusive_price_id"]


def _wait_in_queue(job, seconds):
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET queue_entered_at=clock_timestamp()-make_interval(secs=>%s) WHERE job_id=%s",
            (seconds, job["job_id"]),
        )


def test_native_premium_is_ahead_of_every_promoted_standard_bid(paid_fleet):
    promoted = submit(paid_fleet, queue_bid_units=40)
    _wait_in_queue(promoted, 7200)
    premium = submit(paid_fleet, "premium")
    paid_claim = claim(paid_fleet)
    assert paid_claim.job_id == premium["job_id"]
    reserve(paid_fleet, paid_claim)
    claimed = claim(paid_fleet)
    assert claimed.job_id == promoted["job_id"] and claimed.effective_priority == 2
    with _get_pg_pool().connection() as conn:
        log = conn.execute(
            "SELECT effective_tier,bid_micros,aging_credit_micros,ordering_snapshot FROM queue_bid_events WHERE job_id=%s AND event_type='claimed'",
            (promoted["job_id"],),
        ).fetchone()
    assert log[0] == 2 and log[1] == 400000 and log[2] == 320000
    assert log[3]["promoted"] is True
    assert promoted["tier"] == "standard" and promoted["host_exclusive"] is False


def test_a_second_scheduler_cannot_pass_a_locked_paying_premium_job(paid_fleet):
    promoted = submit(paid_fleet, queue_bid_units=40)
    native = submit(paid_fleet, "premium")
    with _get_pg_pool().connection() as first:
        with _get_pg_pool().connection() as second:
            claimed = claim_next_job(
                first, replica_id="native-first", scope_gpu_models=[paid_fleet["model"]]
            )
            assert claimed.job_id == native["job_id"]
            assert (
                claim_next_job(
                    second, replica_id="promoted-second", scope_gpu_models=[paid_fleet["model"]]
                )
                is None
            )
            first.commit()
            second.commit()
    # An in-flight paid claim stays protected until placement or backoff.
    assert claim(paid_fleet) is None
    reserve(paid_fleet, claimed)
    assert claim(paid_fleet).job_id == promoted["job_id"]


def test_native_job_in_capacity_backoff_does_not_block_usable_capacity(paid_fleet):
    promoted = submit(paid_fleet, queue_bid_units=40)
    native = submit(paid_fleet, "premium")
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET next_schedule_at=clock_timestamp()+interval '1 minute' WHERE job_id=%s",
            (native["job_id"],),
        )
    assert claim(paid_fleet).job_id == promoted["job_id"]


def test_promoted_jobs_use_promotion_fifo_instead_of_bid_rank(paid_fleet):
    # .20 reaches the .294 Premium gap after five 15-minute intervals.
    first = submit(paid_fleet, queue_bid_units=20)
    _wait_in_queue(first, 6 * 900 + 10)
    # .30 promotes immediately, but its qualification time is later.
    second = submit(paid_fleet, queue_bid_units=30)
    _wait_in_queue(second, 200)
    assert claim(paid_fleet).job_id == first["job_id"]
    assert claim(paid_fleet).job_id == second["job_id"]


def test_paid_dedicated_is_protected_and_promotion_never_buys_host_exclusivity(paid_fleet):
    promoted = submit(paid_fleet, "premium", queue_bid_units=100)
    _wait_in_queue(promoted, 10 * 900)
    dedicated = submit(paid_fleet, "dedicated", num_gpus=2)
    paid_claim = claim(paid_fleet)
    assert paid_claim.job_id == dedicated["job_id"]
    # Model a completed placement decision that cannot currently use capacity;
    # an in-flight native claim remains protected until that decision finishes.
    from control_plane.scheduler.claim import release_claim

    run_transaction(
        lambda conn: release_claim(
            conn,
            paid_claim.job_id,
            paid_claim.claim_token,
            reason_code="no_capacity",
            requeue_delay_sec=60,
        )
    )
    claimed = claim(paid_fleet)
    assert claimed.job_id == promoted["job_id"] and claimed.effective_priority == 3
    reservation = reserve(paid_fleet, claimed)
    assert len(reservation.gpu_device_ids) == 1
    with _get_pg_pool().connection() as conn:
        scope = conn.execute(
            "SELECT host_exclusive FROM job_attempts WHERE attempt_id=%s", (reservation.attempt_id,)
        ).fetchone()[0]
    assert scope is False


def test_aging_changes_order_only_after_full_fifteen_minute_intervals(paid_fleet):
    aged = submit(paid_fleet, queue_bid_units=20)
    _wait_in_queue(aged, 4 * 900 + 800)
    from control_plane.queue_bids import ordering

    assert ordering(aged["service_tier_quote"], 4 * 900 + 899)["promoted"] is False
    assert ordering(aged["service_tier_quote"], 5 * 900)["promoted"] is True
    newer_larger_bid = submit(paid_fleet, queue_bid_units=29)
    assert claim(paid_fleet).job_id == newer_larger_bid["job_id"]
    # It remains in Standard at four whole aging intervals.
    assert claim(paid_fleet).effective_priority == 1


def test_bid_rank_then_fifo_and_zero_bid_has_no_monetary_aging(paid_fleet):
    zero = submit(paid_fleet)
    _wait_in_queue(zero, 100000)
    one = submit(paid_fleet, queue_bid_units=10)
    _wait_in_queue(one, 2)
    two = submit(paid_fleet, queue_bid_units=10)
    assert [claim(paid_fleet).job_id for _ in range(3)] == [
        one["job_id"],
        two["job_id"],
        zero["job_id"],
    ]


def test_bid_is_funded_and_metered_but_free_aging_credit_is_not_charged(paid_fleet):
    job = submit(paid_fleet, "premium", queue_bid_units=20)
    quote = job["service_tier_quote"]
    assert quote["total_rate_micros"] == 984000
    _wait_in_queue(job, 100 * 900)
    claimed = claim(paid_fleet)
    assert claimed.effective_priority == 3
    reservation = reserve(paid_fleet, claimed)
    start(paid_fleet, reservation)
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET started_at=clock_timestamp()-make_interval(secs=>%s) WHERE attempt_id=%s",
            (3600, reservation.attempt_id),
        )
        started = float(
            conn.execute(
                "SELECT EXTRACT(EPOCH FROM started_at) FROM job_attempts WHERE attempt_id=%s",
                (reservation.attempt_id,),
            ).fetchone()[0]
        )
        hold = conn.execute(
            "SELECT amount_micros FROM wallet_holds WHERE job_id=%s AND status='held'",
            (job["job_id"],),
        ).fetchone()[0]
    assert hold == 984000
    result = get_billing_engine().bill_running_period(job["job_id"], period_end=started + 3600)
    assert result["amount_cad"] == 0.984
    with _get_pg_pool().connection() as conn:
        events = conn.execute(
            "SELECT event_type,bid_price_id,bid_units,policy_version FROM queue_bid_events WHERE job_id=%s ORDER BY recorded_at",
            (job["job_id"],),
        ).fetchall()
        cycle = conn.execute(
            "SELECT pricing_snapshot FROM billing_cycles WHERE job_id=%s", (job["job_id"],)
        ).fetchone()[0]
    assert [row[0] for row in events] == ["submitted", "claimed"]
    assert all(row[1] == quote["queue_bid_price_id"] and row[2] == 20 for row in events)
    assert cycle["queue_bid_micros"] == 200000


def test_claim_rollback_also_rolls_back_bid_order_evidence(paid_fleet):
    job = submit(paid_fleet, queue_bid_units=10)
    with pytest.raises(RuntimeError):
        with _get_pg_pool().connection() as conn:
            got = claim_next_job(
                conn, replica_id="rollback-bid", scope_gpu_models=[paid_fleet["model"]]
            )
            assert got.job_id == job["job_id"]
            raise RuntimeError("fail claim transaction")
    with _get_pg_pool().connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM queue_bid_events WHERE job_id=%s AND event_type='claimed'",
                (job["job_id"],),
            ).fetchone()[0]
            == 0
        )
    assert claim(paid_fleet).job_id == job["job_id"]


def test_queue_clock_survives_updates_and_resets_on_requeue(paid_fleet):
    job = submit(paid_fleet, queue_bid_units=20)
    _wait_in_queue(job, 9000)
    with _get_pg_pool().connection() as conn:
        original = conn.execute(
            "SELECT queue_entered_at FROM jobs WHERE job_id=%s", (job["job_id"],)
        ).fetchone()[0]
        conn.execute("UPDATE jobs SET reason_code='no_capacity' WHERE job_id=%s", (job["job_id"],))
        assert (
            conn.execute(
                "SELECT queue_entered_at FROM jobs WHERE job_id=%s", (job["job_id"],)
            ).fetchone()[0]
            == original
        )
        conn.execute(
            "UPDATE jobs SET phase='succeeded',status='stopped' WHERE job_id=%s", (job["job_id"],)
        )
        conn.execute(
            "UPDATE jobs SET phase='pending',status='queued' WHERE job_id=%s", (job["job_id"],)
        )
        current = conn.execute(
            "SELECT queue_entered_at FROM jobs WHERE job_id=%s", (job["job_id"],)
        ).fetchone()[0]
    assert (current - original).total_seconds() >= 9000
    assert claim(paid_fleet).effective_priority == 1


def test_bid_admission_releases_money_if_audit_cannot_commit(paid_fleet, monkeypatch):
    import control_plane.queue_bids as audit
    from scheduler import submit_job

    jid = f"bid-rollback-{uuid.uuid4().hex[:8]}"

    def failed_audit(*args):
        raise RuntimeError("audit failed")

    monkeypatch.setattr(audit, "record_submission", failed_audit)
    with pytest.raises(RuntimeError, match="audit failed"):
        submit_job(
            "bid-rollback",
            8,
            gpu_model=paid_fleet["model"],
            owner=paid_fleet["owner"],
            job_id=jid,
            queue_bid_units=10,
        )
    with _get_pg_pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM jobs WHERE job_id=%s", (jid,)).fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT count(*) FROM wallet_holds WHERE job_id=%s AND status='held'", (jid,)
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("units", [-1, True, 1.5, 1000001])
def test_invalid_bid_quantity_is_refused(paid_fleet, units):
    with pytest.raises(ValueError, match="Stripe bid units"):
        quote_tier(paid_fleet["model"], queue_bid_units=units)


def test_rest_quote_and_submission_pin_the_same_funded_additive_price(paid_fleet, monkeypatch):
    from api import app
    from fastapi.testclient import TestClient
    import routes.instances as routes
    import security

    client = TestClient(app)
    # Authenticate a fixed isolated tenant; financial and queue operations remain real.
    monkeypatch.setattr(
        routes,
        "_require_auth",
        lambda request: {
            "customer_id": paid_fleet["owner"],
            "billing_customer_id": paid_fleet["owner"],
            "role": "customer",
            "scopes": ["*"],
        },
    )
    monkeypatch.setattr(security, "probe_image_exists", lambda image: None)
    model = paid_fleet["model"]
    catalog = client.get("/api/pricing/service-tiers", params={"gpu_model": model}).json()
    price = next(item for item in catalog["tiers"] if item["tier"] == "premium")
    addon = catalog["gpu_exclusive_addon"]
    response = client.get(
        "/api/pricing/rates",
        params={
            "gpu_model": model,
            "tier": "premium",
            "exclusive_gpu": True,
            "num_gpus": 2,
            "province": "ON",
            "priority": "critical",
        },
    )
    assert response.status_code == 200, response.text
    rates = response.json()
    assert rates["total_per_hour"] == 1.96
    assert rates["total_with_tax"] == 2.2148
    assert rates["priority"] == 2
    launch = client.post(
        "/instance",
        json={
            "name": "paid-rest-integration",
            "image": "pytorch/pytorch:latest",
            "gpu_model": model,
            "tier": "premium",
            "num_gpus": 2,
            "priority": 10,
            "service_tier_price_id": price["stripe_price_id"],
            "exclusive_gpu": True,
            "gpu_exclusive_price_id": addon["stripe_price_id"],
        },
    )
    assert launch.status_code == 200, launch.text
    body = launch.json()
    job = body.get("job") or body.get("instance") or body
    jid = job.get("job_id") or body.get("job_id")
    assert jid, body
    paid_fleet["jobs"].append(jid)
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT j.spec,h.amount_micros FROM jobs j JOIN wallet_holds h ON h.hold_id=j.wallet_hold_id WHERE j.job_id=%s",
            (jid,),
        ).fetchone()
    assert row[1] == 1960000
    assert row[0]["service_tier_quote"]["total_rate_micros"] == 1960000
    assert row[0]["service_tier_quote"]["gpu_exclusive_price_id"] == addon["stripe_price_id"]
    assert row[0]["service_tier_quote"]["priority"] == 2


@pytest.mark.parametrize("amount", ["0.00001", "0", "-1", "NaN", "Infinity"])
def test_stripe_import_refuses_unrepresentable_or_invalid_money(amount):
    with pytest.raises(ValueError):
        stripe_amount_micros(
            {"currency": "cad", "billing_scheme": "per_unit", "unit_amount_decimal": amount}
        )


def test_mutable_payload_cannot_reprice_bid_promote_or_buy_host_isolation(paid_fleet):
    job = submit(paid_fleet, queue_bid_units=10)
    with _get_pg_pool().connection() as conn:
        spec, digest = conn.execute(
            "SELECT spec,spec_hash FROM jobs WHERE job_id=%s", (job["job_id"],)
        ).fetchone()
        forged = {
            **job,
            "tier": "dedicated",
            "host_exclusive": True,
            "service_tier_quote": {
                **job["service_tier_quote"],
                "tier": "dedicated",
                "queue_bid_micros": 99000000,
            },
        }
        conn.execute(
            "UPDATE jobs SET payload=%s,priority=999,effective_priority=999 WHERE job_id=%s",
            (Jsonb(forged), job["job_id"]),
        )
    selected = claim(paid_fleet)
    assert selected.effective_priority == 1
    assert selected.spec == spec
    assert selected.spec_hash == digest
    bound = reserve(paid_fleet, selected)
    with _get_pg_pool().connection() as conn:
        assert conn.execute(
            "SELECT host_exclusive FROM job_attempts WHERE attempt_id=%s", (bound.attempt_id,)
        ).fetchone() == (False,)
        recorded = conn.execute(
            "SELECT pricing_snapshot FROM queue_bid_events WHERE job_id=%s AND event_type='claimed'",
            (job["job_id"],),
        ).fetchone()[0]
    assert recorded == spec["service_tier_quote"]


def test_hold_renewal_uses_immutable_quote_despite_payload_tampering(paid_fleet):
    from service_tier_pricing import refresh_pending_tier_funding

    job = submit(paid_fleet, "premium", queue_bid_units=20)
    with _get_pg_pool().connection() as conn:
        conn.execute("UPDATE wallet_holds SET expires_at=0 WHERE job_id=%s", (job["job_id"],))
        conn.execute(
            "UPDATE jobs SET payload=payload||%s::jsonb WHERE job_id=%s",
            (Jsonb({"service_tier_quote": {"total_rate_micros": 1}}), job["job_id"]),
        )
    refresh_pending_tier_funding(scope_gpu_models=[paid_fleet["model"]])
    with _get_pg_pool().connection() as conn:
        held = conn.execute(
            "SELECT h.amount_micros FROM jobs j JOIN wallet_holds h ON h.hold_id=j.wallet_hold_id WHERE j.job_id=%s",
            (job["job_id"],),
        ).fetchone()[0]
    assert held == 984000
    assert claim(paid_fleet).job_id == job["job_id"]


def test_pending_internal_spec_with_json_null_does_not_create_an_instance_hold(paid_fleet):
    from control_plane.job_specs import make_execution_spec, canonical_spec_hash
    from service_tier_pricing import refresh_pending_tier_funding

    job_id = "internal-" + uuid.uuid4().hex
    spec = make_execution_spec({"name": job_id, "gpu_model": paid_fleet["model"], "service_tier_quote": None})
    with _get_pg_pool().connection() as conn:
        conn.execute("INSERT INTO jobs(job_id,status,phase,desired_state,priority,submitted_at,payload,spec,spec_hash,queued_at) VALUES(%s,'queued','pending','running',0,0,'{}',%s,%s,clock_timestamp())", (job_id, Jsonb(spec), canonical_spec_hash(spec)))
    paid_fleet["jobs"].append(job_id)
    refresh_pending_tier_funding(scope_gpu_models=[paid_fleet["model"]])
    with _get_pg_pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM wallet_holds WHERE job_id=%s", (job_id,)).fetchone() == (0,)
