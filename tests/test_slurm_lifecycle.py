"""A Slurm attempt's life from the reconciler's observations (blueprint 2.4).

workers/slurm_lifecycle.py moves an attempt to running when Slurm runs it,
closes it when its mapping settles, and charges it only when the placement was
made in a charging mode and charging is on. A shadow placement is never
charged, whatever the switch says later.
"""

from __future__ import annotations

import json
import uuid

import pytest
from psycopg.types.json import Jsonb

import slurm_adapter as sa
import slurm_billing as sb
import slurm_lifecycle as lc
from control_plane.scheduler.config import slurm_charging_enabled, slurm_placement_mode
from db import pg_connection, pg_transaction

CARD = {"gpu_hour_micros": 3_600_000}  # 1,000 micros per GPU-second


def _run(**overrides):
    record = {
        "state": "COMPLETED",
        "raw_state": "COMPLETED",
        "started_at": 1_794_000_005,
        "ended_at": 1_794_000_065,
        "elapsed_sec": 60,
        "suspended_sec": 0,
        "time_limit_sec": 600,
        "alloc_tres": "cpu=4,gres/gpu=1,node=1",
        "submitted_at": 1_794_000_000,
        "name": "x",
        "exit_code": "0:0",
        "derived_exit_code": "0:0",
        "node_list": "tower",
    }
    record.update(overrides)
    return record


@pytest.fixture
def placed():
    """An assigned job with a reserved attempt on a cluster, submitted to Slurm."""
    made: list[dict] = []

    def make(*, charge: bool, balance: int = 100_000_000) -> dict:
        marker = uuid.uuid4().hex[:8]
        job_id, cluster_id, customer = f"jlife{marker}", f"clife{marker}", f"life-{marker}"
        name = f"{job_id}-a1"
        with pg_transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, priority, submitted_at, payload, phase,
                                  desired_state, effective_priority, spec, tenant_id)
                VALUES (%s, 'assigned', 10, 1794000000, %s, 'scheduled', 'running', 10, %s, %s)
                """,
                (
                    job_id,
                    json.dumps({"owner": customer, "num_gpus": 1}),
                    Jsonb({"num_gpus": 1, "name": job_id}),
                    customer,
                ),
            )
            attempt_id, fence = conn.execute(
                """
                INSERT INTO job_attempts
                    (job_id, attempt_number, status, fencing_token, job_generation, created_by)
                VALUES (%s, 1, 'reserved', nextval('placement_fencing_token_seq'), 1, 'test')
                RETURNING attempt_id, fencing_token
                """,
                (job_id,),
            ).fetchone()
            conn.execute(
                "UPDATE jobs SET active_attempt_id = %s WHERE job_id = %s", (attempt_id, job_id)
            )
            conn.execute(
                """
                INSERT INTO slurm_clusters (cluster_id, profile_key, display_name, enabled,
                                            poll_interval_seconds, rate_card, suspend_holds_gres)
                VALUES (%s, 'generic', 'test', true, 60, %s, true)
                """,
                (cluster_id, Jsonb(CARD)),
            )
            conn.execute(
                """
                INSERT INTO slurm_submissions (submission_name, cluster_id, xcelsior_job_id,
                                               attempt_id, tenant_id, state, slurm_job_id, charge)
                VALUES (%s, %s, %s, %s, %s, 'submitted', '7001', %s)
                """,
                (name, cluster_id, job_id, attempt_id, customer, charge),
            )
            conn.execute(
                "INSERT INTO wallets (customer_id, balance_micros, status, created_at, updated_at) "
                "VALUES (%s, %s, 'active', 1794000000, 1794000000)",
                (customer, balance),
            )
        sa.register_slurm_job(
            job_id,
            "7001",
            cluster_id=cluster_id,
            tenant_id=customer,
            attempt_id=str(attempt_id),
            submission_name=name,
        )
        ctx = {
            "job_id": job_id,
            "cluster_id": cluster_id,
            "customer": customer,
            "attempt_id": str(attempt_id),
            "fence": fence,
            "name": name,
        }
        made.append(ctx)
        return ctx

    yield make
    with pg_transaction() as conn:
        for ctx in made:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id = %s", (ctx["job_id"],))
            conn.execute(
                "DELETE FROM wallet_transactions WHERE customer_id = %s", (ctx["customer"],)
            )
            conn.execute("DELETE FROM wallets WHERE customer_id = %s", (ctx["customer"],))
            conn.execute(
                "DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (ctx["cluster_id"],)
            )
            conn.execute(
                "DELETE FROM slurm_submissions WHERE cluster_id = %s", (ctx["cluster_id"],)
            )
            conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (ctx["cluster_id"],))
            conn.execute("DELETE FROM job_attempts WHERE job_id = %s", (ctx["job_id"],))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (ctx["job_id"],))


def _observe(ctx, state: str, *, settle_runs: list[dict] | None = None) -> None:
    """What the reconciler writes: the observed state, and on settling the ledger."""
    with pg_transaction() as conn:
        conn.execute(
            """
            UPDATE slurm_job_mappings
               SET observed_state = %s, started_at = to_timestamp(1794000005),
                   terminal_at = CASE WHEN %s THEN clock_timestamp() END,
                   ended_at = CASE WHEN %s THEN to_timestamp(1794000065) END
             WHERE submission_name = %s
            """,
            (state, settle_runs is not None, settle_runs is not None, ctx["name"]),
        )
        if settle_runs:
            mapping = conn.execute(
                """
                SELECT mapping_id, cluster_id, slurm_job_id, xcelsior_job_id,
                       xcelsior_attempt_id, tenant_id
                  FROM slurm_job_mappings WHERE submission_name = %s
                """,
                (ctx["name"],),
            ).fetchone()
            keys = (
                "mapping_id",
                "cluster_id",
                "slurm_job_id",
                "xcelsior_job_id",
                "xcelsior_attempt_id",
                "tenant_id",
            )
            sb.write_segments(
                conn,
                dict(zip(keys, mapping)),
                settle_runs,
                {"rate_card": CARD, "suspend_holds_gres": True},
            )


def _state(ctx) -> dict:
    with pg_connection() as conn:
        job = conn.execute(
            "SELECT status, phase FROM jobs WHERE job_id = %s", (ctx["job_id"],)
        ).fetchone()
        attempt = conn.execute(
            "SELECT status, started_at IS NOT NULL FROM job_attempts WHERE attempt_id = %s",
            (ctx["attempt_id"],),
        ).fetchone()
        debit = conn.execute(
            "SELECT COALESCE(SUM(-amount_micros), 0) FROM wallet_transactions "
            "WHERE customer_id = %s AND tx_type = 'charge'",
            (ctx["customer"],),
        ).fetchone()[0]
        charged = conn.execute(
            "SELECT bool_and(charged) FROM slurm_billing_segments WHERE attempt_id = %s",
            (ctx["attempt_id"],),
        ).fetchone()[0]
        events = [
            r[0]["status"]
            for r in conn.execute(
                "SELECT payload FROM outbox_events WHERE aggregate_id = %s "
                "AND event_type = 'job.v1.attempt_status_changed' ORDER BY created_at",
                (ctx["job_id"],),
            ).fetchall()
        ]
    return {
        "job": tuple(job),
        "attempt": attempt[0],
        "attempt_started": attempt[1],
        "debit": int(debit),
        "charged": charged,
        "events": events,
    }


def _mine(result: dict, ctx: dict) -> dict:
    """The parts of a pass that are about this test's attempt."""
    aid = ctx["attempt_id"]
    return {
        "running": [a for a in result["running"] if a == aid],
        "closed": [c for c in result["closed"] if c[0] == aid],
        "charged": [c for c in result["charged"] if c[0] == aid],
        "errors": [e for e in result["errors"] if e[0] == aid],
    }


class TestRunning:
    def test_a_running_slurm_job_reads_running(self, placed):
        ctx = placed(charge=False)
        _observe(ctx, "running")
        mine = _mine(lc.advance_attempts(charging=False), ctx)
        assert mine["running"] == [ctx["attempt_id"]]
        state = _state(ctx)
        assert state["job"] == ("running", "running")
        assert state["attempt"] == "running" and state["attempt_started"]
        assert state["events"] == ["running"]
        # Already running: the next pass has nothing to do.
        assert _mine(lc.advance_attempts(charging=False), ctx)["running"] == []

    def test_a_pending_slurm_job_stays_assigned(self, placed):
        ctx = placed(charge=False)
        _observe(ctx, "pending")
        lc.advance_attempts(charging=False)
        assert _state(ctx)["job"] == ("assigned", "scheduled")


class TestFinished:
    def test_a_shadow_placement_closes_and_is_never_charged(self, placed):
        ctx = placed(charge=False)
        _observe(ctx, "running")
        lc.advance_attempts(charging=True)
        _observe(ctx, "completed", settle_runs=[_run()])
        mine = _mine(lc.advance_attempts(charging=True), ctx)  # charging on: still not charged
        assert mine["closed"] == [(ctx["attempt_id"], "succeeded")] and mine["charged"] == []
        state = _state(ctx)
        assert state["job"][0] == "completed" and state["attempt"] == "succeeded"
        assert state["debit"] == 0 and state["charged"] is False
        assert state["events"] == ["running", "succeeded"]
        assert _mine(lc.advance_attempts(charging=True), ctx) == {
            "running": [],
            "closed": [],
            "charged": [],
            "errors": [],
        }

    def test_a_chargeable_placement_is_charged_its_ledger_sum(self, placed):
        ctx = placed(charge=True)
        _observe(ctx, "running")
        lc.advance_attempts(charging=True)
        _observe(ctx, "completed", settle_runs=[_run()])
        mine = _mine(lc.advance_attempts(charging=True), ctx)
        assert mine["charged"] == [(ctx["attempt_id"], "succeeded", 60_000)]
        state = _state(ctx)
        assert state["job"][0] == "completed" and state["attempt"] == "succeeded"
        assert state["debit"] == 60_000 and state["charged"] is True
        assert state["events"] == ["running", "succeeded"]
        lc.advance_attempts(charging=True)  # nothing left: charged once
        assert _state(ctx)["debit"] == 60_000

    def test_with_charging_off_it_closes_now_and_charges_once_charging_is_on(self, placed):
        ctx = placed(charge=True)
        _observe(
            ctx,
            "failed",
            settle_runs=[
                _run(
                    state="FAILED",
                    raw_state="FAILED",
                    exit_code="1:0",
                    elapsed_sec=30,
                    ended_at=1_794_000_035,
                )
            ],
        )
        mine = _mine(lc.advance_attempts(charging=False), ctx)
        assert mine["closed"] == [(ctx["attempt_id"], "failed")]
        assert _state(ctx)["debit"] == 0 and _state(ctx)["job"][0] == "failed"
        mine = _mine(lc.advance_attempts(charging=True), ctx)
        assert mine["charged"] == [(ctx["attempt_id"], "failed", 30_000)]
        state = _state(ctx)
        assert state["debit"] == 30_000 and state["charged"] is True
        assert state["events"] == ["failed"]  # closed once, not twice

    def test_each_requeued_run_is_charged(self, placed):
        ctx = placed(charge=True)
        runs = [
            _run(state="REQUEUED", raw_state="REQUEUED", elapsed_sec=20, ended_at=1_794_000_025),
            _run(started_at=1_794_000_100, ended_at=1_794_000_150, elapsed_sec=50),
        ]
        _observe(ctx, "completed", settle_runs=runs)
        mine = _mine(lc.advance_attempts(charging=True), ctx)
        assert mine["charged"] == [(ctx["attempt_id"], "succeeded", 70_000)]

    def test_a_mapping_lost_to_id_reuse_closes_as_lost_and_is_not_charged(self, placed):
        ctx = placed(charge=True)
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_job_mappings SET observed_state = 'unknown', "
                "state_reason = 'slurm_job_id_reused', terminal_at = clock_timestamp() "
                "WHERE submission_name = %s",
                (ctx["name"],),
            )
        mine = _mine(lc.advance_attempts(charging=True), ctx)
        assert mine["closed"] == [(ctx["attempt_id"], "lost")] and mine["charged"] == []
        state = _state(ctx)
        assert state["attempt"] == "lost" and state["job"][0] == "failed"
        assert state["debit"] == 0

    def test_a_wallet_that_refuses_is_retried_not_dropped(self, placed):
        ctx = placed(charge=True, balance=0)
        _observe(ctx, "completed", settle_runs=[_run()])
        mine = _mine(lc.advance_attempts(charging=True), ctx)
        state = _state(ctx)
        if mine["errors"]:
            # Refused: nothing moved, and the next pass tries again.
            assert state["attempt"] == "reserved" and state["charged"] is False
            assert _mine(lc.advance_attempts(charging=True), ctx)["errors"]
        else:
            # The wallet allows a negative balance: charged, exactly once.
            assert state["debit"] == 60_000 and state["charged"] is True


class TestPlacementMode:
    @pytest.mark.parametrize(
        "raw,mode",
        [
            (None, "off"),
            ("", "off"),
            ("0", "off"),
            ("off", "off"),
            ("1", "shadow"),
            ("true", "shadow"),
            ("on", "shadow"),  # an old "on" never starts charging
            ("shadow", "shadow"),
            ("admin", "admin"),
            ("Allowlist", "allowlist"),
            ("all", "all"),
            ("everyone", "off"),
        ],
    )
    def test_the_switch_reads_as_a_mode(self, monkeypatch, raw, mode):
        if raw is None:
            monkeypatch.delenv("XCELSIOR_SLURM_PLACEMENT", raising=False)
        else:
            monkeypatch.setenv("XCELSIOR_SLURM_PLACEMENT", raw)
        assert slurm_placement_mode() == mode
        assert slurm_charging_enabled() is (mode in ("admin", "allowlist", "all"))


def test_charging_happens_only_through_the_lifecycle_task():
    """The scheduler, the reconciler and the drift report never charge. The
    one task that can is the lifecycle pass, and only for a chargeable
    placement while a charging mode is on."""
    import inspect

    import bg_worker
    import slurm_reconciler
    from control_plane.scheduler import slurm_placement

    for source in (
        inspect.getsource(slurm_placement),
        inspect.getsource(slurm_reconciler),
        inspect.getsource(sb.report_due_clusters),
    ):
        assert "settle_finished_attempt" not in source
        assert "_charge_on_connection" not in source
    worker = inspect.getsource(bg_worker)
    assert "settle_finished_attempt" not in worker and "_charge_on_connection" not in worker
    assert 'register_task("slurm_lifecycle"' in worker
    life = inspect.getsource(lc._advance_one)
    assert 'charging and row["charge"]' in life
