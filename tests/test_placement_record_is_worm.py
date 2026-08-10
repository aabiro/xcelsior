"""Gate P5 clause 3: the placement record, and the two properties that matter.

The clause asks that the chosen host's reputation and SLA *at time of placement*
are recorded. Two things make that claim worth anything, and both are asserted
here rather than described:

1. **The row cannot be changed.** A record an operator can rewrite is not
   evidence. The WORM trigger is probed with a real UPDATE and a real DELETE.
2. **Refusals are recorded too.** A preference that refused was honoured *by* the
   refusal; a table of successes only would hold no evidence of the behaviour
   this gate exists to produce.

These tests deliberately leave their rows behind — the table refuses DELETE, and
a test that could clean up would be proving the opposite of the point. Each run
uses a fresh tenant id.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = _c.execute("SELECT to_regclass('placement_decisions')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 105")

from control_plane.scheduler.placement_record import (  # noqa: E402
    list_placements,
    premium_pct,
    price_micros,
    read_placement,
    record_placement,
)
from control_plane.scheduler.preference import (  # noqa: E402
    PlacementPreference,
    choose_host,
)

DAY = 86400.0
FLOOR = 40 * DAY   # comfortably over MIN_OBSERVATION_SECONDS


@pytest.fixture
def tenant():
    return f"tenant-{uuid.uuid4().hex[:12]}"


def _host(host_id, price_cents, *, downtime=0.0, state="verified", now=None):
    now = time.time() if now is None else now
    return {
        "host_id": host_id,
        "price_cents_per_hour": price_cents,
        "verification_state": state,
        "verified_at": now - 3 * DAY,
        "deverified_at": None,
        "last_check_at": now - 3600.0,
        "next_check_at": now + 82800.0,
        "verification_unavailable": False,
        "reputation_tier": "new_user",
        "reputation_score": 50.0,
        "sla_total_seconds": FLOOR,
        "sla_downtime_seconds": downtime,
    }


# ── The record itself ─────────────────────────────────────────────────


def test_a_placement_records_what_was_true_at_the_time(tenant):
    """Copied, not referenced — the point of the clause."""
    candidates = [_host("cheap", 20, downtime=FLOOR * 0.05), _host("reliable", 22)]
    decision = choose_host(candidates, PlacementPreference(min_uptime_pct=99.5))
    assert decision.host["host_id"] == "reliable"

    with pg_transaction() as conn:
        decision_id = record_placement(
            conn, tenant_id=tenant, decision=decision, candidates=candidates,
            preference=PlacementPreference(min_uptime_pct=99.5), job_id="job-1",
        )
        stored = read_placement(conn, decision_id, tenant_id=tenant)

    assert stored is not None
    assert stored["outcome"] == "placed"
    assert stored["host_id"] == "reliable"
    assert stored["job_id"] == "job-1"
    assert stored["asked"]["min_uptime_pct"] == 99.5
    assert stored["evidence"]["reputation_score"] == 50.0
    assert stored["evidence"]["verification_status"] == "verified"
    assert stored["evidence"]["uptime_pct"] == pytest.approx(100.0)
    assert stored["candidate_count"] == 2
    assert stored["baseline_price_micros"] == 200_000, "20 cents/hour in micros"
    assert stored["chosen_price_micros"] == 220_000
    assert stored["premium_pct"] == pytest.approx(10.0)


def test_a_refusal_is_recorded_with_the_number_that_failed(tenant):
    """The half a successes-only table would drop."""
    candidates = [_host("a", 20, state="unverified"), _host("b", 22, state="unverified")]
    preference = PlacementPreference(require_verified=True)
    decision = choose_host(candidates, preference)

    with pg_transaction() as conn:
        decision_id = record_placement(
            conn, tenant_id=tenant, decision=decision,
            candidates=candidates, preference=preference,
        )
        stored = read_placement(conn, decision_id, tenant_id=tenant)

    assert stored["outcome"] == "refused"
    assert stored["host_id"] is None
    assert stored["refusal_code"] == "verification_unsatisfiable"
    assert "no candidate host is verified" in stored["refusal_detail"]
    assert stored["asked"]["require_verified"] is True
    assert stored["candidate_count"] == 2, (
        "a refusal is only interpretable against the field it refused over"
    )
    assert stored["candidates"][0]["verification_status"] == "unverified"
    assert stored["premium_pct"] is None


# ── WORM, probed rather than described ────────────────────────────────


def test_the_record_cannot_be_updated(tenant):
    """A record an operator can rewrite is not evidence."""
    candidates = [_host("only", 20)]
    with pg_transaction() as conn:
        decision_id = record_placement(
            conn, tenant_id=tenant, decision=choose_host(candidates),
            candidates=candidates,
        )

    with pytest.raises(Exception, match="append-only"):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE placement_decisions SET host_id = 'somewhere-else' "
                "WHERE decision_id = %s",
                (decision_id,),
            )

    with pg_transaction() as conn:
        assert read_placement(conn, decision_id, tenant_id=tenant)["host_id"] == "only"


def test_the_record_cannot_be_deleted(tenant):
    candidates = [_host("only", 20)]
    with pg_transaction() as conn:
        decision_id = record_placement(
            conn, tenant_id=tenant, decision=choose_host(candidates),
            candidates=candidates,
        )

    with pytest.raises(Exception, match="append-only"):
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM placement_decisions WHERE decision_id = %s", (decision_id,)
            )

    with pg_transaction() as conn:
        assert read_placement(conn, decision_id, tenant_id=tenant) is not None


# ── WORM rows, prunable table ─────────────────────────────────────────


def test_the_table_is_partitioned_by_month():
    """`105` took the trigger from `072` and the partitioning from neither.

    `072_audit_events_v2.py` says it outright: *partition drops (retention) are
    DDL and are unaffected* by the trigger. That is the whole mechanism by which
    an append-only table stays prunable. Without it the trigger forbids the only
    statement that could ever shrink a per-request table.
    """
    with pg_transaction() as conn:
        kind = conn.execute(
            "SELECT relkind FROM pg_class WHERE relname = 'placement_decisions'"
        ).fetchone()[0]
        children = conn.execute(
            """SELECT count(*) FROM pg_inherits i
                 JOIN pg_class p ON p.oid = i.inhparent
                WHERE p.relname = 'placement_decisions'"""
        ).fetchone()[0]

    assert kind == "p", "placement_decisions is a plain table; it cannot be pruned"
    assert children >= 2, "no monthly partitions and no default partition"


def test_a_partition_can_be_dropped_even_though_a_row_cannot_be_deleted(tenant):
    """The pair that makes retention possible without making evidence mutable.

    Both halves in one test on purpose: either alone is a property nobody wants.
    A table you can DELETE from is not an audit trail; a table you cannot prune
    grows until someone has to make it mutable.
    """
    old = "2019-03-01"
    with pg_transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS placement_decisions_201903
                PARTITION OF placement_decisions
                FOR VALUES FROM ('{old}') TO ('2019-04-01')"""
        )
        conn.execute(
            """INSERT INTO placement_decisions
                 (tenant_id, outcome, host_id, decided_at)
               VALUES (%s, 'placed', 'ancient', %s::timestamptz)""",
            (tenant, f"{old} 12:00:00+00"),
        )
        assert conn.execute(
            "SELECT count(*) FROM placement_decisions_201903"
        ).fetchone()[0] == 1

    # The row itself is immutable.
    with pytest.raises(Exception, match="append-only"):
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM placement_decisions WHERE tenant_id = %s", (tenant,)
            )

    # The partition holding it is not.
    with pg_transaction() as conn:
        conn.execute("DROP TABLE placement_decisions_201903")
        assert conn.execute(
            "SELECT count(*) FROM placement_decisions WHERE tenant_id = %s", (tenant,)
        ).fetchone()[0] == 0


def test_partition_maintenance_keeps_the_window_full():
    """One maintainer for every partitioned table, not one per table.

    A maintainer duplicated per table is how one of them silently stops
    advancing while the other looks fine — the DEFAULT partition absorbs the
    writes and nothing complains until someone tries to prune.
    """
    import datetime as dt

    from control_plane.audit_partitions import (
        PARTITIONED_TABLES,
        ensure_monthly_partitions,
    )

    assert "placement_decisions" in PARTITIONED_TABLES
    assert "audit_events_v2" in PARTITIONED_TABLES

    far = dt.date(2030, 5, 1)
    with pg_transaction() as conn:
        ensured = ensure_monthly_partitions(
            conn, "placement_decisions", months_ahead=1, today=far
        )
        assert ensured == ["203005", "203006"]
        present = conn.execute(
            """SELECT count(*) FROM pg_class
                WHERE relname IN ('placement_decisions_203005',
                                  'placement_decisions_203006')"""
        ).fetchone()[0]
        conn.execute("DROP TABLE IF EXISTS placement_decisions_203005")
        conn.execute("DROP TABLE IF EXISTS placement_decisions_203006")
    assert present == 2


def test_an_unknown_table_name_is_not_interpolated_into_ddl():
    """The table name reaches a CREATE TABLE, so it is never taken on trust."""
    from control_plane.audit_partitions import ensure_monthly_partitions

    with pg_transaction() as conn:
        with pytest.raises(ValueError, match="not a known partitioned table"):
            ensure_monthly_partitions(conn, "users; DROP TABLE wallets")


# ── Shape constraints, probed against the database ────────────────────


def test_a_placement_on_no_host_is_not_representable(tenant):
    """`ck_placement_shape`. "Placed on nothing" is a hole where a reader looks."""
    with pytest.raises(Exception, match="ck_placement_shape"):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO placement_decisions (tenant_id, outcome) VALUES (%s, 'placed')",
                (tenant,),
            )


def test_a_refusal_without_a_code_is_not_representable(tenant):
    with pytest.raises(Exception, match="ck_placement_shape"):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO placement_decisions (tenant_id, outcome) VALUES (%s, 'refused')",
                (tenant,),
            )


def test_a_zero_price_is_not_representable(tenant):
    """A zero baseline is what silently turns every premium into 0%."""
    with pytest.raises(Exception, match="ck_placement_prices"):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO placement_decisions "
                " (tenant_id, outcome, host_id, baseline_price_micros) "
                " VALUES (%s, 'placed', 'h', 0)",
                (tenant,),
            )


# ── Reads are tenant-scoped ───────────────────────────────────────────


def test_another_tenants_decision_is_not_readable(tenant):
    candidates = [_host("only", 20)]
    with pg_transaction() as conn:
        decision_id = record_placement(
            conn, tenant_id=tenant, decision=choose_host(candidates),
            candidates=candidates,
        )
        assert read_placement(conn, decision_id, tenant_id="someone-else") is None


def test_the_listing_is_scoped_and_newest_first(tenant):
    candidates = [_host("only", 20)]
    with pg_transaction() as conn:
        for _ in range(3):
            record_placement(
                conn, tenant_id=tenant, decision=choose_host(candidates),
                candidates=candidates,
            )
        record_placement(
            conn, tenant_id=f"{tenant}-other", decision=choose_host(candidates),
            candidates=candidates,
        )
        listed = list_placements(conn, tenant_id=tenant)

    assert len(listed) == 3
    times = [row["decided_at"] for row in listed]
    assert times == sorted(times, reverse=True)


def test_an_unattributed_decision_is_refused(tenant):
    """The index leads with tenant_id; a row nobody owns is never returned."""
    candidates = [_host("only", 20)]
    with pg_transaction() as conn:
        with pytest.raises(ValueError, match="attributed to a tenant"):
            record_placement(
                conn, tenant_id="  ", decision=choose_host(candidates),
                candidates=candidates,
            )


# ── Money stays integral ──────────────────────────────────────────────


def test_prices_convert_to_micros_and_refuse_zero():
    assert price_micros(20) == 200_000
    assert price_micros(20.5) == 205_000
    assert price_micros(0) is None
    assert price_micros(None) is None


def test_the_premium_is_recomputed_not_stored():
    assert premium_pct(200_000, 220_000) == pytest.approx(10.0)
    assert premium_pct(0, 220_000) is None
    assert premium_pct(200_000, None) is None
