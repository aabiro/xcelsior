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
