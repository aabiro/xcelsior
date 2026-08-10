"""C1's projection, **against the real schema**.

`tests/test_placement_preference_refuses.py` drives `choose_host` over dicts. It
proves the refusal semantics and cannot prove a single field name, because it
writes both sides of the contract. Every name in `choose_host` was an assumption
until this file, and a wrong name there does not raise — it produces a gate that
refuses everything, which is indistinguishable from the gate working.

Naming is the one class of defect no fixture here has ever caught, and it has
happened three times in this phase: the invented tier vocabulary, `verified` read
as a tier rather than `host_verifications.state`, and `price_cents_per_hour` on
candidate rows that carry `cost_per_hour` in dollars. So these tests insert real
rows into `host_verifications`, `reputation_scores` and `sla_monthly`, and assert
the values come back out.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = all(
            _c.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0] is not None
            for t in ("host_verifications", "reputation_scores", "sla_monthly")
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database lacks the placement evidence tables")

from control_plane.scheduler.host_projection import (  # noqa: E402
    PRICE_FIELD,
    REQUIRED_EVIDENCE_FIELDS,
    ProjectionError,
    assert_evidence_shape,
    attach_placement_evidence,
    normalise_price_cents,
    project_placement_evidence,
)
from control_plane.scheduler.preference import (  # noqa: E402
    MIN_OBSERVATION_SECONDS,
    PlacementPreference,
    PlacementRefused,
    choose_host,
    host_uptime_pct,
)

DAY = 86400.0


@pytest.fixture
def hosts():
    """Unique host ids, and their rows removed afterwards."""
    tag = uuid.uuid4().hex[:12]
    ids = [f"h-{tag}-{n}" for n in range(4)]
    yield ids
    with pg_transaction() as conn:
        for table, column in (
            ("host_verifications", "host_id"),
            ("reputation_scores", "entity_id"),
            ("sla_monthly", "host_id"),
        ):
            conn.execute(f"DELETE FROM {table} WHERE {column} = ANY(%s)", (ids,))


def _verification(conn, host_id, *, state, verified_at=None, last_check_at=None,
                  next_check_at=None, deverified_at=None):
    conn.execute(
        """INSERT INTO host_verifications
             (host_id, verification_id, state, verified_at, deverified_at,
              last_check_at, next_check_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (host_id, f"v-{host_id}", state, verified_at, deverified_at,
         last_check_at, next_check_at),
    )


def _reputation(conn, entity_id, *, tier, final_score, raw_score=None, entity_type="host"):
    conn.execute(
        """INSERT INTO reputation_scores
             (entity_id, entity_type, tier, final_score, raw_score)
           VALUES (%s, %s, %s, %s, %s)""",
        (entity_id, entity_type, tier, final_score,
         final_score if raw_score is None else raw_score),
    )


def _sla(conn, host_id, month, *, total_seconds, downtime_seconds):
    conn.execute(
        """INSERT INTO sla_monthly
             (host_id, month, tier, total_seconds, downtime_seconds)
           VALUES (%s, %s, 'community', %s, %s)""",
        (host_id, month, total_seconds, downtime_seconds),
    )


def _month(offset_months: int, now: float) -> str:
    """`YYYY-MM` `offset_months` before the month containing `now`."""
    ref = datetime.fromtimestamp(now)
    year, month = ref.year, ref.month - offset_months
    while month <= 0:
        month += 12
        year -= 1
    return f"{year:04d}-{month:02d}"


def _month_start(month: str) -> float:
    return datetime.strptime(month, "%Y-%m").timestamp()


# ── The naming test ───────────────────────────────────────────────────


def test_every_field_choose_host_reads_comes_back_from_the_real_tables(hosts):
    """The one thing a dict fixture cannot assert.

    Each value is deliberately distinct so a field crossed with its neighbour
    fails rather than coincidentally matching.
    """
    host_id = hosts[0]
    now = time.time()
    month = _month(1, now)
    with pg_transaction() as conn:
        _verification(
            conn, host_id, state="verified",
            verified_at=now - 10 * DAY, deverified_at=now - 40 * DAY,
            last_check_at=now - 2 * DAY, next_check_at=now + 1 * DAY,
        )
        _reputation(conn, host_id, tier="gold", final_score=61.5, raw_score=99.0)
        _sla(conn, host_id, month, total_seconds=30 * DAY, downtime_seconds=1234.0)

        row = project_placement_evidence(conn, [host_id], now=now)[host_id]

    assert row["verification_state"] == "verified", "read from host_verifications.state"
    assert row["verified_at"] == pytest.approx(now - 10 * DAY)
    assert row["deverified_at"] == pytest.approx(now - 40 * DAY)
    assert row["last_check_at"] == pytest.approx(now - 2 * DAY)
    assert row["next_check_at"] == pytest.approx(now + 1 * DAY)
    assert row["reputation_tier"] == "gold"
    assert row["reputation_score"] == pytest.approx(61.5), (
        "reputation_score must be final_score; raw_score is 99.0 and would put "
        "this host two tiers higher on penalties it has already taken"
    )
    assert row["sla_total_seconds"] == pytest.approx(30 * DAY)
    assert row["sla_downtime_seconds"] == pytest.approx(1234.0)
    assert row["verification_unavailable"] is False


def test_a_users_reputation_is_not_read_as_a_hosts(hosts):
    """`entity_type` discriminates, and the projection filters on it."""
    host_id = hosts[0]
    with pg_transaction() as conn:
        _reputation(conn, host_id, tier="diamond", final_score=95.0, entity_type="user")
        row = project_placement_evidence(conn, [host_id])[host_id]
    assert row["reputation_tier"] is None
    assert row["reputation_score"] is None


# ── The observation window ────────────────────────────────────────────


def test_sla_is_summed_across_the_window_and_older_months_are_excluded(hosts):
    """Three months in, one month out."""
    host_id = hosts[0]
    now = _month_start(_month(0, time.time())) + 15 * DAY   # the 16th of this month
    with pg_transaction() as conn:
        _sla(conn, host_id, _month(1, now), total_seconds=31 * DAY, downtime_seconds=100.0)
        _sla(conn, host_id, _month(2, now), total_seconds=30 * DAY, downtime_seconds=50.0)
        _sla(conn, host_id, _month(5, now), total_seconds=31 * DAY, downtime_seconds=9999.0)
        row = project_placement_evidence(conn, [host_id], now=now)[host_id]

    # The five-months-ago row is outside a 90-day window; its downtime dwarfs
    # the rest, so its exclusion is visible in the number rather than implied.
    assert row["sla_downtime_seconds"] == pytest.approx(150.0)
    assert row["sla_total_seconds"] == pytest.approx(61 * DAY)


def test_the_in_progress_month_is_clamped_to_elapsed_time(hosts):
    """`sla.py` writes the whole calendar month; the rest has not happened.

    Without the clamp a host down for a day on the 1st reads ~96.8% on the 10th
    of a 31-day month instead of ~90%, and `min_uptime_pct` — the entire point
    of the gate — is measured against time that does not exist yet.
    """
    host_id = hosts[0]
    this_month = _month(0, time.time())
    now = _month_start(this_month) + 10 * DAY
    with pg_transaction() as conn:
        _sla(conn, host_id, this_month, total_seconds=31 * DAY, downtime_seconds=1 * DAY)
        row = project_placement_evidence(conn, [host_id], now=now)[host_id]

    assert row["sla_total_seconds"] == pytest.approx(10 * DAY), (
        "the unelapsed remainder of the month was counted as observed uptime"
    )
    assert host_uptime_pct(row) == pytest.approx(90.0, abs=0.01)


def test_downtime_cannot_exceed_the_time_it_is_measured_against(hosts):
    """A clamped total with an unclamped downtime would report negative uptime."""
    host_id = hosts[0]
    this_month = _month(0, time.time())
    now = _month_start(this_month) + 2 * DAY
    with pg_transaction() as conn:
        _sla(conn, host_id, this_month, total_seconds=31 * DAY, downtime_seconds=5 * DAY)
        row = project_placement_evidence(conn, [host_id], now=now)[host_id]
    assert row["sla_downtime_seconds"] <= row["sla_total_seconds"]
    assert host_uptime_pct(row) == pytest.approx(0.0)


# ── Absent evidence is a fact; absent columns are a bug ───────────────


def test_a_host_with_no_rows_gets_every_key_and_no_claims(hosts):
    """"No evidence" must still be a complete row."""
    host_id = hosts[0]
    with pg_transaction() as conn:
        row = project_placement_evidence(conn, [host_id])[host_id]
    for field in REQUIRED_EVIDENCE_FIELDS:
        assert field in row
    assert row["verification_state"] is None
    assert row["reputation_tier"] is None
    assert row["sla_total_seconds"] == 0.0
    assert row["verification_unavailable"] is False, (
        "no rows is not an unreadable store; conflating them would refuse every "
        "constrained request on an empty database"
    )


def test_a_complete_row_passes_the_shape_check():
    """The green half of the pair below."""
    assert_evidence_shape([{f: None for f in REQUIRED_EVIDENCE_FIELDS}
                           | {"verification_unavailable": False}])


def test_a_missing_key_raises_rather_than_reading_as_no_evidence():
    """The red half. This is the whole reason the check exists."""
    row = {f: None for f in REQUIRED_EVIDENCE_FIELDS} | {"verification_unavailable": False}
    del row["sla_total_seconds"]
    with pytest.raises(ProjectionError, match="sla_total_seconds"):
        assert_evidence_shape([row])


def test_verification_unavailable_must_be_a_boolean():
    """The one fail-open field, so a null would silently disable the refusal."""
    row = {f: None for f in REQUIRED_EVIDENCE_FIELDS}
    with pytest.raises(ProjectionError, match="verification_unavailable"):
        assert_evidence_shape([row])


def test_a_missing_price_raises_when_the_price_is_required():
    row = {f: None for f in REQUIRED_EVIDENCE_FIELDS} | {"verification_unavailable": False}
    with pytest.raises(ProjectionError, match=PRICE_FIELD):
        assert_evidence_shape([row], require_price=True)


# ── Which store failures are tolerated ────────────────────────────────


class _Broken:
    """A connection that fails one table and delegates the rest."""

    def __init__(self, real, table):
        self._real, self._table = real, table

    def execute(self, sql, params=None):
        if self._table in sql:
            raise RuntimeError(f"{self._table} is unreadable")
        return self._real.execute(sql, params)


def test_an_unreadable_verification_store_is_stated_not_swallowed(hosts):
    """Fail open with a flag, because `choose_host` refuses only on constraint."""
    host_id = hosts[0]
    with pg_transaction() as conn:
        _reputation(conn, host_id, tier="gold", final_score=61.5)
        row = project_placement_evidence(_Broken(conn, "host_verifications"), [host_id])[host_id]

    assert row["verification_unavailable"] is True
    assert row["verification_state"] is None
    assert row["reputation_tier"] == "gold", "the other evidence still projects"

    refusal = choose_host([row | {PRICE_FIELD: 100}], PlacementPreference(require_verified=True))
    assert isinstance(refusal, PlacementRefused)
    assert refusal.code == "verification_unreadable"


def test_an_unreadable_reputation_store_raises(hosts):
    """No flag exists for it, so silence would mean a gate that always refuses.

    Zeros read as "no tier" and "no history" — a universally-refusing gate that
    looks exactly like a working one. The caller has to see the failure.
    """
    with pg_transaction() as conn:
        with pytest.raises(RuntimeError, match="reputation_scores"):
            project_placement_evidence(_Broken(conn, "reputation_scores"), [hosts[0]])


def test_an_unreadable_sla_store_raises(hosts):
    with pg_transaction() as conn:
        with pytest.raises(RuntimeError, match="sla_monthly"):
            project_placement_evidence(_Broken(conn, "sla_monthly"), [hosts[0]])


# ── The price, in one unit ────────────────────────────────────────────


def test_dollars_per_hour_becomes_cents_per_hour():
    """The alias gap that would have refused every real candidate.

    `usable_price` reads `price_cents_per_hour` and `ask_cents_per_hour`; the
    host dicts `scheduler.allocate_best_host` ranks carry `cost_per_hour` in
    dollars, which it reads as neither.
    """
    assert normalise_price_cents({"cost_per_hour": 0.20}) == pytest.approx(20.0)
    assert normalise_price_cents({"ask_cents_per_hour": 20}) == pytest.approx(20.0)
    assert normalise_price_cents({PRICE_FIELD: 20.0}) == pytest.approx(20.0)
    assert normalise_price_cents({}) is None
    assert normalise_price_cents({"cost_per_hour": "not a number"}) is None


def test_mixed_units_would_have_made_the_premium_wrong_by_100x():
    """Both shapes in one list, normalised to one unit before ranking."""
    candidates = [
        {"host_id": "cheap", "cost_per_hour": 0.20},          # 20 cents
        {"host_id": "dear", "ask_cents_per_hour": 25},        # 25 cents
    ]
    prices = [normalise_price_cents(c) for c in candidates]
    assert prices == [pytest.approx(20.0), pytest.approx(25.0)]


# ── End to end, on real rows ──────────────────────────────────────────


def test_a_scheduler_shaped_candidate_list_places_on_real_evidence(hosts):
    """`attach_placement_evidence` → `choose_host`, nothing hand-written.

    The candidates carry `cost_per_hour`, the shape the legacy scheduler builds.
    """
    cheap, reliable = hosts[0], hosts[1]
    now = _month_start(_month(0, time.time())) + 20 * DAY
    with pg_transaction() as conn:
        # Two full past months each, so both clear the observation floor.
        for host_id, downtime in ((cheap, 5 * DAY), (reliable, 60.0)):
            _sla(conn, host_id, _month(1, now), total_seconds=31 * DAY,
                 downtime_seconds=downtime)
            _sla(conn, host_id, _month(2, now), total_seconds=30 * DAY,
                 downtime_seconds=0.0)

        merged = attach_placement_evidence(
            conn,
            [
                {"host_id": cheap, "cost_per_hour": 0.20},
                {"host_id": reliable, "cost_per_hour": 0.22},
            ],
            now=now,
        )

    assert all(m["sla_total_seconds"] >= MIN_OBSERVATION_SECONDS for m in merged)

    unconstrained = choose_host(merged)
    assert unconstrained.host["host_id"] == cheap

    constrained = choose_host(merged, PlacementPreference(min_uptime_pct=99.5))
    assert not isinstance(constrained, PlacementRefused), getattr(constrained, "detail", "")
    assert constrained.host["host_id"] == reliable
    assert constrained.premium_pct == pytest.approx(10.0), (
        "the premium is measured in one unit against the cheapest eligible host"
    )
    assert constrained.evidence["uptime_pct"] == pytest.approx(
        host_uptime_pct(merged[1])
    )


def test_a_candidate_with_no_host_id_is_a_caller_bug_not_a_refusal(hosts):
    """Nothing can be read for it, and it must not compete for a placement.

    Filling it with no-evidence defaults would let an unidentifiable row into
    the ranking; refusing it quietly would dress a malformed shortlist up as
    policy.
    """
    with pg_transaction() as conn:
        with pytest.raises(ProjectionError, match="no host_id"):
            attach_placement_evidence(conn, [{"host_id": "", "cost_per_hour": 0.20}])


def test_a_verified_stamp_nobody_has_rechecked_refuses_as_stale(hosts):
    """Real rows, the production condition: verified, and 111 days unchecked."""
    host_id = hosts[0]
    now = time.time()
    with pg_transaction() as conn:
        _verification(
            conn, host_id, state="verified",
            verified_at=now - 130 * DAY, last_check_at=now - 111 * DAY,
            next_check_at=now - 110 * DAY,
        )
        merged = attach_placement_evidence(
            conn, [{"host_id": host_id, "cost_per_hour": 0.20}], now=now
        )

    refusal = choose_host(merged, PlacementPreference(require_verified=True))
    assert isinstance(refusal, PlacementRefused)
    assert refusal.code == "verification_stale"
