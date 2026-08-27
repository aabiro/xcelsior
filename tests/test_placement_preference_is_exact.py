"""A premium bound gates a price, so its conversion must be exact.

`max_premium_pct` says "pay at most 15% more than the cheapest eligible host".
The API takes a percent, the schema stores integer basis points, and the whole
value of that choice is lost if the conversion between them goes through binary
floating point.

## The trap, with the actual numbers

`int(8.7 * 100)` is **869**, not 870, because 8.7 has no exact binary
representation and the product lands at 869.9999999999999. `int(1.15 * 100)` is
**114**. A user asking to pay at most 8.7% more would silently get a bound of
8.69%, and a host priced exactly at the stated bound would be refused — or
accepted — depending on a rounding artefact nobody typed.

`money.cad_to_micros` already solved this for currency: read the number the
caller *wrote* with `Decimal(str(...))` rather than the float it became, then
quantize with `ROUND_HALF_UP`. `pct_to_bps` is the same discipline on a
different unit, and this file is the proof it holds where the naive version
fails.

`min_uptime_pct` earns the same care for a different reason: 99.95% and 99.9%
are different SLAs and the difference lives in the third decimal place.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from control_plane.scheduler.preference import bps_to_pct, pct_to_bps


#: Values where naive `int(pct * 100)` is provably wrong. Not hypothetical —
#: each was checked against this interpreter before being written down.
FLOAT_TRAPS = [
    (8.7, 870, 869),
    (1.15, 115, 114),
]


@pytest.mark.parametrize("pct,exact,naive", FLOAT_TRAPS)
def test_the_naive_conversion_really_is_wrong(pct: float, exact: int, naive: int):
    """Calibration: if this stops failing, the trap has moved and the test below
    is no longer proving anything."""
    assert int(pct * 100) == naive, (
        f"int({pct} * 100) is no longer {naive} on this interpreter; re-derive "
        "the trap values before trusting the assertion below"
    )
    assert naive != exact


@pytest.mark.parametrize("pct,exact,_naive", FLOAT_TRAPS)
def test_pct_to_bps_is_exact_where_floats_are_not(pct: float, exact: int, _naive: int):
    assert pct_to_bps(pct) == exact


@pytest.mark.parametrize(
    "pct,bps",
    [
        (99.5, 9950),      # the gate's own example
        (99.95, 9995),     # three decimal places, a real SLA distinction
        (99.9, 9990),
        (15, 1500),        # an int, not a float
        ("15.005", 1501),  # a string, rounded half up rather than truncated
        (0, 0),
        (100, 10000),
        (None, None),
    ],
)
def test_known_values(pct, bps):
    assert pct_to_bps(pct) == bps


def test_the_round_trip_returns_what_was_asked_for():
    """A user who typed 99.5 must read 99.5 back, not 99.49999999999999."""
    for pct in (99.5, 99.95, 15.0, 0.01, 100.0):
        assert bps_to_pct(pct_to_bps(pct)) == pct


def test_decimal_input_is_not_degraded_to_float():
    """Passing a Decimal must not round-trip through binary float on the way in."""
    assert pct_to_bps(Decimal("8.7")) == 870


def test_half_up_not_bankers_rounding():
    """Python's built-in `round` is banker's rounding: `round(0.5)` is 0.

    A bound that rounds to even would move a user's stated cap up or down
    depending on whether the neighbouring digit happened to be even, which is
    not a rule anyone would agree to if it were stated out loud.
    """
    assert pct_to_bps("0.005") == 1   # 0.5 bps -> 1, not 0
    assert pct_to_bps("0.015") == 2   # 1.5 bps -> 2, not 2 by luck
    assert pct_to_bps("0.025") == 3   # 2.5 bps -> 3, banker's would give 2


# ── Through the real storage path ─────────────────────────────────────
#
# The conversions above are exact in isolation. What matters is that they stay
# exact through `upsert_job` and the database, because that is the path a launch
# takes and it is where a unit could quietly be re-derived.


def _pg():
    from db import _get_pg_pool

    return _get_pg_pool()


def test_a_preference_round_trips_through_the_database_exactly():
    import time
    import uuid

    from db import DatabaseOps
    from psycopg.rows import dict_row

    job_id = f"pref-exact-{uuid.uuid4().hex[:8]}"
    job = {
        "job_id": job_id,
        "status": "queued",
        "priority": 0,
        "submitted_at": time.time(),
        "placement_preference": {
            "min_uptime_pct": 8.7,   # naive float conversion yields 869
            "min_tier": "  GOLD  ",  # whitespace and case the caller did not mean
            "require_verified": True,
            "max_premium_pct": 1.15,  # naive float conversion yields 114
        },
    }
    pool = _pg()
    try:
        with pool.connection() as conn:
            DatabaseOps.upsert_job(conn, job, backend="postgres")
            conn.commit()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            row = conn.execute(
                """SELECT placement_min_uptime_bps, placement_min_tier,
                          placement_require_verified, placement_max_premium_bps
                   FROM jobs WHERE job_id = %s""",
                (job_id,),
            ).fetchone()

        assert row["placement_min_uptime_bps"] == 870, "the float trap reached the database"
        assert row["placement_max_premium_bps"] == 115, "the float trap reached the database"
        # Normalised before the CHECK sees it: `'  GOLD  '` would fail the shape
        # constraint, and refusing a launch over whitespace is not the lesson.
        assert row["placement_min_tier"] == "gold"
        assert row["placement_require_verified"] is True
    finally:
        with pool.connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
            conn.commit()


def test_resubmitting_without_a_preference_clears_it():
    """The stale-constraint hazard.

    `upsert_job` is an upsert: a job resubmitted without a preference must come
    back unconstrained. Leaving the previous values attached would apply a
    constraint nobody asked for on this submission — and it would be invisible,
    because the request that produced it said nothing about placement.
    """
    import time
    import uuid

    from db import DatabaseOps
    from psycopg.rows import dict_row

    job_id = f"pref-clear-{uuid.uuid4().hex[:8]}"
    base = {"job_id": job_id, "status": "queued", "priority": 0, "submitted_at": time.time()}
    pool = _pg()
    try:
        with pool.connection() as conn:
            DatabaseOps.upsert_job(
                conn,
                {**base, "placement_preference": {"min_uptime_pct": 99.5, "require_verified": True}},
                backend="postgres",
            )
            conn.commit()
        with pool.connection() as conn:
            DatabaseOps.upsert_job(conn, dict(base), backend="postgres")
            conn.commit()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            row = conn.execute(
                """SELECT placement_min_uptime_bps, placement_min_tier,
                          placement_require_verified
                   FROM jobs WHERE job_id = %s""",
                (job_id,),
            ).fetchone()

        assert row["placement_min_uptime_bps"] is None
        assert row["placement_min_tier"] is None
        assert row["placement_require_verified"] is False
    finally:
        with pool.connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
            conn.commit()


# ── The chain, end to end ─────────────────────────────────────────────


def test_the_launch_path_carries_a_preference_all_the_way_down():
    """Every hop between the request body and the column.

    The defect this replaces was not a broken hop — it was a **missing** one:
    the control evaluated a preference and nothing carried it, so a user could
    state a constraint, see it satisfied, and launch without it. A test that
    only checked the ends would have passed against that.
    """
    import inspect

    from routes.instances import JobIn
    from scheduler import submit_job
    from control_plane.launch import service

    assert "placement_preference" in JobIn.model_fields
    assert "placement_preference" in inspect.signature(submit_job).parameters
    assert "placement_preference" in inspect.signature(service._submit_from_spec).parameters


def test_the_boundary_bounds_match_the_evaluate_route():
    """One validator disagreeing with another is how a request that previewed
    fine gets refused at launch — the user did nothing wrong and cannot tell."""
    from routes.action_plans import PlacementPreferenceIn
    from routes.instances import PlacementPreferenceBody

    for field in ("min_uptime_pct", "min_tier", "require_verified", "max_premium_pct"):
        assert field in PlacementPreferenceBody.model_fields, field
        assert field in PlacementPreferenceIn.model_fields, field

    def bounds(model, name):
        meta = model.model_fields[name].metadata
        return sorted((type(m).__name__, getattr(m, "ge", getattr(m, "le", None))) for m in meta)

    for field in ("min_uptime_pct", "max_premium_pct"):
        assert bounds(PlacementPreferenceBody, field) == bounds(PlacementPreferenceIn, field), (
            f"{field} is bounded differently on the launch body than on the "
            "evaluate route, so a preview can succeed where the launch refuses"
        )


def test_the_preference_never_enters_the_spec():
    """`spec` feeds `canonicalize`/`spec_hash`.

    If the preference were folded in, two identical workloads asking for
    different reliability would hash apart and an approved plan would stop
    matching its rerun — the reason `PlacementPreferenceIn` says it is
    "deliberately not part of JobIn"'s spec in the first place.
    """
    import pathlib

    service_src = pathlib.Path("control_plane/launch/service.py").read_text(encoding="utf-8")
    body = service_src.split("def _submit_from_spec", 1)[1].split("\ndef ", 1)[0]
    assert 'spec.get("placement_preference")' not in body, (
        "the preference is being read out of the spec; it must travel beside it"
    )
    assert "placement_preference=placement_preference" in body
