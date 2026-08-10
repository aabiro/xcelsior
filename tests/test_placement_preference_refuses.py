"""A preference that cannot be satisfied refuses. It does not settle.

Gate P5's second clause, and the plan flags it as *"the failure mode that would
quietly destroy trust"*. A user who asks for 99.5% uptime and silently gets a
97% host has not been served approximately — they have been answered a different
question, and they will stop checking.

## The two halves that are not interchangeable

A **preference over eligible hosts** ranks. A **constraint** gates eligibility.
The plan's own example contains both: *"above 99.5% uptime"* gates, *"even at 15%
more"* bounds what the gate may cost. Treating the constraint as a ranking is
exactly how silent fallback happens — the 99.1% host sorts last and still wins
when nothing better exists.

## Why every refusal carries a number

"No host matched" is not something a user can act on. "The best measured uptime
is 99.10%, you asked for 99.5%" lets them decide whether to relax the constraint
or wait. A refusal that cannot be turned into a decision is only marginally
better than a silent fallback.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")


def _host(host_id, price, *, up=None, total=None, tier=None, score=None):
    """A host as the scheduler sees it, with SLA and reputation joined on."""
    h = {"host_id": host_id, "price_cents_per_hour": price}
    if total is not None:
        h["sla_total_seconds"] = total
        h["sla_downtime_seconds"] = total * (1 - up / 100.0) if up is not None else 0
    if tier is not None:
        h["reputation_tier"] = tier
    if score is not None:
        h["reputation_score"] = score
    return h


#: A year of observation on every host, so these tests exercise the *constraint*
#: rather than the observation window. The original fixtures used 1,000,000
#: seconds — 11.5 days — which silently sat under the 30-day minimum once that
#: was added, and made every uptime test refuse for a reason it was not testing.
YEAR = 365 * 24 * 3600

FLEET = [
    _host("cheap-flaky", 100, up=97.0, total=YEAR, tier="bronze", score=40),
    _host("mid-good", 110, up=99.6, total=YEAR, tier="gold", score=80),
    _host("dear-great", 200, up=99.95, total=YEAR, tier="platinum", score=95),
]


def test_no_preference_takes_the_cheapest(_=None):
    """Calibration. If everything refused, the refusals below prove nothing."""
    from control_plane.scheduler.preference import choose_host

    result = choose_host(FLEET)
    assert result.as_dict()["host_id"] == "cheap-flaky"
    assert result.as_dict()["premium_pct"] == 0.0


def test_a_satisfiable_uptime_constraint_picks_the_cheapest_that_meets_it():
    """Not the most reliable — the cheapest that clears the bar.

    Over-delivering costs the user money they did not agree to spend.
    """
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host(FLEET, PlacementPreference(min_uptime_pct=99.5))
    assert result.as_dict()["host_id"] == "mid-good"


def test_an_unsatisfiable_uptime_constraint_refuses_with_the_number():
    """The clause. Silent fallback to `cheap-flaky` is what this prevents."""
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host(FLEET, PlacementPreference(min_uptime_pct=99.99))
    d = result.as_dict()
    assert d["refused"] is True, "an unsatisfiable preference placed a host anyway"
    assert d["code"] == "uptime_unsatisfiable"
    assert d["asked"] == 99.99
    assert 99.9 < d["best_available"] < 100.0, (
        "the refusal does not say what was actually available, so the user "
        "cannot decide whether to relax it"
    )


def test_a_host_with_no_uptime_history_does_not_satisfy_a_reliability_constraint():
    """Unmeasured is not perfect.

    Otherwise a host with no track record beats one with a year of evidence, at
    exactly the moment the user asked for evidence.
    """
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    newcomer = [_host("brand-new", 50, tier="gold", score=50)]
    result = choose_host(newcomer, PlacementPreference(min_uptime_pct=99.0))
    d = result.as_dict()
    assert d["refused"] is True, "a host with no SLA history satisfied an uptime constraint"
    assert d["best_available"] is None, (
        "the refusal reported a best-available figure for hosts that have none"
    )
    # No history and *too little* history are the same refusal — both mean
    # "there is not enough evidence to judge this", and both are answered by
    # waiting rather than by relaxing the number.
    assert d["code"] == "insufficient_history"


def test_the_premium_is_measured_against_the_cheapest_eligible_host():
    """`mid-good` is 10% dearer than `cheap-flaky`, so a 15% allowance admits it."""
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host(
        FLEET, PlacementPreference(min_uptime_pct=99.5, max_premium_pct=15)
    )
    d = result.as_dict()
    assert d["refused"] is False
    assert d["host_id"] == "mid-good"
    assert d["premium_pct"] == pytest.approx(10.0, abs=0.1)


def test_a_preference_costing_more_than_allowed_refuses():
    """"Even at 15% more" is a bound, and a bound that is not enforced is a hint."""
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host(
        FLEET, PlacementPreference(min_uptime_pct=99.9, max_premium_pct=15)
    )
    d = result.as_dict()
    assert d["refused"] is True
    assert d["code"] == "premium_exceeded"
    assert d["best_available"] == pytest.approx(100.0, abs=0.1), (
        "dear-great is 100% dearer than the baseline; the refusal should say so"
    )


def test_the_premium_baseline_is_not_the_chosen_host():
    """Measured against the chosen host, any bound is vacuous — it is always 0%.

    Asserted separately because it is the mistake that would make every other
    premium test pass while the bound did nothing.
    """
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host(
        FLEET, PlacementPreference(min_uptime_pct=99.9, max_premium_pct=15)
    )
    assert result.as_dict()["refused"] is True, (
        "the premium was measured against the chosen host, so the bound can "
        "never be exceeded and never refuses"
    )


def test_a_tier_constraint_gates_rather_than_ranks():
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host(FLEET, PlacementPreference(min_tier="platinum"))
    assert result.as_dict()["host_id"] == "dear-great"

    refused = choose_host(
        [_host("only-bronze", 10, up=99.9, total=YEAR, tier="bronze")],
        PlacementPreference(min_tier="gold"),
    ).as_dict()
    assert refused["refused"] is True
    assert refused["code"] == "tier_unsatisfiable"
    assert refused["best_available"] == "bronze"


def test_an_unknown_tier_is_refused_rather_than_ignored():
    """A typo must not silently drop the constraint.

    Ignoring an unrecognised tier would place *something* while the user
    believed they had constrained the choice — the silent-fallback failure
    arriving through a spelling mistake.
    """
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host(FLEET, PlacementPreference(min_tier="platnium")).as_dict()
    assert result["refused"] is True
    assert result["code"] == "unknown_tier"


def test_no_eligible_hosts_is_its_own_refusal():
    """Distinct from "the preference was unsatisfiable" — the user should not
    relax a constraint that was never the problem."""
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host([], PlacementPreference(min_uptime_pct=99.0)).as_dict()
    assert result["code"] == "no_eligible_hosts"


def test_the_evidence_is_copied_not_referenced():
    """Gate P5's third clause.

    A trail storing only a host id answers "what is this host's reputation
    *now*", which is a different question and a useless one when reconstructing
    an incident weeks later.
    """
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    host = _host("mid-good", 110, up=99.6, total=YEAR, tier="gold", score=80)
    result = choose_host([host], PlacementPreference(min_uptime_pct=99.5))
    evidence = result.as_dict()["evidence"]

    # The host's score changes after placement, as reputation does.
    host["reputation_score"] = 12
    host["reputation_tier"] = "new_user"

    assert evidence["reputation_score"] == 80, "the record followed the host's later score"
    assert evidence["reputation_tier"] == "gold"
    assert evidence["uptime_pct"] == pytest.approx(99.6, abs=0.01)


# ── What review caught: two ways the gate went vacuous ───────────────


def test_a_short_observation_window_does_not_beat_a_year_of_evidence():
    """§3.1 separated `None` from a number, which was not enough.

    A host with a two-hour window and no downtime reports **100.0%** and cleared
    a 99.5% gate, beating a host with a year at 99.6% — the exact inversion §3.1
    exists to prevent, because `total in (None, 0)` admitted any nonzero window.
    "Has a measurement" was standing in for "has enough measurement".
    """
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    young = _host("young", 50, up=100.0, total=7200)          # 2 hours, spotless
    veteran = _host("veteran", 60, up=99.6, total=YEAR)

    chosen = choose_host([young, veteran], PlacementPreference(min_uptime_pct=99.5))
    assert chosen.as_dict()["host_id"] == "veteran", (
        "two hours of evidence outranked a year — a brand new host reads as "
        "100% and wins a reliability preference"
    )


def test_insufficient_history_is_its_own_refusal():
    """"Wait for history" and "relax the number" are different next steps.

    Collapsing them into `uptime_unsatisfiable` would tell a user to lower a
    constraint that no amount of lowering can satisfy.
    """
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host(
        [_host("young", 50, up=100.0, total=7200)],
        PlacementPreference(min_uptime_pct=99.5),
    ).as_dict()
    assert result["code"] == "insufficient_history"
    assert result["best_available"] is None
    assert "days" in result["detail"], "the refusal does not say how much history exists"


def test_a_host_with_no_usable_price_cannot_become_the_premium_baseline():
    """The premium bound's second vacuous path, and the worse one.

    `_price` fell back to `0` for a missing or zero price. That host sorted
    first, became the baseline, `baseline > 0` failed, and the premium was
    reported as **0.0** — so the same two hosts that correctly refuse at 200%
    over a 15% cap would place on the dearest and report the cap held.

    The trigger is **provider-controlled**: an ask of `0` from one provider
    changed what every other tenant's bound meant, which makes it cross-tenant
    rather than cosmetic.
    """
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    free = _host("free", 0, up=0.0, total=YEAR)
    cheap = _host("cheap", 100, up=97.0, total=YEAR)
    dear = _host("dear", 300, up=99.95, total=YEAR)
    pref = PlacementPreference(min_uptime_pct=99.9, max_premium_pct=15)

    without_free = choose_host([cheap, dear], pref).as_dict()
    with_free = choose_host([free, cheap, dear], pref).as_dict()

    assert without_free["code"] == "premium_exceeded"
    assert with_free["refused"] is True, (
        "a zero-priced host made the premium cap vacuous — the user is told "
        "their cap held while paying 3x"
    )
    assert with_free["code"] == "premium_exceeded"
    assert with_free["best_available"] == without_free["best_available"], (
        "an unpriced third-party host changed the premium this tenant was quoted"
    )


def test_hosts_without_any_usable_price_are_refused_rather_than_placed():
    from control_plane.scheduler.preference import PlacementPreference, choose_host

    result = choose_host(
        [_host("free", 0, up=99.99, total=YEAR)],
        PlacementPreference(max_premium_pct=15),
    ).as_dict()
    assert result["refused"] is True
    assert result["code"] == "no_priced_hosts"


def test_a_negative_or_unparsable_price_is_not_usable():
    """A provider can put anything in that column."""
    from control_plane.scheduler.preference import usable_price

    assert usable_price({"price_cents_per_hour": -5}) is None
    assert usable_price({"price_cents_per_hour": "free"}) is None
    assert usable_price({"price_cents_per_hour": None, "ask_cents_per_hour": 42}) == 42.0


def test_the_tier_vocabulary_is_the_one_production_actually_uses():
    """The invented-vocabulary bug, guarded.

    This module first shipped `("unverified", "basic", "verified", "trusted")`,
    which appears nowhere in the system. Production stores `new_user`, so every
    `min_tier` constraint would have refused `tier_unsatisfiable` against real
    data — and a gate that always refuses is indistinguishable from a broken
    one. It was found by querying production, not by any test here, which is why
    this asserts against the enum rather than a list.
    """
    from reputation import ReputationTier
    from control_plane.scheduler.preference import TIER_ORDER

    assert set(TIER_ORDER) == {t.value for t in ReputationTier}, (
        "the placement tier vocabulary has drifted from ReputationTier; a "
        "constraint naming a tier the system does not store refuses every host"
    )
    # Ordering is by threshold, not by declaration order.
    assert TIER_ORDER[0] == "new_user"
    assert TIER_ORDER[-1] == "diamond"
