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


FLEET = [
    _host("cheap-flaky", 100, up=97.0, total=1_000_000, tier="basic", score=40),
    _host("mid-good", 110, up=99.6, total=1_000_000, tier="verified", score=80),
    _host("dear-great", 200, up=99.95, total=1_000_000, tier="trusted", score=95),
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

    newcomer = [_host("brand-new", 50, tier="verified", score=50)]
    result = choose_host(newcomer, PlacementPreference(min_uptime_pct=99.0))
    d = result.as_dict()
    assert d["refused"] is True, "a host with no SLA history satisfied an uptime constraint"
    assert d["best_available"] is None, (
        "the refusal reported a best-available figure for hosts that have none"
    )
    # The distinction the user needs: nothing *measured*, as opposed to measured
    # and insufficient. Those call for different next steps — wait for history,
    # versus relax the number.
    assert "measured uptime history" in d["detail"]


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

    result = choose_host(FLEET, PlacementPreference(min_tier="trusted"))
    assert result.as_dict()["host_id"] == "dear-great"

    refused = choose_host(
        [_host("only-basic", 10, up=99.9, total=1000, tier="basic")],
        PlacementPreference(min_tier="verified"),
    ).as_dict()
    assert refused["refused"] is True
    assert refused["code"] == "tier_unsatisfiable"
    assert refused["best_available"] == "basic"


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

    host = _host("mid-good", 110, up=99.6, total=1_000_000, tier="verified", score=80)
    result = choose_host([host], PlacementPreference(min_uptime_pct=99.5))
    evidence = result.as_dict()["evidence"]

    # The host's score changes after placement, as reputation does.
    host["reputation_score"] = 12
    host["reputation_tier"] = "unverified"

    assert evidence["reputation_score"] == 80, "the record followed the host's later score"
    assert evidence["reputation_tier"] == "verified"
    assert evidence["uptime_pct"] == pytest.approx(99.6, abs=0.01)
