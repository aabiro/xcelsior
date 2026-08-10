"""One ranker, two paths — and the parity that proves nothing moved.

`allocate_best_host` ranks by compute efficiency weighted by reputation.
`choose_host` used to take `survivors[0]` in price order. Those are two
**selection policies**, not two implementations of one, and folding constrained
requests into the second would have changed where those jobs land with nothing
erroring — the failure mode Gate P5 clause 2 exists to close, arriving through
the fix for Gate P5 clause 2.

So the ranker was extracted to `control_plane.scheduler.ranking` and neither
path owns it. Three things have to hold, and each is asserted rather than
assumed:

1. **Unconstrained placement is unchanged.** Driven against an oracle that is a
   verbatim copy of the closure that used to live in `allocate_best_host`.
2. **A constrained request ranks by efficiency too**, not by price — otherwise
   asking for verification silently also asks for the cheapest machine.
3. **`allocate_best_host` still returns `dict | None` on every path**, so a
   later "let's unify these" refactor goes red instead of reintroducing the
   truthiness hazard: `PlacementRefused` is a frozen dataclass with no
   `__bool__`, so `if not host: continue` would pass it straight through to
   `host["host_id"]`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from control_plane.scheduler.preference import (  # noqa: E402
    PlacementPreference,
    PlacementRefused,
    choose_host,
)
from control_plane.scheduler.ranking import host_efficiency_score  # noqa: E402
from scheduler import allocate_best_host, estimate_compute_score  # noqa: E402

YEAR = 365 * 24 * 3600


def _legacy_score_host(h):
    """**Verbatim** the closure that lived in `allocate_best_host` before the
    extraction, minus the reputation lookup (which is unchanged and would make
    this an oracle for the database rather than for the arithmetic).

    Kept as a literal copy on purpose: an oracle rewritten in the new style
    proves the new style agrees with itself.
    """
    compute = h.get("compute_score") or estimate_compute_score(h.get("gpu_model", ""))
    price = h.get("cost_per_hour", 0.20) or 0.20
    return compute / price


def _legacy_host(host_id, *, gpu_model, cost_per_hour, compute_score=None, vram=48):
    host = {
        "host_id": host_id,
        "gpu_model": gpu_model,
        "cost_per_hour": cost_per_hour,
        "total_vram_gb": vram,
        "free_vram_gb": vram,
    }
    if compute_score is not None:
        host["compute_score"] = compute_score
    return host


#: Deliberately a fleet where cheapest and most-efficient are different hosts.
#: A fleet where they agree cannot tell the two policies apart, which is exactly
#: why the 33 existing preference tests all still passed after the change.
FLEET = [
    _legacy_host("slow-cheap", gpu_model="RTX 3060", cost_per_hour=0.10, compute_score=1.0),
    _legacy_host("fast-dearer", gpu_model="RTX 4090", cost_per_hour=0.20, compute_score=8.3),
    _legacy_host("mid", gpu_model="RTX 4080", cost_per_hour=0.18, compute_score=4.0),
]


# ── 1. Unconstrained placement is unchanged ───────────────────────────


def test_the_ranker_orders_the_fleet_exactly_as_the_old_closure_did():
    """Parity, against a literal copy of the code that was replaced."""
    by_new = sorted(FLEET, key=host_efficiency_score, reverse=True)
    by_old = sorted(FLEET, key=_legacy_score_host, reverse=True)
    assert [h["host_id"] for h in by_new] == [h["host_id"] for h in by_old]


def test_dollars_and_cents_do_not_change_the_order():
    """The move to cents is a constant factor, and `max` cannot see it.

    Mixing the two units *within* one list would change the order by 100×, which
    is why `normalise_price_cents` is the only price reader in the ranker.
    """
    ratios = [
        host_efficiency_score(h) / _legacy_score_host(h)
        for h in FLEET
        if _legacy_score_host(h)
    ]
    assert len(set(round(r, 9) for r in ratios)) == 1, (
        "hosts are being scaled by different factors, so the two paths would "
        "rank the same fleet differently"
    )


def test_allocate_best_host_places_the_efficient_host_not_the_cheapest():
    """The behaviour the oracle above is protecting."""
    chosen = allocate_best_host({"vram_needed_gb": 24}, list(FLEET))
    assert chosen["host_id"] == "fast-dearer"


# ── 2. A constrained request ranks the same way ───────────────────────


def _candidate(host_id, price_cents, *, gpu_model, compute_score, state="verified"):
    """A projected row, as `attach_placement_evidence` produces one."""
    import time

    now = time.time()
    return {
        "host_id": host_id,
        "price_cents_per_hour": price_cents,
        "gpu_model": gpu_model,
        "compute_score": compute_score,
        "verification_state": state,
        "verified_at": now - 3600,
        "deverified_at": None,
        "last_check_at": now - 3600,
        "next_check_at": now + 82800,
        "verification_unavailable": False,
        "reputation_tier": "new_user",
        "reputation_score": 50.0,
        "sla_total_seconds": YEAR,
        "sla_downtime_seconds": 0.0,
    }


CONSTRAINED = [
    _candidate("slow-cheap", 10, gpu_model="RTX 3060", compute_score=1.0),
    _candidate("fast-dearer", 20, gpu_model="RTX 4090", compute_score=8.3),
]


def test_a_constrained_request_gets_the_efficient_survivor():
    """Asking for verification is not asking for the cheapest machine.

    Before the extraction this returned `slow-cheap` — `survivors[0]` in price
    order — so stating a preference silently switched selection policy.
    """
    cheapest = min(CONSTRAINED, key=lambda h: h["price_cents_per_hour"])["host_id"]
    assert cheapest == "slow-cheap", (
        "this fixture no longer distinguishes the two policies, so the "
        "assertion below would pass under either"
    )

    result = choose_host(list(CONSTRAINED), PlacementPreference(require_verified=True))
    assert not isinstance(result, PlacementRefused), getattr(result, "detail", "")
    assert result.host["host_id"] == "fast-dearer", (
        "the constrained path ranked by price while the unconstrained path "
        "ranks by efficiency"
    )


def test_the_premium_is_still_measured_against_the_cheapest_eligible_host():
    """Cheapest-eligible → chosen-efficient. That *is* what the constraint cost."""
    result = choose_host(list(CONSTRAINED), PlacementPreference(require_verified=True))
    assert result.baseline_price == 10, "the baseline moved off the cheapest eligible host"
    assert result.chosen_price == 20
    assert result.premium_pct == pytest.approx(100.0)


#: A premium only exists when the cheapest *eligible* host is not a survivor —
#: otherwise the baseline is itself the answer and the premium is 0%. So the
#: cheapest host here fails the constraint.
PREMIUM_FLEET = [
    _candidate("unverified-cheap", 10, gpu_model="RTX 3060",
               compute_score=1.0, state="unverified"),
    _candidate("mid", 11, gpu_model="RTX 4080", compute_score=4.0),
    _candidate("fast-dearer", 20, gpu_model="RTX 4090", compute_score=8.3),
]


def test_a_premium_bound_picks_the_best_host_that_fits_it():
    """The bound filters, then the ranker chooses among what is left.

    `fast-dearer` wins on efficiency but costs 100% over the baseline. Refusing
    on that while `mid` sat inside the cap would be over-refusing; placing on
    `fast-dearer` anyway would ignore a bound the user stated.
    """
    result = choose_host(
        list(PREMIUM_FLEET),
        PlacementPreference(require_verified=True, max_premium_pct=15),
    )
    assert not isinstance(result, PlacementRefused), getattr(result, "detail", "")
    assert result.host["host_id"] == "mid", (
        "a survivor inside the premium bound was passed over for one outside it"
    )
    assert result.premium_pct == pytest.approx(10.0)


def test_a_premium_bound_refuses_with_the_cheapest_survivors_number():
    """When no survivor fits, the number reported is the smallest overage.

    If the cheapest survivor is over the cap then every survivor is, so quoting
    it tells the user the least they would have to relax by.
    """
    result = choose_host(
        list(PREMIUM_FLEET),
        PlacementPreference(require_verified=True, max_premium_pct=5),
    )
    assert isinstance(result, PlacementRefused)
    assert result.code == "premium_exceeded"
    assert result.best_available == pytest.approx(10.0), (
        "the refusal quoted the efficient host's premium rather than the "
        "cheapest survivor's, overstating what the user must relax by"
    )


# ── 3. The truthiness hazard stays closed ─────────────────────────────


@pytest.mark.parametrize(
    "hosts, vram",
    [
        ([], 0),                                    # nothing to place on
        (FLEET, 999),                               # nothing fits the VRAM ask
        (FLEET, 24),                                # normal
        ([_legacy_host("only", gpu_model="", cost_per_hour=0)], 0),  # unpriced
    ],
)
def test_allocate_best_host_returns_a_dict_or_none_on_every_path(hosts, vram):
    """`PlacementRefused` must never reach this function's callers.

    It is a frozen dataclass with no `__bool__`, so `if not host: continue`
    passes it through and `host["host_id"]` raises TypeError on the next line.
    The temptation is to give it a falsy `__bool__` — do not: that converts a
    loud crash into a silently dropped job, which is the same silent-fallback
    family this gate exists to close.
    """
    result = allocate_best_host({"vram_needed_gb": vram}, list(hosts))
    assert result is None or isinstance(result, dict)


def test_neither_path_carries_its_own_ranker():
    """The drift guard. Two rankers in two files is how placement diverges."""
    source = Path("scheduler.py").read_text()
    assert "def score_host" not in source, (
        "allocate_best_host grew a local ranker again; both paths must rank "
        "through control_plane.scheduler.ranking"
    )
    assert "host_efficiency_score" in source
    assert "host_efficiency_score" in Path(
        "control_plane/scheduler/preference.py"
    ).read_text()
