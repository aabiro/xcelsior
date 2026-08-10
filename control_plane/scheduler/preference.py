"""Placement preference: constraints that gate, and a premium that bounds.

C0 of `docs/placement-preference-plan.md`. Pure functions over a host list, so
the refusal semantics can be driven directly rather than through a scheduler.

The clause this exists for, from Gate P5: *"a placement preference that cannot be
satisfied **refuses clearly** rather than silently falling back to the cheapest
host. This is the failure mode that would quietly destroy trust."*

## The distinction the whole module turns on

A **preference over eligible hosts** ranks them. A **constraint** decides who is
eligible at all. Both are useful; they are not interchangeable, and the plan's
own example contains both — *"above 99.5% uptime"* gates, *"even at 15% more"*
bounds what the gate may cost.

Treating a constraint as a ranking is what produces the silent fallback: the
99.1% host sorts last but still wins when nothing better exists, and the user is
told their reliable placement succeeded.

## Why "no history" fails a reliability constraint

A host with no `sla_monthly` row has no *measured* uptime. Treating unmeasured as
perfect lets a host with no track record beat one with a year of evidence, at
precisely the moment a user asked for evidence. It is harsh on new providers and
it is the right default; earning history is the provider surface's job, not
something granted here by omission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

def _tier_order() -> tuple[str, ...]:
    """Tier names worst-to-best, **derived from `reputation.TIER_THRESHOLDS`**.

    The first version of this module invented `("unverified", "basic",
    "verified", "trusted")`. Production uses `new_user`/`bronze`/`silver`/
    `gold`/`platinum`/`diamond`, so every `min_tier` constraint would have
    refused with `tier_unsatisfiable` against real data — a gate that always
    refuses is indistinguishable from a broken one, and it was found by querying
    production rather than by any test here.

    Deriving it also fixes the ordering by *threshold* rather than by the order
    someone typed the names, so a tier inserted between two others sorts
    correctly without this file being touched.
    """
    from reputation import TIER_THRESHOLDS

    return tuple(
        str(getattr(tier, "value", tier))
        for tier, _ in sorted(TIER_THRESHOLDS.items(), key=lambda kv: kv[1])
    )


#: Snapshot for callers that want the vocabulary without importing `reputation`.
TIER_ORDER = _tier_order()

#: The shortest observation window that may satisfy a reliability constraint.
#:
#: Separating `None` from a number was not enough, and the gap was the exact
#: inversion §3.1 exists to prevent: a host with a two-hour window and no
#: downtime reports **100.0%** and clears a 99.5% gate, beating a host with a
#: year of evidence at 99.6%. `total in (None, 0)` admitted any nonzero window,
#: so "has a measurement" was standing in for "has enough measurement".
#:
#: **A floor on a summed trailing window, not on one month's row.** "Thirty days
#: is one `sla_monthly` period" was wrong arithmetic: rows are per calendar
#: month, and February's maximum is 2,419,200s against this 2,592,000s minimum,
#: so **no February row could ever satisfy it** — and no current month until day
#: 30. The threshold was only ever reachable by summing across rows, which
#: quietly settled the aggregation question C1 had written down as still open.
#:
#: So it is settled deliberately instead: the projection sums `total_seconds`
#: and `downtime_seconds` over a **trailing window** and hands C0 the totals.
#: That also stops `min_uptime_pct` lurching every 1st of the month, which a
#: per-month reading would cause — a preference that passes on the 31st and
#: refuses on the 1st is a preference nobody can rely on.
MIN_OBSERVATION_SECONDS = 30 * 24 * 3600

#: How far back the projection sums. Stated here because C0's threshold is
#: meaningless without it: 30 days of evidence out of 90 is a different claim
#: from 30 out of 30.
OBSERVATION_WINDOW_DAYS = 90


@dataclass(frozen=True)
class PlacementPreference:
    """What the user asked for, separated into gates and bounds."""

    min_uptime_pct: float | None = None
    min_tier: str | None = None
    #: The gate's own example is *"prefer **verified** hosts above 99.5% uptime"*,
    #: and `verified` is not a reputation tier in this system — it is
    #: `host_verifications.state`, a four-state machine with its own checker and
    #: recheck schedule. Expressing that example needs this axis; `min_tier`
    #: cannot carry it, and pointing `min_tier` at reputation only moved the
    #: mistake onto a real vocabulary on the wrong dimension.
    require_verified: bool = False
    #: How much more than the cheapest *eligible* host they will pay to satisfy
    #: the constraints. `None` means no bound — they will pay what it costs.
    max_premium_pct: float | None = None

    def is_empty(self) -> bool:
        return (
            self.min_uptime_pct is None
            and self.min_tier is None
            and self.max_premium_pct is None
            and not self.require_verified
        )


@dataclass(frozen=True)
class PlacementRefused:
    """A typed refusal carrying the number that failed.

    Not an exception: the launch route and the MCP tool have to render this as a
    trade-off the user can act on, and a string cannot be rendered into a
    choice. "No host matched" is not an answer anyone can do anything with;
    "the best available uptime is 99.1%, you asked for 99.5%" is.
    """

    code: str
    detail: str
    asked: Any = None
    best_available: Any = None

    def as_dict(self) -> dict:
        return {
            "refused": True,
            "code": self.code,
            "detail": self.detail,
            "asked": self.asked,
            "best_available": self.best_available,
        }


@dataclass(frozen=True)
class PlacementChoice:
    """The host chosen, and what was true about it when it was chosen."""

    host: dict
    baseline_price: float
    chosen_price: float
    premium_pct: float
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "refused": False,
            "host_id": str(self.host.get("host_id")),
            "baseline_price": self.baseline_price,
            "chosen_price": self.chosen_price,
            "premium_pct": round(self.premium_pct, 2),
            "evidence": self.evidence,
        }


def host_uptime_pct(host: dict) -> float | None:
    """Measured uptime, or None when there is no measurement.

    `None` is not 100. Every caller has to decide what to do with "unknown", and
    making it explicit is what stops it silently becoming "fine".
    """
    total = host.get("sla_total_seconds")
    if total in (None, 0):
        return None
    down = float(host.get("sla_downtime_seconds") or 0)
    return max(0.0, min(100.0, (1.0 - down / float(total)) * 100.0))


def usable_price(host: dict) -> float | None:
    """The host's price, or None when it does not have a usable one.

    **A host without a usable price is ineligible**, and that is a correctness
    fix rather than tidiness. The previous version fell back to `0`, which sorted
    such a host first and made it the premium baseline; `baseline > 0` then
    failed and the premium was reported as **0.0**. The same two hosts that
    correctly refuse at 200% over a 15% cap would place on the dearest one and
    report that the cap held.

    The input is **provider-controlled**. An ask of `0` from one provider
    silently changed what every other tenant's premium bound meant, which makes
    this a cross-tenant defect rather than a display bug.
    """
    for key in ("price_cents_per_hour", "ask_cents_per_hour"):
        raw = host.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def host_tier(host: dict) -> str | None:
    """The host's tier, **derived from its score** when a score is present.

    `reputation_scores.tier` carries `server_default 'bronze'` while
    `final_score` defaults to `0`, and `score_to_tier(0)` is `new_user`. A row
    inserted without an explicit tier therefore sits at **bronze on zero earned
    score**, and would satisfy `min_tier="bronze"` on no evidence — §3.1's
    argument in the reputation dimension, fail-open, written into the schema.

    Deriving is the conservative direction: where the stored column and the
    score disagree, the score wins and the host ranks lower. The stored value is
    used only when there is no score to derive from.
    """
    score = host.get("reputation_score")
    if score is not None:
        try:
            from reputation import score_to_tier

            return str(getattr(score_to_tier(float(score)), "value", score_to_tier(float(score))))
        except (TypeError, ValueError):
            pass
    tier = host.get("reputation_tier")
    return str(tier).strip().lower() if tier else None


#: How many reverification intervals a stamp may lapse before it stops counting.
#:
#: Refusing the instant a sweep is late would make the constraint hostage to the
#: sweep's punctuality; seven intervals is slack enough to absorb an outage and
#: short enough that a four-month-old stamp never passes.
VERIFICATION_STALE_AFTER_INTERVALS = 7


def verification_max_age_seconds() -> float:
    """The staleness tolerance, **derived from the interval policy it depends on**.

    This was a typed `7 * 24 * 3600` whose docstring explained it as "a generous
    multiple of `reverify_interval_sec`" — with the multiple living in the
    comment and not in the code. That is the `TIER_ORDER` defect again: raise
    the interval to a week and the tolerance silently becomes 1×; raise it to a
    month and this gate refuses hosts that are perfectly current. Deriving it
    means the relationship cannot drift out of the sentence describing it.
    """
    try:
        from verification import VERIFICATION_THRESHOLDS

        interval = float(VERIFICATION_THRESHOLDS.get("reverify_interval_sec") or 0)
    except Exception:  # pragma: no cover - verification module is optional here
        interval = 0.0
    if interval <= 0:
        interval = 86400.0
    return interval * VERIFICATION_STALE_AFTER_INTERVALS


#: Snapshot for readers; the function above is the authority.
VERIFICATION_MAX_AGE_SECONDS = verification_max_age_seconds()


def verification_status(host: dict, *, now: float | None = None) -> str:
    """`verified`, `stale`, or the raw state — the third instance of one shape.

    `state = 'verified'` means "was verified once and nothing has revoked it
    since". It is not a current fact: `reverify_interval_sec` is 86400 and
    `list_hosts_needing_reverification()` implements the due query, but its
    only wrapper `get_hosts_needing_reverification()` **has no callers** — no
    worker, no scheduler pass, no timer. `next_check_at` is written and never
    read, so nothing ever moves a host out of `verified`.

    On production both verified hosts are overdue by **111 and 124 days**
    against a one-day interval.

    That is the same shape as "unmeasured is not perfect" and "a two-hour window
    is not a year": **measured, but not recently enough to be evidence.** So it
    gets the same treatment and its own code, because "no verified hosts exist"
    and "the checker is not running" call for different next steps — the first
    is answered by verifying a host, the second by starting a sweep.
    """
    import time as _time

    state = str(host.get("verification_state") or "").strip().lower()
    if state != "verified":
        return state or "unknown"
    checked = host.get("last_check_at")
    if checked is None:
        return "stale"
    try:
        age = (now if now is not None else _time.time()) - float(checked)
    except (TypeError, ValueError):
        return "stale"
    return "verified" if age <= verification_max_age_seconds() else "stale"


def host_verified(host: dict, *, now: float | None = None) -> bool:
    """True only for a host verified *and* checked recently enough to mean it."""
    return verification_status(host, now=now) == "verified"


def _price(host: dict) -> float:
    """Sort key. Only ever called on hosts already known to have a price."""
    return usable_price(host) or 0.0


def choose_host(
    hosts: Sequence[dict],
    preference: PlacementPreference | None = None,
) -> PlacementChoice | PlacementRefused:
    """Pick a host, or refuse and say why.

    `hosts` are already past the hard filters — this decides among hosts that
    *could* run the job, not whether any can.

    The baseline is computed **before** the constraints are applied, because
    "15% more" has to be 15% more than something the user would recognise: the
    cheapest host that could have run the job. Measuring the premium against the
    chosen host instead would make the bound vacuous, and against the cheapest
    host overall would compare against something ineligible.
    """
    if not hosts:
        return PlacementRefused(
            code="no_eligible_hosts",
            detail="no host satisfies the job's requirements",
        )

    # A constrained request cannot be honoured on unreadable evidence.
    #
    # `scheduler.py` logs and proceeds when the verification store is
    # unavailable, which is right for the *unconstrained* path — a cold start
    # should not stop every launch. On a request that asked for verification it
    # is the same silent fallback one layer down, and C1 would go on to write
    # "verified at placement" onto evidence nobody managed to read.
    if preference is not None and preference.require_verified:
        unreadable = [h for h in hosts if h.get("verification_unavailable")]
        if unreadable:
            return PlacementRefused(
                code="verification_unreadable",
                detail=(
                    "verification state could not be read for "
                    f"{len(unreadable)} candidate host(s), so `require_verified` "
                    "cannot be honoured. Placing anyway would record a verified "
                    "placement that nobody checked."
                ),
                asked="verified",
            )

    priced = [h for h in hosts if usable_price(h) is not None]
    if not priced:
        return PlacementRefused(
            code="no_priced_hosts",
            detail=(
                f"{len(hosts)} host(s) match the job but none publishes a usable "
                "price, so no premium bound can be honoured"
            ),
        )

    ranked = sorted(priced, key=_price)
    baseline = _price(ranked[0])
    if baseline <= 0:
        # Unreachable given the filter above, and asserted anyway: a zero
        # baseline is what silently turns every premium into 0%.
        return PlacementRefused(
            code="no_priced_hosts",
            detail="the cheapest eligible host has no usable price",
        )

    if preference is None or preference.is_empty():
        chosen = ranked[0]
        return PlacementChoice(
            host=chosen,
            baseline_price=baseline,
            chosen_price=_price(chosen),
            premium_pct=0.0,
            evidence=placement_evidence(chosen),
        )

    survivors = list(ranked)

    if preference.min_uptime_pct is not None:
        # Two separate refusals, because they call for different next steps:
        # "wait for history" is not "relax the number".
        observed = [
            h for h in survivors
            if float(h.get("sla_total_seconds") or 0) >= MIN_OBSERVATION_SECONDS
        ]
        if not observed:
            longest = max(
                (float(h.get("sla_total_seconds") or 0) for h in survivors), default=0.0
            )
            return PlacementRefused(
                code="insufficient_history",
                detail=(
                    "no candidate has enough measured history to judge "
                    f"reliability: the longest observation window is "
                    f"{longest / 86400:.1f} days, and "
                    f"{MIN_OBSERVATION_SECONDS / 86400:.0f} are required. A short "
                    "window with no downtime reads as 100% and would otherwise "
                    "beat a host with a year of evidence."
                ),
                asked=preference.min_uptime_pct,
                best_available=None,
            )
        measured = [(h, host_uptime_pct(h)) for h in observed]
        survivors = [h for h, up in measured if up is not None and up >= preference.min_uptime_pct]
        if not survivors:
            known = [up for _, up in measured if up is not None]
            return PlacementRefused(
                code="uptime_unsatisfiable",
                detail=(
                    f"no host meets {preference.min_uptime_pct}% uptime; "
                    + (
                        f"the best measured is {max(known):.2f}%"
                        if known
                        else "no candidate has any measured uptime history"
                    )
                ),
                asked=preference.min_uptime_pct,
                best_available=max(known) if known else None,
            )

    if preference.require_verified:
        statuses = [(h, verification_status(h)) for h in survivors]
        survivors = [h for h, status in statuses if status == "verified"]
        if not survivors:
            seen = sorted({st for _, st in statuses if st}) or ["none recorded"]
            # Stale is its own refusal. "No verified hosts exist" is answered by
            # verifying one; "the checker is not running" is answered by
            # starting the sweep. Collapsing them sends the operator after the
            # wrong problem.
            if "stale" in seen:
                return PlacementRefused(
                    code="verification_stale",
                    detail=(
                        "candidate hosts carry a `verified` state but no recent "
                        f"check — the stamp must be under "
                        f"{verification_max_age_seconds() / 86400:.0f} days old to "
                        "count as current. `get_hosts_needing_reverification()` "
                        "has no callers, so nothing moves a host out of "
                        "`verified` and the stamp ages indefinitely."
                    ),
                    asked="verified",
                    best_available="stale",
                )
            return PlacementRefused(
                code="verification_unsatisfiable",
                detail=(
                    "no candidate host is verified; the states present are "
                    + ", ".join(seen)
                    + ". A deverified host is one whose verification was revoked, "
                    "which is further from acceptable than one never checked."
                ),
                asked="verified",
                best_available=seen[0] if seen != ["none recorded"] else None,
            )

    if preference.min_tier is not None:
        want = preference.min_tier.strip().lower()
        if want not in TIER_ORDER:
            return PlacementRefused(
                code="unknown_tier",
                detail=f"{preference.min_tier!r} is not a tier; expected one of {list(TIER_ORDER)}",
                asked=preference.min_tier,
            )
        floor = TIER_ORDER.index(want)
        rated = [(h, host_tier(h) or "") for h in survivors]
        survivors = [h for h, tier in rated if tier in TIER_ORDER and TIER_ORDER.index(tier) >= floor]
        if not survivors:
            present = [t for _, t in rated if t in TIER_ORDER]
            best = max(present, key=TIER_ORDER.index) if present else None
            return PlacementRefused(
                code="tier_unsatisfiable",
                detail=(
                    f"no host is {want} or better"
                    + (f"; the best available is {best}" if best else "")
                ),
                asked=want,
                best_available=best,
            )

    chosen = survivors[0]
    chosen_price = _price(chosen)
    premium = ((chosen_price - baseline) / baseline * 100.0) if baseline > 0 else 0.0

    if preference.max_premium_pct is not None and premium > preference.max_premium_pct:
        # The user said what reliability was worth to them, and this exceeds it.
        # Placing anyway would be answering a question they did not ask.
        return PlacementRefused(
            code="premium_exceeded",
            detail=(
                f"the cheapest host meeting the preference costs {premium:.1f}% "
                f"more than the cheapest eligible host, above the "
                f"{preference.max_premium_pct}% you allowed"
            ),
            asked=preference.max_premium_pct,
            best_available=round(premium, 1),
        )

    return PlacementChoice(
        host=chosen,
        baseline_price=baseline,
        chosen_price=chosen_price,
        premium_pct=premium,
        evidence=placement_evidence(chosen),
    )


def placement_evidence(host: dict) -> dict:
    """What was true about this host at the moment it was chosen.

    Gate P5 asks that reputation and SLA "at time of placement" are recorded.
    Storing a host id and re-reading its score later answers "what is this
    host's reputation *now*" — a different question, and a useless one when
    reconstructing an incident weeks afterwards. So the numbers are copied.
    """
    return {
        # Derived, not the stored column — see `host_tier`.
        "reputation_tier": host_tier(host),
        "reputation_score": host.get("reputation_score"),
        "uptime_pct": host_uptime_pct(host),
        "sla_total_seconds": host.get("sla_total_seconds"),
        "sla_downtime_seconds": host.get("sla_downtime_seconds"),
        # **Verification is revocable**, which makes it precisely the fact that
        # goes stale: a host verified at placement can be deverified the next
        # day, and an incident review needs what was true then. `verified_at`
        # comes with it, because "verified" alone does not say how recently.
        "verification_state": host.get("verification_state"),
        "verified_at": host.get("verified_at"),
        "deverified_at": host.get("deverified_at"),
        # `verified_at` alone does not say whether anyone was still *looking*.
        # These two are what distinguish a live check from a stamp nobody has
        # revisited, and an incident review needs the difference.
        "last_check_at": host.get("last_check_at"),
        "next_check_at": host.get("next_check_at"),
        "verification_status": verification_status(host),
    }
