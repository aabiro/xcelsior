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

#: Tier ordering, worst to best. A `min_tier` constraint admits its own tier and
#: everything above it.
TIER_ORDER = ("unverified", "basic", "verified", "trusted")


@dataclass(frozen=True)
class PlacementPreference:
    """What the user asked for, separated into gates and bounds."""

    min_uptime_pct: float | None = None
    min_tier: str | None = None
    #: How much more than the cheapest *eligible* host they will pay to satisfy
    #: the constraints. `None` means no bound — they will pay what it costs.
    max_premium_pct: float | None = None

    def is_empty(self) -> bool:
        return (
            self.min_uptime_pct is None
            and self.min_tier is None
            and self.max_premium_pct is None
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


def _price(host: dict) -> float:
    return float(host.get("price_cents_per_hour") or host.get("ask_cents_per_hour") or 0)


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

    ranked = sorted(hosts, key=_price)
    baseline = _price(ranked[0])

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
        measured = [(h, host_uptime_pct(h)) for h in survivors]
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

    if preference.min_tier is not None:
        want = preference.min_tier.strip().lower()
        if want not in TIER_ORDER:
            return PlacementRefused(
                code="unknown_tier",
                detail=f"{preference.min_tier!r} is not a tier; expected one of {list(TIER_ORDER)}",
                asked=preference.min_tier,
            )
        floor = TIER_ORDER.index(want)
        rated = [(h, str(h.get("reputation_tier") or "").strip().lower()) for h in survivors]
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
        "reputation_tier": host.get("reputation_tier"),
        "reputation_score": host.get("reputation_score"),
        "uptime_pct": host_uptime_pct(host),
        "sla_total_seconds": host.get("sla_total_seconds"),
        "sla_downtime_seconds": host.get("sla_downtime_seconds"),
    }
