"""How a host is ranked, in one place, so two paths cannot rank differently.

`scheduler.allocate_best_host` ranks by **compute efficiency weighted by
reputation** — XCU per unit price, times the host's search boost.
`preference.choose_host` sorted by price and took the first survivor, which is a
**different selection policy**, not a second implementation of the same one.

That difference is invisible and it matters: a user who states
`require_verified` and nothing else has asked for verification. Answering with
*the cheapest* verified host rather than the best one they could have had is
answering a question they did not ask — the same objection `choose_host`'s own
`premium_exceeded` refusal raises about placing over a stated bound.

So the ranker lives here and neither path owns it. The premium is still measured
against the **cheapest eligible** host, because "15% more" has to be 15% more
than a number the user would recognise; what changes is only which survivor is
picked once the constraints have been applied.

## Why this is unit-safe

The legacy path reads `cost_per_hour` in dollars and the preference path reads
`price_cents_per_hour` in cents. Ranking is by `max`, so a constant factor
cannot change the order — but *mixing* the two within one list would, by 100×.
`normalise_price_cents` is therefore the only price reader here, exactly as it
is in the projection.
"""

from __future__ import annotations

import logging

from control_plane.scheduler.host_projection import normalise_price_cents

log = logging.getLogger(__name__)

#: Stand-in when a host publishes no usable price, in cents per hour.
#:
#: Preserves `allocate_best_host`'s existing `h.get("cost_per_hour", 0.20)`
#: fallback exactly — $0.20/hour — so unconstrained placement ranks identically
#: to how it does today. It is a fail-open default and it is kept only for that
#: parity: the preference path never reaches it, because `usable_price` has
#: already made an unpriced host ineligible.
DEFAULT_PRICE_CENTS = 20.0


def host_efficiency_score(host: dict) -> float:
    """Compute per unit price, weighted by reputation. Higher is better.

    A faithful move of `allocate_best_host`'s inner `score_host`, including the
    reputation lookup and its `except: boost = 1.0`. Unconstrained placement
    must land on the same host it lands on today, so the behaviour is preserved
    rather than improved in passing.
    """
    from scheduler import estimate_compute_score

    compute = host.get("compute_score") or estimate_compute_score(host.get("gpu_model", ""))
    price = normalise_price_cents(host) or DEFAULT_PRICE_CENTS
    efficiency = compute / price

    boost = 1.0
    try:
        from scheduler import get_reputation_engine

        boost = get_reputation_engine().compute_score(host["host_id"]).search_boost
    except Exception as exc:
        # Unchanged from the original: an unreachable reputation store ranks
        # every host neutrally rather than failing the placement.
        log.debug("reputation boost unavailable for %s: %s", host.get("host_id"), exc)

    return efficiency * boost
