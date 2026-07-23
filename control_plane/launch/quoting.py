"""Versioned quoting for a canonical launch spec (§14.1 steps 4–5, §15.4).

A quote binds a *price to a version*. The action plan stores the estimate
and the pricing version; execute (B2.5) re-quotes and, if the version moved
the price beyond the plan's tolerance, refuses with ``quote_changed`` rather
than silently charging the new rate (§15.4). Money is integer micro-CAD, the
ledger unit — the same unit the wallet hold is created in, so the tolerance
comparison never round-trips through a float.

The rate itself is not reinvented here: it comes from the existing
:meth:`billing.BillingEngine.estimate_launch_hold_cad`, the one authority
for "what does one hour of this launch cost". Quoting only versions it,
scales it to a runway, and converts to micro-CAD. Quoting performs no write
and must be called *outside* any open transaction (§10.4 — no external
pricing call inside a control-plane transaction).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from money import cad_to_micros

# Bump when the quoting *method* changes (not when a rate changes). The
# per-quote ``pricing_version`` below also encodes the rate, so a rate move
# is detectable without bumping this.
PRICING_CATALOG_VERSION = "v1"

# Default worst-case runway a preview reserves when the caller does not
# specify one: one hour of burn (matches the Track A launch-hold estimate).
DEFAULT_RUNWAY_HOURS = 1.0


@dataclass(frozen=True)
class Quote:
    quote_id: str
    pricing_version: str
    currency: str
    # Burn rate and the worst-case amount the plan authorizes (the runway).
    hourly_burn_micros: int
    estimate_micros: int
    runway_hours: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "quote_id": self.quote_id,
            "pricing_version": self.pricing_version,
            "currency": self.currency,
            "hourly_burn_micros": self.hourly_burn_micros,
            "estimate_micros": self.estimate_micros,
            "runway_hours": self.runway_hours,
        }


def _hourly_burn_cad(spec: dict[str, Any], host: dict[str, Any] | None) -> float:
    """One hour of this launch, in CAD — reuses the billing estimate authority."""
    from billing import get_billing_engine

    return get_billing_engine().estimate_launch_hold_cad(
        pricing_mode=str(spec.get("pricing_mode") or "on_demand"),
        gpu_model=str(spec.get("gpu_model") or "") or None,
        num_gpus=int(spec.get("num_gpus") or 1),
        host=host,
    )


def quote_launch(
    spec: dict[str, Any],
    *,
    runway_hours: float = DEFAULT_RUNWAY_HOURS,
    host: dict[str, Any] | None = None,
    currency: str = "CAD",
) -> Quote:
    """Quote a canonical spec. Pure with respect to persistence; no writes.

    ``runway_hours`` is the worst-case duration the plan authorizes funds
    for; the estimate is ``hourly_burn * runway_hours``. The ``pricing_version``
    encodes both the catalog version and the exact hourly rate, so a later
    re-quote at a different rate produces a different version — which is how
    B3.4 detects a price change without a second pricing store.
    """
    runway = max(0.0, float(runway_hours))
    hourly_cad = _hourly_burn_cad(spec, host)
    hourly_micros = cad_to_micros(hourly_cad)
    estimate_micros = cad_to_micros(hourly_cad * runway)
    # Version is deterministic in the inputs: same rate -> same version, so a
    # replayed quote is byte-stable and a moved rate is visibly different.
    pricing_version = f"{PRICING_CATALOG_VERSION}:{hourly_micros}:{currency}"
    return Quote(
        quote_id=f"q_{uuid.uuid4().hex}",
        pricing_version=pricing_version,
        currency=currency,
        hourly_burn_micros=hourly_micros,
        estimate_micros=estimate_micros,
        runway_hours=runway,
    )


def price_moved_beyond_tolerance(
    old_estimate_micros: int,
    new_estimate_micros: int,
    tolerance_bps: int,
) -> bool:
    """True when the re-quote exceeds the plan's allowed tolerance (§15.4).

    Basis points of the original estimate. A drop in price is never a
    violation — the customer is only protected against paying *more* than the
    quote they approved.
    """
    if new_estimate_micros <= old_estimate_micros:
        return False
    if old_estimate_micros <= 0:
        return new_estimate_micros > 0
    increase = new_estimate_micros - old_estimate_micros
    # increase / old > tolerance/10000  <=>  increase*10000 > tolerance*old
    return increase * 10_000 > int(tolerance_bps) * old_estimate_micros
