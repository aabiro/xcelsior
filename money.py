"""Money representation helpers (Track B B9.5, companion §4.4 rule 6).

"Monetary values use integer minor units or ``NUMERIC``, never binary
floats." Binary floating point cannot represent most decimal cent values
exactly, so a running balance accumulates error: measured against this
system's own database, 1000 postings of $0.07 sum to 69.99999999999966
rather than 70.00. The per-operation error is negligible; what breaks is
that ``sum(amount)`` over a ledger stops equalling the stored balance, and
that equality is what finance reconciliation checks.

These helpers convert at the boundary. Arithmetic happens on the integer
side.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

# CAD has two minor digits. Kept explicit rather than assumed, so a
# future zero-decimal currency (JPY) does not silently inherit a x100.
MINOR_UNITS_PER_MAJOR = 100


def cad_to_minor(amount_cad: float | str | Decimal) -> int:
    """CAD dollars -> integer cents, without a binary-float round trip.

    ``Decimal(str(x))`` reads the *decimal* value the caller meant (25.0)
    rather than the float's binary expansion (24.999999999999996), and
    ROUND_HALF_UP is the convention for currency. Note that the naive
    ``int(x * 100)`` is wrong in ordinary cases, not exotic ones:
    ``int(1.15 * 100)`` is 114.
    """
    return int(
        (Decimal(str(amount_cad)) * MINOR_UNITS_PER_MAJOR).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def minor_to_cad(amount_minor: int) -> float:
    """Integer cents -> CAD dollars, for float-typed legacy interfaces.

    Exact for every value a wallet can hold: cents/100 is representable in
    binary floating point well past any realistic balance. Use this only
    at a boundary that demands a float — never as an intermediate step in
    a calculation, which would reintroduce the drift.
    """
    return int(amount_minor) / MINOR_UNITS_PER_MAJOR


# ── Micro-CAD: the ledger unit ────────────────────────────────────────
# Cents are too coarse for this product. Xcelsior meters GPU-seconds and
# tokens, which produce sub-cent charges — a real per-tick charge of
# $0.0073 rounds to $0.01 in cents, a 37% overcharge repeated every tick.
# The ledger's unit must be at least as fine as the smallest amount the
# business meters, so the wallet stores micro-CAD (1e-6). A BIGINT of
# micro-CAD tops out around $9.2 trillion.
#
# `ln_deposits` keeps cents (migration 066): a deposit has a $1 minimum
# and is never sub-cent, and converting whole cents to micros is exact,
# so the two scales meet cleanly at the deposit->wallet boundary.
MICROS_PER_CAD = 1_000_000


def cad_to_micros(amount_cad: float | str | Decimal) -> int:
    """CAD dollars -> integer micro-CAD, without a binary-float round trip."""
    return int(
        (Decimal(str(amount_cad)) * MICROS_PER_CAD).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def micros_to_cad(amount_micros: int) -> float:
    """Integer micro-CAD -> dollars, for float-typed legacy interfaces."""
    return int(amount_micros) / MICROS_PER_CAD
