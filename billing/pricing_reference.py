"""GPU rates read from the imported Stripe compute catalog.

Every amount here is a Stripe Price that
``api/scripts/sync_stripe_service_tiers.py::import_catalog`` wrote into
``gpu_pricing`` from ``billing/stripe_catalog.json`` (deploy runs it through
``run_db_migrations.py --import-compute-catalog``; the test session imports the
same snapshot). Nothing in this module derives a price from a multiplier or a
static table: a GPU, variant or mode without an imported Price has no rate.

Free-text names resolve to one variant the same way a launch quote does
(``service_tier_pricing.tier_catalog``): the requested VRAM and form factor when
given, otherwise the smallest VRAM, then the standard-frequency pairing, then
form factor.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from money import micros_to_cad

log = logging.getLogger("xcelsior.pricing_reference")

PRICING_MODES = ("on_demand", "spot", "reserved_1mo", "reserved_3mo", "reserved_1yr")

_VRAM_RE = re.compile(r"\b(\d{1,3})\s*GB\b", re.IGNORECASE)
_FORM_FACTOR_RE = re.compile(r"\b(SXM\d?|PCIe|OAM)\b", re.IGNORECASE)

_MODE_FIELDS = {
    "on_demand": "base_rate_cad",
    "spot": "spot_cad",
    "reserved_1mo": "reserved_1mo_cad",
    "reserved_3mo": "reserved_3mo_cad",
    "reserved_1yr": "reserved_1yr_cad",
}

_COMMITMENT_TO_MODE = {
    "1_month": "reserved_1mo",
    "3_month": "reserved_3mo",
    "1_year": "reserved_1yr",
}


def commitment_pricing_mode(commitment_type: str) -> str | None:
    return _COMMITMENT_TO_MODE.get(commitment_type)


def parse_gpu_name(raw: str) -> tuple[str, int | None, str | None]:
    """Split a free-text GPU name into (canonical model, VRAM GB, form factor).

    ``"NVIDIA A100 80GB SXM"`` -> ``("A100", 80, "SXM")``;
    ``"rtx 4090"`` -> ``("RTX 4090", None, None)``. The model is canonical only
    when the catalogue knows it; otherwise it is returned as given and has no
    Price.
    """
    from host_metadata import normalize_gpu_model

    text = str(raw or "")
    vram = _VRAM_RE.search(text)
    form = _FORM_FACTOR_RE.search(text)
    model = normalize_gpu_model(" ".join(_FORM_FACTOR_RE.sub(" ", _VRAM_RE.sub(" ", text)).split()))
    form_factor = None
    if form:
        token = form.group(1).upper()
        form_factor = "PCIe" if token == "PCIE" else ("OAM" if token == "OAM" else "SXM")
    return model, (int(vram.group(1)) if vram else None), form_factor


def _variant_price(
    gpu_model: str,
    pricing_mode: str,
    *,
    tier: str,
    vram_gb: int | None,
    form_factor: str | None,
) -> dict[str, Any] | None:
    """The imported Price row for one model/mode/tier, on the quoted variant."""
    from db import pg_connection
    from psycopg.rows import dict_row

    with pg_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        # Same variant order as service_tier_pricing.tier_catalog.
        return cur.execute(
            """SELECT gpu_model, vram_gb, form_factor, high_frequency,
                      base_rate_micros, stripe_price_id
                 FROM gpu_pricing
                WHERE gpu_model=%s AND tier=%s AND pricing_mode=%s AND active
                  AND stripe_price_id IS NOT NULL
                  AND (%s::integer IS NULL OR vram_gb=%s)
                  AND (%s::text IS NULL OR form_factor=%s)
                ORDER BY vram_gb, high_frequency, form_factor
                LIMIT 1""",
            (gpu_model, tier, pricing_mode, vram_gb, vram_gb, form_factor, form_factor),
        ).fetchone()


def get_rate(
    gpu_model: str,
    pricing_mode: str = "on_demand",
    *,
    tier: str = "standard",
    vram_gb: int | None = None,
) -> float:
    """Imported Stripe rate per GPU-hour in CAD, or 0.0 when the catalog has none."""
    model, parsed_vram, form_factor = parse_gpu_name(gpu_model)
    if not model:
        return 0.0
    try:
        row = _variant_price(
            model,
            pricing_mode,
            tier=tier,
            vram_gb=vram_gb if vram_gb is not None else parsed_vram,
            form_factor=form_factor,
        )
    except Exception as exc:
        log.warning("gpu_pricing lookup failed for %s/%s: %s", model, pricing_mode, exc)
        return 0.0
    return micros_to_cad(row["base_rate_micros"]) if row else 0.0


def quote_gpu(
    gpu_name: str,
    pricing_mode: str = "on_demand",
    *,
    num_gpus: int = 1,
    tier: str = "standard",
) -> dict[str, Any]:
    """The launch quote (``service_tier_pricing.quote_tier``) for a free-text GPU name.

    Raises ``TierPricingUnavailable`` when no Stripe Price covers the GPU,
    the requested VRAM or form factor, or the mode.
    """
    from service_tier_pricing import TierPricingUnavailable, quote_tier

    model, vram_gb, form_factor = parse_gpu_name(gpu_name)
    price_id = None
    if form_factor:
        # quote_tier selects a variant by VRAM; pin the form factor by its Price.
        row = _variant_price(
            model, pricing_mode, tier="standard", vram_gb=vram_gb, form_factor=form_factor
        )
        if row is None:
            raise TierPricingUnavailable("Stripe compute pricing is unavailable for this GPU")
        price_id = row["stripe_price_id"]
    return quote_tier(
        model, pricing_mode, tier=tier, num_gpus=num_gpus, vram_gb=vram_gb, price_id=price_id
    )


def estimate_job_cost(
    gpu_model: str,
    duration_hours: float,
    *,
    pricing_mode: str = "on_demand",
    num_gpus: int = 1,
) -> dict[str, Any]:
    """What ``duration_hours`` of a GPU costs at its Standard Stripe Price.

    The rate is the quote a launch is charged (``quote_gpu``) and the total uses
    the metering rounding (``usage_charge_micros``), so an estimate and the
    eventual usage charge agree. Raises ``TierPricingUnavailable`` for a GPU
    or mode without a Price; there is no stand-in rate.
    """
    from service_tier_pricing import usage_charge_micros

    quote = quote_gpu(gpu_model, pricing_mode, num_gpus=num_gpus)
    cost_micros = usage_charge_micros(quote["total_rate_micros"], float(duration_hours) * 3600)
    return {
        "gpu_model": quote["gpu_model"],
        "vram_gb": quote["vram_gb"],
        "form_factor": quote["form_factor"],
        "pricing_mode": pricing_mode,
        "spot": pricing_mode == "spot",
        "num_gpus": num_gpus,
        "duration_hours": duration_hours,
        "stripe_price_id": quote["stripe_price_id"],
        "rate_per_gpu_cad": micros_to_cad(quote["unit_amount_micros"]),
        "rate_cad_per_hour": micros_to_cad(quote["total_rate_micros"]),
        "gross_cost_cad": micros_to_cad(cost_micros),
        "currency": "CAD",
    }


def _cad(micros: int | None) -> float | None:
    return None if micros is None else micros_to_cad(micros)


def build_gpu_pricing_reference() -> dict[str, dict[str, Any]]:
    """Per-GPU Stripe rates for each model's default variant, cheapest first.

    Standard-tier rates for every pricing mode, plus the Premium and Dedicated
    on-demand Prices. Empty when the catalog is not imported or unreadable.
    """
    from db import pg_connection
    from psycopg.rows import dict_row

    try:
        with pg_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            # Variant order matches service_tier_pricing.tier_catalog.
            rows = cur.execute("""SELECT gpu_model, vram_gb, form_factor, high_frequency,
                          tier, pricing_mode, base_rate_micros
                     FROM gpu_pricing
                    WHERE active AND stripe_price_id IS NOT NULL
                    ORDER BY gpu_model, vram_gb, high_frequency, form_factor""").fetchall()
    except Exception as exc:
        log.warning("gpu_pricing reference query failed: %s", exc)
        return {}

    variants: dict[str, tuple] = {}
    micros: dict[str, dict[tuple[str, str], int]] = {}
    for row in rows:
        model = str(row["gpu_model"])
        variant = (row["vram_gb"], row["form_factor"], row["high_frequency"])
        if variants.setdefault(model, variant) != variant:
            continue
        micros.setdefault(model, {})[(row["tier"], row["pricing_mode"])] = int(
            row["base_rate_micros"]
        )

    pricing: dict[str, dict[str, Any]] = {}
    for model, rates in micros.items():
        if ("standard", "on_demand") not in rates:
            continue
        vram_gb, form_factor, high_frequency = variants[model]
        entry: dict[str, Any] = {
            "vram_gb": vram_gb,
            "form_factor": form_factor,
            "high_frequency": high_frequency,
        }
        for mode, field in _MODE_FIELDS.items():
            entry[field] = _cad(rates.get(("standard", mode)))
        entry["premium_rate_cad"] = _cad(rates.get(("premium", "on_demand")))
        entry["dedicated_rate_cad"] = _cad(rates.get(("dedicated", "on_demand")))
        pricing[model] = entry
    return dict(sorted(pricing.items(), key=lambda kv: (kv[1]["base_rate_cad"], kv[0])))


def build_reference_list(pricing: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Flat list shape consumed by ``frontend/src/lib/api.ts``."""
    data = pricing if pricing is not None else build_gpu_pricing_reference()
    out: list[dict[str, Any]] = []
    for model, rates in data.items():
        out.append(
            {
                "gpu_model": model,
                "vram_gb": rates.get("vram_gb"),
                "on_demand_cad": rates.get("base_rate_cad"),
                "spot_cad": rates.get("spot_cad"),
                "reserved_1mo_cad": rates.get("reserved_1mo_cad"),
                "reserved_3mo_cad": rates.get("reserved_3mo_cad"),
                "reserved_1yr_cad": rates.get("reserved_1yr_cad"),
            }
        )
    return out


def get_on_demand_rate(gpu_model: str, *, tier: str = "standard") -> float:
    """On-demand Stripe rate for reserve/commitment flows (0.0 when unpriced)."""
    return get_rate(gpu_model, "on_demand", tier=tier)


def get_reserved_rate(gpu_model: str, commitment_type: str, *, tier: str = "standard") -> float:
    """Reserved Stripe rate for a commitment term (0.0 when unpriced)."""
    mode = commitment_pricing_mode(commitment_type)
    return get_rate(gpu_model, mode, tier=tier) if mode else 0.0
