#!/usr/bin/env python3
"""Seed Stripe Products & Prices for the Xcelsior GPU catalog.

Creates one Stripe Product per canonical GPU variant (model × VRAM × form factor
× high-frequency) and attaches metered hourly CAD prices for every tier × mode
combination — matching db.py ``_GPU_PRICING_BASE`` expansion exactly.

Idempotent: uses stable ``lookup_key`` on prices and ``metadata.xcelsior_sku`` on
products. Re-run safe.

Usage:
  python3 scripts/seed_stripe_products.py --dry-run
  python3 scripts/seed_stripe_products.py --apply
  python3 scripts/seed_stripe_products.py --apply --mode sandbox
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Repo root on path for db imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from db import _GPU_PRICING_BASE, _generate_gpu_pricing_rows  # noqa: E402

TIER_LABELS = {
    "standard": "Standard",
    "premium": "Premium",
    "sovereign": "Sovereign",
}

MODE_LABELS = {
    "on_demand": "On-Demand",
    "spot": "Spot",
    "reserved_1mo": "Reserved 1 Month",
    "reserved_3mo": "Reserved 3 Months",
    "reserved_1yr": "Reserved 1 Year",
}

WALLET_PRODUCT_SKU = "xcelsior-compute-credits"
METER_EVENT_NAME = "xcelsior_gpu_hour"
SERVERLESS_METER_EVENT = "xcelsior_serverless_worker_second"
STORAGE_METER_EVENT = "xcelsior_storage_gb_month"
MANIFEST_PATH = ROOT / "config" / "stripe_catalog.json"

# Preset wallet top-up amounts (CAD) — applied dynamically at checkout, not a product picker.
WALLET_TOPUP_PRESETS_CAD = [5, 10, 25, 50, 100, 250, 500]

# Non-GPU platform billables (wallet-debited today; Stripe SKUs for dynamic line items).
PLATFORM_SERVICES: list[dict] = [
    {
        "sku": "xcelsior-serverless-inference",
        "name": "Xcelsior Serverless Inference",
        "description": (
            "Serverless GPU endpoints — billed per worker running second at the "
            "selected GPU tier rate (Novita-aligned metering)."
        ),
        "product_type": "serverless",
        "meter_event": SERVERLESS_METER_EVENT,
        "meter_display": "Xcelsior Serverless Worker Seconds",
        "meter_value_key": "seconds",
    },
    {
        "sku": "xcelsior-persistent-storage",
        "name": "Xcelsior Persistent Storage",
        "description": "Persistent block volumes — $0.03 CAD per GB per month.",
        "product_type": "volume_storage",
        "meter_event": STORAGE_METER_EVENT,
        "meter_display": "Xcelsior Storage GB-Months",
        "meter_value_key": "gb_months",
        "rate_cad_per_gb_month": 0.03,
    },
    {
        "sku": "xcelsior-serverless-input-tokens",
        "name": "Xcelsior Serverless Input Tokens",
        "description": "Observability / future billing — input tokens per million (reference rate).",
        "product_type": "serverless_tokens_input",
        "rate_cad_per_million": 0.50,
    },
    {
        "sku": "xcelsior-serverless-output-tokens",
        "name": "Xcelsior Serverless Output Tokens",
        "description": "Observability / future billing — output tokens per million (reference rate).",
        "product_type": "serverless_tokens_output",
        "rate_cad_per_million": 1.50,
    },
]

# Reserved commitment plan products (discount tiers — GPU hourly rates use reserved_* modes).
RESERVED_COMMITMENT_PRODUCTS: list[dict] = [
    {
        "sku": "xcelsior-reserved-1-month",
        "name": "Xcelsior Reserved — 1 Month",
        "description": "20% off on-demand GPU rates · 1-month commitment · min 4 hrs/day.",
        "commitment_type": "1_month",
        "discount_pct": 20,
    },
    {
        "sku": "xcelsior-reserved-3-month",
        "name": "Xcelsior Reserved — 3 Months",
        "description": "30% off on-demand GPU rates · 3-month commitment · min 4 hrs/day.",
        "commitment_type": "3_month",
        "discount_pct": 30,
    },
    {
        "sku": "xcelsior-reserved-1-year",
        "name": "Xcelsior Reserved — 1 Year",
        "description": "45% off on-demand GPU rates · 1-year commitment.",
        "commitment_type": "1_year",
        "discount_pct": 45,
    },
]


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _resolve_secret_key(mode: str) -> str:
    if mode == "sandbox":
        return (
            os.environ.get("XCELSIOR_STRIPE_SANDBOX_SECRET_KEY", "")
            or os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")
        )
    return os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")


def _slug(*parts: str) -> str:
    raw = ":".join(parts)
    return re.sub(r"[^a-z0-9:]+", "-", raw.lower().replace(" ", "-"))


def _variant_key(model: str, vram: int, form_factor: str, high_freq: bool) -> str:
    return _slug("xcelsior", model, str(vram), form_factor, "hf1" if high_freq else "hf0")


def _price_lookup_key(
    model: str,
    vram: int,
    form_factor: str,
    high_freq: bool,
    tier: str,
    mode: str,
) -> str:
    return _slug("xc", model, str(vram), form_factor, "hf1" if high_freq else "hf0", tier, mode)


def _product_name(model: str, vram: int, form_factor: str, high_freq: bool) -> str:
    """Customer-facing product title — canonical short GPU name."""
    name = model
    if high_freq:
        name = f"{name} (High-Frequency CPU)"
    return name


def _product_description(model: str, vram: int, form_factor: str, high_freq: bool) -> str:
    bits = [f"{vram}GB {form_factor}"]
    if high_freq:
        bits.append("high-frequency CPU pairing")
    bits.append("Xcelsior GPU compute — billed per GPU-hour from wallet balance")
    return " · ".join(bits)


def _group_variants() -> dict[str, dict]:
    """Group pricing rows by hardware variant."""
    groups: dict[str, dict] = {}
    for row in _generate_gpu_pricing_rows():
        model, vram, form_factor, high_freq, tier, mode, rate, *_ = row
        key = _variant_key(model, vram, form_factor, high_freq)
        if key not in groups:
            groups[key] = {
                "model": model,
                "vram_gb": vram,
                "form_factor": form_factor,
                "high_frequency": high_freq,
                "prices": [],
            }
        groups[key]["prices"].append(
            {
                "tier": tier,
                "mode": mode,
                "rate_cad": rate,
                "lookup_key": _price_lookup_key(model, vram, form_factor, high_freq, tier, mode),
                "nickname": f"{TIER_LABELS[tier]} · {MODE_LABELS[mode]}",
            }
        )
    return groups


def _cad_to_decimal_cents(rate_cad: float) -> str:
    """Stripe unit_amount_decimal is cents as a decimal string."""
    cents = rate_cad * 100
    # Avoid float noise — 4 dp on rate → up to 6 dp on cents
    return f"{cents:.4f}".rstrip("0").rstrip(".")


def _stripe_metadata(obj) -> dict[str, str]:
    """Normalize Stripe metadata StripeObject → plain dict."""
    raw = getattr(obj, "metadata", None)
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    try:
        return {str(k): str(raw[k]) for k in raw.keys()}
    except Exception:
        return {}


def _find_product_by_sku(stripe, sku: str):
    # Paginate active products with our metadata
    starting_after = None
    while True:
        params: dict = {"limit": 100, "active": True}
        if starting_after:
            params["starting_after"] = starting_after
        page = stripe.Product.list(**params)
        for prod in page.data:
            if _stripe_metadata(prod).get("xcelsior_sku") == sku:
                return prod
        if not page.has_more:
            break
        starting_after = page.data[-1].id
    return None


def _find_price_by_lookup(stripe, lookup_key: str):
    try:
        prices = stripe.Price.list(lookup_keys=[lookup_key], limit=1)
        if prices.data:
            return prices.data[0]
    except Exception:
        pass
    return None


def _ensure_gpu_meter(stripe, dry_run: bool, manifest: dict) -> str:
    """Platform meter for GPU-hour usage (required for metered prices since 2025-03-31)."""
    if dry_run:
        meter_id = "mtr_DRY_GPU_HOUR"
        manifest["gpu_meter"] = {"id": meter_id, "event_name": METER_EVENT_NAME}
        print(f"  [meter] would create: {METER_EVENT_NAME}")
        return meter_id

    try:
        page = stripe.billing.Meter.list(limit=100)
        for m in page.data:
            if getattr(m, "event_name", None) == METER_EVENT_NAME:
                print(f"  [meter] exists: {m.id}")
                manifest["gpu_meter"] = {"id": m.id, "event_name": METER_EVENT_NAME}
                return m.id
    except Exception as exc:
        print(f"  [meter] list failed: {exc}", file=sys.stderr)

    meter = stripe.billing.Meter.create(
        display_name="Xcelsior GPU Hours",
        event_name=METER_EVENT_NAME,
        default_aggregation={"formula": "sum"},
        customer_mapping={"type": "by_id", "event_payload_key": "stripe_customer_id"},
        value_settings={"event_payload_key": "hours"},
    )
    print(f"  [meter] created: {meter.id}")
    manifest["gpu_meter"] = {"id": meter.id, "event_name": METER_EVENT_NAME}
    return meter.id


def _ensure_wallet_product(stripe, dry_run: bool, manifest: dict) -> None:
    sku = WALLET_PRODUCT_SKU
    existing = None if dry_run else _find_product_by_sku(stripe, sku)
    product_data = {
        "name": "Xcelsior Compute Credits",
        "description": (
            "Prepaid CAD wallet balance for GPU compute on Xcelsior. "
            "Deposits credit your account; usage is metered per GPU-hour."
        ),
        "metadata": {
            "xcelsior_sku": sku,
            "xcelsior_catalog": "platform",
            "product_type": "wallet_deposit",
        },
        "statement_descriptor": "XCELSIOR CREDITS",
        "tax_code": "txcd_10000000",  # General - Electronically Supplied Services
    }
    if existing:
        product_id = existing.id
        print(f"  [wallet] product exists: {product_id}")
    elif dry_run:
        product_id = "prod_DRY_WALLET"
        print("  [wallet] would create: Xcelsior Compute Credits")
    else:
        prod = stripe.Product.create(**product_data)
        product_id = prod.id
        print(f"  [wallet] created product: {product_id}")

    manifest["wallet_product"] = {"sku": sku, "product_id": product_id, **product_data}
    presets = _seed_wallet_topup_presets(stripe, dry_run, product_id)
    manifest["wallet_product"]["preset_prices"] = presets


def _ensure_meter(
    stripe,
    dry_run: bool,
    *,
    event_name: str,
    display_name: str,
    value_key: str,
    manifest_key: str,
    manifest: dict,
) -> str:
    if dry_run:
        mid = f"mtr_DRY_{event_name[:20]}"
        manifest[manifest_key] = {"id": mid, "event_name": event_name}
        print(f"  [meter] would create: {event_name}")
        return mid
    try:
        page = stripe.billing.Meter.list(limit=100)
        for m in page.data:
            if getattr(m, "event_name", None) == event_name:
                manifest[manifest_key] = {"id": m.id, "event_name": event_name}
                print(f"  [meter] exists: {m.id} ({event_name})")
                return m.id
    except Exception as exc:
        print(f"  [meter] list failed for {event_name}: {exc}", file=sys.stderr)
    meter = stripe.billing.Meter.create(
        display_name=display_name,
        event_name=event_name,
        default_aggregation={"formula": "sum"},
        customer_mapping={"type": "by_id", "event_payload_key": "stripe_customer_id"},
        value_settings={"event_payload_key": value_key},
    )
    manifest[manifest_key] = {"id": meter.id, "event_name": event_name}
    print(f"  [meter] created: {meter.id} ({event_name})")
    return meter.id


def _seed_wallet_topup_presets(stripe, dry_run: bool, product_id: str) -> list[dict]:
    out: list[dict] = []
    for amount in WALLET_TOPUP_PRESETS_CAD:
        lk = f"xc-wallet-topup-{amount}-cad"
        existing = None if dry_run else _find_price_by_lookup(stripe, lk)
        if existing:
            out.append({"amount_cad": amount, "price_id": existing.id, "lookup_key": lk})
            continue
        if dry_run:
            out.append({"amount_cad": amount, "price_id": f"price_DRY_{amount}", "lookup_key": lk})
            continue
        pr = stripe.Price.create(
            product=product_id,
            currency="cad",
            unit_amount=int(amount * 100),
            lookup_key=lk,
            nickname=f"Wallet top-up ${amount} CAD",
            metadata={"product_type": "wallet_deposit", "amount_cad": str(amount)},
        )
        out.append({"amount_cad": amount, "price_id": pr.id, "lookup_key": lk})
        time.sleep(0.03)
    print(f"  [wallet] {len(out)} preset top-up prices")
    return out


def _seed_platform_services(stripe, dry_run: bool, manifest: dict) -> int:
    created = 0
    manifest["platform_services"] = []
    for svc in PLATFORM_SERVICES:
        sku = svc["sku"]
        existing = None if dry_run else _find_product_by_sku(stripe, sku)
        payload = {
            "name": svc["name"],
            "description": svc["description"],
            "metadata": {
                "xcelsior_sku": sku,
                "xcelsior_catalog": "platform",
                "product_type": svc["product_type"],
            },
            "statement_descriptor": "XCELSIOR",
            "tax_code": "txcd_10000000",
        }
        if existing:
            product_id = existing.id
        elif dry_run:
            product_id = f"prod_DRY_{sku[:30]}"
            created += 1
        else:
            prod = stripe.Product.create(**payload)
            product_id = prod.id
            created += 1
            time.sleep(0.05)

        entry: dict = {"sku": sku, "product_id": product_id, **svc, "prices": []}
        meter_id = None
        if svc.get("meter_event"):
            meter_id = _ensure_meter(
                stripe,
                dry_run,
                event_name=svc["meter_event"],
                display_name=svc["meter_display"],
                value_key=svc["meter_value_key"],
                manifest_key=f"meter_{svc['product_type']}",
                manifest=manifest,
            )
        if meter_id and svc.get("rate_cad_per_gb_month"):
            lk = f"xc-{sku}-gb-month"
            rate = svc["rate_cad_per_gb_month"]
            if not dry_run and not _find_price_by_lookup(stripe, lk):
                pr = stripe.Price.create(
                    product=product_id,
                    currency="cad",
                    unit_amount_decimal=_cad_to_decimal_cents(rate),
                    lookup_key=lk,
                    nickname="Per GB-month",
                    recurring={"interval": "month", "usage_type": "metered", "meter": meter_id},
                    metadata={"rate_cad_per_gb_month": str(rate)},
                )
                entry["prices"].append({"price_id": pr.id, "lookup_key": lk})
            elif dry_run:
                entry["prices"].append({"price_id": f"price_DRY_{lk}", "lookup_key": lk})
        elif svc.get("rate_cad_per_million"):
            lk = f"xc-{sku}-per-million"
            cents = int(svc["rate_cad_per_million"] * 100)
            if not dry_run and not _find_price_by_lookup(stripe, lk):
                pr = stripe.Price.create(
                    product=product_id,
                    currency="cad",
                    unit_amount=cents,
                    lookup_key=lk,
                    nickname="Per 1M tokens (reference)",
                    metadata={"rate_cad_per_million": str(svc["rate_cad_per_million"])},
                )
                entry["prices"].append({"price_id": pr.id, "lookup_key": lk})
            elif dry_run:
                entry["prices"].append({"price_id": f"price_DRY_{lk}", "lookup_key": lk})

        manifest["platform_services"].append(entry)
        print(f"  [{'exists' if existing else 'created' if not dry_run else 'dry-run'}] {svc['name']}")
    return created


def _seed_reserved_commitments(stripe, dry_run: bool, manifest: dict) -> int:
    created = 0
    manifest["reserved_commitments"] = []
    for plan in RESERVED_COMMITMENT_PRODUCTS:
        sku = plan["sku"]
        existing = None if dry_run else _find_product_by_sku(stripe, sku)
        payload = {
            "name": plan["name"],
            "description": plan["description"],
            "metadata": {
                "xcelsior_sku": sku,
                "xcelsior_catalog": "reserved_commitment",
                "commitment_type": plan["commitment_type"],
                "discount_pct": str(plan["discount_pct"]),
            },
            "statement_descriptor": "XCELSIOR RSVD",
            "tax_code": "txcd_10000000",
        }
        if existing:
            product_id = existing.id
        elif dry_run:
            product_id = f"prod_DRY_{sku[:30]}"
            created += 1
        else:
            prod = stripe.Product.create(**payload)
            product_id = prod.id
            created += 1
            time.sleep(0.05)
        manifest["reserved_commitments"].append({**plan, "product_id": product_id})
        print(f"  [{'exists' if existing else 'created' if not dry_run else 'dry-run'}] {plan['name']}")
    return created


def _seed_gpu_catalog(stripe, dry_run: bool, manifest: dict, meter_id: str) -> tuple[int, int]:
    groups = _group_variants()
    created_products = 0
    created_prices = 0
    manifest["gpu_variants"] = []

    for key, variant in sorted(groups.items(), key=lambda x: (x[1]["model"], x[1]["vram_gb"])):
        model = variant["model"]
        vram = variant["vram_gb"]
        ff = variant["form_factor"]
        hf = variant["high_frequency"]
        sku = key

        product_payload = {
            "name": _product_name(model, vram, ff, hf),
            "description": _product_description(model, vram, ff, hf),
            "metadata": {
                "xcelsior_sku": sku,
                "xcelsior_catalog": "gpu_compute",
                "gpu_model": model,
                "vram_gb": str(vram),
                "form_factor": ff,
                "high_frequency": "true" if hf else "false",
            },
            "statement_descriptor": "XCELSIOR GPU",
            "tax_code": "txcd_10000000",
        }

        existing = None if dry_run else _find_product_by_sku(stripe, sku)
        if existing:
            product_id = existing.id
        elif dry_run:
            product_id = f"prod_DRY_{sku[:40]}"
            created_products += 1
        else:
            prod = stripe.Product.create(**product_payload)
            product_id = prod.id
            created_products += 1
            time.sleep(0.05)  # gentle rate limit

        variant_entry = {
            "sku": sku,
            "product_id": product_id,
            "gpu_model": model,
            "vram_gb": vram,
            "form_factor": ff,
            "high_frequency": hf,
            "prices": [],
        }

        for price_row in variant["prices"]:
            lk = price_row["lookup_key"]
            rate = price_row["rate_cad"]
            existing_price = None if dry_run else _find_price_by_lookup(stripe, lk)

            price_payload = {
                "product": product_id,
                "currency": "cad",
                "nickname": price_row["nickname"],
                "lookup_key": lk,
                "unit_amount_decimal": _cad_to_decimal_cents(rate),
                # Catalog price: CAD per GPU-hour. Applied dynamically as checkout line items
                # (never a customer-facing product picker). Meter backs future usage events.
                "recurring": {
                    "interval": "month",
                    "usage_type": "metered",
                    "meter": meter_id,
                },
                "billing_scheme": "per_unit",
                "metadata": {
                    "xcelsior_sku": sku,
                    "gpu_model": model,
                    "tier": price_row["tier"],
                    "pricing_mode": price_row["mode"],
                    "rate_cad_per_hr": str(rate),
                    "vram_gb": str(vram),
                    "form_factor": ff,
                    "high_frequency": "true" if hf else "false",
                },
                "tax_behavior": "exclusive",
            }

            if existing_price:
                price_id = existing_price.id
            elif dry_run:
                price_id = f"price_DRY_{lk[:40]}"
                created_prices += 1
            else:
                pr = stripe.Price.create(**price_payload)
                price_id = pr.id
                created_prices += 1
                time.sleep(0.05)

            variant_entry["prices"].append(
                {
                    "price_id": price_id,
                    "lookup_key": lk,
                    "tier": price_row["tier"],
                    "mode": price_row["mode"],
                    "rate_cad_per_hr": rate,
                    "nickname": price_row["nickname"],
                }
            )

        manifest["gpu_variants"].append(variant_entry)
        status = "exists" if existing else ("dry-run" if dry_run else "created")
        print(
            f"  [{status}] {model} {vram}GB {ff}"
            f"{' HF' if hf else ''} — {len(variant['prices'])} prices"
        )

    return created_products, created_prices


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Stripe GPU product catalog for Xcelsior")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without Stripe API writes")
    parser.add_argument("--apply", action="store_true", help="Create/update products in Stripe")
    parser.add_argument(
        "--mode",
        choices=["live", "sandbox"],
        default=None,
        help="Override XCELSIOR_STRIPE_MODE from .env",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        parser.error("Specify --dry-run or --apply")

    _load_dotenv()
    mode = args.mode or os.environ.get("XCELSIOR_STRIPE_MODE", "live").lower()
    secret = _resolve_secret_key(mode)
    if not secret.startswith("sk_"):
        print(f"ERROR: No valid Stripe secret key for mode={mode}", file=sys.stderr)
        return 1

    try:
        import stripe as stripe_mod
    except ImportError:
        print("ERROR: pip install stripe", file=sys.stderr)
        return 1

    stripe_mod.api_key = secret
    base_count = len(_GPU_PRICING_BASE)
    row_count = len(_generate_gpu_pricing_rows())
    variant_count = len(_group_variants())

    print(f"Stripe mode: {mode}")
    print(f"Catalog: {base_count} base variants → {variant_count} products × 12 prices = {row_count} price rows")
    print()

    manifest: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stripe_mode": mode,
        "base_variants": base_count,
        "products": variant_count + 1,  # + wallet
        "price_rows": row_count,
    }

    print("Platform meter + wallet:")
    meter_id = _ensure_gpu_meter(stripe_mod, args.dry_run, manifest)
    _ensure_wallet_product(stripe_mod, args.dry_run, manifest)
    print()
    print("Platform services (serverless, storage, tokens):")
    plat_created = _seed_platform_services(stripe_mod, args.dry_run, manifest)
    print()
    print("Reserved commitment plans:")
    rsv_created = _seed_reserved_commitments(stripe_mod, args.dry_run, manifest)
    print()
    print("GPU compute catalog (dynamic line-item SKUs — no checkout product picker):")
    created_p, created_pr = _seed_gpu_catalog(stripe_mod, args.dry_run, manifest, meter_id)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print()
    print(f"Manifest written: {MANIFEST_PATH}")

    if args.dry_run:
        print(
            f"DRY RUN — GPU ~{created_p} products / ~{created_pr} prices; "
            f"platform +{plat_created}; reserved +{rsv_created}"
        )
    else:
        print(
            f"Done — GPU {created_p} products, {created_pr} prices; "
            f"platform {plat_created}; reserved {rsv_created}"
        )

    print()
    print("── Stripe Dashboard polish ──")
    print("  Run: python3 scripts/polish_stripe_dashboard.py --apply")
    print("  Then paste brand file IDs in Dashboard → Settings → Branding (platform Account.modify is N/A)")
    print("  Webhooks: /api/providers/webhook + /api/connect/webhooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())