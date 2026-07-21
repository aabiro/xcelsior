"""Stripe catalog helpers — SKU resolution and invoice line-item metadata.

Products exist in Stripe for reconciliation and future Stripe Invoice sync.
Customers never pick from a product list. Usage is wallet-debited continuously;
itemized lines appear on generated invoices (on request or monthly).
Top-up is the only customer-facing checkout (Payment Element / wallet deposit).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("xcelsior.stripe_catalog")

_ROOT = Path(__file__).resolve().parent
_MANIFEST_PATH = _ROOT / "config" / "stripe_catalog.json"

_TIER_LABELS = {
    "standard": "Standard",
    "premium": "Premium",
    "sovereign": "Sovereign",
}

_MODE_LABELS = {
    "on_demand": "On-Demand",
    "spot": "Spot",
    "reserved_1mo": "Reserved 1 Month",
    "reserved_3mo": "Reserved 3 Months",
    "reserved_1yr": "Reserved 1 Year",
}

_PLATFORM_SKUS = {
    "serverless": "xcelsior-serverless-inference",
    "storage": "xcelsior-persistent-storage",
}


def _slug(*parts: str) -> str:
    raw = ":".join(parts)
    return re.sub(r"[^a-z0-9:]+", "-", raw.lower().replace(" ", "-"))


def price_lookup_key(
    gpu_model: str,
    vram_gb: int,
    form_factor: str,
    high_frequency: bool,
    tier: str,
    mode: str,
) -> str:
    return _slug(
        "xc",
        gpu_model,
        str(vram_gb),
        form_factor,
        "hf1" if high_frequency else "hf0",
        tier,
        mode,
    )


def variant_sku(gpu_model: str, vram_gb: int, form_factor: str, high_frequency: bool) -> str:
    return _slug("xcelsior", gpu_model, str(vram_gb), form_factor, "hf1" if high_frequency else "hf0")


def load_manifest() -> dict[str, Any]:
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(_MANIFEST_PATH.read_text())
    except Exception as exc:
        log.warning("Could not load stripe catalog manifest: %s", exc)
        return {}


def resolve_product_id(
    gpu_model: str,
    *,
    vram_gb: int = 0,
    form_factor: str = "PCIe",
    high_frequency: bool = False,
    manifest: dict | None = None,
) -> str | None:
    data = manifest if manifest is not None else load_manifest()
    sku = variant_sku(gpu_model, vram_gb, form_factor, high_frequency)
    for variant in data.get("gpu_variants", []):
        if variant.get("sku") == sku:
            return variant.get("product_id")
    for variant in data.get("gpu_variants", []):
        if variant.get("gpu_model") == gpu_model:
            return variant.get("product_id")
    return None


def resolve_platform_product_id(line_type: str, manifest: dict | None = None) -> str | None:
    data = manifest if manifest is not None else load_manifest()
    sku = _PLATFORM_SKUS.get(line_type)
    if not sku:
        return None
    for svc in data.get("platform_services", []):
        if svc.get("sku") == sku:
            return svc.get("product_id")
    wp = data.get("wallet_product", {})
    if line_type == "wallet_deposit" and wp.get("sku") == "xcelsior-compute-credits":
        return wp.get("product_id")
    return None


def format_line_name(gpu_model: str, tier: str, mode: str) -> str:
    return f"{gpu_model} — {_TIER_LABELS.get(tier, tier)} · {_MODE_LABELS.get(mode, mode)}"


def enrich_invoice_lines_with_catalog(
    line_items: list[dict[str, Any]],
    *,
    manifest: dict | None = None,
) -> None:
    """Attach stripe_product_id to invoice line dicts in-place."""
    data = manifest if manifest is not None else load_manifest()
    for li in line_items:
        line_type = li.get("line_type") or "compute"
        if line_type == "storage":
            pid = resolve_platform_product_id("storage", data)
        elif line_type == "serverless":
            pid = resolve_platform_product_id("serverless", data) or resolve_product_id(
                li.get("gpu_model") or "",
                manifest=data,
            )
        else:
            pid = resolve_product_id(li.get("gpu_model") or "", manifest=data)
        if pid:
            li["stripe_product_id"] = pid


def build_invoice_line_items(
    invoice_lines: list[dict[str, Any]],
    *,
    manifest: dict | None = None,
) -> list[dict[str, Any]]:
    """Map invoice line items to Stripe Invoice/Checkout price_data (for sync jobs)."""
    out: list[dict[str, Any]] = []
    data = manifest if manifest is not None else load_manifest()
    for li in invoice_lines:
        subtotal = float(li.get("subtotal_cad") or 0)
        if subtotal <= 0:
            continue
        line_type = li.get("line_type") or "compute"
        product_id = li.get("stripe_product_id")
        if not product_id:
            if line_type in ("storage", "serverless"):
                product_id = resolve_platform_product_id(line_type, data)
            elif li.get("gpu_model"):
                product_id = resolve_product_id(str(li.get("gpu_model")), manifest=data)
        if product_id:
            out.append(
                {
                    "price_data": {
                        "currency": "cad",
                        "unit_amount": max(1, int(round(subtotal * 100))),
                        "product": product_id,
                    },
                    "quantity": 1,
                }
            )
        else:
            out.append(
                {
                    "price_data": {
                        "currency": "cad",
                        "unit_amount": max(1, int(round(subtotal * 100))),
                        "product_data": {
                            "name": li.get("description") or "Xcelsior usage",
                            "metadata": {
                                "line_type": line_type,
                                "gpu_model": li.get("gpu_model") or "",
                            },
                        },
                    },
                    "quantity": 1,
                }
            )
    return out


# Back-compat alias for any callers still using the old name.
build_usage_line_items = build_invoice_line_items