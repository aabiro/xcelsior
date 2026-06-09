#!/usr/bin/env python3
"""Archive duplicate Xcelsior Compute Credits products in Stripe.

Keeps the canonical product (metadata xcelsior_sku=xcelsior-compute-credits from
config/stripe_catalog.json, or the oldest active match) and sets active=false on
duplicates from failed seed runs.

Usage:
  python3 scripts/cleanup_stripe_wallet_products.py --dry-run
  python3 scripts/cleanup_stripe_wallet_products.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "stripe_catalog.json"
CANONICAL_SKU = "xcelsior-compute-credits"
WALLET_NAME = "Xcelsior Compute Credits"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _stripe_metadata(obj) -> dict[str, str]:
    raw = getattr(obj, "metadata", None)
    if not raw:
        return {}
    try:
        return {str(k): str(raw[k]) for k in raw.keys()}
    except Exception:
        return {}


def _resolve_key(mode: str | None) -> str:
    if mode == "sandbox":
        return (
            os.environ.get("XCELSIOR_STRIPE_SANDBOX_SECRET_KEY", "")
            or os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")
        )
    return os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--mode", choices=["live", "sandbox"], default=None)
    args = parser.parse_args()
    if not args.dry_run and not args.apply:
        parser.error("Specify --dry-run or --apply")

    _load_dotenv()
    import stripe

    mode = args.mode or os.environ.get("XCELSIOR_STRIPE_MODE", "live").lower()
    stripe.api_key = _resolve_key(mode)
    if not stripe.api_key.startswith("sk_"):
        print("ERROR: missing Stripe secret key", file=sys.stderr)
        return 1

    canonical_id = None
    if MANIFEST.exists():
        try:
            canonical_id = json.loads(MANIFEST.read_text()).get("wallet_product", {}).get("product_id")
        except Exception:
            pass

    candidates = []
    starting_after = None
    while True:
        params: dict = {"limit": 100, "active": True}
        if starting_after:
            params["starting_after"] = starting_after
        page = stripe.Product.list(**params)
        for prod in page.data:
            meta = _stripe_metadata(prod)
            if prod.name == WALLET_NAME or meta.get("xcelsior_sku") == CANONICAL_SKU:
                candidates.append(prod)
        if not page.has_more:
            break
        starting_after = page.data[-1].id

    if not candidates:
        print("No active wallet products found.")
        return 0

    keep = None
    if canonical_id:
        keep = next((p for p in candidates if p.id == canonical_id), None)
    if not keep:
        keep = next(
            (p for p in candidates if _stripe_metadata(p).get("xcelsior_sku") == CANONICAL_SKU),
            None,
        )
    if not keep:
        keep = sorted(candidates, key=lambda p: p.created)[0]

    print(f"Keeping: {keep.id} ({keep.name})")
    archived = 0
    for prod in candidates:
        if prod.id == keep.id:
            continue
        print(f"  [{'would archive' if args.dry_run else 'archive'}] {prod.id} created={prod.created}")
        if args.apply:
            stripe.Product.modify(prod.id, active=False)
            archived += 1

    if args.dry_run:
        print(f"DRY RUN — would archive {len(candidates) - 1} duplicate(s)")
    else:
        print(f"Archived {archived} duplicate wallet product(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())