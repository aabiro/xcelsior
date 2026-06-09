#!/usr/bin/env python3
"""Apply Xcelsior Stripe Dashboard polish via API.

Automates: brand asset upload, customer portal configuration, webhook URL/event
consistency, and tax-settings verification.

Note: Stripe's Account.modify API only works on *connected* accounts, not the
platform account itself. Account-level branding and default portal must be set
once in Dashboard → Settings (file IDs from this script are printed for paste).

Usage:
  python3 scripts/polish_stripe_dashboard.py --dry-run
  python3 scripts/polish_stripe_dashboard.py --apply
  python3 scripts/polish_stripe_dashboard.py --apply --mode sandbox
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "stripe_dashboard.json"
ICON_PATH = ROOT / "frontend" / "public" / "xcelsior_icon_circle_512x512.png"
LOGO_PATH = ROOT / "frontend" / "public" / "xcelsior_icon_circle_1024x1024.png"

BRAND_PRIMARY = "#00d4ff"
BRAND_SECONDARY = "#060a13"

WEBHOOK_TARGETS = {
    "xcelsior-platform-webhook": {
        "url_path": "/api/providers/webhook",
        "description": "Xcelsior platform — wallet top-ups, Connect account updates, payouts",
        "enabled_events": [
            "account.updated",
            "checkout.session.completed",
            "payment_intent.payment_failed",
            "payment_intent.succeeded",
            "payout.failed",
            "payout.paid",
            "transfer.created",
            "transfer.reversed",
        ],
        "connect": False,
    },
    "xcelsior-connect-thin-webhook": {
        "url_path": "/api/connect/webhooks",
        "description": "Xcelsior Connect v2 thin events — requirements and capability changes",
        "enabled_events": [
            "account.updated",
            "capability.updated",
        ],
        "connect": True,
    },
}


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


def _resolve_key(mode: str | None) -> str:
    if mode == "sandbox":
        return (
            os.environ.get("XCELSIOR_STRIPE_SANDBOX_SECRET_KEY", "")
            or os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")
        )
    return os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")


def _base_url() -> str:
    return os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca").rstrip("/")


def _stripe_metadata(obj) -> dict[str, str]:
    raw = getattr(obj, "metadata", None)
    if not raw:
        return {}
    try:
        return {str(k): str(raw[k]) for k in raw.keys()}
    except Exception:
        return {}


def _upload_brand_file(stripe_mod, path: Path, purpose: str, dry_run: bool) -> str | None:
    if not path.exists():
        print(f"  skip upload — missing {path.name}")
        return None
    if dry_run:
        print(f"  [dry-run] would upload {path.name} as {purpose}")
        return f"file_dry_{purpose}"
    with path.open("rb") as fh:
        uploaded = stripe_mod.File.create(file=fh, purpose=purpose)
    print(f"  uploaded {path.name} → {uploaded.id}")
    return uploaded.id


def _apply_branding(stripe_mod, dry_run: bool, state: dict) -> None:
    print("Branding (platform account — Dashboard paste for icon/logo)")
    icon_id = _upload_brand_file(stripe_mod, ICON_PATH, "business_icon", dry_run)
    logo_id = _upload_brand_file(stripe_mod, LOGO_PATH, "business_logo", dry_run)
    if icon_id and not icon_id.startswith("file_dry_"):
        state["brand_icon_file_id"] = icon_id
    if logo_id and not logo_id.startswith("file_dry_"):
        state["brand_logo_file_id"] = logo_id
    state["brand_colors"] = {"primary": BRAND_PRIMARY, "secondary": BRAND_SECONDARY}
    print(f"  colors: primary={BRAND_PRIMARY} secondary={BRAND_SECONDARY}")
    print("  manual: Dashboard → Settings → Branding")
    print(f"    icon file: {state.get('brand_icon_file_id', '(upload with --apply)')}")
    print(f"    logo file: {state.get('brand_logo_file_id', '(upload with --apply)')}")
    print("    statement descriptor: XCELSIOR")


def _portal_features() -> dict:
    return {
        "customer_update": {
            "enabled": True,
            "allowed_updates": ["email", "address", "phone", "tax_id"],
        },
        "invoice_history": {"enabled": True},
        "payment_method_update": {"enabled": True},
    }


def _apply_customer_portal(stripe_mod, dry_run: bool, state: dict) -> None:
    print("Customer portal")
    params = {
        "business_profile": {
            "headline": "Xcelsior — manage billing & payment methods",
            "privacy_policy_url": f"{_base_url()}/privacy",
            "terms_of_service_url": f"{_base_url()}/terms",
        },
        "features": _portal_features(),
        "default_return_url": f"{_base_url()}/dashboard/billing",
    }

    if dry_run:
        print(f"  [dry-run] portal configuration: {json.dumps(params, indent=2)}")
        print("  manual: Dashboard → Settings → Billing → Customer portal → set as default")
        return

    existing_id = state.get("billing_portal_configuration_id")
    if existing_id:
        try:
            stripe_mod.billing_portal.Configuration.modify(existing_id, **params)
            print(f"  updated portal configuration {existing_id}")
            print("  manual: confirm it is the default portal in Dashboard → Billing → Customer portal")
            return
        except Exception as exc:
            print(f"  could not update {existing_id}: {exc}")

    configs = stripe_mod.billing_portal.Configuration.list(limit=20)
    managed = [
        c for c in configs.data if _stripe_metadata(c).get("xcelsior_managed") == "true"
    ]
    if managed:
        cfg_id = managed[0].id
        stripe_mod.billing_portal.Configuration.modify(cfg_id, **params)
        state["billing_portal_configuration_id"] = cfg_id
        print(f"  updated managed portal configuration {cfg_id}")
    else:
        created = stripe_mod.billing_portal.Configuration.create(
            **params,
            metadata={"xcelsior_managed": "true"},
        )
        state["billing_portal_configuration_id"] = created.id
        print(f"  created portal configuration {created.id}")
    print("  manual: Dashboard → Settings → Billing → Customer portal → activate & set default")


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def _apply_webhooks(stripe_mod, dry_run: bool, state: dict) -> None:
    print("Webhooks")
    base = _base_url()
    known = {w.id: w for w in stripe_mod.WebhookEndpoint.list(limit=100).auto_paging_iter()}

    for key, spec in WEBHOOK_TARGETS.items():
        target_url = f"{base}{spec['url_path']}"
        match = None
        for wh in known.values():
            if _normalize_url(wh.url) == _normalize_url(target_url):
                match = wh
                break
        if match:
            print(f"  {key}: {match.id} → {match.url}")
            missing = set(spec["enabled_events"]) - set(match.enabled_events or [])
            if missing:
                if dry_run:
                    print(f"    [dry-run] would add events: {sorted(missing)}")
                else:
                    merged = sorted(set(match.enabled_events or []) | set(spec["enabled_events"]))
                    stripe_mod.WebhookEndpoint.modify(
                        match.id,
                        enabled_events=merged,
                        description=spec["description"],
                    )
                    print(f"    added {len(missing)} event(s); {len(merged)} total")
            state.setdefault("webhook_endpoints", {})[key] = match.id
            continue

        print(f"  {key}: missing endpoint at {target_url}")
        if dry_run:
            print(f"    [dry-run] would create ({len(spec['enabled_events'])} events)")
            continue

        created = stripe_mod.WebhookEndpoint.create(
            url=target_url,
            enabled_events=spec["enabled_events"],
            description=spec["description"],
            connect=spec.get("connect", False),
            metadata={"xcelsior_webhook_key": key},
        )
        state.setdefault("webhook_endpoints", {})[key] = created.id
        print(f"    created {created.id} — add signing secret to .env")


def _verify_tax(stripe_mod, dry_run: bool, state: dict) -> None:
    print("Tax")
    if dry_run:
        print("  [dry-run] would retrieve tax.settings and report status")
        return
    try:
        settings = json.loads(str(stripe_mod.tax.Settings.retrieve()))
        status = settings.get("status")
        defaults = settings.get("defaults") or {}
        addr = (settings.get("head_office") or {}).get("address") or {}
        state["tax_settings"] = {
            "status": status,
            "provider": defaults.get("provider"),
            "head_office_country": addr.get("country"),
        }
        print(f"  Stripe Tax status: {status}")
        if addr:
            print(f"  head office: {addr.get('city')}, {addr.get('state')} {addr.get('country')}")
        if status != "active":
            print("  manual: Dashboard → Settings → Tax → complete Canada registration")
    except Exception as exc:
        print(f"  tax settings check skipped: {exc}")


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

    state: dict = {}
    if CONFIG_PATH.exists():
        try:
            state = json.loads(CONFIG_PATH.read_text())
        except Exception:
            state = {}

    dry = args.dry_run
    print(f"Stripe dashboard polish ({mode}) base={_base_url()}")
    _apply_branding(stripe, dry, state)
    print()
    _apply_customer_portal(stripe, dry, state)
    print()
    _apply_webhooks(stripe, dry, state)
    print()
    _verify_tax(stripe, dry, state)

    if args.apply:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        state["stripe_mode"] = mode
        state["base_url"] = _base_url()
        CONFIG_PATH.write_text(json.dumps(state, indent=2) + "\n")
        print()
        print(f"State written: {CONFIG_PATH}")

    if dry:
        print()
        print("DRY RUN — no Stripe changes applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())