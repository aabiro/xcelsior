"""Routes: Stripe Connect V2 — Sample Integration.

This module implements a complete Stripe Connect integration using the V2 API:
  - Create connected accounts (platform responsible for pricing/fees)
  - Onboard connected accounts via Account Links
  - Create products at the platform level
  - Display a storefront of all products
  - Process destination charges with application fees
  - Handle thin webhook events for requirements changes

All Stripe calls use a single StripeClient instance.
"""

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from db import _get_pg_pool
from psycopg.rows import dict_row

log = logging.getLogger("xcelsior.stripe_connect_v2")

router = APIRouter(tags=["Stripe Connect V2"])

# ── Environment / Configuration ───────────────────────────────────────
# Stripe secret key — must start with "sk_test_" or "sk_live_".
# Set via .env or environment. Missing key → routes return 503.
STRIPE_SECRET_KEY = os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")
if not STRIPE_SECRET_KEY:
    log.warning(
        "XCELSIOR_STRIPE_SECRET_KEY is not set. "
        "Stripe Connect V2 routes will return errors until a valid key is provided."
    )

# Stripe webhook secret for thin events. Obtain from
# Stripe Dashboard → Developers → Webhooks. Missing secret → webhook
# signature verification will fail (webhooks will be rejected).
STRIPE_WEBHOOK_SECRET = os.environ.get("XCELSIOR_STRIPE_WEBHOOK_SECRET", "")
if not STRIPE_WEBHOOK_SECRET:
    log.warning(
        "XCELSIOR_STRIPE_WEBHOOK_SECRET is not set. " "Webhook signature verification will fail."
    )

# Base URL for redirect URLs (onboarding return/refresh, checkout success).
BASE_URL = os.environ.get("XCELSIOR_BASE_URL", "http://localhost:8000")

# Platform application fee in cents (applied to each checkout).
PLATFORM_FEE_CENTS = int(os.environ.get("XCELSIOR_PLATFORM_FEE_CENTS", "200"))


# ── Stripe Client (singleton) ────────────────────────────────────────
# We use the StripeClient class (not the legacy module-level stripe.api_key).
# The SDK automatically uses the latest preview API version (2026-03-25.dahlia).

_stripe_client = None


def _get_stripe_client():
    """Return a lazily-initialised StripeClient.

    Raises HTTPException 503 if the secret key is missing.
    """
    global _stripe_client
    if _stripe_client is not None:
        return _stripe_client

    if not STRIPE_SECRET_KEY:
        raise HTTPException(
            503,
            "Stripe is not configured. Set XCELSIOR_STRIPE_SECRET_KEY in your environment.",
        )

    from stripe import StripeClient

    # Create the client with our secret key.
    # The SDK will automatically use the latest API version.
    _stripe_client = StripeClient(STRIPE_SECRET_KEY)
    return _stripe_client


# ── Database helpers ──────────────────────────────────────────────────
# We store connected-account and product mappings in Postgres alongside
# the rest of the Xcelsior schema.


def _ensure_connect_tables():
    """Create the connect_accounts and connect_products tables if missing."""
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS connect_accounts (
                id              SERIAL PRIMARY KEY,
                display_name    TEXT NOT NULL,
                contact_email   TEXT NOT NULL,
                stripe_account_id TEXT UNIQUE NOT NULL,
                created_at      TIMESTAMPTZ DEFAULT now()
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS connect_products (
                id              SERIAL PRIMARY KEY,
                stripe_product_id TEXT UNIQUE NOT NULL,
                stripe_price_id TEXT NOT NULL,
                name            TEXT NOT NULL,
                description     TEXT DEFAULT '',
                price_cents     INTEGER NOT NULL,
                currency        TEXT DEFAULT 'usd',
                account_id      TEXT NOT NULL,  -- stripe connected-account ID
                created_at      TIMESTAMPTZ DEFAULT now()
            );
        """)
        conn.commit()


# Run table creation once at import time.
try:
    _ensure_connect_tables()
except Exception as exc:
    log.warning("Could not ensure connect tables (will retry on first request): %s", exc)


# ── Pydantic request models ──────────────────────────────────────────


class CreateAccountRequest(BaseModel):
    """Body for POST /api/connect/accounts."""

    display_name: str  # Human-readable name for the connected account
    contact_email: str  # Email address used for onboarding communications


class CreateProductRequest(BaseModel):
    """Body for POST /api/connect/products."""

    name: str  # Product name shown to customers
    description: str = ""  # Optional product description
    price_cents: int  # Price in the smallest currency unit (e.g. cents)
    currency: str = "usd"  # Three-letter ISO currency code
    account_id: str  # Stripe connected-account ID that owns this product


class CheckoutRequest(BaseModel):
    """Body for POST /api/connect/checkout."""

    product_id: str  # Stripe product ID (prod_xxx)
    quantity: int = 1  # Number of items to purchase


# ═══════════════════════════════════════════════════════════════════════
# 1. CONNECTED ACCOUNTS — Create & Onboard
# ═══════════════════════════════════════════════════════════════════════


@router.post("/api/connect/accounts")
def create_connected_account(req: CreateAccountRequest):
    """Create a new Stripe connected account using the V2 API.

    The platform is responsible for pricing and fee collection
    (fees_collector: 'application', losses_collector: 'application').
    The account is configured as a *recipient* that can receive
    stripe_balance transfers.

    Steps:
      1. Call stripeClient.v2.core.accounts.create(...) with V2 params.
      2. Store the mapping (display_name → stripe account ID) in Postgres.
      3. Return the new account ID.
    """
    client = _get_stripe_client()

    # Create the connected account via the V2 Core Accounts API.
    # - dashboard='express' gives the user a Stripe-hosted Express dashboard.
    # - identity.country sets the default country for the account.
    # - responsibilities.fees_collector / losses_collector = 'application'
    #   means the *platform* (us) handles pricing and absorbs losses.
    # - recipient configuration with stripe_balance.stripe_transfers
    #   lets the account receive transfers from the platform.
    account = client.v2.core.accounts.create(
        params={
            "display_name": req.display_name,
            "contact_email": req.contact_email,
            "identity": {
                "country": "us",
            },
            "dashboard": "express",
            "defaults": {
                "responsibilities": {
                    "fees_collector": "application",
                    "losses_collector": "application",
                },
            },
            "configuration": {
                "recipient": {
                    "capabilities": {
                        "stripe_balance": {
                            "stripe_transfers": {
                                "requested": True,
                            },
                        },
                    },
                },
            },
        }
    )

    stripe_account_id = account.id

    # Persist the mapping so we can look it up later (storefront, payouts, etc.).
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO connect_accounts (display_name, contact_email, stripe_account_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (stripe_account_id) DO NOTHING
            """,
            (req.display_name, req.contact_email, stripe_account_id),
        )
        conn.commit()

    log.info("Created connected account %s for %s", stripe_account_id, req.display_name)

    return {
        "ok": True,
        "account_id": stripe_account_id,
        "display_name": req.display_name,
    }


@router.get("/api/connect/accounts/{account_id}/onboarding-link")
def create_onboarding_link(account_id: str):
    """Generate a Stripe Account Link for onboarding a connected account.

    Uses the V2 Account Links API with use_case='account_onboarding'.
    The user is redirected through Stripe-hosted onboarding and then back
    to our return_url.

    refresh_url — where Stripe sends the user if the link expires.
    return_url  — where the user lands after completing onboarding.
    """
    client = _get_stripe_client()

    # Build the account link via the V2 Core AccountLinks API.
    account_link = client.v2.core.account_links.create(
        params={
            "account": account_id,
            "use_case": {
                "type": "account_onboarding",
                "account_onboarding": {
                    "configurations": ["recipient"],
                    # Refresh URL: user is sent here if the link expires or
                    # they need to restart onboarding.
                    "refresh_url": f"{BASE_URL}/connect/dashboard?accountId={account_id}",
                    # Return URL: user lands here after completing onboarding.
                    "return_url": f"{BASE_URL}/connect/dashboard?accountId={account_id}",
                },
            },
        }
    )

    return {
        "ok": True,
        "url": account_link.url,
        "account_id": account_id,
    }


@router.get("/api/connect/accounts/{account_id}/status")
def get_account_status(account_id: str):
    """Retrieve the current onboarding / requirements status of an account.

    Always fetches fresh data from the Stripe API (no caching).
    Returns:
      - ready_to_receive_payments: whether stripe_transfers capability is active
      - onboarding_complete: whether there are no currently_due / past_due requirements
      - requirements_status: the raw status string from Stripe
    """
    client = _get_stripe_client()

    # Retrieve the account with expanded configuration and requirements.
    account = client.v2.core.accounts.retrieve(
        account_id,
        params={
            "include": ["configuration.recipient", "requirements"],
        },
    )

    # Check if the recipient configuration's stripe_transfers capability is active.
    ready_to_receive_payments = False
    try:
        status = account.configuration.recipient.capabilities.stripe_balance.stripe_transfers.status
        ready_to_receive_payments = status == "active"
    except (AttributeError, TypeError):
        pass

    # Check whether there are outstanding requirements.
    requirements_status = None
    onboarding_complete = False
    try:
        requirements_status = account.requirements.summary.minimum_deadline.status
        onboarding_complete = requirements_status not in ("currently_due", "past_due")
    except (AttributeError, TypeError):
        # If requirements or summary is None, treat as complete.
        onboarding_complete = True

    return {
        "ok": True,
        "account_id": account_id,
        "ready_to_receive_payments": ready_to_receive_payments,
        "onboarding_complete": onboarding_complete,
        "requirements_status": requirements_status,
    }


@router.get("/api/connect/accounts")
def list_connected_accounts():
    """List all connected accounts stored in our database."""
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        rows = conn.execute(
            "SELECT display_name, contact_email, stripe_account_id, created_at "
            "FROM connect_accounts ORDER BY created_at DESC"
        ).fetchall()
    return {"ok": True, "accounts": rows}


# ═══════════════════════════════════════════════════════════════════════
# 2. PRODUCTS — Create & List
# ═══════════════════════════════════════════════════════════════════════


@router.post("/api/connect/products")
def create_product(req: CreateProductRequest):
    """Create a Stripe Product at the *platform* level.

    The product is created on the platform's own Stripe account (not on the
    connected account) using the standard Products API.  We store the mapping
    from product → connected-account ID in our database so we know which
    seller to pay when a customer buys this product.

    Steps:
      1. Create the product with default_price_data via the Stripe Client.
      2. Store (product_id, price_id, account_id) in connect_products.
    """
    client = _get_stripe_client()

    # Create the product (and its default price) on the platform account.
    product = client.products.create(
        params={
            "name": req.name,
            "description": req.description,
            "default_price_data": {
                "unit_amount": req.price_cents,
                "currency": req.currency,
            },
        }
    )

    stripe_product_id = product.id
    stripe_price_id = product.default_price  # The auto-created Price ID

    # Persist the product ↔ connected-account mapping.
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO connect_products
                (stripe_product_id, stripe_price_id, name, description,
                 price_cents, currency, account_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stripe_product_id) DO NOTHING
            """,
            (
                stripe_product_id,
                stripe_price_id,
                req.name,
                req.description,
                req.price_cents,
                req.currency,
                req.account_id,
            ),
        )
        conn.commit()

    log.info(
        "Created product %s (%s) for connected account %s",
        stripe_product_id,
        req.name,
        req.account_id,
    )

    return {
        "ok": True,
        "product_id": stripe_product_id,
        "price_id": stripe_price_id,
        "name": req.name,
    }


@router.get("/api/connect/products")
def list_products(account_id: Optional[str] = None):
    """List all products, optionally filtered by connected-account ID."""
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        if account_id:
            rows = conn.execute(
                "SELECT * FROM connect_products WHERE account_id = %s " "ORDER BY created_at DESC",
                (account_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM connect_products ORDER BY created_at DESC"
            ).fetchall()
    return {"ok": True, "products": rows}


# ═══════════════════════════════════════════════════════════════════════
# 3. CHECKOUT — Destination Charges
# ═══════════════════════════════════════════════════════════════════════


@router.post("/api/connect/checkout")
def create_checkout_session(req: CheckoutRequest):
    """Create a Stripe Checkout Session using a *destination charge*.

    A destination charge means:
      - The payment is created on the *platform* account.
      - An application_fee_amount is kept by the platform.
      - The remainder is automatically transferred to the connected account
        specified in transfer_data.destination.

    This uses Stripe Hosted Checkout for simplicity — the customer is
    redirected to Stripe's payment page.

    Steps:
      1. Look up the product in our DB to find the connected-account ID and price.
      2. Create a checkout.Session with line_items, payment_intent_data, success/cancel URLs.
      3. Return the Checkout URL for the frontend to redirect the customer.
    """
    client = _get_stripe_client()

    # Look up the product to find the connected-account and pricing info.
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        product = conn.execute(
            "SELECT * FROM connect_products WHERE stripe_product_id = %s",
            (req.product_id,),
        ).fetchone()

    if not product:
        raise HTTPException(404, f"Product {req.product_id} not found")

    # Create a Checkout Session with a destination charge.
    # payment_intent_data.application_fee_amount — the platform's fee in cents.
    # payment_intent_data.transfer_data.destination — the connected account
    #   that receives the funds (minus the application fee).
    session = client.checkout.sessions.create(
        params={
            "line_items": [
                {
                    "price_data": {
                        "currency": product["currency"],
                        "product_data": {
                            "name": product["name"],
                            "description": product["description"] or product["name"],
                        },
                        "unit_amount": product["price_cents"],
                    },
                    "quantity": req.quantity,
                },
            ],
            "payment_intent_data": {
                # Platform keeps this fee (in cents).
                "application_fee_amount": PLATFORM_FEE_CENTS,
                "transfer_data": {
                    # The connected account that will receive the payout.
                    "destination": product["account_id"],
                },
            },
            "mode": "payment",
            # {CHECKOUT_SESSION_ID} is a Stripe template variable that gets
            # replaced with the actual session ID after payment.
            "success_url": f"{BASE_URL}/connect/success?session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{BASE_URL}/connect/storefront",
        }
    )

    return {
        "ok": True,
        "checkout_url": session.url,
        "session_id": session.id,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. WEBHOOKS — Thin Events for V2 Account Changes
# ═══════════════════════════════════════════════════════════════════════


@router.post("/api/connect/webhooks")
async def handle_thin_webhook(request: Request):
    """Handle Stripe thin webhook events for V2 account changes.

    Thin events are lightweight notifications that contain only the event ID
    and type — you must call v2.core.events.retrieve() to get the full payload.

    This endpoint handles:
      - v2.core.account[requirements].updated
        → The account's requirements have changed (e.g. new KYC needed).
      - v2.core.account[.recipient].capability_status_updated
        → A capability on the recipient configuration changed status.

    Setup (Stripe Dashboard):
      1. Go to Developers → Webhooks → + Add destination.
      2. Select "Connected accounts" in Events from.
      3. Select "Show advanced options" → Payload style: Thin.
      4. Search for "v2" and add the event types listed above.
      5. Set the endpoint URL to: {BASE_URL}/api/connect/webhooks

    Local testing with Stripe CLI:
      stripe listen \\
        --thin-events 'v2.core.account[requirements].updated,v2.core.account[.recipient].capability_status_updated' \\
        --forward-thin-to http://localhost:8000/api/connect/webhooks
    """
    client = _get_stripe_client()

    # Read the raw body and the Stripe-Signature header.
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if not sig_header:
        raise HTTPException(400, "Missing Stripe-Signature header")

    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(
            503,
            "Webhook secret not configured. Set XCELSIOR_STRIPE_WEBHOOK_SECRET.",
        )

    # Step 1: Parse the thin event.
    # parse_event_notification verifies the signature and returns a lightweight
    # event object containing { id, type, ... }.
    try:
        thin_event = client.parse_event_notification(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        log.warning("Webhook signature verification failed: %s", e)
        raise HTTPException(400, "Invalid webhook signature") from e

    event_type = thin_event.type
    log.info("Received thin event: %s (id=%s)", event_type, thin_event.id)

    # Step 2: Retrieve the full event data from the V2 Events API.
    try:
        event = client.v2.core.events.retrieve(thin_event.id)
    except Exception as e:
        log.error("Failed to retrieve event %s: %s", thin_event.id, e)
        raise HTTPException(502, "Failed to retrieve event details") from e

    # Step 3: Handle each event type.
    if event_type == "v2.core.account[requirements].updated":
        # The account's requirements changed — possibly new documents or
        # information needed due to regulatory changes.
        _handle_requirements_updated(event)

    elif "capability_status_updated" in event_type:
        # A capability (e.g. stripe_transfers) changed status.
        # Could be: active, inactive, pending, restricted, etc.
        _handle_capability_status_updated(event)

    else:
        log.info("Unhandled event type: %s", event_type)

    return {"ok": True, "event_type": event_type}


def _handle_requirements_updated(event):
    """Process a requirements.updated event.

    When requirements change, you may need to:
      - Notify the connected account owner to complete onboarding.
      - Restrict the account's ability to receive payouts.
      - Log the change for compliance tracking.
    """
    try:
        account_id = event.related_object.id if event.related_object else "unknown"
        log.info(
            "Requirements updated for account %s. "
            "The account owner may need to provide additional information.",
            account_id,
        )
        # In production, you would:
        # 1. Fetch the account to check what's currently_due
        # 2. Send an email/notification to the account owner
        # 3. Update your internal records
    except Exception as e:
        log.error("Error handling requirements update: %s", e)


def _handle_capability_status_updated(event):
    """Process a capability_status_updated event.

    When a capability status changes, the connected account may gain or lose
    the ability to receive transfers, process payments, etc.
    """
    try:
        account_id = event.related_object.id if event.related_object else "unknown"
        log.info(
            "Capability status changed for account %s. "
            "Check the account's configuration for updated capability statuses.",
            account_id,
        )
        # In production, you would:
        # 1. Check which capability changed and its new status
        # 2. Enable/disable features in your app accordingly
        # 3. Notify the account owner if action is needed
    except Exception as e:
        log.error("Error handling capability status update: %s", e)


# ═══════════════════════════════════════════════════════════════════════
# 5. HTML PAGES — Dashboard, Storefront, Success
# ═══════════════════════════════════════════════════════════════════════


@router.get("/connect/dashboard", response_class=HTMLResponse)
def connect_dashboard_page(request: Request, accountId: Optional[str] = None):
    """Serve the Connect Dashboard HTML page.

    This page lets users:
      - Create new connected accounts
      - See onboarding status
      - Start or resume the onboarding flow
      - Create products for their account
    """
    return HTMLResponse(content=_DASHBOARD_HTML)


@router.get("/connect/storefront", response_class=HTMLResponse)
def storefront_page(request: Request):
    """Serve the Storefront HTML page.

    Displays all products across all connected accounts. Customers can
    click "Buy" to be redirected to Stripe Hosted Checkout.
    """
    return HTMLResponse(content=_STOREFRONT_HTML)


@router.get("/connect/success", response_class=HTMLResponse)
def success_page(request: Request, session_id: Optional[str] = None):
    """Serve the payment success page after a completed checkout."""
    return HTMLResponse(content=_SUCCESS_HTML)


# ═══════════════════════════════════════════════════════════════════════
# 6. HTML TEMPLATES (inline for simplicity)
# ═══════════════════════════════════════════════════════════════════════

_COMMON_STYLES = """
<style>
  :root {
    --bg: #0a0e17;
    --surface: #141926;
    --border: #1e2736;
    --text: #e4e8f1;
    --text-muted: #8892a4;
    --accent: #6c63ff;
    --accent-hover: #5a52d5;
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #f87171;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }
  .container { max-width: 960px; margin: 0 auto; }
  h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }
  h2 { font-size: 1.25rem; margin-bottom: 0.75rem; color: var(--text-muted); }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }
  .card h3 { margin-bottom: 1rem; }
  label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.25rem; }
  input, select {
    width: 100%;
    padding: 0.6rem 0.75rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 0.9rem;
    margin-bottom: 0.75rem;
  }
  input:focus, select:focus { outline: none; border-color: var(--accent); }
  button, .btn {
    display: inline-block;
    padding: 0.6rem 1.25rem;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.9rem;
    text-decoration: none;
    transition: background 0.2s;
  }
  button:hover, .btn:hover { background: var(--accent-hover); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-outline {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
  }
  .btn-outline:hover { background: var(--accent); color: #fff; }
  .badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-success { background: rgba(52,211,153,0.15); color: var(--success); }
  .badge-warning { background: rgba(251,191,36,0.15); color: var(--warning); }
  .badge-danger { background: rgba(248,113,113,0.15); color: var(--danger); }
  .badge-muted { background: rgba(136,146,164,0.15); color: var(--text-muted); }
  .grid { display: grid; gap: 1rem; }
  .grid-2 { grid-template-columns: 1fr 1fr; }
  .grid-3 { grid-template-columns: 1fr 1fr 1fr; }
  .text-muted { color: var(--text-muted); }
  .text-success { color: var(--success); }
  .mt-1 { margin-top: 0.5rem; }
  .mt-2 { margin-top: 1rem; }
  .mb-2 { margin-bottom: 1rem; }
  .flex { display: flex; align-items: center; gap: 0.75rem; }
  .flex-between { display: flex; justify-content: space-between; align-items: center; }
  .nav { display: flex; gap: 1rem; margin-bottom: 2rem; }
  .nav a { color: var(--text-muted); text-decoration: none; font-size: 0.9rem; }
  .nav a:hover, .nav a.active { color: var(--accent); }
  #toast {
    position: fixed; bottom: 2rem; right: 2rem;
    padding: 0.75rem 1.25rem; border-radius: 8px;
    background: var(--surface); border: 1px solid var(--border);
    display: none; z-index: 999; font-size: 0.9rem;
  }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--border); }
  th { color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
  tr:hover td { background: rgba(108,99,255,0.04); }
  @media (max-width: 640px) {
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
    body { padding: 1rem; }
  }
</style>
"""

# ── Dashboard HTML ────────────────────────────────────────────────────

_DASHBOARD_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Xcelsior — Connect Dashboard</title>
  {_COMMON_STYLES}
</head>
<body>
<div class="container">

  <!-- Navigation -->
  <nav class="nav">
    <a href="/connect/dashboard" class="active">Dashboard</a>
    <a href="/connect/storefront">Storefront</a>
  </nav>

  <h1>Stripe Connect Dashboard</h1>
  <h2>Manage connected accounts , onboarding, and products</h2>

  <!-- ── Create Account Card ── -->
  <div class="card">
    <h3>Create Connected Account</h3>
    <p class="text-muted mb-2">
      Create a new Express account. The platform handles pricing and fee collection.
    </p>
    <div class="grid grid-2">
      <div>
        <label for="acct-name">Display Name</label>
        <input type="text" id="acct-name" placeholder="Acme Widgets">
      </div>
      <div>
        <label for="acct-email">Contact Email</label>
        <input type="email" id="acct-email" placeholder="seller@example.com">
      </div>
    </div>
    <button onclick="createAccount()">Create Account</button>
  </div>

  <!-- ── Accounts List ── -->
  <div class="card">
    <div class="flex-between">
      <h3>Connected Accounts</h3>
      <button class="btn-outline" onclick="loadAccounts()">Refresh</button>
    </div>
    <div id="accounts-list" class="mt-1">
      <p class="text-muted">Loading...</p>
    </div>
  </div>

  <!-- ── Account Detail / Onboarding ── -->
  <div class="card" id="account-detail" style="display:none;">
    <h3 id="detail-title">Account Detail</h3>
    <div id="detail-status" class="mb-2"></div>
    <div id="detail-actions" class="flex"></div>
  </div>

  <!-- ── Create Product Card ── -->
  <div class="card" id="create-product-card" style="display:none;">
    <h3>Create Product</h3>
    <p class="text-muted mb-2">
      Products are created at the platform level. Select the connected account that
      will receive the payout when a customer purchases this product.
    </p>
    <input type="hidden" id="prod-account-id">
    <div class="grid grid-2">
      <div>
        <label for="prod-name">Product Name</label>
        <input type="text" id="prod-name" placeholder="Premium Widget">
      </div>
      <div>
        <label for="prod-price">Price (cents)</label>
        <input type="number" id="prod-price" placeholder="2500" min="50">
      </div>
    </div>
    <div>
      <label for="prod-desc">Description</label>
      <input type="text" id="prod-desc" placeholder="A high-quality widget">
    </div>
    <div class="grid grid-2">
      <div>
        <label for="prod-currency">Currency</label>
        <select id="prod-currency">
          <option value="usd">USD</option>
          <option value="cad">CAD</option>
          <option value="eur">EUR</option>
          <option value="gbp">GBP</option>
        </select>
      </div>
    </div>
    <button onclick="createProduct()">Create Product</button>
  </div>

  <!-- ── Products for Selected Account ── -->
  <div class="card" id="products-card" style="display:none;">
    <h3 id="products-title">Products</h3>
    <div id="products-list"></div>
  </div>

</div><!-- /container -->

<div id="toast"></div>

<script>
  // ── Toast notifications ──
  function toast(msg, type = 'info') {{
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.style.borderColor = type === 'error' ? 'var(--danger)' : 'var(--accent)';
    el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 4000);
  }}

  // ── Create a connected account ──
  async function createAccount() {{
    const name = document.getElementById('acct-name').value.trim();
    const email = document.getElementById('acct-email').value.trim();
    if (!name || !email) return toast('Please fill in both fields', 'error');

    try {{
      const res = await fetch('/api/connect/accounts', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ display_name: name, contact_email: email }}),
      }});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed');
      toast('Account created: ' + data.account_id);
      document.getElementById('acct-name').value = '';
      document.getElementById('acct-email').value = '';
      loadAccounts();
    }} catch (e) {{
      toast(e.message, 'error');
    }}
  }}

  // ── Load all connected accounts ──
  async function loadAccounts() {{
    try {{
      const res = await fetch('/api/connect/accounts');
      const data = await res.json();
      const container = document.getElementById('accounts-list');
      if (!data.accounts || data.accounts.length === 0) {{
        container.innerHTML = '<p class="text-muted">No accounts yet. Create one above.</p>';
        return;
      }}
      let html = '<table><thead><tr><th>Name</th><th>Email</th><th>Account ID</th><th></th></tr></thead><tbody>';
      for (const a of data.accounts) {{
        html += `<tr>
          <td>${{a.display_name}}</td>
          <td class="text-muted">${{a.contact_email}}</td>
          <td class="text-muted" style="font-size:0.8rem">${{a.stripe_account_id}}</td>
          <td><button class="btn-outline" onclick="selectAccount('${{a.stripe_account_id}}', '${{a.display_name}}')">Manage</button></td>
        </tr>`;
      }}
      html += '</tbody></table>';
      container.innerHTML = html;
    }} catch (e) {{
      toast('Failed to load accounts', 'error');
    }}
  }}

  // ── Select an account to manage ──
  async function selectAccount(accountId, name) {{
    document.getElementById('account-detail').style.display = 'block';
    document.getElementById('create-product-card').style.display = 'block';
    document.getElementById('products-card').style.display = 'block';
    document.getElementById('detail-title').textContent = 'Account: ' + name;
    document.getElementById('prod-account-id').value = accountId;
    document.getElementById('products-title').textContent = 'Products for ' + name;

    // Fetch account status from Stripe
    try {{
      const res = await fetch(`/api/connect/accounts/${{accountId}}/status`);
      const data = await res.json();
      let statusHtml = '';

      if (data.ready_to_receive_payments) {{
        statusHtml += '<span class="badge badge-success">Ready to receive payments</span> ';
      }} else {{
        statusHtml += '<span class="badge badge-warning">Not yet ready</span> ';
      }}

      if (data.onboarding_complete) {{
        statusHtml += '<span class="badge badge-success">Onboarding complete</span>';
      }} else {{
        statusHtml += '<span class="badge badge-danger">Onboarding incomplete</span>';
      }}

      if (data.requirements_status) {{
        statusHtml += ` <span class="badge badge-muted">${{data.requirements_status}}</span>`;
      }}

      document.getElementById('detail-status').innerHTML = statusHtml;

      // Show onboarding button if not complete
      let actionsHtml = '';
      if (!data.onboarding_complete || !data.ready_to_receive_payments) {{
        actionsHtml += `<button onclick="startOnboarding('${{accountId}}')">Onboard to collect payments</button>`;
      }}
      actionsHtml += `<button class="btn-outline" onclick="selectAccount('${{accountId}}', '${{name}}')">Refresh Status</button>`;
      document.getElementById('detail-actions').innerHTML = actionsHtml;
    }} catch (e) {{
      document.getElementById('detail-status').innerHTML =
        '<span class="badge badge-danger">Error fetching status</span>';
    }}

    // Load products for this account
    loadProducts(accountId);
  }}

  // ── Start onboarding (get an Account Link and redirect) ──
  async function startOnboarding(accountId) {{
    try {{
      const res = await fetch(`/api/connect/accounts/${{accountId}}/onboarding-link`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed');
      // Redirect the user to Stripe-hosted onboarding
      window.location.href = data.url;
    }} catch (e) {{
      toast(e.message, 'error');
    }}
  }}

  // ── Create a product ──
  async function createProduct() {{
    const name = document.getElementById('prod-name').value.trim();
    const priceCents = parseInt(document.getElementById('prod-price').value) || 0;
    const description = document.getElementById('prod-desc').value.trim();
    const currency = document.getElementById('prod-currency').value;
    const accountId = document.getElementById('prod-account-id').value;

    if (!name || priceCents < 50) return toast('Name and price (>= 50 cents) required', 'error');
    if (!accountId) return toast('Select an account first', 'error');

    try {{
      const res = await fetch('/api/connect/products', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          name, description,
          price_cents: priceCents,
          currency,
          account_id: accountId,
        }}),
      }});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed');
      toast('Product created: ' + data.product_id);
      document.getElementById('prod-name').value = '';
      document.getElementById('prod-price').value = '';
      document.getElementById('prod-desc').value = '';
      loadProducts(accountId);
    }} catch (e) {{
      toast(e.message, 'error');
    }}
  }}

  // ── Load products for an account ──
  async function loadProducts(accountId) {{
    try {{
      const res = await fetch(`/api/connect/products?account_id=${{accountId}}`);
      const data = await res.json();
      const container = document.getElementById('products-list');
      if (!data.products || data.products.length === 0) {{
        container.innerHTML = '<p class="text-muted">No products yet.</p>';
        return;
      }}
      let html = '<table><thead><tr><th>Name</th><th>Price</th><th>Product ID</th></tr></thead><tbody>';
      for (const p of data.products) {{
        const price = (p.price_cents / 100).toFixed(2);
        html += `<tr>
          <td>${{p.name}}</td>
          <td>${{price}} ${{p.currency.toUpperCase()}}</td>
          <td class="text-muted" style="font-size:0.8rem">${{p.stripe_product_id}}</td>
        </tr>`;
      }}
      html += '</tbody></table>';
      container.innerHTML = html;
    }} catch (e) {{
      toast('Failed to load products', 'error');
    }}
  }}

  // ── Check for onboarding return (accountId in URL) ──
  const params = new URLSearchParams(window.location.search);
  if (params.get('accountId')) {{
    // User returned from Stripe onboarding — select that account
    const aid = params.get('accountId');
    loadAccounts().then(() => selectAccount(aid, aid));
  }} else {{
    loadAccounts();
  }}
</script>
</body>
</html>"""


# ── Storefront HTML ───────────────────────────────────────────────────

_STOREFRONT_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Xcelsior — Storefront</title>
  {_COMMON_STYLES}
  <style>
    .product-card {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .product-card .price {{
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--accent);
      margin: 0.5rem 0;
    }}
    .seller {{ font-size: 0.8rem; color: var(--text-muted); }}
  </style>
</head>
<body>
<div class="container">

  <!-- Navigation -->
  <nav class="nav">
    <a href="/connect/dashboard">Dashboard</a>
    <a href="/connect/storefront" class="active">Storefront</a>
  </nav>

  <h1>Storefront</h1>
  <h2>Browse products from all sellers</h2>

  <div id="storefront-grid" class="grid grid-3">
    <p class="text-muted">Loading products...</p>
  </div>

</div><!-- /container -->

<div id="toast"></div>

<script>
  function toast(msg, type = 'info') {{
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.style.borderColor = type === 'error' ? 'var(--danger)' : 'var(--accent)';
    el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 4000);
  }}

  // ── Load all products and all accounts ──
  async function loadStorefront() {{
    try {{
      // Fetch products and accounts in parallel
      const [prodRes, acctRes] = await Promise.all([
        fetch('/api/connect/products'),
        fetch('/api/connect/accounts'),
      ]);
      const prodData = await prodRes.json();
      const acctData = await acctRes.json();

      // Build a lookup: account_id → display_name
      const accountNames = {{}};
      for (const a of (acctData.accounts || [])) {{
        accountNames[a.stripe_account_id] = a.display_name;
      }}

      const container = document.getElementById('storefront-grid');
      const products = prodData.products || [];

      if (products.length === 0) {{
        container.innerHTML = '<p class="text-muted">No products available yet. Sellers can add them from the <a href="/connect/dashboard" style="color:var(--accent)">Dashboard</a>.</p>';
        return;
      }}

      let html = '';
      for (const p of products) {{
        const price = (p.price_cents / 100).toFixed(2);
        const seller = accountNames[p.account_id] || p.account_id;
        html += `
          <div class="card product-card">
            <div>
              <h3>${{p.name}}</h3>
              <p class="text-muted">${{p.description || 'No description'}}</p>
              <p class="seller">Sold by: ${{seller}}</p>
              <p class="price">${{price}} ${{p.currency.toUpperCase()}}</p>
            </div>
            <button onclick="buyProduct('${{p.stripe_product_id}}')">Buy Now</button>
          </div>`;
      }}
      container.innerHTML = html;
    }} catch (e) {{
      toast('Failed to load storefront', 'error');
    }}
  }}

  // ── Buy a product — redirect to Stripe Checkout ──
  async function buyProduct(productId) {{
    try {{
      const res = await fetch('/api/connect/checkout', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ product_id: productId, quantity: 1 }}),
      }});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to create checkout');
      // Redirect the customer to Stripe Hosted Checkout
      window.location.href = data.checkout_url;
    }} catch (e) {{
      toast(e.message, 'error');
    }}
  }}

  loadStorefront();
</script>
</body>
</html>"""


# ── Success HTML ──────────────────────────────────────────────────────

_SUCCESS_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Xcelsior — Payment Successful</title>
  {_COMMON_STYLES}
  <style>
    .success-icon {{
      font-size: 4rem;
      color: var(--success);
      margin-bottom: 1rem;
    }}
    .center {{ text-align: center; padding: 3rem 1rem; }}
  </style>
</head>
<body>
<div class="container">
  <div class="card center">
    <div class="success-icon">&#10003;</div>
    <h1>Payment Successful!</h1>
    <p class="text-muted mt-1">Thank you for your purchase. Your payment has been processed.</p>
    <p class="text-muted mt-1" id="session-info"></p>
    <div class="mt-2">
      <a href="/connect/storefront" class="btn">Continue Shopping</a>
    </div>
  </div>
</div>
<script>
  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get('session_id');
  if (sessionId) {{
    document.getElementById('session-info').textContent = 'Session: ' + sessionId;
  }}
</script>
</body>
</html>"""
