# Xcelsior Stripe Connect Integration v2.0.0
# Marketplace payment processing, KYC onboarding, and payout management.
#
# Per REPORT_FEATURE_1.md (Report #1.B):
# - Stripe Connect for provider onboarding (identity, bank, tax)
# - Credit-first billing: users deposit CAD, providers withdraw
# - Automated GST/HST collection per province
# - Platform commission split (default 10-15%)
# - Provider incorporation verification
#
# NOTE: This module is a functional stub. Replace XCELSIOR_STRIPE_SECRET_KEY
# in .env with a live Stripe key to activate real payment processing.

import os
import time
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

log = logging.getLogger("xcelsior.stripe")

# ── Configuration ─────────────────────────────────────────────────────

STRIPE_SECRET_KEY = os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("XCELSIOR_STRIPE_WEBHOOK_SECRET", "")
PLATFORM_CUT_PCT = float(os.environ.get("XCELSIOR_PLATFORM_CUT", "15"))
STRIPE_ENABLED = bool(STRIPE_SECRET_KEY and STRIPE_SECRET_KEY.startswith("sk_"))

DB_PATH = os.environ.get("XCELSIOR_STRIPE_DB", "xcelsior_stripe.db")

# Only import stripe if API key is configured
stripe = None
if STRIPE_ENABLED:
    try:
        import stripe as _stripe
        _stripe.api_key = STRIPE_SECRET_KEY
        stripe = _stripe
        log.info("Stripe Connect ENABLED (key prefix: %s...)", STRIPE_SECRET_KEY[:7])
    except ImportError:
        log.warning("stripe package not installed — pip install stripe")
        STRIPE_ENABLED = False


# ── Enums and Data Models ────────────────────────────────────────────

class AccountStatus(str, Enum):
    PENDING = "pending"           # Onboarding started
    ONBOARDING = "onboarding"     # Stripe hosted KYC in progress
    ACTIVE = "active"             # Fully verified, can receive payouts
    RESTRICTED = "restricted"     # Missing info or compliance issue
    SUSPENDED = "suspended"       # Platform-level suspension


class ProviderType(str, Enum):
    INDIVIDUAL = "individual"     # Solo GPU provider
    COMPANY = "company"           # Incorporated Canadian business


@dataclass
class ProviderAccount:
    """A provider's Stripe Connect account and company details."""
    provider_id: str
    provider_type: str = "individual"
    stripe_account_id: str = ""
    status: str = "pending"
    # Canadian company details
    corporation_name: str = ""
    business_number: str = ""          # CRA Business Number (BN)
    incorporation_file_id: str = ""    # Reference to uploaded file in artifacts
    gst_hst_number: str = ""           # GST/HST registration number
    # Contact
    email: str = ""
    legal_name: str = ""
    # Location
    country: str = "CA"
    province: str = ""
    # Timestamps
    created_at: float = 0.0
    onboarded_at: float = 0.0
    # Payout
    default_currency: str = "cad"
    payout_schedule: str = "weekly"    # daily, weekly, monthly


@dataclass
class PaymentIntent:
    """A payment intent for compute credits."""
    intent_id: str
    customer_id: str
    amount_cents: int
    currency: str = "cad"
    status: str = "created"
    stripe_intent_id: str = ""
    description: str = ""
    created_at: float = 0.0


@dataclass
class PayoutSplit:
    """A split payment between provider and platform."""
    job_id: str
    provider_id: str
    total_cad: float
    provider_share_cad: float
    platform_share_cad: float
    gst_hst_cad: float = 0.0
    stripe_transfer_id: str = ""
    created_at: float = 0.0


# ── Stripe Connect Manager ───────────────────────────────────────────

class StripeConnectManager:
    """Manages Stripe Connect accounts, payments, and payouts."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS provider_accounts (
                    provider_id TEXT PRIMARY KEY,
                    provider_type TEXT DEFAULT 'individual',
                    stripe_account_id TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    corporation_name TEXT DEFAULT '',
                    business_number TEXT DEFAULT '',
                    incorporation_file_id TEXT DEFAULT '',
                    gst_hst_number TEXT DEFAULT '',
                    email TEXT DEFAULT '',
                    legal_name TEXT DEFAULT '',
                    country TEXT DEFAULT 'CA',
                    province TEXT DEFAULT '',
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    onboarded_at REAL DEFAULT 0,
                    default_currency TEXT DEFAULT 'cad',
                    payout_schedule TEXT DEFAULT 'weekly'
                );
                CREATE TABLE IF NOT EXISTS payment_intents (
                    intent_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL,
                    currency TEXT DEFAULT 'cad',
                    status TEXT DEFAULT 'created',
                    stripe_intent_id TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );
                CREATE TABLE IF NOT EXISTS payout_splits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    total_cad REAL NOT NULL,
                    provider_share_cad REAL NOT NULL,
                    platform_share_cad REAL NOT NULL,
                    gst_hst_cad REAL DEFAULT 0,
                    stripe_transfer_id TEXT DEFAULT '',
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_payouts_provider
                    ON payout_splits(provider_id, created_at);
            """)

    # ── Provider Onboarding ───────────────────────────────────────────

    def create_provider_account(
        self,
        provider_id: str,
        email: str,
        provider_type: str = "individual",
        corporation_name: str = "",
        business_number: str = "",
        gst_hst_number: str = "",
        province: str = "",
        legal_name: str = "",
    ) -> dict:
        """Create a Stripe Connect Express account for a provider.

        Per Report #1.B "Five Pillars of Compliance":
        1. Identity Verification (Stripe Identity)
        2. Financial Enrollment (bank details)
        3. Credentialing (GPU/bandwidth thresholds)
        4. Tax Compliance (GST/HST)
        """
        now = time.time()
        stripe_account_id = ""
        onboarding_url = ""

        if STRIPE_ENABLED and stripe:
            try:
                # Create Stripe Connect Express account
                acct = stripe.Account.create(
                    type="express",
                    country="CA",
                    email=email,
                    capabilities={
                        "card_payments": {"requested": True},
                        "transfers": {"requested": True},
                    },
                    business_type=provider_type,
                    metadata={
                        "xcelsior_provider_id": provider_id,
                        "corporation_name": corporation_name,
                        "business_number": business_number,
                    },
                )
                stripe_account_id = acct.id

                # Generate onboarding link
                link = stripe.AccountLink.create(
                    account=stripe_account_id,
                    refresh_url=f"https://xcelsior.ca/onboarding/refresh?provider={provider_id}",
                    return_url=f"https://xcelsior.ca/onboarding/complete?provider={provider_id}",
                    type="account_onboarding",
                )
                onboarding_url = link.url
                log.info("Stripe Connect account created: %s for provider %s",
                         stripe_account_id, provider_id)
            except Exception as e:
                log.error("Stripe account creation failed for %s: %s", provider_id, e)
                # Continue with local-only record
        else:
            # Stub mode — generate placeholder
            stripe_account_id = f"acct_stub_{provider_id[:8]}"
            onboarding_url = f"https://xcelsior.ca/onboarding/stub?provider={provider_id}"
            log.info("Stripe STUB: created placeholder account for %s", provider_id)

        # Persist locally
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO provider_accounts
                   (provider_id, provider_type, stripe_account_id, status,
                    corporation_name, business_number, gst_hst_number,
                    email, legal_name, country, province, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'CA', ?, ?)""",
                (provider_id, provider_type, stripe_account_id, "onboarding",
                 corporation_name, business_number, gst_hst_number,
                 email, legal_name, province, now),
            )

        return {
            "provider_id": provider_id,
            "stripe_account_id": stripe_account_id,
            "onboarding_url": onboarding_url,
            "status": "onboarding",
        }

    def upload_incorporation_file(self, provider_id: str, file_id: str) -> dict:
        """Link an uploaded incorporation document to a provider account.

        The actual file is stored via artifacts.py (B2/R2/local).
        This method just records the reference.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE provider_accounts SET incorporation_file_id=? WHERE provider_id=?",
                (file_id, provider_id),
            )
        log.info("Incorporation file %s linked to provider %s", file_id, provider_id)
        return {"provider_id": provider_id, "incorporation_file_id": file_id}

    def get_provider(self, provider_id: str) -> Optional[dict]:
        """Get provider account details."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM provider_accounts WHERE provider_id=?",
                (provider_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_providers(self, status: str = "") -> list[dict]:
        """List all provider accounts, optionally filtered by status."""
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM provider_accounts WHERE status=? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM provider_accounts ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def complete_onboarding(self, provider_id: str) -> dict:
        """Mark a provider's onboarding as complete (webhook callback)."""
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE provider_accounts SET status='active', onboarded_at=? WHERE provider_id=?",
                (now, provider_id),
            )
        log.info("Provider %s onboarding COMPLETE", provider_id)
        return {"provider_id": provider_id, "status": "active"}

    # ── Payment Processing ────────────────────────────────────────────

    def create_credit_deposit(self, customer_id: str, amount_cad: float,
                              description: str = "Compute credits") -> dict:
        """Create a payment intent for depositing compute credits.

        Per Report #1.B: "Credit-first model where users deposit CAD
        into an account. As compute services are delivered, providers
        withdraw funds."
        """
        import secrets
        intent_id = f"pi_{secrets.token_hex(12)}"
        amount_cents = int(amount_cad * 100)
        stripe_intent_id = ""

        if STRIPE_ENABLED and stripe:
            try:
                pi = stripe.PaymentIntent.create(
                    amount=amount_cents,
                    currency="cad",
                    metadata={
                        "xcelsior_customer_id": customer_id,
                        "xcelsior_intent_id": intent_id,
                    },
                    description=description,
                )
                stripe_intent_id = pi.id
            except Exception as e:
                log.error("Stripe PaymentIntent failed: %s", e)
        else:
            stripe_intent_id = f"pi_stub_{intent_id[:8]}"

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO payment_intents
                   (intent_id, customer_id, amount_cents, currency, status,
                    stripe_intent_id, description, created_at)
                   VALUES (?, ?, ?, 'cad', 'created', ?, ?, ?)""",
                (intent_id, customer_id, amount_cents,
                 stripe_intent_id, description, time.time()),
            )

        return {
            "intent_id": intent_id,
            "stripe_intent_id": stripe_intent_id,
            "amount_cad": amount_cad,
            "client_secret": f"stub_secret_{intent_id}" if not STRIPE_ENABLED else "",
        }

    # ── Payout Splitting ──────────────────────────────────────────────

    def split_payout(self, job_id: str, provider_id: str, total_cad: float,
                     province: str = "ON") -> dict:
        """Split a job's revenue between provider and platform.

        Per Report #1.B:
        - Platform takes flat 10-15% commission
        - GST/HST applied per province
        - Provider receives remainder
        """
        from billing import get_tax_rate_for_province

        platform_share = round(total_cad * PLATFORM_CUT_PCT / 100.0, 2)
        pre_tax_provider = round(total_cad - platform_share, 2)
        tax_rate = get_tax_rate_for_province(province)
        gst_hst = round(total_cad * tax_rate, 2)
        provider_share = round(pre_tax_provider, 2)

        stripe_transfer_id = ""
        if STRIPE_ENABLED and stripe:
            provider = self.get_provider(provider_id)
            if provider and provider.get("stripe_account_id"):
                try:
                    transfer = stripe.Transfer.create(
                        amount=int(provider_share * 100),
                        currency="cad",
                        destination=provider["stripe_account_id"],
                        metadata={"job_id": job_id, "provider_id": provider_id},
                    )
                    stripe_transfer_id = transfer.id
                except Exception as e:
                    log.error("Stripe Transfer failed for job %s: %s", job_id, e)
        else:
            stripe_transfer_id = f"tr_stub_{job_id[:8]}"

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO payout_splits
                   (job_id, provider_id, total_cad, provider_share_cad,
                    platform_share_cad, gst_hst_cad, stripe_transfer_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (job_id, provider_id, total_cad, provider_share,
                 platform_share, gst_hst, stripe_transfer_id),
            )

        return {
            "job_id": job_id,
            "total_cad": total_cad,
            "provider_share_cad": provider_share,
            "platform_share_cad": platform_share,
            "gst_hst_cad": gst_hst,
            "province": province,
            "tax_rate": tax_rate,
            "stripe_transfer_id": stripe_transfer_id,
        }

    def get_provider_payouts(self, provider_id: str, limit: int = 50) -> list[dict]:
        """Get payout history for a provider."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM payout_splits WHERE provider_id=? ORDER BY created_at DESC LIMIT ?",
                (provider_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_provider_earnings(self, provider_id: str) -> dict:
        """Get aggregate earnings for a provider."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) as total_jobs,
                    COALESCE(SUM(provider_share_cad), 0) as total_earned_cad,
                    COALESCE(SUM(platform_share_cad), 0) as total_platform_cad,
                    COALESCE(SUM(gst_hst_cad), 0) as total_tax_cad
                   FROM payout_splits WHERE provider_id=?""",
                (provider_id,),
            ).fetchone()
            return dict(row) if row else {
                "total_jobs": 0, "total_earned_cad": 0,
                "total_platform_cad": 0, "total_tax_cad": 0,
            }

    # ── Webhook Handling ──────────────────────────────────────────────

    def handle_webhook(self, payload: bytes, sig_header: str) -> dict:
        """Process a Stripe webhook event.

        Handles:
        - account.updated → Provider KYC status change
        - payment_intent.succeeded → Credit deposit confirmed
        - payout.paid → Provider payout completed
        """
        if not STRIPE_ENABLED or not stripe:
            return {"handled": False, "reason": "Stripe not enabled"}

        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET,
            )
        except Exception as e:
            log.error("Webhook signature verification failed: %s", e)
            return {"handled": False, "error": str(e)}

        event_type = event["type"]
        data = event["data"]["object"]

        if event_type == "account.updated":
            acct_id = data["id"]
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT provider_id FROM provider_accounts WHERE stripe_account_id=?",
                    (acct_id,),
                ).fetchone()
                if row:
                    charges_enabled = data.get("charges_enabled", False)
                    payouts_enabled = data.get("payouts_enabled", False)
                    if charges_enabled and payouts_enabled:
                        self.complete_onboarding(row["provider_id"])
            return {"handled": True, "type": "account.updated"}

        elif event_type == "payment_intent.succeeded":
            si_id = data["id"]
            with self._conn() as conn:
                conn.execute(
                    "UPDATE payment_intents SET status='succeeded' WHERE stripe_intent_id=?",
                    (si_id,),
                )
            return {"handled": True, "type": "payment_intent.succeeded"}

        elif event_type == "payout.paid":
            return {"handled": True, "type": "payout.paid"}

        return {"handled": False, "type": event_type}


# ── Singleton ─────────────────────────────────────────────────────────

_stripe_manager: Optional[StripeConnectManager] = None


def get_stripe_manager() -> StripeConnectManager:
    global _stripe_manager
    if _stripe_manager is None:
        _stripe_manager = StripeConnectManager()
    return _stripe_manager
