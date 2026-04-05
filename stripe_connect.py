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
# Requires XCELSIOR_STRIPE_SECRET_KEY to be set with a valid Stripe key.
# Operations that require Stripe will raise errors if not configured.

import json
import os
import time
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

log = logging.getLogger("xcelsior.stripe")

# ── Configuration ─────────────────────────────────────────────────────

STRIPE_SECRET_KEY = os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("XCELSIOR_STRIPE_WEBHOOK_SECRET", "")
_raw_cut = float(os.environ.get("XCELSIOR_PLATFORM_CUT", "0.15"))
PLATFORM_CUT_FRAC = _raw_cut if _raw_cut <= 1.0 else _raw_cut / 100.0
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
    PENDING = "pending"  # Onboarding started
    ONBOARDING = "onboarding"  # Stripe hosted KYC in progress
    ACTIVE = "active"  # Fully verified, can receive payouts
    RESTRICTED = "restricted"  # Missing info or compliance issue
    SUSPENDED = "suspended"  # Platform-level suspension


class ProviderType(str, Enum):
    INDIVIDUAL = "individual"  # Solo GPU provider
    COMPANY = "company"  # Incorporated Canadian business


@dataclass
class ProviderAccount:
    """A provider's Stripe Connect account and company details."""

    provider_id: str
    provider_type: str = "individual"
    stripe_account_id: str = ""
    status: str = "pending"
    # Canadian company details
    corporation_name: str = ""
    business_number: str = ""  # CRA Business Number (BN)
    incorporation_file_id: str = ""  # Reference to uploaded file in artifacts
    gst_hst_number: str = ""  # GST/HST registration number
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
    payout_schedule: str = "weekly"  # daily, weekly, monthly


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
        self.db_path = db_path  # Legacy compat

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

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

        if not (STRIPE_ENABLED and stripe):
            raise RuntimeError(
                "Stripe Connect is not configured. Set XCELSIOR_STRIPE_SECRET_KEY "
                "in .env with a valid Stripe secret key to enable provider onboarding."
            )

        _base_url = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
        refresh_url = f"{_base_url}/dashboard/earnings?stripe=refresh&provider={provider_id}"
        return_url = f"{_base_url}/dashboard/earnings?stripe=return&provider={provider_id}"

        def _create_hosted_stripe_url(account_id: str, status_hint: str) -> tuple[str, str]:
            try:
                link = stripe.AccountLink.create(
                    account=account_id,
                    refresh_url=refresh_url,
                    return_url=return_url,
                    type="account_onboarding",
                )
                return link.url, status_hint
            except Exception as e:
                log.warning(
                    "Stripe AccountLink creation failed for provider %s (acct=%s): %s",
                    provider_id,
                    account_id,
                    e,
                )

            # Fallback: if account is already fully enabled, open Stripe Express dashboard.
            try:
                acct = stripe.Account.retrieve(account_id)
                charges_enabled = bool(acct.get("charges_enabled", False))
                payouts_enabled = bool(acct.get("payouts_enabled", False))
                if charges_enabled and payouts_enabled:
                    login_link = stripe.Account.create_login_link(account_id)
                    return login_link.url, "active"
            except Exception as e:
                log.warning(
                    "Stripe login-link fallback failed for provider %s (acct=%s): %s",
                    provider_id,
                    account_id,
                    e,
                )

            raise RuntimeError(
                "Unable to open Stripe onboarding right now. Please try again in a moment."
            )

        # Check if provider already has a Stripe account
        existing = None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT stripe_account_id, status FROM provider_accounts WHERE provider_id=%s",
                (provider_id,),
            ).fetchone()
            if row:
                existing = {
                    "stripe_account_id": row["stripe_account_id"],
                    "status": row["status"],
                }

        if existing and existing["stripe_account_id"]:
            # Re-generate onboarding link for existing account
            stripe_account_id = existing["stripe_account_id"]
            onboarding_url, status = _create_hosted_stripe_url(
                stripe_account_id,
                existing["status"],
            )
            with self._conn() as conn:
                conn.execute(
                    "UPDATE provider_accounts SET status=%s WHERE provider_id=%s",
                    (status, provider_id),
                )

            log.info(
                "Stripe onboarding/login link generated for existing account %s (provider %s)",
                stripe_account_id,
                provider_id,
            )

            return {
                "provider_id": provider_id,
                "stripe_account_id": stripe_account_id,
                "onboarding_url": onboarding_url,
                "status": status,
            }

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
            onboarding_url, status = _create_hosted_stripe_url(stripe_account_id, "onboarding")
            log.info(
                "Stripe Connect account created: %s for provider %s",
                stripe_account_id,
                provider_id,
            )
        except Exception as e:
            log.error("Stripe account creation failed for %s: %s", provider_id, e)
            raise RuntimeError(
                "Failed to start Stripe onboarding. Please try again in a moment."
            ) from e

        # Persist locally
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO provider_accounts
                   (provider_id, provider_type, stripe_account_id, status,
                    corporation_name, business_number, gst_hst_number,
                    email, legal_name, country, province, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'CA', %s, %s)
                   ON CONFLICT (provider_id) DO UPDATE SET
                     provider_type = EXCLUDED.provider_type, stripe_account_id = EXCLUDED.stripe_account_id,
                     status = EXCLUDED.status, corporation_name = EXCLUDED.corporation_name,
                     business_number = EXCLUDED.business_number, gst_hst_number = EXCLUDED.gst_hst_number,
                     email = EXCLUDED.email, legal_name = EXCLUDED.legal_name,
                     province = EXCLUDED.province, created_at = EXCLUDED.created_at""",
                (
                    provider_id,
                    provider_type,
                    stripe_account_id,
                    status,
                    corporation_name,
                    business_number,
                    gst_hst_number,
                    email,
                    legal_name,
                    province,
                    now,
                ),
            )

        return {
            "provider_id": provider_id,
            "stripe_account_id": stripe_account_id,
            "onboarding_url": onboarding_url,
            "status": status,
        }

    def upload_incorporation_file(self, provider_id: str, file_id: str) -> dict:
        """Link an uploaded incorporation document to a provider account.

        The actual file is stored via artifacts.py (B2/R2/local).
        This method just records the reference.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE provider_accounts SET incorporation_file_id=%s WHERE provider_id=%s",
                (file_id, provider_id),
            )
        log.info("Incorporation file %s linked to provider %s", file_id, provider_id)
        return {"provider_id": provider_id, "incorporation_file_id": file_id}

    def get_provider(self, provider_id: str) -> Optional[dict]:
        """Get provider account details."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM provider_accounts WHERE provider_id=%s",
                (provider_id,),
            ).fetchone()
            if not row:
                return None

            provider = dict(row)

            # Best-effort status sync in case webhook delivery is delayed or missed.
            stripe_account_id = provider.get("stripe_account_id")
            if STRIPE_ENABLED and stripe and stripe_account_id:
                try:
                    acct = stripe.Account.retrieve(stripe_account_id)
                    charges_enabled = bool(acct.get("charges_enabled", False))
                    payouts_enabled = bool(acct.get("payouts_enabled", False))
                    disabled_reason = (acct.get("requirements") or {}).get("disabled_reason")

                    if charges_enabled and payouts_enabled:
                        new_status = "active"
                    elif disabled_reason:
                        new_status = "restricted"
                    else:
                        new_status = "onboarding"

                    updates: dict[str, float | str] = {}
                    if provider.get("status") != new_status:
                        updates["status"] = new_status
                    if new_status == "active" and not provider.get("onboarded_at"):
                        updates["onboarded_at"] = time.time()

                    if updates:
                        if "onboarded_at" in updates:
                            conn.execute(
                                "UPDATE provider_accounts SET status=%s, onboarded_at=%s WHERE provider_id=%s",
                                (updates["status"], updates["onboarded_at"], provider_id),
                            )
                        else:
                            conn.execute(
                                "UPDATE provider_accounts SET status=%s WHERE provider_id=%s",
                                (updates["status"], provider_id),
                            )
                        provider["status"] = updates.get("status", provider.get("status"))
                        if "onboarded_at" in updates:
                            provider["onboarded_at"] = updates["onboarded_at"]
                except Exception as e:
                    log.warning(
                        "Stripe status sync failed for provider %s (acct=%s): %s",
                        provider_id,
                        stripe_account_id,
                        e,
                    )

            return provider

    def list_providers(self, status: str = "") -> list[dict]:
        """List all provider accounts, optionally filtered by status."""
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM provider_accounts WHERE status=%s ORDER BY created_at DESC",
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
                "UPDATE provider_accounts SET status='active', onboarded_at=%s WHERE provider_id=%s",
                (now, provider_id),
            )
        log.info("Provider %s onboarding COMPLETE", provider_id)
        return {"provider_id": provider_id, "status": "active"}

    # ── Payment Processing ────────────────────────────────────────────

    def create_credit_deposit(
        self, customer_id: str, amount_cad: float, description: str = "Compute credits"
    ) -> dict:
        """Create a payment intent for depositing compute credits.

        Per Report #1.B: "Credit-first model where users deposit CAD
        into an account. As compute services are delivered, providers
        withdraw funds."
        """
        import secrets

        intent_id = f"pi_{secrets.token_hex(12)}"
        amount_cents = int(amount_cad * 100)
        stripe_intent_id = ""
        client_secret = ""

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
                client_secret = pi.client_secret
            except Exception as e:
                log.error("Stripe PaymentIntent failed: %s", e)
        else:
            raise RuntimeError(
                "Stripe is not configured. Set XCELSIOR_STRIPE_SECRET_KEY "
                "in .env to enable payment processing."
            )

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO payment_intents
                   (intent_id, customer_id, amount_cents, currency, status,
                    stripe_intent_id, description, created_at)
                   VALUES (%s, %s, %s, 'cad', 'created', %s, %s, %s)""",
                (intent_id, customer_id, amount_cents, stripe_intent_id, description, time.time()),
            )

        return {
            "intent_id": intent_id,
            "stripe_intent_id": stripe_intent_id,
            "amount_cad": amount_cad,
            "client_secret": client_secret,
        }

    # ── Payout Splitting ──────────────────────────────────────────────

    def split_payout(
        self, job_id: str, provider_id: str, total_cad: float, province: str = "ON"
    ) -> dict:
        """Split a job's revenue between provider and platform.

        Per Report #1.B:
        - Platform takes flat 10-15% commission
        - GST/HST applied per province
        - Provider receives remainder
        """
        from billing import get_tax_rate_for_province

        platform_share = round(total_cad * PLATFORM_CUT_FRAC, 2)
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
            raise RuntimeError(
                "Stripe is not configured. Set XCELSIOR_STRIPE_SECRET_KEY "
                "in .env to enable provider payouts."
            )

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO payout_splits
                   (job_id, provider_id, total_cad, provider_share_cad,
                    platform_share_cad, gst_hst_cad, stripe_transfer_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    job_id,
                    provider_id,
                    total_cad,
                    provider_share,
                    platform_share,
                    gst_hst,
                    stripe_transfer_id,
                ),
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
                "SELECT * FROM payout_splits WHERE provider_id=%s ORDER BY created_at DESC LIMIT %s",
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
                   FROM payout_splits WHERE provider_id=%s""",
                (provider_id,),
            ).fetchone()
            return (
                dict(row)
                if row
                else {
                    "total_jobs": 0,
                    "total_earned_cad": 0,
                    "total_platform_cad": 0,
                    "total_tax_cad": 0,
                }
            )

    # ── Webhook Handling (Inbox Pattern) ─────────────────────────────

    def handle_webhook(self, payload: bytes, sig_header: str) -> dict:
        """Receive a Stripe webhook event into the inbox for idempotent processing.

        Two-phase approach:
        1. Verify signature, write to stripe_event_inbox (dedup on event_id)
        2. Background processor picks up pending events and processes them

        This guarantees at-least-once delivery with exactly-once semantics
        because Stripe retries are deduped by event_id primary key.
        """
        if not STRIPE_ENABLED or not stripe:
            return {"handled": False, "reason": "Stripe not enabled"}

        try:
            event = stripe.Webhook.construct_event(
                payload,
                sig_header,
                STRIPE_WEBHOOK_SECRET,
            )
        except Exception as e:
            log.error("Webhook signature verification failed: %s", e)
            return {"handled": False, "error": str(e)}

        event_id = event["id"]
        event_type = event["type"]
        now = time.time()

        # Phase 1: Write to inbox (idempotent via PK)
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT event_id, status FROM stripe_event_inbox WHERE event_id = %s",
                (event_id,),
            ).fetchone()
            if existing:
                log.info("Webhook event %s already in inbox (status=%s), skipping", event_id, existing["status"])
                return {"handled": True, "type": event_type, "dedup": True}

            from psycopg.types.json import Jsonb
            conn.execute(
                """INSERT INTO stripe_event_inbox
                   (event_id, event_type, stripe_account, livemode, api_version,
                    created_unix, received_at, payload, status, attempts, next_retry_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', 0, %s)""",
                (
                    event_id,
                    event_type,
                    event.get("account", ""),
                    event.get("livemode", True),
                    event.get("api_version", ""),
                    event.get("created", 0),
                    now,
                    Jsonb(dict(event)),
                    now,  # process immediately
                ),
            )

        log.info("Webhook event %s (%s) written to inbox", event_id, event_type)

        # Try eager processing (best-effort, background processor is the safety net)
        try:
            self._process_single_event(event_id)
        except Exception as e:
            log.warning("Eager processing failed for %s, will retry: %s", event_id, e)

        return {"handled": True, "type": event_type, "event_id": event_id}

    def _process_single_event(self, event_id: str) -> bool:
        """Process one event from the inbox. Returns True if processed."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM stripe_event_inbox WHERE event_id = %s AND status = 'pending' FOR UPDATE SKIP LOCKED",
                (event_id,),
            ).fetchone()
            if not row:
                return False

            event_type = row["event_type"]
            payload = row["payload"]
            data = payload.get("data", {}).get("object", {})
            attempts = (row["attempts"] or 0) + 1

            try:
                self._dispatch_event(event_type, data, event_id)
                conn.execute(
                    "UPDATE stripe_event_inbox SET status = 'processed', attempts = %s, processed_at = %s WHERE event_id = %s",
                    (attempts, time.time(), event_id),
                )
                log.info("Event %s (%s) processed successfully", event_id, event_type)
                return True
            except Exception as e:
                # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                backoff = min(30 * (2 ** (attempts - 1)), 3600)
                next_retry = time.time() + backoff
                max_attempts = 8
                new_status = "failed" if attempts >= max_attempts else "pending"
                conn.execute(
                    """UPDATE stripe_event_inbox
                       SET status = %s, attempts = %s, last_error = %s, next_retry_at = %s
                       WHERE event_id = %s""",
                    (new_status, attempts, str(e)[:500], next_retry, event_id),
                )
                log.error("Event %s processing failed (attempt %d/%d): %s", event_id, attempts, max_attempts, e)
                return False

    def _dispatch_event(self, event_type: str, data: dict, event_id: str):
        """Route an event to its handler. Raises on failure for retry."""
        if event_type == "account.updated":
            self._handle_account_updated(data)
        elif event_type == "payment_intent.succeeded":
            self._handle_payment_succeeded(data, event_id)
        elif event_type == "payment_intent.payment_failed":
            self._handle_payment_failed(data)
        elif event_type == "transfer.created":
            self._handle_transfer_created(data)
        elif event_type == "transfer.reversed":
            self._handle_transfer_reversed(data)
        elif event_type == "payout.paid":
            self._handle_payout_paid(data)
        elif event_type == "payout.failed":
            self._handle_payout_failed(data)
        else:
            log.debug("Unhandled event type: %s", event_type)

    def _handle_account_updated(self, data: dict):
        acct_id = data["id"]
        with self._conn() as conn:
            row = conn.execute(
                "SELECT provider_id FROM provider_accounts WHERE stripe_account_id=%s",
                (acct_id,),
            ).fetchone()
            if row:
                charges_enabled = data.get("charges_enabled", False)
                payouts_enabled = data.get("payouts_enabled", False)
                if charges_enabled and payouts_enabled:
                    self.complete_onboarding(row["provider_id"])
                elif not charges_enabled or not payouts_enabled:
                    # Stripe disabled capabilities — mark restricted
                    reqs = data.get("requirements", {})
                    if reqs.get("disabled_reason"):
                        conn.execute(
                            "UPDATE provider_accounts SET status='restricted' WHERE provider_id=%s",
                            (row["provider_id"],),
                        )
                        log.warning("Provider %s restricted: %s", row["provider_id"], reqs.get("disabled_reason"))

    def _handle_payment_succeeded(self, data: dict, event_id: str):
        si_id = data["id"]
        with self._conn() as conn:
            conn.execute(
                "UPDATE payment_intents SET status='succeeded' WHERE stripe_intent_id=%s",
                (si_id,),
            )
            row = conn.execute(
                "SELECT customer_id, amount_cents FROM payment_intents WHERE stripe_intent_id=%s",
                (si_id,),
            ).fetchone()
        if row:
            from billing import get_billing_engine
            amount_cad = round(row["amount_cents"] / 100.0, 2)
            engine = get_billing_engine()
            # Idempotent deposit using event_id as idempotency_key
            engine.deposit(
                row["customer_id"],
                amount_cad,
                description=f"Stripe deposit {si_id}",
                idempotency_key=f"stripe:{event_id}",
            )
            log.info("Wallet credited: %s +$%.2f from %s", row["customer_id"], amount_cad, si_id)

            # If wallet was suspended and balance is now positive, reactivate
            wallet = engine.get_wallet(row["customer_id"])
            if wallet.get("status") == "suspended" and wallet.get("balance_cad", 0) > 0:
                engine.reactivate_wallet(row["customer_id"])

    def _handle_payment_failed(self, data: dict):
        si_id = data["id"]
        failure_code = data.get("last_payment_error", {}).get("code", "unknown")
        with self._conn() as conn:
            conn.execute(
                "UPDATE payment_intents SET status='failed' WHERE stripe_intent_id=%s",
                (si_id,),
            )
        log.warning("Payment failed: %s reason=%s", si_id, failure_code)

    def _handle_transfer_created(self, data: dict):
        transfer_id = data["id"]
        meta = data.get("metadata", {})
        job_id = meta.get("job_id", "")
        log.info("Transfer created: %s for job %s", transfer_id, job_id)

    def _handle_transfer_reversed(self, data: dict):
        transfer_id = data["id"]
        meta = data.get("metadata", {})
        job_id = meta.get("job_id", "")
        provider_id = meta.get("provider_id", "")
        amount_cents = data.get("amount_reversed", 0)
        log.warning("Transfer REVERSED: %s job=%s provider=%s amount=%d cents", transfer_id, job_id, provider_id, amount_cents)
        # Claw back from provider's pending balance tracking
        if provider_id:
            with self._conn() as conn:
                conn.execute(
                    """UPDATE payout_splits SET platform_share_cad = total_cad, provider_share_cad = 0
                       WHERE job_id = %s AND provider_id = %s AND stripe_transfer_id = %s""",
                    (job_id, provider_id, transfer_id),
                )

    def _handle_payout_paid(self, data: dict):
        log.info("Payout paid: %s", data.get("id", ""))

    def _handle_payout_failed(self, data: dict):
        payout_id = data.get("id", "")
        failure_code = data.get("failure_code", "unknown")
        failure_message = data.get("failure_message", "")
        log.error("Payout FAILED: %s code=%s msg=%s", payout_id, failure_code, failure_message)

    # ── Background Event Processor ────────────────────────────────────

    def process_pending_events(self, batch_size: int = 20) -> int:
        """Process pending events from the inbox. Returns count processed."""
        now = time.time()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT event_id FROM stripe_event_inbox
                   WHERE status = 'pending' AND next_retry_at <= %s
                   ORDER BY next_retry_at ASC LIMIT %s""",
                (now, batch_size),
            ).fetchall()

        processed = 0
        for row in rows:
            if self._process_single_event(row["event_id"]):
                processed += 1
        return processed


# ── Singleton ─────────────────────────────────────────────────────────

_stripe_manager: Optional[StripeConnectManager] = None


def get_stripe_manager() -> StripeConnectManager:
    global _stripe_manager
    if _stripe_manager is None:
        _stripe_manager = StripeConnectManager()
    return _stripe_manager
