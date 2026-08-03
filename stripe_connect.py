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
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, cast

from money import cad_to_micros
from provider_settlement import (
    SettlementConflict,
    claim_settlements,
    get_settlement,
    mark_settlement_paid,
    mark_settlement_retry,
    prepare_settlement,
    settlement_response,
)

log = logging.getLogger("xcelsior.stripe")

# ── Configuration ─────────────────────────────────────────────────────

# When XCELSIOR_STRIPE_MODE=sandbox, use the *_SANDBOX_* keys instead of live keys.
_STRIPE_MODE = os.environ.get("XCELSIOR_STRIPE_MODE", "live").lower()
if _STRIPE_MODE == "sandbox":
    STRIPE_SECRET_KEY = os.environ.get("XCELSIOR_STRIPE_SANDBOX_SECRET_KEY", "") or os.environ.get(
        "XCELSIOR_STRIPE_SECRET_KEY", ""
    )
    STRIPE_WEBHOOK_SECRET = os.environ.get(
        "XCELSIOR_STRIPE_SANDBOX_WEBHOOK_SECRET", ""
    ) or os.environ.get("XCELSIOR_STRIPE_WEBHOOK_SECRET", "")
else:
    STRIPE_SECRET_KEY = os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.environ.get("XCELSIOR_STRIPE_WEBHOOK_SECRET", "")
_raw_cut = float(os.environ.get("XCELSIOR_PLATFORM_CUT", "0.15"))
PLATFORM_CUT_FRAC = _raw_cut if _raw_cut <= 1.0 else _raw_cut / 100.0
STRIPE_ENABLED = bool(STRIPE_SECRET_KEY and STRIPE_SECRET_KEY.startswith("sk_"))


def _webhook_secret_candidates() -> list:
    """All configured Stripe webhook signing secrets, in priority order.

    A single endpoint can be the target of multiple Stripe destinations — e.g.
    a general account webhook (``xcelsior-webhook``) and a Connect-specific one
    (``xcelsior-connect-snapshot``) — and each destination has its own signing
    secret. We try every configured secret until one verifies, so an event is
    never rejected just because it arrived from a sibling destination.
    """
    raw = [
        STRIPE_WEBHOOK_SECRET,  # mode-resolved primary (sandbox or live)
        os.environ.get("XCELSIOR_STRIPE_WEBHOOK_SECRET", ""),
        os.environ.get("XCELSIOR_STRIPE_CONNECT_WEBHOOK_SECRET", ""),
        os.environ.get("XCELSIOR_STRIPE_SANDBOX_WEBHOOK_SECRET", ""),
    ]
    seen = set()
    out = []
    for secret in raw:
        if secret and secret not in seen:
            seen.add(secret)
            out.append(secret)
    return out


DB_PATH = os.environ.get("XCELSIOR_STRIPE_DB", "xcelsior_stripe.db")

# Only import stripe if API key is configured
stripe = None
# Pin to the recommended Dahlia API version shipped with stripe-python >= 15.3.
STRIPE_API_VERSION = "2026-06-24.dahlia"
if STRIPE_ENABLED:
    try:
        import stripe as _stripe

        _stripe.api_key = STRIPE_SECRET_KEY
        # SDK 15.3+ defaults to 2026-06-24.dahlia; set explicitly for older installs.
        try:
            _stripe.api_version = STRIPE_API_VERSION
        except Exception:
            pass
        stripe = _stripe
        log.info(
            "Stripe Connect ENABLED (mode=%s, api=%s, key prefix: %s...)",
            _STRIPE_MODE,
            getattr(stripe, "api_version", STRIPE_API_VERSION),
            STRIPE_SECRET_KEY[:7],
        )
    except ImportError:
        log.warning("stripe package not installed — pip install stripe")
        STRIPE_ENABLED = False


def _available_cad_cents(stripe_mod=None) -> int:
    """Platform Stripe available balance in CAD cents (read-only API)."""
    client = stripe_mod if stripe_mod is not None else stripe
    if not (STRIPE_ENABLED and client):
        return 0
    try:
        bal = client.Balance.retrieve()
        available = getattr(bal, "available", None)
        if available is None and isinstance(bal, dict):
            available = bal.get("available")
        total = 0
        for entry in available or []:
            currency = entry.get("currency") if isinstance(entry, dict) else getattr(entry, "currency", "")
            amount = entry.get("amount") if isinstance(entry, dict) else getattr(entry, "amount", 0)
            if str(currency).lower() == "cad":
                total += int(amount or 0)
        return total
    except Exception as exc:
        log.warning("Stripe Balance.retrieve failed: %s", exc)
        return 0


def evaluate_settlement(
    *,
    provider: dict | None,
    provider_share_micros: int | None = None,
    provider_share_cad: float | None = None,
    available_cad_cents: int | None = None,
    stripe_mod=None,
) -> dict:
    """Decide pay vs queue for an exact provider share.

    Returns ``{status, error, need_cents, available_cents}`` where status is
    ``paid_eligible`` (float OK, account active) or ``queued``.
    """
    if provider_share_micros is None:
        if provider_share_cad is None:
            raise ValueError("provider_share_micros is required")
        # Compatibility boundary for pure callers. Production settlement
        # always supplies the integer field loaded from PostgreSQL.
        provider_share_micros = cad_to_micros(provider_share_cad)
    if int(provider_share_micros) % 10_000 != 0:
        raise ValueError("provider share must be payable in whole CAD cents")
    need = int(provider_share_micros) // 10_000
    if need <= 0:
        return {
            "status": "queued",
            "error": "zero_amount",
            "need_cents": need,
            "available_cents": 0,
        }
    if not provider or not provider.get("stripe_account_id"):
        return {
            "status": "queued",
            "error": "no_stripe_account",
            "need_cents": need,
            "available_cents": 0,
        }
    if (provider.get("status") or "") != "active":
        return {
            "status": "queued",
            "error": "provider_not_active",
            "need_cents": need,
            "available_cents": 0,
        }
    if available_cad_cents is None:
        available_cad_cents = _available_cad_cents(stripe_mod)
    if available_cad_cents < need:
        return {
            "status": "queued",
            "error": "insufficient_platform_balance",
            "need_cents": need,
            "available_cents": available_cad_cents,
        }
    return {
        "status": "paid_eligible",
        "error": "",
        "need_cents": need,
        "available_cents": available_cad_cents,
    }


def _stripe_create_transfer(
    *,
    amount_cents: int,
    destination: str,
    job_id: str,
    provider_id: str,
    settlement: str,
    idempotency_key: str,
    stripe_mod=None,
):
    """Create a Transfer with idempotency key (request option, not body field)."""
    client = stripe_mod if stripe_mod is not None else stripe
    if client is None:
        raise RuntimeError("Stripe SDK unavailable — cannot create Transfer")
    kwargs = {
        "amount": amount_cents,
        "currency": "cad",
        "destination": destination,
        "metadata": {
            "job_id": job_id,
            "provider_id": provider_id,
            "settlement": settlement,
        },
    }
    return client.Transfer.create(**kwargs, idempotency_key=idempotency_key)


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
        country: str = "CA",
    ) -> dict:
        """Create a Stripe Connect Express account for a provider.

        Per Report #1.B "Five Pillars of Compliance":
        1. Identity Verification (Stripe Identity)
        2. Financial Enrollment (bank details)
        3. Credentialing (GPU/bandwidth thresholds)
        4. Tax Compliance (GST/HST)

        ``country`` is ISO-3166 alpha-2. Cross-border providers are supported when
        the platform Connect settings allow payouts to that country.
        """
        now = time.time()
        stripe_account_id = ""
        onboarding_url = ""
        country_code = (country or "CA").strip().upper()[:2] or "CA"

        if not (STRIPE_ENABLED and stripe):
            raise RuntimeError(
                "Stripe Connect is not configured. Set XCELSIOR_STRIPE_SECRET_KEY "
                "in .env with a valid Stripe secret key to enable provider onboarding."
            )
        stripe_api = stripe

        _base_url = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
        refresh_url = f"{_base_url}/dashboard/earnings?stripe=refresh&provider={provider_id}"
        return_url = f"{_base_url}/dashboard/earnings?stripe=return&provider={provider_id}"

        def _create_hosted_stripe_url(account_id: str, status_hint: str) -> tuple[str, str]:
            # Check live with Stripe first — if the account is already fully
            # enabled, skip creating another onboarding link and return the
            # Express dashboard login link instead.  This prevents the loop
            # where the user completes onboarding but the webhook hasn't landed
            # yet, then clicks "Continue" and gets sent back into the flow.
            try:
                acct_check = stripe_api.Account.retrieve(account_id)
                acct_check_dict = json.loads(str(acct_check))
                if acct_check_dict.get("charges_enabled") and acct_check_dict.get(
                    "payouts_enabled"
                ):
                    login_link = stripe_api.Account.create_login_link(account_id)
                    return login_link.url, "active"
            except Exception as check_err:
                log.warning(
                    "Pre-flight Stripe account check failed for provider %s (acct=%s): %s",
                    provider_id,
                    account_id,
                    check_err,
                )

            try:
                link = stripe_api.AccountLink.create(
                    account=account_id,
                    refresh_url=refresh_url,
                    return_url=return_url,
                    type="account_onboarding",
                )
                return link.url, status_hint
            except Exception as link_err:
                log.warning(
                    "Stripe AccountLink creation failed for provider %s (acct=%s): %s",
                    provider_id,
                    account_id,
                    link_err,
                )

            # Verify the account still exists on Stripe before deciding next steps.
            try:
                acct = stripe_api.Account.retrieve(account_id)
            except Exception as retrieve_err:
                # Account not found on Stripe (e.g. deleted, wrong API mode key).
                # Signal to the caller that this is a stale reference so a fresh
                # account can be created.
                log.warning(
                    "Stripe account retrieval failed for provider %s (acct=%s): %s — "
                    "treating as stale reference",
                    provider_id,
                    account_id,
                    retrieve_err,
                )
                raise RuntimeError("__STALE_ACCOUNT__") from retrieve_err

            # Account exists on Stripe.  If it's already fully enabled, open the
            # Express dashboard instead of the onboarding flow.
            acct_dict = json.loads(str(acct))
            charges_enabled = bool(acct_dict.get("charges_enabled", False))
            payouts_enabled = bool(acct_dict.get("payouts_enabled", False))
            if charges_enabled and payouts_enabled:
                try:
                    login_link = stripe_api.Account.create_login_link(account_id)
                    return login_link.url, "active"
                except Exception as ll_err:
                    log.warning(
                        "Stripe login-link fallback failed for provider %s (acct=%s): %s",
                        provider_id,
                        account_id,
                        ll_err,
                    )

            # Account exists but we cannot generate any Stripe URL right now.
            # Do NOT clear the DB entry — preserve it so the user can retry later.
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
            try:
                onboarding_url, status = _create_hosted_stripe_url(
                    stripe_account_id,
                    existing["status"],
                )
            except RuntimeError as exc:
                if "__STALE_ACCOUNT__" not in str(exc):
                    # Stripe account exists but can't generate a link right now.
                    # Preserve the DB entry so the user can retry later.
                    raise
                # Stripe account was deleted / belongs to a different API-key mode.
                # Clear the stale reference so a fresh account can be created below.
                log.warning(
                    "Clearing stale Stripe account %s for provider %s — will re-create",
                    stripe_account_id,
                    provider_id,
                )
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE provider_accounts SET stripe_account_id='', status='pending' "
                        "WHERE provider_id=%s",
                        (provider_id,),
                    )
                existing = None  # fall through to create a new account below

        if existing and existing.get("stripe_account_id"):
            # Successfully generated a link for an existing account.
            # If the account is now fully active, use complete_onboarding so
            # onboarded_at and notifications are recorded (webhook may not have
            # arrived yet or was missed entirely).
            if status == "active":
                self.complete_onboarding(provider_id)
            else:
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
            # Express account: marketplace provider receives Transfers (not direct charges).
            # Country is provider-supplied for global cross-border payouts.
            acct = stripe_api.Account.create(
                type="express",
                country=country_code,
                email=email,
                capabilities={
                    "card_payments": {"requested": True},
                    "transfers": {"requested": True},
                },
                business_type=cast(Any, provider_type),
                metadata={
                    "xcelsior_provider_id": provider_id,
                    "corporation_name": corporation_name,
                    "business_number": business_number,
                    "country": country_code,
                },
            )
            stripe_account_id = acct.id
            onboarding_url, status = _create_hosted_stripe_url(stripe_account_id, "onboarding")
            log.info(
                "Stripe Connect account created: %s for provider %s country=%s",
                stripe_account_id,
                provider_id,
                country_code,
            )
        except Exception as e:
            log.error("Stripe account creation failed for %s: %s", provider_id, e)
            err_msg = str(e)
            if "signed up for Connect" in err_msg or "platform" in err_msg.lower():
                raise RuntimeError(
                    "Stripe Connect is not yet activated on the platform account. "
                    "The platform administrator needs to enable Connect at "
                    "https://dashboard.stripe.com/connect before providers can onboard."
                ) from e
            raise RuntimeError(f"Failed to start Stripe onboarding: {err_msg}") from e

        # Persist locally
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO provider_accounts
                   (provider_id, provider_type, stripe_account_id, status,
                    corporation_name, business_number, gst_hst_number,
                    email, legal_name, country, province, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (provider_id) DO UPDATE SET
                     provider_type = EXCLUDED.provider_type, stripe_account_id = EXCLUDED.stripe_account_id,
                     status = EXCLUDED.status, corporation_name = EXCLUDED.corporation_name,
                     business_number = EXCLUDED.business_number, gst_hst_number = EXCLUDED.gst_hst_number,
                     email = EXCLUDED.email, legal_name = EXCLUDED.legal_name,
                     country = EXCLUDED.country, province = EXCLUDED.province,
                     created_at = EXCLUDED.created_at""",
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
                    country_code,
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
                    acct_dict = json.loads(str(acct))
                    charges_enabled = bool(acct_dict.get("charges_enabled", False))
                    payouts_enabled = bool(acct_dict.get("payouts_enabled", False))
                    disabled_reason = (acct_dict.get("requirements") or {}).get("disabled_reason")

                    if charges_enabled and payouts_enabled:
                        new_status = "active"
                    elif disabled_reason:
                        new_status = "restricted"
                    else:
                        # Not complete — mark abandoned so the UI can show a
                        # distinct "Resume Setup" CTA. Only downgrade from
                        # onboarding→abandoned, never touch active/restricted.
                        current = provider.get("status", "pending")
                        if current in ("onboarding", "pending", "abandoned"):
                            new_status = "abandoned"
                        else:
                            new_status = current

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

                        # Send in-app notification when status just transitioned to active
                        if new_status == "active" and provider.get("email"):
                            try:
                                from db import NotificationStore

                                NotificationStore.create(
                                    user_email=provider["email"],
                                    notif_type="stripe_connected",
                                    title="Stripe Account Connected",
                                    body="Your Stripe account has been connected successfully! You can now receive payouts for completed GPU jobs.",
                                    data={"provider_id": provider_id},
                                )
                            except Exception as ne:
                                log.warning(
                                    "Failed to create stripe notification for %s: %s",
                                    provider_id,
                                    ne,
                                )
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

    def mark_abandoned(self, provider_id: str) -> dict:
        """Mark a provider's onboarding as abandoned (user left Stripe mid-flow)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM provider_accounts WHERE provider_id=%s",
                (provider_id,),
            ).fetchone()
            if not row:
                return {"provider_id": provider_id, "status": "not_found"}
            # Never downgrade an active or restricted account
            if row["status"] in ("active", "restricted", "suspended"):
                return {"provider_id": provider_id, "status": row["status"]}
            conn.execute(
                "UPDATE provider_accounts SET status='abandoned' WHERE provider_id=%s",
                (provider_id,),
            )
        log.info("Provider %s onboarding ABANDONED", provider_id)
        return {"provider_id": provider_id, "status": "abandoned"}

    def disconnect_provider(self, provider_id: str) -> dict:
        """Unlink the provider's Stripe account so they can re-run setup.

        Best-effort deletes the Express account at Stripe (fails if it holds a
        balance — we keep the local detach either way so the UI returns to the
        connect flow).
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT stripe_account_id FROM provider_accounts WHERE provider_id=%s",
                (provider_id,),
            ).fetchone()
        if not row:
            return {"provider_id": provider_id, "status": "not_found"}

        stripe_account_id = row.get("stripe_account_id")
        if STRIPE_ENABLED and stripe and stripe_account_id:
            try:
                stripe.Account.delete(stripe_account_id)
            except Exception as e:
                log.warning(
                    "Stripe account delete failed for %s (acct=%s): %s — detaching locally",
                    provider_id,
                    stripe_account_id,
                    e,
                )

        with self._conn() as conn:
            conn.execute(
                "UPDATE provider_accounts SET stripe_account_id='', status='pending', onboarded_at=0 "
                "WHERE provider_id=%s",
                (provider_id,),
            )
        log.info("Stripe account unlinked for provider %s", provider_id)
        return {"provider_id": provider_id, "status": "pending"}

    def complete_onboarding(self, provider_id: str) -> dict:
        """Mark a provider's onboarding as complete (webhook callback)."""
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE provider_accounts SET status='active', onboarded_at=%s WHERE provider_id=%s",
                (now, provider_id),
            )
        log.info("Provider %s onboarding COMPLETE", provider_id)

        # Send in-app notification
        try:
            from db import NotificationStore, UserStore

            # Look up the user email from provider_id
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT email FROM provider_accounts WHERE provider_id=%s",
                    (provider_id,),
                ).fetchone()
            if row and row.get("email"):
                NotificationStore.create(
                    user_email=row["email"],
                    notif_type="stripe_connected",
                    title="Stripe Account Connected",
                    body="Your Stripe account has been connected successfully! You can now receive payouts for completed GPU jobs.",
                    data={"provider_id": provider_id},
                )
        except Exception as e:
            log.warning(
                "Failed to create Stripe onboarding notification for %s: %s", provider_id, e
            )

        return {"provider_id": provider_id, "status": "active"}

    # ── Payment Processing ────────────────────────────────────────────

    def create_credit_deposit(
        self,
        customer_id: str,
        amount_cad: float,
        description: str = "Compute credits",
        *,
        address: dict | None = None,
        ip_address: str = "",
        email: str = "",
    ) -> dict:
        """Create a payment intent for depositing compute credits.

        Per Report #1.B: credit-first model. Amount is pretax wallet credits (CAD).
        When Stripe Tax is enabled, tax is calculated exclusively on top and the
        PaymentIntent charges amount_total; the wallet is credited pretax only
        (stored as payment_intents.amount_cents).
        """
        import secrets

        intent_id = f"pi_{secrets.token_hex(12)}"
        credit_cents = int(round(float(amount_cad) * 100))
        credit_cad = round(credit_cents / 100.0, 2)
        # Dashboard-facing description — keep human-readable like historical deposits.
        pi_description = (description or "").strip() or "Compute credits"
        if pi_description.lower() in ("compute credits", "credit deposit", "wallet deposit"):
            pi_description = f"Compute credits — ${credit_cad:.2f} CAD"
        stripe_intent_id = ""
        client_secret = ""
        tax_info: dict[str, Any] = {
            "tax_calculation_id": "",
            "amount_total": credit_cents,
            "tax_amount_cents": 0,
            "credit_amount_cents": credit_cents,
            "tax_enabled": False,
            "breakdown": [],
        }

        if STRIPE_ENABLED and stripe:
            try:
                from stripe_tax import calculate_wallet_deposit_tax

                tax_info = calculate_wallet_deposit_tax(
                    amount_cents=credit_cents,
                    address=address,
                    ip_address=ip_address,
                    currency="cad",
                    reference=f"wallet:{customer_id}:{intent_id}",
                    stripe_mod=stripe,
                )
                charge_cents = int(tax_info.get("amount_total") or credit_cents)
                tax_cents = int(tax_info.get("tax_amount_cents") or 0)
                if tax_cents > 0:
                    pi_description = (
                        f"Compute credits — ${credit_cad:.2f} CAD"
                        f" + tax ${tax_cents / 100.0:.2f}"
                    )

                # Always attach a Stripe Customer so Dashboard shows email/name.
                from billing import get_billing_engine

                cust_id = get_billing_engine().ensure_stripe_customer(
                    customer_id, email=email or ""
                )

                # Dynamic payment methods (Dashboard-configured).
                pi_kwargs: dict[str, Any] = {
                    "amount": charge_cents,
                    "currency": "cad",
                    "customer": cust_id,
                    "automatic_payment_methods": {"enabled": True},
                    "metadata": {
                        "xcelsior_customer_id": customer_id,
                        "xcelsior_intent_id": intent_id,
                        "product_type": "wallet_deposit",
                        "xcelsior_sku": "xcelsior-compute-credits",
                        "credit_amount_cents": str(credit_cents),
                        "tax_amount_cents": str(tax_cents),
                        "tax_calculation_id": str(tax_info.get("tax_calculation_id") or ""),
                    },
                    "description": pi_description,
                    "statement_descriptor_suffix": "CREDITS",
                }
                # Simplified Stripe Tax: link calculation so Stripe records tax
                # transactions automatically on success / refunds.
                calc_id = tax_info.get("tax_calculation_id") or ""
                if calc_id and tax_info.get("tax_enabled"):
                    pi_kwargs["hooks"] = {"inputs": {"tax": {"calculation": calc_id}}}

                try:
                    from stripe_catalog import load_manifest

                    wallet = (load_manifest() or {}).get("wallet_product") or {}
                    if wallet.get("product_id"):
                        pi_kwargs["metadata"]["stripe_product_id"] = wallet["product_id"]
                except Exception:
                    pass

                pi = stripe.PaymentIntent.create(**pi_kwargs)
                stripe_intent_id = pi.id
                client_secret = pi.client_secret
            except Exception as e:
                log.error("Stripe PaymentIntent failed: %s", e)
                raise RuntimeError(f"Failed to create payment intent: {e}") from e
        else:
            raise RuntimeError(
                "Stripe is not configured. Set XCELSIOR_STRIPE_SECRET_KEY "
                "in .env to enable payment processing."
            )

        with self._conn() as conn:
            # amount_cents = pretax credits to deposit into wallet
            conn.execute(
                """INSERT INTO payment_intents
                   (intent_id, customer_id, amount_cents, currency, status,
                    stripe_intent_id, description, created_at)
                   VALUES (%s, %s, %s, 'cad', 'created', %s, %s, %s)""",
                (
                    intent_id,
                    customer_id,
                    credit_cents,
                    stripe_intent_id,
                    pi_description,
                    time.time(),
                ),
            )

        return {
            "intent_id": intent_id,
            "stripe_intent_id": stripe_intent_id,
            "amount_cad": credit_cad,
            "credit_amount_cad": credit_cad,
            "tax_amount_cad": round(int(tax_info.get("tax_amount_cents") or 0) / 100.0, 2),
            "charge_amount_cad": round(int(tax_info.get("amount_total") or credit_cents) / 100.0, 2),
            "tax_calculation_id": tax_info.get("tax_calculation_id") or "",
            "tax_breakdown": tax_info.get("breakdown") or [],
            "tax_enabled": bool(tax_info.get("tax_enabled")),
            "tax_location_source": tax_info.get("location_source") or "",
            "description": pi_description,
            "client_secret": client_secret,
        }

    # ── Payout Splitting ──────────────────────────────────────────────

    def _process_stripe_claim(
        self,
        claimed: dict,
        *,
        stripe_mod=None,
        settlement_label: str,
    ) -> dict:
        """Perform one external transfer outside the claim transaction."""

        client = stripe_mod if stripe_mod is not None else stripe
        provider_id = str(claimed["provider_id"])
        job_id = str(claimed["job_id"])
        provider = self.get_provider(provider_id)
        decision = evaluate_settlement(
            provider=provider,
            provider_share_micros=int(claimed["provider_share_micros"]),
            stripe_mod=client,
        )
        if decision["status"] != "paid_eligible" or provider is None:
            reason = str(decision.get("error") or "not_eligible")
            with self._conn() as conn:
                mark_settlement_retry(
                    conn,
                    settlement_id=int(claimed["id"]),
                    claim_token=str(claimed["claim_token"]),
                    error=reason,
                )
                current = get_settlement(conn, job_id=job_id)
            log.warning(
                "Payout queued job=%s provider=%s reason=%s need=%s available=%s",
                job_id,
                provider_id,
                reason,
                decision.get("need_cents"),
                decision.get("available_cents"),
            )
            return settlement_response(current or claimed)

        try:
            transfer = _stripe_create_transfer(
                amount_cents=int(decision["need_cents"]),
                destination=str(provider["stripe_account_id"]),
                job_id=job_id,
                provider_id=provider_id,
                settlement=settlement_label,
                idempotency_key=str(claimed["rail_idempotency_key"]),
                stripe_mod=client,
            )
            transfer_id = getattr(transfer, "id", None) or (
                transfer.get("id", "") if isinstance(transfer, dict) else ""
            )
            if not transfer_id:
                raise RuntimeError("Stripe transfer returned no identifier")
        except Exception as exc:
            with self._conn() as conn:
                mark_settlement_retry(
                    conn,
                    settlement_id=int(claimed["id"]),
                    claim_token=str(claimed["claim_token"]),
                    error=str(exc),
                )
                current = get_settlement(conn, job_id=job_id)
            log.error("Stripe Transfer failed for job %s: %s", job_id, exc)
            return settlement_response(current or claimed)

        with self._conn() as conn:
            paid = mark_settlement_paid(
                conn,
                settlement_id=int(claimed["id"]),
                claim_token=str(claimed["claim_token"]),
                stripe_transfer_id=str(transfer_id),
            )
        return settlement_response(paid)

    def split_payout(self, job_id: str, provider_id: str) -> dict:
        """Settle a job from PostgreSQL authority; the caller supplies no money."""

        if not (STRIPE_ENABLED and stripe):
            raise RuntimeError(
                "Stripe is not configured. Set XCELSIOR_STRIPE_SECRET_KEY "
                "in .env to enable provider payouts."
            )
        with self._conn() as conn:
            prepared = prepare_settlement(
                conn,
                job_id=job_id,
                provider_id=provider_id,
                rail="stripe",
            )
        if prepared.get("settlement_status") == "paid":
            return settlement_response(prepared)

        owner = f"stripe-instant:{uuid.uuid4()}"
        with self._conn() as conn:
            claimed = claim_settlements(
                conn,
                rail="stripe",
                owner=owner,
                limit=1,
                job_id=job_id,
            )
            current = get_settlement(conn, job_id=job_id)
        if not claimed:
            if current is None:
                raise SettlementConflict(
                    "settlement_missing",
                    "Prepared settlement disappeared before it could be claimed",
                )
            return settlement_response(current)
        return self._process_stripe_claim(
            claimed[0],
            stripe_mod=stripe,
            settlement_label="instant",
        )

    def create_account_session(self, provider_id: str) -> dict:
        """Create a Connect AccountSession for embedded onboarding components.

        Components: account_onboarding, notification_banner, account_management, payouts.
        """
        if not (STRIPE_ENABLED and stripe):
            raise RuntimeError("Stripe is not configured")
        provider = self.get_provider(provider_id)
        if not provider or not provider.get("stripe_account_id"):
            raise RuntimeError("Provider has no Stripe account — start onboarding first")
        account_id = provider["stripe_account_id"]
        session = stripe.AccountSession.create(
            account=account_id,
            components={
                "account_onboarding": {"enabled": True},
                "notification_banner": {"enabled": True},
                "account_management": {"enabled": True},
                "payouts": {"enabled": True},
            },
        )
        return {
            "client_secret": session.client_secret,
            "account_id": account_id,
            "provider_id": provider_id,
            "expires_at": getattr(session, "expires_at", None),
        }

    def settle_queued_payouts(self, *, limit: int = 100, stripe_mod=None) -> dict:
        """Daily settlement: attempt Transfers for queued payout_splits.

        Safe to run repeatedly; skips rows already paid. Does not create new
        commercial charges — only moves already-earned platform float.
        """
        client = stripe_mod if stripe_mod is not None else stripe
        if not (STRIPE_ENABLED and client):
            return {"settled": 0, "failed": 0, "skipped": 0, "reason": "stripe_disabled"}
        settled = failed = skipped = 0
        owner = f"stripe-daily:{uuid.uuid4()}"
        with self._conn() as conn:
            rows = claim_settlements(
                conn,
                rail="stripe",
                owner=owner,
                limit=limit,
            )
        for row in rows:
            try:
                result = self._process_stripe_claim(
                    row,
                    stripe_mod=client,
                    settlement_label="daily",
                )
                if result.get("settlement_status") == "paid":
                    settled += 1
                elif result.get("settlement_error"):
                    failed += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                log.error("Daily settlement failed job=%s: %s", row.get("job_id"), exc)
        return {"settled": settled, "failed": failed, "skipped": skipped}

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
                    COALESCE(SUM(ps.provider_share_micros) / 1000000.0, 0) as total_earned_cad,
                    COALESCE(SUM(ps.platform_share_micros) / 1000000.0, 0) as total_platform_cad,
                    COALESCE(SUM(ps.gst_hst_micros) / 1000000.0, 0) as total_tax_cad,
                    COALESCE(SUM(CASE WHEN COALESCE(j.pricing_mode, 'on_demand') = 'spot'
                        THEN ps.provider_share_micros / 1000000.0 ELSE 0 END), 0) as spot_earned_cad,
                    COALESCE(SUM(CASE WHEN COALESCE(j.pricing_mode, 'on_demand') != 'spot'
                        THEN ps.provider_share_micros / 1000000.0 ELSE 0 END), 0) as on_demand_earned_cad
                   FROM payout_splits ps
                   LEFT JOIN jobs j ON j.job_id = ps.job_id
                   WHERE ps.provider_id=%s""",
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
                    "spot_earned_cad": 0,
                    "on_demand_earned_cad": 0,
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

        verification_candidates = _webhook_secret_candidates()
        if not verification_candidates:
            log.error("No Stripe webhook secret configured — rejecting event")
            return {"handled": False, "error": "no webhook secret configured"}

        event = None
        last_err = None
        for candidate in verification_candidates:
            try:
                event = stripe.Webhook.construct_event(payload, sig_header, candidate)
                break
            except Exception as e:  # SignatureVerificationError or ValueError
                last_err = e
        if event is None:
            log.error(
                "Webhook signature verification failed against %d secret(s): %s",
                len(verification_candidates),
                last_err,
            )
            return {"handled": False, "error": str(last_err)}

        # Convert StripeObject to plain dict for safe attribute access
        # str(event) returns JSON reliably across all stripe SDK versions
        import json as _json

        event_dict = _json.loads(str(event))

        event_id = event_dict["id"]
        event_type = event_dict["type"]
        now = time.time()

        # Phase 1: Write to inbox (idempotent via PK)
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT event_id, status FROM stripe_event_inbox WHERE event_id = %s",
                (event_id,),
            ).fetchone()
            if existing:
                log.info(
                    "Webhook event %s already in inbox (status=%s), skipping",
                    event_id,
                    existing["status"],
                )
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
                    event_dict.get("account", ""),
                    event_dict.get("livemode", True),
                    event_dict.get("api_version", ""),
                    event_dict.get("created", 0),
                    now,
                    Jsonb(event_dict),
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
                log.error(
                    "Event %s processing failed (attempt %d/%d): %s",
                    event_id,
                    attempts,
                    max_attempts,
                    e,
                )
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
        elif event_type == "checkout.session.completed":
            self._handle_checkout_completed(data, event_id)
        else:
            log.debug("Unhandled event type: %s", event_type)

    def _handle_account_updated(self, data: dict):
        acct_id = data["id"]
        with self._conn() as conn:
            row = conn.execute(
                "SELECT provider_id FROM provider_accounts WHERE stripe_account_id=%s",
                (acct_id,),
            ).fetchone()
            if not row:
                log.debug(
                    "account.updated for %s — no matching provider_accounts row, skipping", acct_id
                )
                return
            charges_enabled = data.get("charges_enabled", False)
            payouts_enabled = data.get("payouts_enabled", False)
            log.info(
                "account.updated for provider %s: charges=%s payouts=%s",
                row["provider_id"],
                charges_enabled,
                payouts_enabled,
            )
            if charges_enabled and payouts_enabled:
                self.complete_onboarding(row["provider_id"])
            else:
                # Stripe disabled capabilities — mark restricted if there's a reason
                reqs = data.get("requirements", {})
                if reqs.get("disabled_reason"):
                    conn.execute(
                        "UPDATE provider_accounts SET status='restricted' WHERE provider_id=%s",
                        (row["provider_id"],),
                    )
                    log.warning(
                        "Provider %s restricted: %s",
                        row["provider_id"],
                        reqs.get("disabled_reason"),
                    )

    def _handle_payment_succeeded(self, data: dict, event_id: str):
        """Credit the wallet for a confirmed charge.

        This is the only place a card payment turns into wallet balance, which
        makes a missed match here indistinguishable from theft: the customer is
        charged and receives nothing.

        The local `payment_intents` row is the normal way to identify who to
        credit, but it must not be the *only* way. It is written after Stripe
        confirms the charge, so a fast webhook can arrive before the insert
        commits, and a crash between the two loses the row permanently. The
        event itself carries everything needed — `metadata.customer_id` and the
        confirmed `amount` — so it is used whenever the row is absent.

        Both paths share one idempotency key, derived from the event id, so a
        redelivery credits once no matter which path handled the original.
        """
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
            customer_id, amount_cents = row["customer_id"], row["amount_cents"]
            source = "payment_intents"
        else:
            metadata = data.get("metadata") or {}
            customer_id = (metadata.get("customer_id") or "").strip()
            amount_cents = data.get("amount_received") or data.get("amount") or 0
            if not customer_id or metadata.get("product_type") != "wallet_deposit":
                # Not a wallet deposit of ours — a Connect charge, or an intent
                # created outside this system. Nothing to credit.
                log.debug("payment_intent.succeeded %s matched no wallet deposit", si_id)
                return
            source = "event metadata"
            log.warning(
                "No payment_intents row for %s; crediting %s from %s. The intent "
                "was not registered at charge time — check the top-up path.",
                si_id,
                customer_id,
                source,
            )

        if not amount_cents:
            log.error("Refusing to credit %s for %s: amount is zero", customer_id, si_id)
            return

        from billing import get_billing_engine

        amount_cad = round(amount_cents / 100.0, 2)
        engine = get_billing_engine()
        # Idempotent deposit using event_id as idempotency_key
        engine.deposit(
            customer_id,
            amount_cad,
            description=f"Stripe deposit {si_id}",
            idempotency_key=f"stripe:{event_id}",
        )
        log.info(
            "Wallet credited: %s +$%.2f from %s (via %s)",
            customer_id,
            amount_cad,
            si_id,
            source,
        )

        # If wallet was suspended and balance is now positive, reactivate
        wallet = engine.get_wallet(customer_id)
        if wallet.get("status") == "suspended" and wallet.get("balance_cad", 0) > 0:
            engine.reactivate_wallet(customer_id)

    def _handle_payment_failed(self, data: dict):
        si_id = data["id"]
        failure_code = data.get("last_payment_error", {}).get("code", "unknown")
        with self._conn() as conn:
            conn.execute(
                "UPDATE payment_intents SET status='failed' WHERE stripe_intent_id=%s",
                (si_id,),
            )
        log.warning("Payment failed: %s reason=%s", si_id, failure_code)

    def _handle_checkout_completed(self, data: dict, event_id: str):
        """Record a completed marketplace Checkout (destination charge).

        Fired when a customer finishes paying on a Connect Checkout Session
        created by the marketplace storefront. The session is a *destination
        charge*: the platform kept an application fee and the remainder is
        transferred to the connected account named in
        ``payment_intent_data.transfer_data.destination``. We persist the sale
        idempotently (``session_id`` is the primary key) so the storefront can
        show fulfilment and we have a provider-level ledger of marketplace
        revenue. The inbox already dedupes on ``event_id``; the ON CONFLICT is a
        second guard against Stripe's at-least-once retries.
        """
        session_id = data.get("id", "")
        payment_status = data.get("payment_status")
        if payment_status not in ("paid", "no_payment_required"):
            log.info(
                "checkout.session.completed %s payment_status=%s — not recording",
                session_id,
                payment_status,
            )
            return
        meta = data.get("metadata") or {}
        amount_total = data.get("amount_total", 0) or 0
        currency = (data.get("currency") or "").lower()

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO marketplace_sales
                     (session_id, payment_intent_id, destination_account, product_id,
                      amount_total_cents, currency, customer_email, event_id, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO NOTHING""",
                (
                    session_id,
                    data.get("payment_intent", "") or "",
                    meta.get("destination_account", ""),
                    meta.get("product_id", ""),
                    amount_total,
                    currency,
                    (data.get("customer_details") or {}).get("email", ""),
                    event_id,
                    time.time(),
                ),
            )
        log.info(
            "Marketplace sale recorded: session=%s intent=%s dest=%s amount=%d %s",
            session_id,
            data.get("payment_intent", ""),
            meta.get("destination_account", ""),
            amount_total,
            currency.upper(),
        )

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
        log.warning(
            "Transfer REVERSED: %s job=%s provider=%s amount=%d cents",
            transfer_id,
            job_id,
            provider_id,
            amount_cents,
        )
        # Claw back from provider's pending balance tracking
        if provider_id:
            with self._conn() as conn:
                conn.execute(
                    """UPDATE payout_splits
                          SET platform_share_micros = total_micros,
                              provider_share_micros = 0,
                              settlement_status = 'manual_review',
                              settlement_error = 'stripe_transfer_reversed',
                              updated_at = clock_timestamp()
                        WHERE job_id = %s
                          AND provider_id = %s
                          AND settlement_key IS NOT NULL
                          AND stripe_transfer_id = %s""",
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
