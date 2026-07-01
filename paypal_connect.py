"""PayPal Complete Payments — provider onboarding and marketplace payouts."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Optional

import httpx

log = logging.getLogger("xcelsior.paypal_connect")

_PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID", "")
_PAYPAL_CLIENT_SECRET = os.environ.get("PAYPAL_CLIENT_SECRET", "")
_PAYPAL_MODE = os.environ.get("PAYPAL_MODE", "sandbox")
_PAYPAL_PARTNER_ATTRIBUTION_ID = os.environ.get("PAYPAL_PARTNER_ATTRIBUTION_ID", "")
_PAYPAL_PLATFORM_MERCHANT_ID = os.environ.get("PAYPAL_PLATFORM_MERCHANT_ID", "")
_PAYPAL_PLATFORM_PAYEE_EMAIL = os.environ.get("PAYPAL_PLATFORM_PAYEE_EMAIL", "")
_PAYPAL_PARTNER_MERCHANT_ID = os.environ.get(
    "PAYPAL_PARTNER_MERCHANT_ID", _PAYPAL_PLATFORM_MERCHANT_ID
)
_BASE_URL = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
_PAYPAL_BASE = (
    "https://api-m.paypal.com" if _PAYPAL_MODE == "live" else "https://api-m.sandbox.paypal.com"
)

_raw_cut = float(os.environ.get("XCELSIOR_PLATFORM_CUT", "0.15"))
PLATFORM_CUT_FRAC = _raw_cut if _raw_cut <= 1.0 else _raw_cut / 100.0

PAYPAL_ENABLED = bool(_PAYPAL_CLIENT_ID and _PAYPAL_CLIENT_SECRET)


def paypal_enabled() -> bool:
    return PAYPAL_ENABLED


def _access_token() -> str:
    resp = httpx.post(
        f"{_PAYPAL_BASE}/v1/oauth2/token",
        data={"grant_type": "client_credentials"},
        auth=(_PAYPAL_CLIENT_ID, _PAYPAL_CLIENT_SECRET),
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _b64_json(obj: dict) -> str:
    return base64.b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode()


def auth_assertion(*, merchant_id: str = "", email: str = "") -> str:
    """PayPal-Auth-Assertion JWT for acting on behalf of a connected seller."""
    header = _b64_json({"alg": "none"})
    payload: dict[str, str] = {"iss": _PAYPAL_CLIENT_ID}
    if merchant_id:
        payload["payer_id"] = merchant_id
    elif email:
        payload["email"] = email
    return f"{header}.{_b64_json(payload)}."


def _headers(token: str, *, seller_merchant_id: str = "", seller_email: str = "") -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if _PAYPAL_PARTNER_ATTRIBUTION_ID:
        headers["PayPal-Partner-Attribution-Id"] = _PAYPAL_PARTNER_ATTRIBUTION_ID
    if seller_merchant_id or seller_email:
        headers["PayPal-Auth-Assertion"] = auth_assertion(
            merchant_id=seller_merchant_id, email=seller_email
        )
    return headers


def platform_payee() -> dict[str, str] | None:
    """Payee for platform wallet deposits (not provider marketplace orders)."""
    merchant_id = _PAYPAL_PLATFORM_MERCHANT_ID.strip()
    email = _PAYPAL_PLATFORM_PAYEE_EMAIL.strip()
    if merchant_id and merchant_id.isdigit() and len(merchant_id) > 15:
        merchant_id = ""
    if _PAYPAL_MODE == "live":
        if merchant_id:
            return {"merchant_id": merchant_id}
        if email:
            return {"email_address": email}
        return None
    if email:
        return {"email_address": email}
    if merchant_id:
        return {"merchant_id": merchant_id}
    return None


def wallet_purchase_unit(customer_id: str, amount_cad: float) -> dict:
    unit: dict[str, Any] = {
        "amount": {"currency_code": "CAD", "value": f"{amount_cad:.2f}"},
        "description": f"Xcelsior compute credits — {customer_id}",
        "custom_id": customer_id,
    }
    payee = platform_payee()
    if payee:
        unit["payee"] = payee
    return unit


def _split_amounts(total_cad: float, province: str = "ON") -> dict[str, float]:
    from billing import get_tax_rate_for_province

    platform_share = round(total_cad * PLATFORM_CUT_FRAC, 2)
    provider_share = round(total_cad - platform_share, 2)
    tax_rate = get_tax_rate_for_province(province)
    gst_hst = round(total_cad * tax_rate, 2)
    return {
        "platform_share_cad": platform_share,
        "provider_share_cad": provider_share,
        "gst_hst_cad": gst_hst,
        "tax_rate": tax_rate,
    }


class PayPalConnectManager:
    """Partner referrals, seller status, and marketplace orders with platform fees."""

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

    def tracking_id(self, provider_id: str) -> str:
        # PayPal rejects partner-referral calls that reuse a tracking_id, so a
        # static "xcelsior-{provider_id}" value 502s on every retry after the
        # first attempt (INVALID_RESOURCE_ID / DUPLICATE_REQUEST_ID). Suffix
        # with a timestamp so each onboarding attempt gets a fresh id; the new
        # value is persisted immediately in create_onboarding_link so status
        # refreshes and the completion webhook keep matching correctly.
        return f"xcelsior-{provider_id}-{int(time.time())}"

    def get_paypal_profile(self, provider_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM provider_accounts WHERE provider_id=%s",
                (provider_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "provider_id": provider_id,
            "tracking_id": row.get("paypal_tracking_id") or "",
            "merchant_id": row.get("paypal_merchant_id") or "",
            "payer_id": row.get("paypal_payer_id") or "",
            "status": row.get("paypal_status") or "not_started",
            "onboarded_at": row.get("paypal_onboarded_at") or 0,
        }

    def _referral_body(self, provider_id: str, email: str) -> dict:
        return {
            "tracking_id": self.tracking_id(provider_id),
            "email": email,
            "preferred_language_code": "en-CA",
            "legal_country_code": "CA",
            "operations": [
                {
                    "operation": "API_INTEGRATION",
                    "api_integration_preference": {
                        "rest_api_integration": {
                            "integration_method": "PAYPAL",
                            "integration_type": "THIRD_PARTY",
                            "third_party_details": {
                                "features": [
                                    "PAYMENT",
                                    "REFUND",
                                    "PARTNER_FEE",
                                    "ACCESS_MERCHANT_INFORMATION",
                                ]
                            },
                        }
                    },
                }
            ],
            "products": ["EXPRESS_CHECKOUT"],
            "legal_consents": [{"type": "SHARE_DATA_CONSENT", "granted": True}],
            "partner_config_override": {
                "return_url": f"{_BASE_URL}/dashboard/earnings?paypal=return&provider={provider_id}",
                "return_url_description": "Return to Xcelsior",
                "action_renewal_url": f"{_BASE_URL}/dashboard/earnings?paypal=refresh&provider={provider_id}",
                "show_add_credit_card": True,
            },
        }

    def create_onboarding_link(self, provider_id: str, email: str) -> dict:
        if not PAYPAL_ENABLED:
            raise RuntimeError("PayPal is not configured")
        token = _access_token()
        resp = httpx.post(
            f"{_PAYPAL_BASE}/v2/customer/partner-referrals",
            headers=_headers(token),
            json=self._referral_body(provider_id, email),
            timeout=30,
        )
        if resp.status_code >= 400:
            log.error("PayPal partner referral failed: %s", resp.text)
            raise RuntimeError("PayPal provider onboarding failed")
        links = resp.json().get("links") or []
        action_url = next((l["href"] for l in links if l.get("rel") == "action_url"), "")
        if not action_url:
            raise RuntimeError("PayPal onboarding URL missing from referral response")
        tracking_id = self.tracking_id(provider_id)
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO provider_accounts (provider_id, email, status, paypal_tracking_id, paypal_status, created_at)
                   VALUES (%s, %s, 'pending', %s, 'onboarding', %s)
                   ON CONFLICT (provider_id) DO UPDATE SET
                     paypal_tracking_id=EXCLUDED.paypal_tracking_id,
                     paypal_status='onboarding',
                     email=COALESCE(NULLIF(provider_accounts.email,''), EXCLUDED.email)""",
                (provider_id, email, tracking_id, now),
            )
        log.info("PayPal onboarding link created for provider %s", provider_id)
        return {
            "provider_id": provider_id,
            "onboarding_url": action_url,
            "tracking_id": tracking_id,
            "status": "onboarding",
        }

    def refresh_merchant_status(self, provider_id: str) -> dict:
        profile = self.get_paypal_profile(provider_id)
        if not profile or not profile.get("tracking_id"):
            return {"provider_id": provider_id, "status": "not_started"}
        if not _PAYPAL_PARTNER_MERCHANT_ID:
            return profile
        token = _access_token()
        url = (
            f"{_PAYPAL_BASE}/v1/customer/partners/{_PAYPAL_PARTNER_MERCHANT_ID}"
            f"/merchant-integrations?tracking_id={profile['tracking_id']}"
        )
        resp = httpx.get(url, headers=_headers(token), timeout=20)
        if resp.status_code >= 400:
            log.warning("PayPal merchant-integrations lookup %s: %s", resp.status_code, resp.text[:200])
            return profile
        data = resp.json()
        merchant_id = data.get("merchant_id") or ""
        payer_id = (data.get("primary_email") or {}).get("payer_id") or data.get("payer_id") or ""
        payments_receivable = bool(data.get("payments_receivable"))
        status = "active" if payments_receivable and merchant_id else "onboarding"
        onboarded_at = time.time() if status == "active" else profile.get("onboarded_at") or 0
        with self._conn() as conn:
            conn.execute(
                """UPDATE provider_accounts
                   SET paypal_merchant_id=%s, paypal_payer_id=%s, paypal_status=%s,
                       paypal_onboarded_at=CASE WHEN %s > 0 THEN %s ELSE paypal_onboarded_at END
                   WHERE provider_id=%s""",
                (merchant_id, payer_id, status, onboarded_at, onboarded_at, provider_id),
            )
        return {
            "provider_id": provider_id,
            "merchant_id": merchant_id,
            "payer_id": payer_id,
            "status": status,
            "payments_receivable": payments_receivable,
        }

    def disconnect(self, provider_id: str) -> dict:
        """Unlink the provider's PayPal seller account.

        Clears the merchant link and resets status so the connect flow can be
        run again from scratch. Stripe payouts are unaffected.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT provider_id FROM provider_accounts WHERE provider_id=%s",
                (provider_id,),
            ).fetchone()
            if not row:
                return {"provider_id": provider_id, "status": "not_found"}
            conn.execute(
                """UPDATE provider_accounts
                   SET paypal_merchant_id='', paypal_payer_id='', paypal_tracking_id='',
                       paypal_status='not_started', paypal_onboarded_at=0
                   WHERE provider_id=%s""",
                (provider_id,),
            )
        log.info("PayPal unlinked for provider %s", provider_id)
        return {"provider_id": provider_id, "status": "not_started"}

    def complete_onboarding_from_webhook(
        self,
        *,
        provider_id: str = "",
        tracking_id: str = "",
        merchant_id: str = "",
        payer_id: str = "",
    ) -> None:
        with self._conn() as conn:
            if provider_id:
                conn.execute(
                    """UPDATE provider_accounts
                       SET paypal_merchant_id=COALESCE(NULLIF(%s,''), paypal_merchant_id),
                           paypal_payer_id=COALESCE(NULLIF(%s,''), paypal_payer_id),
                           paypal_status='active', paypal_onboarded_at=%s
                       WHERE provider_id=%s""",
                    (merchant_id, payer_id, time.time(), provider_id),
                )
            elif tracking_id:
                conn.execute(
                    """UPDATE provider_accounts
                       SET paypal_merchant_id=COALESCE(NULLIF(%s,''), paypal_merchant_id),
                           paypal_payer_id=COALESCE(NULLIF(%s,''), paypal_payer_id),
                           paypal_status='active', paypal_onboarded_at=%s
                       WHERE paypal_tracking_id=%s""",
                    (merchant_id, payer_id, time.time(), tracking_id),
                )

    def seller_payee(self, provider_id: str) -> tuple[dict[str, str], str, str]:
        """Return (payee dict, merchant_id, email) for marketplace orders."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT paypal_merchant_id, paypal_payer_id, email, paypal_status FROM provider_accounts WHERE provider_id=%s",
                (provider_id,),
            ).fetchone()
        if not row:
            raise RuntimeError(f"Provider {provider_id} not found")
        if row.get("paypal_status") != "active":
            raise RuntimeError("Provider has not completed PayPal onboarding")
        merchant_id = (row.get("paypal_merchant_id") or "").strip()
        payer_id = (row.get("paypal_payer_id") or "").strip()
        email = (row.get("email") or "").strip()
        auth_id = merchant_id or payer_id
        if _PAYPAL_MODE == "live" and merchant_id:
            return {"merchant_id": merchant_id}, auth_id, email
        if email:
            return {"email_address": email}, auth_id, email
        if merchant_id:
            return {"merchant_id": merchant_id}, auth_id, email
        raise RuntimeError("Provider PayPal merchant identity missing")

    def marketplace_purchase_unit(self, provider_id: str, job_id: str, amount_cad: float) -> dict:
        payee, _, _ = self.seller_payee(provider_id)
        splits = _split_amounts(amount_cad)
        unit: dict[str, Any] = {
            "amount": {"currency_code": "CAD", "value": f"{amount_cad:.2f}"},
            "description": f"Xcelsior marketplace — {job_id}",
            "custom_id": f"{provider_id}:{job_id}",
            "payee": payee,
            "payment_instruction": {
                "disbursement_mode": "INSTANT",
                "platform_fees": [
                    {
                        "amount": {
                            "currency_code": "CAD",
                            "value": f"{splits['platform_share_cad']:.2f}",
                        }
                    }
                ],
            },
        }
        return unit

    def create_marketplace_order(self, provider_id: str, job_id: str, amount_cad: float) -> dict:
        if not PAYPAL_ENABLED:
            raise RuntimeError("PayPal is not configured")
        payee, auth_id, email = self.seller_payee(provider_id)
        token = _access_token()
        resp = httpx.post(
            f"{_PAYPAL_BASE}/v2/checkout/orders",
            headers=_headers(token, seller_merchant_id=auth_id, seller_email=email if not auth_id else ""),
            json={
                "intent": "CAPTURE",
                "purchase_units": [self.marketplace_purchase_unit(provider_id, job_id, amount_cad)],
                "application_context": {
                    "brand_name": "Xcelsior",
                    "shipping_preference": "NO_SHIPPING",
                },
            },
            timeout=20,
        )
        if resp.status_code >= 400:
            log.error("PayPal marketplace create-order failed: %s", resp.text)
            raise RuntimeError("PayPal marketplace order creation failed")
        data = resp.json()
        splits = _split_amounts(amount_cad)
        return {
            "order_id": data["id"],
            "provider_id": provider_id,
            "job_id": job_id,
            "amount_cad": amount_cad,
            **splits,
        }

    def capture_marketplace_order(self, provider_id: str, order_id: str) -> dict:
        payee, auth_id, email = self.seller_payee(provider_id)
        token = _access_token()
        hdrs = _headers(token, seller_merchant_id=auth_id, seller_email=email if not auth_id else "")
        get_resp = httpx.get(f"{_PAYPAL_BASE}/v2/checkout/orders/{order_id}", headers=hdrs, timeout=15)
        get_resp.raise_for_status()
        order = get_resp.json()
        if order.get("status") == "COMPLETED":
            data = order
        elif order.get("status") != "APPROVED":
            raise RuntimeError(f"PayPal order status: {order.get('status', 'unknown')}")
        else:
            cap_resp = httpx.post(
                f"{_PAYPAL_BASE}/v2/checkout/orders/{order_id}/capture",
                headers=hdrs,
                json={},
                timeout=20,
            )
            if cap_resp.status_code >= 400:
                log.error("PayPal marketplace capture failed: %s", cap_resp.text)
                raise RuntimeError("PayPal marketplace capture failed")
            data = cap_resp.json()
        capture = data["purchase_units"][0]["payments"]["captures"][0]
        custom_id = (data["purchase_units"][0].get("custom_id") or "")
        job_id = custom_id.split(":", 1)[-1] if ":" in custom_id else custom_id
        if custom_id and ":" in custom_id:
            custom_provider = custom_id.split(":", 1)[0]
            if custom_provider and custom_provider != provider_id:
                raise RuntimeError("PayPal order provider does not match request")
        amount_cad = float(capture["amount"]["value"])
        splits = _split_amounts(amount_cad)
        capture_id = capture.get("id", "")
        with self._conn() as conn:
            existing = conn.execute(
                """SELECT job_id, provider_id, total_cad, provider_share_cad, platform_share_cad,
                          gst_hst_cad, paypal_capture_id
                   FROM payout_splits
                   WHERE job_id=%s AND payment_rail='paypal'""",
                (job_id,),
            ).fetchone()
            if existing:
                log.info("PayPal marketplace capture idempotent for job %s", job_id)
                return {
                    "order_id": order_id,
                    "job_id": existing["job_id"],
                    "provider_id": existing["provider_id"],
                    "capture_id": existing.get("paypal_capture_id") or capture_id,
                    "amount_cad": float(existing["total_cad"]),
                    "provider_share_cad": float(existing["provider_share_cad"]),
                    "platform_share_cad": float(existing["platform_share_cad"]),
                    "gst_hst_cad": float(existing["gst_hst_cad"]),
                    "tax_rate": splits["tax_rate"],
                }
            conn.execute(
                """INSERT INTO payout_splits
                   (job_id, provider_id, total_cad, provider_share_cad, platform_share_cad,
                    gst_hst_cad, stripe_transfer_id, paypal_capture_id, payment_rail, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, '', %s, 'paypal', %s)""",
                (
                    job_id,
                    provider_id,
                    amount_cad,
                    splits["provider_share_cad"],
                    splits["platform_share_cad"],
                    splits["gst_hst_cad"],
                    capture_id,
                    time.time(),
                ),
            )
        return {
            "order_id": order_id,
            "job_id": job_id,
            "provider_id": provider_id,
            "capture_id": capture_id,
            "amount_cad": amount_cad,
            **splits,
        }


_manager: Optional[PayPalConnectManager] = None


def get_paypal_manager() -> PayPalConnectManager:
    global _manager
    if _manager is None:
        _manager = PayPalConnectManager()
    return _manager