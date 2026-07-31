"""PayPal Complete Payments — provider onboarding and marketplace payouts."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional

import httpx

from money import cad_to_micros, micros_to_cad
from provider_settlement import (
    SettlementConflict,
    SettlementNotFound,
    claim_settlements,
    get_settlement,
    mark_awaiting_paypal_capture,
    mark_settlement_paid,
    mark_settlement_retry,
    platform_cut_bps,
    prepare_settlement,
    settlement_response,
    split_source_micros,
    tax_rate_bps_for_province,
)

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


def _headers(
    token: str,
    *,
    seller_merchant_id: str = "",
    seller_email: str = "",
    request_id: str = "",
) -> dict[str, str]:
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
    if request_id:
        headers["PayPal-Request-Id"] = request_id
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
    """Compatibility projection backed by the exact settlement calculator."""

    tax_bps, _description = tax_rate_bps_for_province(province)
    exact = split_source_micros(
        cad_to_micros(total_cad),
        cut_bps=platform_cut_bps(str(PLATFORM_CUT_FRAC)),
        tax_bps=tax_bps,
    )
    return {
        "platform_share_cad": micros_to_cad(exact.platform_share_micros),
        "provider_share_cad": micros_to_cad(exact.provider_share_micros),
        "gst_hst_cad": micros_to_cad(exact.gst_hst_micros),
        "tax_rate": exact.tax_rate_bps / 10_000,
    }


def _paypal_value(amount_micros: int) -> str:
    amount = int(amount_micros)
    if amount < 0 or amount % 10_000:
        raise ValueError("PayPal settlement amounts must be whole CAD cents")
    return f"{amount // 1_000_000}.{(amount // 10_000) % 100:02d}"


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

    def marketplace_purchase_unit(
        self,
        provider_id: str,
        job_id: str,
        settlement: dict,
    ) -> dict:
        payee, _, _ = self.seller_payee(provider_id)
        unit: dict[str, Any] = {
            "amount": {
                "currency_code": str(settlement["currency"]).upper(),
                "value": _paypal_value(int(settlement["total_micros"])),
            },
            "description": f"Xcelsior marketplace — {job_id}",
            "custom_id": f"{provider_id}:{job_id}",
            "payee": payee,
            "payment_instruction": {
                "disbursement_mode": "INSTANT",
                "platform_fees": [
                    {
                        "amount": {
                            "currency_code": str(settlement["currency"]).upper(),
                            "value": _paypal_value(
                                int(settlement["platform_share_micros"])
                            ),
                        }
                    }
                ],
            },
        }
        return unit

    def create_marketplace_order(
        self,
        provider_id: str,
        job_id: str,
        *,
        expected_customer_id: str | None = None,
    ) -> dict:
        """Create one idempotent PayPal order from PostgreSQL authority."""

        if not PAYPAL_ENABLED:
            raise RuntimeError("PayPal is not configured")
        payee, auth_id, email = self.seller_payee(provider_id)
        with self._conn() as conn:
            prepared = prepare_settlement(
                conn,
                job_id=job_id,
                provider_id=provider_id,
                rail="paypal",
                expected_customer_id=expected_customer_id,
            )
        if prepared.get("paypal_order_id"):
            return {
                "order_id": prepared["paypal_order_id"],
                **settlement_response(prepared),
            }
        if prepared.get("settlement_status") == "paid":
            return {
                "order_id": prepared.get("paypal_order_id") or "",
                **settlement_response(prepared),
            }

        owner = f"paypal-order:{uuid.uuid4()}"
        with self._conn() as conn:
            claims = claim_settlements(
                conn,
                rail="paypal",
                owner=owner,
                limit=1,
                job_id=job_id,
                allowed_statuses=("pending", "queued", "failed"),
            )
            current = get_settlement(conn, job_id=job_id)
        if not claims:
            if current and current.get("paypal_order_id"):
                return {
                    "order_id": current["paypal_order_id"],
                    **settlement_response(current),
                }
            raise SettlementConflict(
                "settlement_busy",
                "The PayPal settlement is already being prepared",
            )
        claimed = claims[0]

        token = _access_token()
        try:
            resp = httpx.post(
                f"{_PAYPAL_BASE}/v2/checkout/orders",
                headers=_headers(
                    token,
                    seller_merchant_id=auth_id,
                    seller_email=email if not auth_id else "",
                    request_id=str(claimed["rail_idempotency_key"]),
                ),
                json={
                    "intent": "CAPTURE",
                    "purchase_units": [
                        self.marketplace_purchase_unit(provider_id, job_id, claimed)
                    ],
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
            order_id = str(data.get("id") or "")
            if not order_id:
                raise RuntimeError("PayPal marketplace order returned no identifier")
        except Exception as exc:
            with self._conn() as conn:
                mark_settlement_retry(
                    conn,
                    settlement_id=int(claimed["id"]),
                    claim_token=str(claimed["claim_token"]),
                    error=str(exc),
                )
            raise

        with self._conn() as conn:
            awaiting = mark_awaiting_paypal_capture(
                conn,
                settlement_id=int(claimed["id"]),
                claim_token=str(claimed["claim_token"]),
                paypal_order_id=order_id,
            )
        return {
            "order_id": order_id,
            **settlement_response(awaiting),
        }

    def capture_marketplace_order(
        self,
        provider_id: str,
        order_id: str,
        *,
        expected_customer_id: str | None = None,
    ) -> dict:
        """Capture the persisted PayPal order under a durable DB claim."""

        payee, auth_id, email = self.seller_payee(provider_id)
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT *
                  FROM payout_splits
                 WHERE paypal_order_id = %s
                   AND payment_rail = 'paypal'
                   AND settlement_key IS NOT NULL
                """,
                (order_id,),
            ).fetchone()
            settlement = dict(row) if row else None
        if settlement is None or str(settlement.get("provider_id") or "") != provider_id:
            raise SettlementNotFound("PayPal settlement order was not found")
        if expected_customer_id and str(settlement.get("customer_id") or "") != expected_customer_id:
            raise SettlementNotFound("PayPal settlement order was not found")
        if settlement.get("settlement_status") == "paid":
            return {
                "order_id": order_id,
                "capture_id": settlement.get("paypal_capture_id") or "",
                **settlement_response(settlement),
            }

        owner = f"paypal-capture:{uuid.uuid4()}"
        with self._conn() as conn:
            claims = claim_settlements(
                conn,
                rail="paypal",
                owner=owner,
                limit=1,
                job_id=str(settlement["job_id"]),
                allowed_statuses=("awaiting_capture", "failed"),
            )
            current = get_settlement(conn, job_id=str(settlement["job_id"]))
        if not claims:
            if current and current.get("settlement_status") == "paid":
                return {
                    "order_id": order_id,
                    "capture_id": current.get("paypal_capture_id") or "",
                    **settlement_response(current),
                }
            raise SettlementConflict(
                "settlement_busy",
                "The PayPal settlement is already being captured",
            )
        claimed = claims[0]

        token = _access_token()
        hdrs = _headers(
            token,
            seller_merchant_id=auth_id,
            seller_email=email if not auth_id else "",
            request_id=f"{claimed['rail_idempotency_key']}:capture",
        )
        try:
            get_resp = httpx.get(
                f"{_PAYPAL_BASE}/v2/checkout/orders/{order_id}",
                headers=hdrs,
                timeout=15,
            )
            get_resp.raise_for_status()
            order = get_resp.json()
            units = order.get("purchase_units") or []
            unit = units[0] if units else {}
            expected_custom_id = f"{provider_id}:{claimed['job_id']}"
            if str(unit.get("custom_id") or "") != expected_custom_id:
                raise RuntimeError("PayPal order identity does not match settlement authority")
            order_amount = unit.get("amount") or {}
            if str(order_amount.get("currency_code") or "").upper() != str(
                claimed["currency"]
            ).upper():
                raise RuntimeError("PayPal order currency does not match settlement authority")
            if cad_to_micros(str(order_amount.get("value") or "0")) != int(
                claimed["total_micros"]
            ):
                raise RuntimeError("PayPal order amount does not match settlement authority")

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
            capture_amount = capture.get("amount") or {}
            capture_currency = str(capture_amount.get("currency_code") or "CAD").upper()
            if capture_currency != str(claimed["currency"]).upper():
                raise RuntimeError("PayPal capture currency does not match settlement authority")
            if cad_to_micros(str(capture_amount.get("value") or "0")) != int(
                claimed["total_micros"]
            ):
                raise RuntimeError("PayPal capture amount does not match settlement authority")
            capture_id = str(capture.get("id") or "")
            if not capture_id:
                raise RuntimeError("PayPal capture returned no identifier")
        except Exception as exc:
            with self._conn() as conn:
                mark_settlement_retry(
                    conn,
                    settlement_id=int(claimed["id"]),
                    claim_token=str(claimed["claim_token"]),
                    error=str(exc),
                    retry_status="awaiting_capture",
                )
            raise

        with self._conn() as conn:
            paid = mark_settlement_paid(
                conn,
                settlement_id=int(claimed["id"]),
                claim_token=str(claimed["claim_token"]),
                paypal_capture_id=capture_id,
            )
        return {
            "order_id": order_id,
            "capture_id": capture_id,
            **settlement_response(paid),
        }


_manager: Optional[PayPalConnectManager] = None


def get_paypal_manager() -> PayPalConnectManager:
    global _manager
    if _manager is None:
        _manager = PayPalConnectManager()
    return _manager
