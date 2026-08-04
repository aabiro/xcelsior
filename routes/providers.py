"""Routes: providers."""

import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _get_current_user,
    _is_platform_admin,
    _require_auth,
    _require_scope,
    broadcast_sse,
    log,
    otel_span,
)
from db import UserStore
from stripe_connect import get_stripe_manager
from paypal_connect import get_paypal_manager, paypal_enabled
from reputation import VerificationType, get_reputation_engine

router = APIRouter()


def _require_provider_access(request: Request, provider_id: str) -> dict:
    """Authn + ownership guard for provider-scoped routes."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    if _is_platform_admin(user):
        return user
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    owned = {str(user.get(k) or "").strip() for k in ("provider_id", "email", "customer_id")}
    owned.discard("")
    if provider_id not in owned and str(provider.get("email") or "").strip() not in owned:
        raise HTTPException(403, "Forbidden")
    return user


# ── Model: ProviderRegisterRequest ──


class ProviderRegisterRequest(BaseModel):
    provider_id: str
    email: str
    provider_type: str = "individual"  # "individual" or "company"
    corporation_name: str = ""  # Required for company type
    business_number: str = ""  # CRA Business Number (BN)
    gst_hst_number: str = ""  # GST/HST registration number
    province: str = ""  # ON, QC, BC, AB, etc. (or region code)
    legal_name: str = ""  # Legal name of individual or entity
    country: str = "CA"  # ISO-3166 alpha-2; global cross-border providers supported


# ── Model: IncorporationUploadRequest ──


class IncorporationUploadRequest(BaseModel):
    file_id: str  # Reference to file uploaded via /api/artifacts/upload


@router.post("/api/providers/register", tags=["Providers"])
def api_register_provider(req: ProviderRegisterRequest, request: Request):
    """Register a GPU provider with Stripe Connect onboarding.

    For Canadian companies, include corporation_name, business_number,
    and gst_hst_number. Returns a Stripe onboarding URL for KYC completion.

    Per Report #1.B "Five Pillars of Compliance":
    1. Identity Verification (Stripe Identity)
    2. Financial Enrollment (bank details via Stripe Express)
    3. Credentialing (GPU/bandwidth checked at admission)
    4. Tax Compliance (GST/HST auto-collected per province)
    """
    user = _require_auth(request)
    _require_scope(user, "providers:write")
    caller_email = str(user.get("email") or "").strip().lower()
    if not caller_email:
        raise HTTPException(401, "Not authenticated")
    if req.email.strip().lower() != caller_email and not _is_platform_admin(user):
        raise HTTPException(403, "You can only register a provider for your own email")
    register_email = caller_email if not _is_platform_admin(user) else req.email.strip()
    if req.provider_type == "company" and not req.corporation_name:
        raise HTTPException(400, "corporation_name required for company providers")

    mgr = get_stripe_manager()
    try:
        result = mgr.create_provider_account(
            provider_id=req.provider_id,
            email=register_email,
            provider_type=req.provider_type,
            corporation_name=req.corporation_name,
            business_number=req.business_number,
            gst_hst_number=req.gst_hst_number,
            province=req.province,
            legal_name=req.legal_name,
            country=(req.country or "CA").strip().upper()[:2] or "CA",
        )
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    except Exception as e:
        log.error("Provider registration failed: %s", e)
        raise HTTPException(502, f"Provider registration failed: {e}") from e
    # Link provider_id to user account and promote role
    from db import UserStore

    UserStore.update_user(register_email, {"provider_id": req.provider_id, "role": "provider"})

    # Create initial reputation record so the provider starts with a score
    try:
        rep_engine = get_reputation_engine()
        rep_engine._ensure_entity(req.provider_id, entity_type="host")
        rep_engine.add_verification(req.provider_id, VerificationType.EMAIL)
        log.info("Initial reputation record created for provider %s", req.provider_id)
    except Exception as e:
        log.warning("Failed to create initial reputation for %s: %s", req.provider_id, e)

    broadcast_sse(
        "provider_registered",
        {
            "provider_id": req.provider_id,
            "type": req.provider_type,
            "corporation_name": req.corporation_name,
        },
    )
    return {"ok": True, **result}


@router.post("/api/providers/{provider_id}/abandon-onboarding", tags=["Providers"])
def api_abandon_onboarding(provider_id: str, request: Request):
    """Mark a provider's onboarding as abandoned.

    Called when the user returns via the Stripe refresh URL (link expired)
    or when the frontend poll times out after a return URL visit.
    Idempotent — safe to call multiple times.
    """
    from routes._deps import _require_scope

    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    mgr = get_stripe_manager()
    result = mgr.mark_abandoned(provider_id)
    return {"ok": True, **result}


@router.post("/api/providers/{provider_id}/resume-onboarding", tags=["Providers"])
def api_resume_onboarding(provider_id: str, request: Request):
    """Generate a fresh Stripe onboarding URL for a provider stuck in onboarding.

    This lets users who closed the Stripe modal mid-flow resume from where
    they left off without re-registering.
    """
    from routes._deps import _require_scope

    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    if provider.get("status") == "active":
        return {"ok": True, "status": "active", "message": "Provider is already fully onboarded"}
    # Re-call create_provider_account which handles re-generating the onboarding link
    try:
        result = mgr.create_provider_account(
            provider_id=provider_id,
            email=provider.get("email", user.get("email", "")),
            provider_type=provider.get("provider_type", "individual"),
            corporation_name=provider.get("corporation_name", ""),
            business_number=provider.get("business_number", ""),
            gst_hst_number=provider.get("gst_hst_number", ""),
            province=provider.get("province", ""),
            legal_name=provider.get("legal_name", ""),
        )
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    return {"ok": True, **result}


@router.get("/api/providers/{provider_id}", tags=["Providers"])
def api_get_provider(provider_id: str, request: Request):
    """Get provider account details including company info and payout status."""
    from routes._deps import _require_scope

    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:read")
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    # Redact sensitive fields
    provider.pop("stripe_account_id", None)
    provider.pop("paypal_merchant_id", None)
    provider.pop("paypal_payer_id", None)
    provider.pop("paypal_tracking_id", None)
    provider["paypal"] = {
        "enabled": paypal_enabled(),
        "status": provider.pop("paypal_status", "") or "not_started",
        "onboarded_at": provider.pop("paypal_onboarded_at", 0) or 0,
    }
    return {"ok": True, "provider": provider}


@router.get("/api/providers", tags=["Providers"])
def api_list_providers(request: Request, status: str = ""):
    """List provider accounts visible to the caller (own account, or all for admins)."""
    user = _require_auth(request)
    _require_scope(user, "providers:read")
    mgr = get_stripe_manager()
    if _is_platform_admin(user):
        providers = mgr.list_providers(status)
    else:
        pid = str(user.get("provider_id") or "").strip()
        if not pid:
            providers = []
        else:
            one = mgr.get_provider(pid)
            if not one:
                providers = []
            elif status and one.get("status") != status:
                providers = []
            else:
                providers = [one]
    for p in providers:
        p.pop("stripe_account_id", None)
        p.pop("paypal_merchant_id", None)
        p.pop("paypal_payer_id", None)
        p.pop("paypal_tracking_id", None)
        p["paypal"] = {
            "enabled": paypal_enabled(),
            "status": p.pop("paypal_status", "") or "not_started",
            "onboarded_at": p.pop("paypal_onboarded_at", 0) or 0,
        }
    return {"ok": True, "providers": providers, "count": len(providers)}


@router.post("/api/providers/{provider_id}/incorporation", tags=["Providers"])
def api_upload_incorporation(provider_id: str, req: IncorporationUploadRequest, request: Request):
    """Link an uploaded incorporation document to a provider account.

    The file itself should first be uploaded via POST /api/artifacts/upload
    with artifact_type='incorporation_doc'. Then pass the resulting file_id here.
    """
    from routes._deps import _require_scope

    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    mgr = get_stripe_manager()
    result = mgr.upload_incorporation_file(provider_id, req.file_id)

    # Also add 'incorporation' verification to reputation
    try:
        re = get_reputation_engine()
        re.add_verification(provider_id, VerificationType.INCORPORATION)
    except Exception as e:
        log.debug("reputation incorporation update failed: %s", e)

    return {"ok": True, **result}


@router.get("/api/providers/{provider_id}/earnings", tags=["Providers"])
def api_provider_earnings(provider_id: str, request: Request):
    """Get aggregate earnings and payout history for a provider."""
    from routes._deps import _require_scope

    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:read")
    mgr = get_stripe_manager()
    earnings = mgr.get_provider_earnings(provider_id)
    payouts = mgr.get_provider_payouts(provider_id, limit=20)
    return {
        "ok": True,
        "earnings": earnings,
        "recent_payouts": payouts,
        "intro_fee": _provider_intro_fee_status(provider_id, user),
    }


def _provider_intro_fee_status(provider_id: str, user: dict) -> dict:
    """New-provider 0% platform-fee window status, from the provider's listings."""
    import math
    import time

    from scheduler import PROVIDER_INTRO_FEE_DAYS, load_marketplace

    status = {"window_days": PROVIDER_INTRO_FEE_DAYS, "active": False, "days_remaining": 0}
    if PROVIDER_INTRO_FEE_DAYS <= 0:
        return status
    ids = {provider_id, str(user.get("user_id") or ""), str(user.get("email") or "")}
    ids.discard("")
    try:
        mine = [
            l for l in load_marketplace()
            if str(l.get("owner") or "") in ids or str(l.get("host_id") or "") in ids
        ]
    except Exception:
        return status
    if not mine:
        return status
    now = time.time()
    earliest = min(float(l.get("listed_at") or now) for l in mine)
    remaining_days = PROVIDER_INTRO_FEE_DAYS - (now - earliest) / 86400
    if remaining_days > 0:
        status["active"] = True
        status["days_remaining"] = int(math.ceil(remaining_days))
    return status


@router.post("/api/providers/{provider_id}/account-session", tags=["Providers"])
def api_provider_account_session(provider_id: str, request: Request):
    """Create a Stripe Connect AccountSession for embedded onboarding UI.

    Returns client_secret for @stripe/connect-js components (account_onboarding,
    notification_banner, account_management, payouts). Free API — no charges.
    """
    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    try:
        result = get_stripe_manager().create_account_session(provider_id)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, **result}


@router.get("/api/providers/{provider_id}/paypal", tags=["Providers"])
def api_provider_paypal_status(provider_id: str, request: Request):
    """PayPal Complete Payments onboarding status for a provider."""
    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:read")
    mgr = get_paypal_manager()
    profile = mgr.get_paypal_profile(provider_id)
    if not profile:
        raise HTTPException(404, f"Provider {provider_id} not found")
    return {"ok": True, "paypal_enabled": paypal_enabled(), "paypal": profile}


@router.post("/api/providers/{provider_id}/paypal/onboard", tags=["Providers"])
def api_provider_paypal_onboard(provider_id: str, request: Request):
    """Start or resume PayPal seller onboarding (partner referral link)."""
    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    stripe_mgr = get_stripe_manager()
    provider = stripe_mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    email = str(provider.get("email") or user.get("email") or "").strip()
    if not email:
        raise HTTPException(400, "Provider email required for PayPal onboarding")
    try:
        result = get_paypal_manager().create_onboarding_link(provider_id, email)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"ok": True, **result}


@router.post("/api/providers/{provider_id}/stripe/disconnect", tags=["Providers"])
def api_provider_stripe_disconnect(provider_id: str, request: Request):
    """Unlink a provider's Stripe account so they can re-run the connect flow."""
    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    mgr = get_stripe_manager()
    result = mgr.disconnect_provider(provider_id)
    if result.get("status") == "not_found":
        raise HTTPException(404, f"Provider {provider_id} not found")
    broadcast_sse("provider_stripe_disconnected", {"provider_id": provider_id})
    return {"ok": True, **result}


@router.post("/api/providers/{provider_id}/paypal/disconnect", tags=["Providers"])
def api_provider_paypal_disconnect(provider_id: str, request: Request):
    """Unlink a provider's PayPal seller account."""
    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    result = get_paypal_manager().disconnect(provider_id)
    if result.get("status") == "not_found":
        raise HTTPException(404, f"Provider {provider_id} not found")
    broadcast_sse("provider_paypal_disconnected", {"provider_id": provider_id})
    return {"ok": True, "paypal": result}


@router.post("/api/providers/{provider_id}/paypal/refresh", tags=["Providers"])
def api_provider_paypal_refresh(provider_id: str, request: Request):
    """Poll PayPal for seller merchant_id after onboarding completes."""
    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    try:
        result = get_paypal_manager().refresh_merchant_status(provider_id)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"ok": True, "paypal": result}


@router.post("/api/providers/{provider_id}/payout", tags=["Providers"])
def api_provider_payout(
    provider_id: str,
    request: Request,
    job_id: str = "",
    payment_rail: str = "stripe",
):
    """Settle an eligible job from PostgreSQL billing authority.

    The caller identifies a job and rail only. Amount, customer, provider
    ownership, currency, terminal state, and prior settlement are derived from
    PostgreSQL under a lock.
    """
    from routes._deps import _require_scope
    from provider_settlement import SettlementError

    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    if not job_id:
        raise HTTPException(400, "job_id is required")
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    rail = (payment_rail or "stripe").strip().lower()
    if rail == "paypal":
        try:
            result = get_paypal_manager().create_marketplace_order(provider_id, job_id)
        except SettlementError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "ok": True,
            "payment_rail": "paypal",
            "order_id": result["order_id"],
            "platform_share_cad": result["platform_share_cad"],
            "provider_share_cad": result["provider_share_cad"],
            "message": "Approve and capture via /api/billing/paypal/marketplace/capture-order",
        }
    if rail != "stripe":
        raise HTTPException(400, "payment_rail must be stripe or paypal")
    try:
        result = mgr.split_payout(job_id, provider_id)
    except SettlementError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    return {"ok": True, "payment_rail": "stripe", **result}


@router.post("/api/providers/webhook", tags=["Providers"])
async def api_stripe_webhook(request: Request):
    """Handle Stripe Connect webhooks (account.updated, payment_intent.succeeded, etc.)."""
    with otel_span("webhook.stripe"):
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")
        mgr = get_stripe_manager()
        result = mgr.handle_webhook(payload, sig_header)

        # This used to return `{"ok": True, **result}` unconditionally, so an
        # event whose signature could not be verified was answered 200.
        #
        # Stripe reads any 2xx as delivered. It stops retrying, and — the part
        # that makes this unrecoverable rather than merely delayed — the event
        # never appears in `GET /v1/events?delivery_success=false`, which is the
        # documented mechanism for finding undelivered events. A rotated or
        # misrouted signing secret therefore does not delay those events, it
        # loses them, and the loss is invisible to the only recovery path there
        # is. Events are retained for 30 days; after that there is nothing to
        # find at all.
        #
        # Mapped from `outcome` rather than from error prose: matching on a
        # message string is how a security decision comes to depend on someone
        # typing a literal correctly.
        outcome = str(result.get("outcome") or "")
        if outcome == "signature_invalid":
            # Stripe's own reference implementation returns 400 here. Retried
            # with exponential backoff for ~3 days, and visible in the
            # undelivered-event sweep throughout.
            raise HTTPException(400, "Webhook signature verification failed")
        if outcome == "no_secret_configured":
            # Not the sender's fault, and a 400 would tell Stripe the request was
            # malformed when the deployment is. 503 keeps it retryable, so events
            # that arrive during the misconfiguration land once it is corrected.
            # `control_plane.startup_validation` refuses the boot for this, so
            # reaching here means the gate was skipped or the secret was removed
            # at runtime.
            raise HTTPException(503, "Webhook signing secret is not configured")
        if outcome == "stripe_disabled":
            # A Stripe destination is pointed at a deployment with Stripe off.
            # Retryable for the same reason.
            raise HTTPException(503, "Stripe is not enabled on this deployment")

        return {"ok": True, **result}
