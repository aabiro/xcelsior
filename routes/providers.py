"""Routes: providers."""

import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    broadcast_sse,
    log,
    otel_span,
)
from db import UserStore
from stripe_connect import get_stripe_manager
from reputation import VerificationType, get_reputation_engine

router = APIRouter()


# ── Model: ProviderRegisterRequest ──

class ProviderRegisterRequest(BaseModel):
    provider_id: str
    email: str
    provider_type: str = "individual"  # "individual" or "company"
    corporation_name: str = ""  # Required for company type
    business_number: str = ""  # CRA Business Number (BN)
    gst_hst_number: str = ""  # GST/HST registration number
    province: str = ""  # ON, QC, BC, AB, etc.
    legal_name: str = ""  # Legal name of individual or entity


# ── Model: IncorporationUploadRequest ──

class IncorporationUploadRequest(BaseModel):
    file_id: str  # Reference to file uploaded via /api/artifacts/upload

@router.post("/api/providers/register", tags=["Providers"])
def api_register_provider(req: ProviderRegisterRequest):
    """Register a GPU provider with Stripe Connect onboarding.

    For Canadian companies, include corporation_name, business_number,
    and gst_hst_number. Returns a Stripe onboarding URL for KYC completion.

    Per Report #1.B "Five Pillars of Compliance":
    1. Identity Verification (Stripe Identity)
    2. Financial Enrollment (bank details via Stripe Express)
    3. Credentialing (GPU/bandwidth checked at admission)
    4. Tax Compliance (GST/HST auto-collected per province)
    """
    if req.provider_type == "company" and not req.corporation_name:
        raise HTTPException(400, "corporation_name required for company providers")

    mgr = get_stripe_manager()
    try:
        result = mgr.create_provider_account(
            provider_id=req.provider_id,
            email=req.email,
            provider_type=req.provider_type,
            corporation_name=req.corporation_name,
            business_number=req.business_number,
            gst_hst_number=req.gst_hst_number,
            province=req.province,
            legal_name=req.legal_name,
        )
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    except Exception as e:
        log.error("Provider registration failed: %s", e)
        raise HTTPException(502, f"Provider registration failed: {e}") from e
    # Link provider_id to user account
    from db import UserStore
    UserStore.update_user(req.email, {"provider_id": req.provider_id})

    broadcast_sse(
        "provider_registered",
        {
            "provider_id": req.provider_id,
            "type": req.provider_type,
            "corporation_name": req.corporation_name,
        },
    )
    return {"ok": True, **result}

@router.get("/api/providers/{provider_id}", tags=["Providers"])
def api_get_provider(provider_id: str):
    """Get provider account details including company info and payout status."""
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    # Redact sensitive fields
    provider.pop("stripe_account_id", None)
    return {"ok": True, "provider": provider}

@router.get("/api/providers", tags=["Providers"])
def api_list_providers(status: str = ""):
    """List all provider accounts, optionally filtered by status."""
    mgr = get_stripe_manager()
    providers = mgr.list_providers(status)
    # Redact Stripe IDs
    for p in providers:
        p.pop("stripe_account_id", None)
    return {"ok": True, "providers": providers, "count": len(providers)}

@router.post("/api/providers/{provider_id}/incorporation", tags=["Providers"])
def api_upload_incorporation(provider_id: str, req: IncorporationUploadRequest):
    """Link an uploaded incorporation document to a provider account.

    The file itself should first be uploaded via POST /api/artifacts/upload
    with artifact_type='incorporation_doc'. Then pass the resulting file_id here.
    """
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    result = mgr.upload_incorporation_file(provider_id, req.file_id)

    # Also add 'incorporation' verification to reputation
    try:
        re = get_reputation_engine()
        re.add_verification(provider_id, VerificationType.INCORPORATION)
    except Exception as e:
        log.debug("reputation incorporation update failed: %s", e)

    return {"ok": True, **result}

@router.get("/api/providers/{provider_id}/earnings", tags=["Providers"])
def api_provider_earnings(provider_id: str):
    """Get aggregate earnings and payout history for a provider."""
    mgr = get_stripe_manager()
    earnings = mgr.get_provider_earnings(provider_id)
    payouts = mgr.get_provider_payouts(provider_id, limit=20)
    return {"ok": True, "earnings": earnings, "recent_payouts": payouts}

@router.post("/api/providers/{provider_id}/payout", tags=["Providers"])
def api_provider_payout(provider_id: str, job_id: str = "", total_cad: float = 0):
    """Split a job payment between provider (85%) and platform (15%).

    Applies province-specific GST/HST. If Stripe is configured,
    creates a real Transfer to the provider's connected account.
    """
    if not job_id or total_cad <= 0:
        raise HTTPException(400, "job_id and total_cad (>0) required")
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    result = mgr.split_payout(job_id, provider_id, total_cad, provider.get("province", "ON"))
    return {"ok": True, **result}

@router.post("/api/providers/webhook", tags=["Providers"])
async def api_stripe_webhook(request: Request):
    """Handle Stripe Connect webhooks (account.updated, payment_intent.succeeded, etc.)."""
    with otel_span("webhook.stripe"):
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")
        mgr = get_stripe_manager()
        result = mgr.handle_webhook(payload, sig_header)
        return {"ok": True, **result}

