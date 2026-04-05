"""Routes: privacy."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _get_current_user,
    log,
)
from privacy import PrivacyConfig, RETENTION_POLICIES, execute_right_to_erasure, get_consent_manager, get_lifecycle_manager

router = APIRouter()

@router.get("/api/privacy/retention-policies", tags=["Privacy"])
def api_retention_policies():
    """Data retention policies per PIPEDA fair information principles."""
    policies = {}
    for cat, policy in RETENTION_POLICIES.items():
        cat_key = cat.value if hasattr(cat, "value") else str(cat)
        policies[cat_key] = {
            "retention_days": policy["retention_sec"] // 86400,
            "description": policy["description"],
            "redact_on_completion": policy.get("redact_on_completion", False),
        }
    return {"policies": policies}

@router.get("/api/privacy/retention-summary", tags=["Privacy"])
def api_retention_summary():
    """Current retention status across all data categories."""
    lm = get_lifecycle_manager()
    return lm.get_retention_summary()

@router.post("/api/privacy/purge-expired", tags=["Privacy"])
def api_purge_expired():
    """Purge all expired retention records (daily maintenance)."""
    lm = get_lifecycle_manager()
    count = lm.purge_expired()
    return {"ok": True, "purged": count}


# ── Model: PrivacyConfigRequest ──

class PrivacyConfigRequest(BaseModel):
    org_id: str
    privacy_level: str = "strict"
    privacy_officer_name: str = ""
    privacy_officer_email: str = ""
    enable_identification: bool = False
    enable_location_tracking: bool = False
    enable_profiling: bool = False
    redact_pii_in_logs: bool = True
    redact_env_vars: bool = True
    redact_ip_addresses: bool = True
    log_retention_days: int = None
    telemetry_retention_days: int = None

@router.post("/api/privacy/config", tags=["Privacy"])
def api_save_privacy_config(req: PrivacyConfigRequest):
    """Save privacy configuration for an organization."""
    lm = get_lifecycle_manager()
    config = PrivacyConfig(
        privacy_level=req.privacy_level,
        privacy_officer_name=req.privacy_officer_name,
        privacy_officer_email=req.privacy_officer_email,
        privacy_officer_designated=bool(req.privacy_officer_name),
        enable_identification=req.enable_identification,
        enable_location_tracking=req.enable_location_tracking,
        enable_profiling=req.enable_profiling,
        redact_pii_in_logs=req.redact_pii_in_logs,
        redact_env_vars=req.redact_env_vars,
        redact_ip_addresses=req.redact_ip_addresses,
        log_retention_days=req.log_retention_days,
        telemetry_retention_days=req.telemetry_retention_days,
    )
    lm.save_config(req.org_id, config)
    return {"ok": True, "org_id": req.org_id, "privacy_level": req.privacy_level}

@router.get("/api/privacy/config/{org_id}", tags=["Privacy"])
def api_get_privacy_config(org_id: str):
    """Get privacy configuration for an organization (defaults to STRICT)."""
    lm = get_lifecycle_manager()
    config = lm.get_config(org_id)
    return config.to_dict()


# ── Model: ConsentRequest ──

class ConsentRequest(BaseModel):
    entity_id: str
    consent_type: str  # "cross_border", "data_collection", "telemetry", "profiling"
    details: dict = None

@router.post("/api/privacy/consent", tags=["Privacy"])
def api_record_consent(req: ConsentRequest):
    """Record explicit consent (PIPEDA principle: Consent)."""
    lm = get_lifecycle_manager()
    consent_id = lm.record_consent(req.entity_id, req.consent_type, req.details)
    return {"ok": True, "consent_id": consent_id}

@router.delete("/api/privacy/consent/{entity_id}/{consent_type}", tags=["Privacy"])
def api_revoke_consent(entity_id: str, consent_type: str):
    """Revoke consent (PIPEDA: individuals can withdraw consent)."""
    lm = get_lifecycle_manager()
    lm.revoke_consent(entity_id, consent_type)
    return {"ok": True, "revoked": consent_type}

@router.get("/api/privacy/consent/{entity_id}", tags=["Privacy"])
def api_get_consents(entity_id: str):
    """Get all consent records for an entity (PIPEDA: Individual Access)."""
    lm = get_lifecycle_manager()
    consents = lm.get_consents(entity_id)
    return {"consents": consents}


# ── Model: ConsentRequest ──

class ConsentRequest(BaseModel):
    purpose: str
    consent_type: str = "express"

@router.post("/api/v2/privacy/consent", tags=["Privacy"])
def api_privacy_record_consent(body: ConsentRequest, request: Request):
    """Record CASL consent for a purpose."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    cm = get_consent_manager()
    client_ip = request.client.host if request.client else ""
    cm.record_consent(
        user_id=user.get("user_id", user.get("email", "")),
        consent_type=body.consent_type,
        purpose=body.purpose,
        source="api",
        ip_address=client_ip,
    )
    return {"ok": True}

@router.delete("/api/v2/privacy/consent/{purpose}", tags=["Privacy"])
def api_privacy_withdraw_consent(purpose: str, request: Request):
    """Withdraw CASL consent for a purpose (unsubscribe)."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    cm = get_consent_manager()
    cm.withdraw_consent(user.get("user_id", user.get("email", "")), purpose)
    return {"ok": True}

@router.get("/api/v2/privacy/consents", tags=["Privacy"])
def api_privacy_list_consents(request: Request):
    """List all consent records for the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    cm = get_consent_manager()
    consents = cm.get_user_consents(user.get("user_id", user.get("email", "")))
    return {"ok": True, "consents": consents}

@router.post("/api/v2/privacy/erase", tags=["Privacy"])
def api_privacy_right_to_erasure(request: Request):
    """Execute right-to-erasure (PIPEDA/Law 25). Irreversible."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    summary = execute_right_to_erasure(user.get("user_id", user.get("email", "")))
    return {"ok": True, "erasure": summary}

