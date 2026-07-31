"""Routes: privacy."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _get_current_user,
    _require_user_grant,
    log,
)
from privacy import (
    PrivacyConfig,
    RETENTION_POLICIES,
    get_consent_manager,
    get_lifecycle_manager,
)

router = APIRouter()


@router.get("/api/privacy/retention-policies", tags=["Privacy"])
def api_retention_policies(request: Request):
    """Data retention policies per PIPEDA fair information principles."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "privacy:read")
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
def api_retention_summary(request: Request):
    """Current retention status across all data categories."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "privacy:read")
    lm = get_lifecycle_manager()
    return lm.get_retention_summary()


@router.post("/api/privacy/purge-expired", tags=["Privacy"])
def api_purge_expired(request: Request):
    """Purge all expired retention records (daily maintenance)."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "privacy:write")
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
def api_save_privacy_config(req: PrivacyConfigRequest, request: Request):
    """Save privacy configuration for an organization."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "privacy:write")
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
def api_get_privacy_config(org_id: str, request: Request):
    """Get privacy configuration for an organization (defaults to STRICT)."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "privacy:read")
    lm = get_lifecycle_manager()
    config = lm.get_config(org_id)
    return config.to_dict()


# ── Model: ConsentRequest ──


class LifecycleConsentRequest(BaseModel):
    entity_id: str
    consent_type: str  # "cross_border", "data_collection", "telemetry", "profiling"
    details: dict = None


@router.post("/api/privacy/consent", tags=["Privacy"])
def api_record_consent(req: LifecycleConsentRequest, request: Request):
    """Record explicit consent (PIPEDA principle: Consent)."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "privacy:write")
    lm = get_lifecycle_manager()
    consent_id = lm.record_consent(req.entity_id, req.consent_type, req.details)
    return {"ok": True, "consent_id": consent_id}


@router.delete("/api/privacy/consent/{entity_id}/{consent_type}", tags=["Privacy"])
def api_revoke_consent(entity_id: str, consent_type: str, request: Request):
    """Revoke consent (PIPEDA: individuals can withdraw consent)."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "privacy:write")
    lm = get_lifecycle_manager()
    lm.revoke_consent(entity_id, consent_type)
    return {"ok": True, "revoked": consent_type}


@router.get("/api/privacy/consent/{entity_id}", tags=["Privacy"])
def api_get_consents(entity_id: str, request: Request):
    """Get all consent records for an entity (PIPEDA: Individual Access)."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "privacy:read")
    lm = get_lifecycle_manager()
    consents = lm.get_consents(entity_id)
    return {"consents": consents}


# ── Model: ConsentRequest ──


class CaslConsentRequest(BaseModel):
    purpose: str
    consent_type: str = "express"


@router.post("/api/v2/privacy/consent", tags=["Privacy"])
def api_privacy_record_consent(body: CaslConsentRequest, request: Request):
    """Record CASL consent for a purpose."""
    user = _require_user_grant(request)
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
    user = _require_user_grant(request)
    cm = get_consent_manager()
    cm.withdraw_consent(user.get("user_id", user.get("email", "")), purpose)
    return {"ok": True}


@router.get("/api/v2/privacy/consents", tags=["Privacy"])
def api_privacy_list_consents(request: Request):
    """List all consent records for the current user."""
    user = _require_user_grant(request)
    cm = get_consent_manager()
    consents = cm.get_user_consents(user.get("user_id", user.get("email", "")))
    return {"ok": True, "consents": consents}


@router.post("/api/v2/privacy/erase", tags=["Privacy"])
def api_privacy_right_to_erasure(request: Request):
    """Create an inspectable, asynchronous right-to-erasure request."""
    from fastapi.responses import JSONResponse
    from privacy_deletion import (
        PrivacyDeletionError,
        create_deletion_request,
    )

    user = _require_user_grant(request)
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    try:
        receipt = create_deletion_request(
            user_id=str(user.get("user_id") or ""),
            email=str(user.get("email") or ""),
            customer_ids=[str(user.get("customer_id") or "")],
            idempotency_key=idempotency_key,
            requested_by=str(user.get("user_id") or user.get("email") or ""),
        )
    except PrivacyDeletionError as exc:
        raise HTTPException(428, str(exc)) from exc
    return JSONResponse(
        status_code=202,
        content={
            "ok": True,
            "request_id": receipt.request_id,
            "state": receipt.state,
            "deadline_at": receipt.deadline_at.isoformat(),
            "status_token": receipt.status_token,
            "already_existed": receipt.already_existed,
            "message": (
                "Deletion is in progress. Keep the status token until every "
                "data store has reported its result."
            ),
        },
    )


@router.get("/api/v2/privacy/erase/{request_id}", tags=["Privacy"])
def api_privacy_erasure_status(request_id: str, request: Request):
    """Return honest per-store status for one erasure request."""
    from routes._deps import _get_current_user, _is_platform_admin
    from privacy_deletion import (
        PrivacyDeletionAccessDenied,
        PrivacyDeletionNotFound,
        get_deletion_status,
    )

    user = _get_current_user(request)
    status_token = request.headers.get("X-Deletion-Status-Token", "").strip()
    if not user and not status_token:
        raise HTTPException(401, "Authentication or a deletion status token is required")
    try:
        status = get_deletion_status(
            request_id,
            caller_user_id=str((user or {}).get("user_id") or "") or None,
            status_token=status_token or None,
            is_admin=bool(user and _is_platform_admin(user)),
        )
    except (PrivacyDeletionNotFound, PrivacyDeletionAccessDenied) as exc:
        raise HTTPException(404, "Deletion request not found") from exc
    return {"ok": True, "deletion": status}
