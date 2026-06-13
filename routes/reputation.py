"""Routes: reputation."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _caller_owner_ids,
    _is_platform_admin,
    _require_admin,
    _require_auth,
    _require_scope,
)
from reputation import (
    VerificationType,
    get_reputation_engine,
    TIER_THRESHOLDS,
    TIER_SEARCH_BOOST,
    TIER_PRICING_PREMIUM,
    TIER_PLATFORM_COMMISSION,
    ReputationTier,
)

router = APIRouter()

# ── Tier unlock requirements and descriptions ──

_TIER_UNLOCK_REQUIREMENTS = {
    ReputationTier.NEW_USER: "Create an account.",
    ReputationTier.BRONZE: "Earn 100 reputation points (complete jobs or add verifications).",
    ReputationTier.SILVER: "Reach 250 points — verify phone or gov ID to accelerate.",
    ReputationTier.GOLD: "Reach 450 points — hardware audit recommended for bonus points.",
    ReputationTier.PLATINUM: "Reach 650 points — consistent uptime & low failure rate required.",
    ReputationTier.DIAMOND: "Reach 850 points — maintained through ongoing activity and reliability.",
}

_TIER_DESCRIPTIONS = {
    ReputationTier.NEW_USER: "Baseline access — build your reputation to unlock perks.",
    ReputationTier.BRONZE: "Established presence — standard marketplace visibility.",
    ReputationTier.SILVER: "Trusted provider — priority payout status and small pricing premium.",
    ReputationTier.GOLD: "Verified provider — verified badge, higher visibility, 20% pricing premium.",
    ReputationTier.PLATINUM: "Elite provider — featured listing placement, 40% pricing premium.",
    ReputationTier.DIAMOND: "Top-tier provider — maximum visibility, 50% premium, reduced platform fee.",
}


def _require_reputation_entity_access(user: dict, entity_id: str) -> None:
    if _is_platform_admin(user):
        return
    eid = (entity_id or "").strip()
    if eid in _caller_owner_ids(user):
        return
    raise HTTPException(403, "Forbidden")


def _reputation_subject_id(user: dict) -> str:
    from routes.instances import _canonical_owner_id

    return (
        str(user.get("provider_id") or "").strip()
        or _canonical_owner_id(user)
        or str(user.get("email") or "").strip()
    )


@router.get("/api/trust-tiers", tags=["Reputation"])
def api_trust_tiers(request: Request):
    """Return all six trust tiers with thresholds, perks, and unlock requirements."""
    user = _require_auth(request)
    _require_scope(user, "reputation:read")
    tiers = []
    for tier in ReputationTier:
        tiers.append(
            {
                "tier": tier.value,
                "threshold": TIER_THRESHOLDS[tier],
                "search_boost": TIER_SEARCH_BOOST[tier],
                "pricing_premium_pct": TIER_PRICING_PREMIUM[tier],
                "platform_commission": TIER_PLATFORM_COMMISSION[tier],
                "description": _TIER_DESCRIPTIONS[tier],
                "unlock_requirements": _TIER_UNLOCK_REQUIREMENTS[tier],
            }
        )
    return {"ok": True, "tiers": tiers}


@router.get("/api/reputation/leaderboard", tags=["Reputation"])
def api_reputation_leaderboard(request: Request, entity_type: str = "host", limit: int = 20):
    """Top hosts/users by reputation score."""
    user = _require_auth(request)
    _require_scope(user, "reputation:read")
    re_engine = get_reputation_engine()
    board = re_engine.get_leaderboard(entity_type, limit)
    return {"ok": True, "entity_type": entity_type, "leaderboard": board}


@router.get("/api/reputation/me", tags=["Reputation"])
def api_reputation_me(request: Request):
    """Get reputation for the currently authenticated user."""
    user = _require_auth(request)
    _require_scope(user, "reputation:read")
    user_id = _reputation_subject_id(user)
    if not user_id:
        return {"ok": True, "score": 0, "tier": "new_user"}
    re_engine = get_reputation_engine()
    score = re_engine.compute_score(user_id)
    return {"ok": True, **score.to_dict()}


def _claimable_verifications(user: dict) -> dict[str, dict]:
    """Map each self-claimable verification to whether the user has earned it.

    These check genuine account state — the user can't claim a badge they
    haven't actually achieved. Admin-only verifications (gov_id, hardware_audit,
    incorporation, data_center) are intentionally excluded.
    """
    from db import UserStore, MfaStore

    email = str(user.get("email") or "").strip()
    full = (UserStore.get_user(email) if email else None) or user

    # Phone is "verified" once an SMS MFA method is enabled.
    has_sms = False
    try:
        has_sms = any(
            m.get("method_type") == "sms" and m.get("enabled")
            for m in MfaStore.list_methods(email)
        )
    except Exception:
        has_sms = False

    profile_complete = bool(
        str(full.get("name") or "").strip()
        and str(full.get("country") or full.get("province") or "").strip()
    )

    return {
        VerificationType.EMAIL.value: {
            "earned": bool(full.get("email_verified")),
            "how": "Verify your email from the link we sent at sign-up.",
        },
        VerificationType.PHONE.value: {
            "earned": has_sms,
            "how": "Add SMS two-factor authentication in Settings → Security.",
        },
    }


@router.get("/api/reputation/me/verifications", tags=["Reputation"])
def api_reputation_my_verifications(request: Request):
    """List self-claimable verifications and whether the caller has earned each."""
    user = _require_auth(request)
    _require_scope(user, "reputation:read")
    return {"ok": True, "claimable": _claimable_verifications(user)}


@router.post("/api/reputation/me/claim", tags=["Reputation"])
def api_reputation_claim(request: Request):
    """Grant any self-claimable verification badges the caller has genuinely earned.

    Validates against real account state (verified email, SMS 2FA) so this is a
    real reward, not a free points button. Idempotent — already-granted badges
    are skipped by add_verification.
    """
    user = _require_auth(request)
    _require_scope(user, "reputation:write")
    subject = _reputation_subject_id(user)
    if not subject:
        raise HTTPException(400, "No reputation subject for this account")

    re_engine = get_reputation_engine()
    re_engine._ensure_entity(subject, entity_type="host")
    claimable = _claimable_verifications(user)
    newly_granted: list[str] = []
    for vtype_value, info in claimable.items():
        if not info["earned"]:
            continue
        try:
            vtype = VerificationType(vtype_value)
        except ValueError:
            continue
        before = re_engine.store.get_score(subject) or {}
        before_v = before.get("verifications", "[]")
        re_engine.add_verification(subject, vtype)
        after = re_engine.store.get_score(subject) or {}
        if str(after.get("verifications")) != str(before_v):
            newly_granted.append(vtype_value)

    score = re_engine.compute_score(subject)
    return {
        "ok": True,
        "newly_granted": newly_granted,
        "claimable": claimable,
        **score.to_dict(),
    }


@router.get("/api/reputation/{entity_id}", tags=["Reputation"])
def api_get_reputation(entity_id: str, request: Request):
    """Get reputation score and tier for a host or user."""
    user = _require_auth(request)
    _require_scope(user, "reputation:read")
    re_engine = get_reputation_engine()
    score = re_engine.compute_score(entity_id)
    return {"ok": True, "reputation": score.to_dict()}


@router.get("/api/reputation/{entity_id}/history", tags=["Reputation"])
def api_reputation_history(entity_id: str, request: Request, limit: int = 50):
    """Get reputation event history."""
    user = _require_auth(request)
    _require_scope(user, "reputation:read")
    _require_reputation_entity_access(user, entity_id)
    re_engine = get_reputation_engine()
    history = re_engine.store.get_event_history(entity_id, limit)
    return {"ok": True, "entity_id": entity_id, "events": history}


# ── Model: VerificationGrant ──


class VerificationGrant(BaseModel):
    entity_id: str
    verification_type: str  # email, phone, gov_id, hardware_audit, incorporation, data_center


@router.post("/api/reputation/verify", tags=["Reputation"])
def api_grant_verification(req: VerificationGrant, request: Request):
    """Grant a verification badge to a host/user (platform admin only)."""
    _require_admin(request)
    try:
        vtype = VerificationType(req.verification_type)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid verification type: {req.verification_type}"
        )
    re_engine = get_reputation_engine()
    score = re_engine.add_verification(req.entity_id, vtype)
    return {"ok": True, "reputation": score.to_dict()}


@router.get("/api/reputation/{entity_id}/breakdown", tags=["Reputation"])
def api_reputation_breakdown(entity_id: str, request: Request):
    """Get a detailed breakdown of how a reputation score is calculated."""
    user = _require_auth(request)
    _require_scope(user, "reputation:read")
    _require_reputation_entity_access(user, entity_id)
    re_engine = get_reputation_engine()
    score_data = re_engine.store.get_score(entity_id) or {}
    history = re_engine.store.get_event_history(entity_id, limit=100)

    jobs_points = 0
    uptime_bonus = 0
    penalties = 0
    decay = 0

    for event in history:
        delta = event.get("score_delta", event.get("delta", 0))
        reason = (event.get("reason", "") or "").lower()
        if "job" in reason or "complete" in reason:
            jobs_points += max(0, delta)
        elif "uptime" in reason or "bonus" in reason:
            uptime_bonus += max(0, delta)
        elif "penalt" in reason or "violat" in reason or "fail" in reason:
            penalties += abs(min(0, delta))
        elif "decay" in reason:
            decay += abs(min(0, delta))
        else:
            if delta >= 0:
                jobs_points += delta
            else:
                penalties += abs(delta)

    total_score = score_data.get("final_score", score_data.get("score", 0))

    return {
        "ok": True,
        "entity_id": entity_id,
        "total_score": total_score,
        "tier": score_data.get("tier", "new_user"),
        "breakdown": {
            "jobs_completed": round(jobs_points, 1),
            "uptime_bonus": round(uptime_bonus, 1),
            "penalties": round(penalties, 1),
            "decay": round(decay, 1),
        },
        "events_analyzed": len(history),
    }
