"""Routes: reputation."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _AUTH_COOKIE_NAME,
    _USE_PERSISTENT_AUTH,
)
from db import UserStore
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
    ReputationTier.DIAMOND: "Top-tier provider — maximum visibility, 50% premium, reduced commission.",
}


@router.get("/api/trust-tiers", tags=["Reputation"])
def api_trust_tiers():
    """Return all six trust tiers with thresholds, perks, and unlock requirements."""
    tiers = []
    for tier in ReputationTier:
        tiers.append({
            "tier": tier.value,
            "threshold": TIER_THRESHOLDS[tier],
            "search_boost": TIER_SEARCH_BOOST[tier],
            "pricing_premium_pct": TIER_PRICING_PREMIUM[tier],
            "platform_commission": TIER_PLATFORM_COMMISSION[tier],
            "description": _TIER_DESCRIPTIONS[tier],
            "unlock_requirements": _TIER_UNLOCK_REQUIREMENTS[tier],
        })
    return {"ok": True, "tiers": tiers}

@router.get("/api/reputation/leaderboard", tags=["Reputation"])
def api_reputation_leaderboard(entity_type: str = "host", limit: int = 20):
    """Top hosts/users by reputation score."""
    re_engine = get_reputation_engine()
    board = re_engine.get_leaderboard(entity_type, limit)
    return {"ok": True, "entity_type": entity_type, "leaderboard": board}

@router.get("/api/reputation/me", tags=["Reputation"])
def api_reputation_me(request: Request):
    """Get reputation for the currently authenticated user."""
    user = getattr(request.state, "user", None)
    user_id = ""
    if user:
        user_id = getattr(user, "user_id", "") or getattr(user, "customer_id", "")
    if not user_id:
        # Try from middleware-set attributes
        user_id = getattr(request.state, "user_id", "") or getattr(request.state, "customer_id", "")
    if not user_id:
        # Try extracting from session cookie
        token = request.cookies.get(_AUTH_COOKIE_NAME, "")
        if token and _USE_PERSISTENT_AUTH:
            session = UserStore.get_session(token)
            if session:
                user_id = session.get("user_id", "")
    if not user_id:
        # Fall back to provider_id if present
        user_id = getattr(request.state, "provider_id", "")
    if not user_id:
        return {"ok": True, "score": 0, "tier": "new_user"}
    re_engine = get_reputation_engine()
    score = re_engine.compute_score(user_id)
    return {"ok": True, **score.to_dict()}

@router.get("/api/reputation/{entity_id}", tags=["Reputation"])
def api_get_reputation(entity_id: str):
    """Get reputation score and tier for a host or user."""
    re_engine = get_reputation_engine()
    score = re_engine.compute_score(entity_id)
    return {"ok": True, "reputation": score.to_dict()}

@router.get("/api/reputation/{entity_id}/history", tags=["Reputation"])
def api_reputation_history(entity_id: str, limit: int = 50):
    """Get reputation event history."""
    re_engine = get_reputation_engine()
    history = re_engine.store.get_event_history(entity_id, limit)
    return {"ok": True, "entity_id": entity_id, "events": history}


# ── Model: VerificationGrant ──

class VerificationGrant(BaseModel):
    entity_id: str
    verification_type: str  # email, phone, gov_id, hardware_audit, incorporation, data_center

@router.post("/api/reputation/verify", tags=["Reputation"])
def api_grant_verification(req: VerificationGrant):
    """Grant a verification badge to a host/user."""
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
def api_reputation_breakdown(entity_id: str):
    """Get a detailed breakdown of how a reputation score is calculated."""
    re_engine = get_reputation_engine()
    score_data = re_engine.store.get_score(entity_id) or {}
    history = re_engine.store.get_event_history(entity_id, limit=100)

    # Calculate component breakdown from history
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

