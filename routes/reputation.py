"""Routes: reputation."""

import logging

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
log = logging.getLogger("xcelsior")

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


# ── Milestone journey ──
#
# The journey turns reputation into a guided, gamified progression. Every
# milestone's progress is computed from REAL account + activity state, so a
# reward can only be claimed once it's genuinely earned. Each milestone is a
# one-time activity grant tagged in the event log (idempotent via the engine's
# grant_milestone / claimed_milestones).

# id, title, description, icon, reward points, target count, what it unlocks,
# and (for the two security milestones) the verification badge it also grants.
_MILESTONES: list[dict] = [
    {
        "id": "verify_email",
        "title": "Verify your email",
        "description": "Confirm your email from the link we sent at sign-up.",
        "icon": "✉️",
        "points": 50,
        "target": 1,
        "unlocks": "Verified-member badge",
        "verification": VerificationType.EMAIL.value,
        "cta": "/dashboard/settings",
    },
    {
        "id": "verify_phone",
        "title": "Enable SMS two-factor",
        "description": "Add SMS 2FA in Settings → Security to harden your account.",
        "icon": "📱",
        "points": 50,
        "target": 1,
        "unlocks": "2FA-secured badge",
        "verification": VerificationType.PHONE.value,
        "cta": "/dashboard/settings",
    },
    {
        "id": "complete_profile",
        "title": "Complete your profile",
        "description": "Add your name and region so we can route compute correctly.",
        "icon": "📝",
        "points": 25,
        "target": 1,
        "unlocks": "+25 reputation",
        "cta": "/dashboard/settings",
    },
    {
        "id": "first_launch",
        "title": "Launch your first instance",
        "description": "Spin up any GPU instance to get started.",
        "icon": "🚀",
        "points": 25,
        "target": 1,
        "unlocks": "+25 reputation",
        "cta": "/dashboard/instances",
    },
    {
        "id": "five_sessions",
        "title": "Run five sessions",
        "description": "Complete five GPU sessions to prove you're a real user.",
        "icon": "🔥",
        "points": 50,
        "target": 5,
        "unlocks": "+50 reputation · faster scheduling",
        "cta": "/dashboard/instances",
    },
    {
        "id": "become_provider",
        "title": "Become a GPU provider",
        "description": "Connect Stripe and list a host to earn from your GPUs.",
        "icon": "💎",
        "points": 75,
        "target": 1,
        "unlocks": "Provider earnings + premium tier boost",
        "cta": "/dashboard/earnings",
    },
]


def _journey_signals(user: dict) -> dict[str, int]:
    """Gather real progress counters for the milestone journey."""
    from db import UserStore, MfaStore

    email = str(user.get("email") or "").strip()
    full = (UserStore.get_user(email) if email else None) or user

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

    # Instance counts from the scheduler, scoped to the caller's owner ids.
    launched = 0
    completed = 0
    try:
        from scheduler import list_jobs

        owners = _caller_owner_ids(user)
        for j in list_jobs():
            if str(j.get("owner") or "").strip() not in owners:
                continue
            launched += 1
            if str(j.get("status")) == "completed":
                completed += 1
    except Exception as exc:
        log.debug("journey instance count failed: %s", exc)

    provider_active = False
    try:
        from stripe_connect import get_stripe_manager

        pid = str(user.get("provider_id") or user.get("customer_id") or "").strip()
        if pid:
            prov = get_stripe_manager().get_provider(pid)
            provider_active = bool(prov and prov.get("status") == "active")
    except Exception as exc:
        log.debug("journey provider lookup failed: %s", exc)

    return {
        "verify_email": 1 if full.get("email_verified") else 0,
        "verify_phone": 1 if has_sms else 0,
        "complete_profile": 1 if profile_complete else 0,
        "first_launch": min(launched, 1),
        "five_sessions": min(completed, 5),
        "become_provider": 1 if provider_active else 0,
    }


def _build_journey(user: dict, subject: str) -> dict:
    """Compose the milestone journey with live progress + claim status."""
    re_engine = get_reputation_engine()
    re_engine._ensure_entity(subject, entity_type="host")
    claimed = re_engine.claimed_milestones(subject)
    signals = _journey_signals(user)

    milestones = []
    earned_unclaimed = 0
    for m in _MILESTONES:
        current = int(signals.get(m["id"], 0))
        target = int(m["target"])
        met = current >= target
        is_claimed = m["id"] in claimed
        if is_claimed:
            status = "claimed"
        elif met:
            status = "claimable"
            earned_unclaimed += 1
        else:
            status = "locked"
        milestones.append(
            {
                "id": m["id"],
                "title": m["title"],
                "description": m["description"],
                "icon": m["icon"],
                "points": m["points"],
                "current": min(current, target),
                "target": target,
                "unlocks": m["unlocks"],
                "cta": m.get("cta"),
                "status": status,
            }
        )

    completed_count = sum(1 for x in milestones if x["status"] == "claimed")
    return {
        "milestones": milestones,
        "claimable_count": earned_unclaimed,
        "completed_count": completed_count,
        "total_count": len(milestones),
    }


@router.get("/api/reputation/me/journey", tags=["Reputation"])
def api_reputation_journey(request: Request):
    """The caller's gamified milestone journey with live progress."""
    user = _require_auth(request)
    _require_scope(user, "reputation:read")
    subject = _reputation_subject_id(user)
    if not subject:
        return {"ok": True, "milestones": [], "claimable_count": 0, "completed_count": 0, "total_count": 0}
    return {"ok": True, **_build_journey(user, subject)}


@router.post("/api/reputation/me/claim", tags=["Reputation"])
def api_reputation_claim(request: Request):
    """Grant every milestone the caller has genuinely earned but not yet claimed.

    Progress is validated against real account + activity state, so this is a
    real reward, not a free points button. Idempotent — already-claimed
    milestones are skipped by the engine.
    """
    user = _require_auth(request)
    _require_scope(user, "reputation:write")
    subject = _reputation_subject_id(user)
    if not subject:
        raise HTTPException(400, "No reputation subject for this account")

    re_engine = get_reputation_engine()
    re_engine._ensure_entity(subject, entity_type="host")
    signals = _journey_signals(user)
    claimed = re_engine.claimed_milestones(subject)

    newly_granted: list[dict] = []
    for m in _MILESTONES:
        if m["id"] in claimed:
            continue
        if int(signals.get(m["id"], 0)) < int(m["target"]):
            continue
        granted, _ = re_engine.grant_milestone(subject, m["id"], float(m["points"]))
        # Security milestones also light up the verification badge.
        if m.get("verification"):
            try:
                re_engine.add_verification(subject, VerificationType(m["verification"]))
            except ValueError:
                pass
        if granted:
            newly_granted.append({"id": m["id"], "title": m["title"], "points": m["points"]})

    journey = _build_journey(user, subject)
    score = re_engine.compute_score(subject)
    return {
        "ok": True,
        "newly_granted": newly_granted,
        **journey,
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
