# Xcelsior Provider Expectations — single source of truth for what we tell
# providers about scoring, incentives, and reliability.
#
# WHY THIS FILE EXISTS
# --------------------
# We say a lot of things to providers ("stay online X hours and you keep more of
# every dollar", "dropping a job costs you a tier"). Those promises live in three
# places at once: the signup email, the public docs (fern/pages/earn.mdx +
# trust.mdx), and the onboarding wizard. If any of them drifts from what
# reputation.py *actually does*, we are lying to the people whose GPUs we depend
# on — the fastest way to lose providers.
#
# So every human-facing number here is DERIVED from the real reputation engine
# constants. tests/test_provider_expectations.py fails if the code and the
# messaging disagree. Change the incentives in reputation.py and this file (and
# every surface that reads it) updates with it.
#
# DESIGN PRINCIPLE — the incentives cannot bankrupt the platform
# --------------------------------------------------------------
# Every reward a provider unlocks is self-funding:
#   * Lower platform commission (15%→8%) is earned by driving reliable volume;
#     8% of a lot beats 15% of a little. It comes out of margin we would not have
#     captured without that reliability.
#   * Higher search boost = more matched, billable hours — allocation, not cash.
#   * Pricing premium is paid by *buyers*, not by us.
# There is no flat cash bonus to subsidize. "Earn more" always means "you became
# more valuable to the network," never "we paid you to show up."

from __future__ import annotations

from reputation import (
    DECAY_GRACE_DAYS,
    MAX_VERIFICATION_POINTS,
    PENALTY_POINTS,
    POINTS_PER_COMPLETED_JOB,
    POINTS_PER_HOSTING_DAY,
    RELIABILITY_WEIGHTS,
    TIER_PLATFORM_COMMISSION,
    TIER_PRICING_PREMIUM,
    TIER_SEARCH_BOOST,
    TIER_THRESHOLDS,
    PenaltyType,
    ReputationTier,
)

# ── Concrete weekly uptime goals ──────────────────────────────────────
# These are the behavioural targets we ask providers to aim for. They are the
# *messaging* layer — reputation.py rewards continuous availability
# (POINTS_PER_HOSTING_DAY) and measured uptime (RELIABILITY_WEIGHTS["uptime_pct"])
# on a continuous basis; these buckets just give a human a number to hit.
#
# A 24/7 rig is 168 h/week. We deliberately do NOT ask for 24/7 — predictable
# part-time availability is worth more to the network than erratic full-time,
# and asking for the impossible just makes people churn.

WEEKLY_UPTIME_GOALS = [
    {
        "key": "casual",
        "label": "Casual",
        "hours_per_week": 20,
        "plain": "~3 hrs a day, or a weekend rig",
        "unlocks": (
            "Stay listed and outrun activity decay. You keep earning hosting "
            "points instead of bleeding them."
        ),
    },
    {
        "key": "reliable",
        "label": "Reliable",
        "hours_per_week": 40,
        "plain": "evenings + weekends, or an always-on desktop overnight",
        "unlocks": (
            "A strong reliability multiplier and a steady climb toward Gold — "
            "where our commission drops and you keep more of every dollar."
        ),
    },
    {
        "key": "pro",
        "label": "Pro",
        "hours_per_week": 100,
        "plain": "a mostly-on dedicated rig",
        "unlocks": (
            "The fastest path to Platinum/Diamond and the 90–92% payout band, "
            "top search placement, and the highest price premium buyers will pay."
        ),
    },
]

# The single headline number we put in front of a new provider. "Reliable" is the
# recommended default: meaningful to the network, achievable on a home rig.
RECOMMENDED_WEEKLY_HOURS = 40
MINIMUM_WEEKLY_HOURS = WEEKLY_UPTIME_GOALS[0]["hours_per_week"]

# Ordered worst→best for display.
_TIER_ORDER = [
    ReputationTier.NEW_USER,
    ReputationTier.BRONZE,
    ReputationTier.SILVER,
    ReputationTier.GOLD,
    ReputationTier.PLATINUM,
    ReputationTier.DIAMOND,
]

_TIER_LABELS = {
    ReputationTier.NEW_USER: "New",
    ReputationTier.BRONZE: "Bronze",
    ReputationTier.SILVER: "Silver",
    ReputationTier.GOLD: "Gold",
    ReputationTier.PLATINUM: "Platinum",
    ReputationTier.DIAMOND: "Diamond",
}


def tier_reward_table() -> list[dict]:
    """The concrete 'what you get for climbing' table, derived from the real
    reputation constants. This is what powers the docs table and the email.

    ``keep_pct`` is the share of each dollar the provider keeps (100 - commission)
    — the number a human actually cares about.
    """
    rows = []
    for tier in _TIER_ORDER:
        commission = TIER_PLATFORM_COMMISSION[tier]
        rows.append(
            {
                "tier": tier.value,
                "label": _TIER_LABELS[tier],
                "min_score": TIER_THRESHOLDS[tier],
                "commission_pct": round(commission * 100, 1),
                "keep_pct": round((1.0 - commission) * 100, 1),
                "pricing_premium_pct": round(TIER_PRICING_PREMIUM[tier] * 100, 1),
                "search_boost": TIER_SEARCH_BOOST[tier],
            }
        )
    return rows


def reliability_breakdown() -> list[dict]:
    """The reliability multiplier components, as percentages, from the engine."""
    labels = {
        "uptime_pct": "Measured uptime (are you online when you say you are)",
        "job_success_rate": "Job success rate (jobs finished vs. host-side failures)",
        "network_stability": "Network stability (low jitter and packet loss)",
    }
    return [
        {"key": k, "weight_pct": round(v * 100, 1), "label": labels.get(k, k)}
        for k, v in RELIABILITY_WEIGHTS.items()
    ]


def key_penalties() -> list[dict]:
    """The penalties a provider is most likely to trigger, worst first."""
    order = [
        (PenaltyType.SLA_BREACH, "Going offline mid-job / breaking an uptime commitment"),
        (PenaltyType.JOB_FAILURE_HOST, "A job fails because of your machine"),
        (PenaltyType.SECURITY_INCIDENT, "A security incident on your host"),
    ]
    return [
        {"type": pt.value, "points": PENALTY_POINTS[pt], "plain": plain}
        for pt, plain in order
    ]


def provider_expectations_summary() -> dict:
    """Structured, machine-readable expectations — safe to expose to the
    frontend/wizard so every surface renders the same, always-current numbers."""
    return {
        "recommended_weekly_hours": RECOMMENDED_WEEKLY_HOURS,
        "minimum_weekly_hours": MINIMUM_WEEKLY_HOURS,
        "weekly_goals": WEEKLY_UPTIME_GOALS,
        "tier_rewards": tier_reward_table(),
        "reliability_components": reliability_breakdown(),
        "key_penalties": key_penalties(),
        "hosting_day_points": POINTS_PER_HOSTING_DAY,
        "completed_job_points": POINTS_PER_COMPLETED_JOB,
        "verification_points_max": MAX_VERIFICATION_POINTS,
        "decay_grace_days": DECAY_GRACE_DAYS,
        "uptime_weight_pct": round(RELIABILITY_WEIGHTS["uptime_pct"] * 100, 1),
    }


# ── Human-facing copy (built from the numbers above) ──────────────────


def _gold_keep_delta() -> int:
    """How many more cents-on-the-dollar a Gold host keeps vs. an entry host.
    This is the concrete 'stay online → get N% more' payoff the incentive
    promises, computed from real commission tiers so it can never overstate."""
    rows = {r["tier"]: r for r in tier_reward_table()}
    return round(rows["gold"]["keep_pct"] - rows["bronze"]["keep_pct"])


def _diamond_keep() -> float:
    return {r["tier"]: r for r in tier_reward_table()}["diamond"]["keep_pct"]


def provider_welcome_email_text(display_name: str) -> str:
    """Plain text (paragraphs separated by blank lines) for the provider welcome
    email. _send_team_email splits on blank lines into styled <p> blocks, so keep
    each idea as its own paragraph."""
    gold_delta = _gold_keep_delta()
    uptime_weight = round(RELIABILITY_WEIGHTS["uptime_pct"] * 100)
    sla = abs(PENALTY_POINTS[PenaltyType.SLA_BREACH])
    return (
        f"Hi {display_name},\n\n"
        "Welcome — and thanks for putting your GPU on Xcelsior. Here's exactly "
        "what we expect and exactly how you get paid more for it, straight out, "
        "so there are no surprises.\n\n"
        f"THE ONE GOAL: aim for {RECOMMENDED_WEEKLY_HOURS} online hours a week. "
        "That's the sweet spot — an evenings-and-weekends rig clears it easily, "
        f"and even {MINIMUM_WEEKLY_HOURS} hours a week keeps you listed and "
        "earning. We are NOT asking you to run 24/7; predictable part-time beats "
        "erratic full-time every time.\n\n"
        "HOW SCORING WORKS, in three lines: (1) You earn points for every "
        "completed job and every day you host. (2) Verifying your account and "
        "hardware adds a big one-time boost. (3) Everything is then multiplied by "
        f"your reliability score — and {uptime_weight}% of that is simply whether "
        "you're online when you said you would be.\n\n"
        f"WHAT IT'S WORTH: climbing tiers means we take a smaller cut. A Gold host "
        f"keeps about {gold_delta}% more of every dollar than a new host, ranks "
        f"higher in search (so you win more jobs), and can charge a premium. Top "
        f"hosts keep {_diamond_keep():.0f}% of what they earn.\n\n"
        "THE ONE THING THAT HURTS EVERYONE: if you disappear in the middle of a "
        "job, that customer's work fails over to someone else — they wait, the "
        "whole network looks less reliable to buyers, and you take a "
        f"-{sla}-point SLA hit PLUS a drop in your reliability multiplier that "
        "scales your ENTIRE score down. One dropped week can cost you a tier. If "
        "you need to take your machine offline, just stop accepting new jobs "
        "first — no penalty for planned downtime.\n\n"
        "The full breakdown — every tier, every number — is on the Earn page. "
        "Reply to this email any time; we read every one."
    )


def provider_welcome_notification() -> tuple[str, str]:
    """(title, body) for the in-app inbox notification. Covers providers who sign
    up via OAuth and never get the email."""
    title = "You're a provider now — here's how to earn"
    body = (
        f"Aim for {RECOMMENDED_WEEKLY_HOURS} online hours a week. You earn points "
        "for every job and every day you host, multiplied by how reliable you are "
        "— so staying online when you said you would is what climbs you through "
        "the tiers. Higher tiers mean we take a smaller cut and you rank higher in "
        "search. The one rule that matters: don't vanish mid-job. If you need to "
        "go offline, stop accepting new jobs first — planned downtime is free, a "
        "dropped job is not. Open the Earn page for the full breakdown."
    )
    return title, body
