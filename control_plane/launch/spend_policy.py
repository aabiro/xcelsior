"""Standing-policy evaluation for a launch (§14.2, §15.4).

A launch always creates spend, so it always needs approval — the question is
*whose*. If the client has a standing ``mcp_client_policies`` row, the plan
is inside every ceiling and allowed capability, and the policy grants
auto-approval, the plan can be server-approved without bouncing a human
(``approval_mode = 'standing_policy'``). Otherwise a human must approve at
the dashboard (``approval_mode = 'human'``). A ``confirm:true`` from the
client is intent, never approval (§14.2) — it does not enter this decision.

Money is micro-CAD throughout, matching the quote and the ledger.

The rolling ceilings (``hourly_spend_max_micros``, ``daily_spend_max_micros``)
are held by live counters in :mod:`control_plane.launch.spend_counters`. This
module reads none of them itself: the caller passes what is already reserved
in the current windows as ``window_usage``, and a plan that would cross either
ceiling earns a human approval instead of an automatic one. That is a
read-time decision; the atomic reservation that actually holds the line is
taken at execute, so two plans racing for the last of a window cannot both
spend it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    # Does the plan violate a hard capability/ceiling in the policy?
    allowed: bool
    # Which approval path the plan takes: 'standing_policy' | 'human'.
    approval_mode: str
    # Human-readable reasons the plan is blocked or cannot auto-approve.
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "approval_mode": self.approval_mode,
            "reasons": list(self.reasons),
        }


def _security_mode(spec: dict[str, Any]) -> str:
    return "encrypted" if spec.get("encrypted_workspace") else "standard"


def evaluate(
    spec: dict[str, Any],
    *,
    estimate_micros: int,
    policy: dict[str, Any] | None,
    window_usage: dict[str, int] | None = None,
) -> PolicyDecision:
    """Evaluate a spec+estimate against an optional standing policy.

    Returns whether the plan is within the policy's hard limits and the
    approval mode it earns. With no policy the plan is not *blocked* (a human
    may still approve it) but it can never auto-approve.

    ``window_usage`` is ``{"hourly_micros": n, "daily_micros": n}`` already
    reserved against this policy. Omitted, the rolling ceilings are not
    judged here and only execute's reservation enforces them.
    """
    if policy is None:
        return PolicyDecision(
            allowed=True,
            approval_mode="human",
            reasons=["no standing policy; requires human approval"],
        )

    reasons: list[str] = []

    per_action = policy.get("per_action_max_micros")
    if per_action is not None and int(estimate_micros) > int(per_action):
        reasons.append(f"estimate {estimate_micros} exceeds per-action ceiling {per_action}")

    allowed_models = policy.get("allowed_gpu_models")
    gpu_model = str(spec.get("gpu_model") or "")
    if allowed_models and gpu_model and gpu_model not in allowed_models:
        reasons.append(f"gpu_model {gpu_model!r} not in allowed set")

    allowed_regions = policy.get("allowed_regions")
    region = str(spec.get("region") or "")
    if allowed_regions and region and region not in allowed_regions:
        reasons.append(f"region {region!r} not in allowed set")

    allowed_modes = policy.get("allowed_security_modes")
    mode = _security_mode(spec)
    if allowed_modes and mode not in allowed_modes:
        reasons.append(f"security mode {mode!r} not allowed")

    max_conc = policy.get("max_concurrency")
    num_gpus = int(spec.get("num_gpus") or 1)
    if max_conc is not None and num_gpus > int(max_conc):
        reasons.append(f"num_gpus {num_gpus} exceeds max_concurrency {max_conc}")

    if window_usage is not None:
        for window, column in (
            ("hourly", "hourly_spend_max_micros"),
            ("daily", "daily_spend_max_micros"),
        ):
            ceiling = policy.get(column)
            if ceiling is None:
                continue
            projected = int(window_usage.get(f"{window}_micros") or 0) + int(estimate_micros)
            if projected > int(ceiling):
                reasons.append(
                    f"{window} ceiling {ceiling} would be exceeded (projected {projected})"
                )

    within_limits = not reasons
    auto = bool(policy.get("auto_approve")) and within_limits
    return PolicyDecision(
        allowed=within_limits,
        approval_mode="standing_policy" if auto else "human",
        reasons=reasons,
    )
